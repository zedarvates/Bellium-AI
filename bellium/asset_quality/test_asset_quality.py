"""Contract, leakage, abstention, model fit and real-file CLI checks."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from bellium.asset_quality.__main__ import inspect_asset
from bellium.asset_quality.benchmark import run_benchmark
from bellium.asset_quality.dataset import dataset_digest, load_dataset, synthetic_rows, validate_splits
from bellium.asset_quality.features import extract_features, rule_advice, validate_vector
from bellium.asset_quality.models import Predictors, digest, train_bundle, validate_bundle


def sprite():
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill=(40, 80, 180, 255))
    return image


class AlphaQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.train = [row for row in cls.rows if row["split"] == "train"]
        cls.bundle = train_bundle(cls.train, epochs=3)

    def test_measured_alpha_geometry(self):
        features = extract_features(sprite())
        self.assertEqual(features, [0.25, 0, 0, 0.25, 0, 0])
        self.assertEqual(rule_advice(features)["label"], "candidate")
        image = sprite()
        image.putpixel((0, 0), (255, 0, 0, 255))
        self.assertAlmostEqual(extract_features(image)[2], 1 / 124)
        self.assertIn("foreground_touches_canvas", rule_advice(extract_features(image))["reasons"])

    def test_empty_and_uncertain_masks_require_review(self):
        empty = Image.new("RGBA", (32, 32))
        self.assertEqual(rule_advice(extract_features(empty))["label"], "review")
        faded = sprite()
        faded.putalpha(faded.getchannel("A").point(lambda value: 160 if value else 0))
        self.assertIn("broad_partial_alpha", rule_advice(extract_features(faded))["reasons"])

    def test_rgb_is_not_silently_treated_as_a_cutout(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            extract_features(Image.new("RGB", (32, 32)))
        with self.assertRaises(ValueError):
            extract_features(Image.new("RGBA", (1, 1)))

    def test_nonfinite_bool_and_wrong_shape_features_rejected(self):
        for values in ([0] * 5, [True] * 6, [math.nan] * 6, [math.inf] * 6, [1.1] * 6):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_vector(values)

    def test_training_refuses_validation_or_test_rows(self):
        with self.assertRaisesRegex(ValueError, "training"):
            train_bundle(self.rows)

    def test_fit_is_reproducible_and_networks_have_declared_size(self):
        repeat = train_bundle(self.train, epochs=3)
        self.assertEqual(repeat, self.bundle)
        for name, size in (("nano", 7), ("micro", 65)):
            model = repeat["networks"][name]
            count = sum(len(row) for row in model["w1"]) + len(model["b1"]) + len(model["w2"]) + 1
            self.assertEqual(count, size)

    def test_parameters_learn_from_labels(self):
        # Separably labeled inputs verify fitting behavior, independently of fixture metrics.
        rows = [{"features": [value] * 6, "split": "train", "label": label,
                 "sha256": hashlib.sha256(f"{value}-{i}".encode()).hexdigest()}
                for value, label in ((0.0, "candidate"), (1.0, "review")) for i in range(5)]
        predictors = Predictors(train_bundle(rows, epochs=100))
        for name in ("knn", "nano", "micro"):
            self.assertEqual(predictors.predict(name, [0.0] * 6)["label"], "candidate")
            self.assertEqual(predictors.predict(name, [1.0] * 6)["label"], "review")

    def test_group_and_pixel_split_leakage_rejected(self):
        for field in ("group", "sha256"):
            rows = copy.deepcopy(self.rows)
            heldout = next(row for row in rows if row["split"] == "test")
            heldout[field] = rows[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "leaks"):
                validate_splits(rows)

    def test_bundle_tampering_and_incompatible_contract_rejected(self):
        broken = copy.deepcopy(self.bundle)
        broken["networks"]["nano"]["b2"] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_bundle(broken)
        for field, value in (("authority", "active"), ("feature_names", ["aesthetic"] * 6)):
            broken = copy.deepcopy(self.bundle)
            broken[field] = value
            broken["bundle_sha256"] = digest({k: v for k, v in broken.items() if k != "bundle_sha256"})
            with self.assertRaises(ValueError):
                validate_bundle(broken)

    def test_self_neighbors_and_duplicates_cannot_manufacture_support(self):
        bundle = copy.deepcopy(self.bundle)
        bundle["support"] = [bundle["support"][0]] * 5
        bundle["training_digest"] = digest(bundle["support"])
        bundle["bundle_sha256"] = digest({k: v for k, v in bundle.items() if k != "bundle_sha256"})
        predictors = Predictors(bundle)
        for name in ("knn", "nano", "micro"):
            self.assertEqual(predictors.predict(name, bundle["support"][0]["features"])["label"], "abstain")
        bundle["support"] = [{**self.bundle["support"][0], "sha256": hashlib.sha256(str(i).encode()).hexdigest()}
                             for i in range(3)]
        bundle["training_digest"] = digest(bundle["support"])
        bundle["bundle_sha256"] = digest({k: v for k, v in bundle.items() if k != "bundle_sha256"})
        predictors = Predictors(bundle)
        result = predictors.predict("knn", bundle["support"][0]["features"], sha256=bundle["support"][0]["sha256"])
        self.assertEqual(result["label"], "abstain")

    def test_out_of_support_abstains_for_all_learned_models(self):
        predictors = Predictors(self.bundle)
        for name in ("knn", "nano", "micro"):
            self.assertEqual(predictors.predict(name, [1, 1, 1, 1, 1, 1])["label"], "abstain")

    def test_report_separates_coverage_and_correctness(self):
        report, bundle = run_benchmark(self.rows, provenance="test synthetic", epochs=2)
        self.assertEqual(report["split_counts"], {"train": 240, "validation": 60, "test": 60, "ood": 12})
        self.assertEqual(report["bundle_sha256"], bundle["bundle_sha256"])
        for model in report["models"].values():
            result = model["by_split"]["test"]
            self.assertEqual(result["answered"] + result["abstentions"], 60)
            self.assertLessEqual(result["correct"], result["answered"])
            self.assertEqual(result["coverage"], result["answered"] / 60)
            self.assertTrue(all(case["split"] != "train" for case in model["cases"]))

    def test_committed_evidence_matches_sources_and_predictions(self):
        root = Path(__file__).resolve().parents[2]
        evidence = root / "docs" / "evidence"
        report = json.loads((evidence / "alpha-qa-synthetic-v1.json").read_text(encoding="utf-8"))
        bundle = json.loads((evidence / "alpha-qa-models-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(report["dataset_sha256"], dataset_digest(self.rows))
        self.assertEqual(report["bundle_sha256"], bundle["bundle_sha256"])
        for name, expected in report["implementation_sha256"].items():
            source = (Path(__file__).parent / name).read_text(encoding="utf-8").encode()
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, name)
        predictors = Predictors(bundle)
        by_id = {row["id"]: row for row in self.rows}
        for name, model in report["models"].items():
            for case in model["cases"]:
                row = by_id[case["id"]]
                prediction = rule_advice(row["features"]) if name == "rules" else predictors.predict(
                    name, row["features"], sha256=row["sha256"])
                self.assertEqual(case["expected"], row["label"])
                self.assertEqual(case["split"], row["split"])
                self.assertEqual(case["predicted"], prediction["label"])

    def test_real_file_inspection_preserves_bytes_and_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sprite.png"
            sprite().save(path)
            payload = path.read_bytes()
            report = inspect_asset(path, self.bundle, hashlib.sha256(payload).hexdigest())
            self.assertEqual(report["advice"], report["rules"]["label"])
            self.assertFalse(report["production_authorized"])
            self.assertFalse(report["acted"])
            self.assertEqual(path.read_bytes(), payload)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                inspect_asset(path, expected_sha256="0" * 64)

    def test_dataset_local_files_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"schema": "bellium.alpha-dataset/v1", "provenance": "human:test", "cases": []}
            for i, (split, label) in enumerate((s, l) for s in ("train", "validation", "test")
                                              for l in ("candidate", "review")):
                image = sprite()
                image.putpixel((10, 10), (40 + i, 80, 180, 255))
                image.save(root / f"sprite-{i}.png")
                data["cases"].append({"id": f"case-{i}", "group": f"group-{i}", "split": split,
                                      "label": label, "reason": "human test label", "path": f"sprite-{i}.png"})
            manifest = root / "dataset.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            rows, provenance = load_dataset(manifest)
            self.assertEqual(len(rows), 6)
            self.assertEqual(provenance, "human:test")
            data["cases"][0]["path"] = "../outside.png"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "directory"):
                load_dataset(manifest)

    def test_cli_json_error_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sprite.png"
            sprite().save(path)
            payload = path.read_bytes()
            cmd = [sys.executable, "-m", "bellium.asset_quality", "inspect", str(path)]
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            success = subprocess.run(cmd, capture_output=True, text=True, env=env)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout)["authority"], "consultative")
            refusal = subprocess.run(cmd + ["--output", str(path)], capture_output=True, text=True, env=env)
            self.assertEqual(refusal.returncode, 2)
            self.assertEqual(json.loads(refusal.stderr)["status"], "invalid_input")
            self.assertEqual(path.read_bytes(), payload)
            path.write_bytes(b"not an image")
            broken = subprocess.run(cmd, capture_output=True, text=True, env=env)
            self.assertEqual(broken.returncode, 2)
            self.assertEqual(json.loads(broken.stderr)["status"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
