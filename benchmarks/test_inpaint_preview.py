"""Replay fixed preview evidence, including hidden-detail losses and abstention."""
import json
from pathlib import Path
import unittest

from benchmarks.inpaint_preview import evaluate


class PreviewEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = evaluate()
        root = Path(__file__).resolve().parents[1]
        cls.recorded = json.loads((root / "docs/evidence/inpaint-preview-v1.json").read_text(encoding="utf-8"))

    def test_inputs_sources_methods_outputs_and_quality_replay(self):
        for key in ("schema", "authority", "baseline_revision", "seed", "parameters",
                    "pre_evaluation_preview_sha256", "source_sha256_lf", "reference_hashes_disjoint"):
            self.assertEqual(self.current[key], self.recorded[key], key)
        local_measurements = {"latency_p50_ms", "latency_p95_ms", "python_allocation_peak_probe_bytes"}
        for split in ("development_48", "new_identities_96"):
            current, recorded = self.current[split], self.recorded[split]
            for key in ("dataset_sha256", "fixture_identities", "paired_by_family"):
                self.assertEqual(current[key], recorded[key], (split, key))
            for method in current["summary"]:
                self.assertEqual({k: v for k, v in current["summary"][method].items() if k not in local_measurements},
                                 {k: v for k, v in recorded["summary"][method].items() if k not in local_measurements})
            self.assertEqual(len(current["cases"]), len(recorded["cases"]))
            for a, b in zip(current["cases"], recorded["cases"]):
                self.assertEqual({k: v for k, v in a.items() if k != "latency_ms"},
                                 {k: v for k, v in b.items() if k != "latency_ms"})

    def test_fallback_and_abstention_preserve_their_contract(self):
        for split in ("development_48", "new_identities_96"):
            for row in self.current[split]["cases"]:
                for key in ("source_unchanged", "mask_unchanged", "outside_unchanged", "alpha_unchanged", "mode_unchanged"):
                    self.assertTrue(row[key])
                if not row["completed"]:
                    self.assertIsNone(row["quality"])
                    self.assertIsNone(row["max_masked_rgb_error"])
                    self.assertEqual(row["filled_pixels"], 0)
                gate = row["interpolation_gate"]
                if gate is not None and not gate["accepted"]:
                    self.assertTrue(row["fallback_matches_baseline"])

    def test_new_smooth_fields_improve_and_old_gradients_are_exact(self):
        new = self.current["new_identities_96"]
        for family in ("affine_integer", "affine_quantized", "bilinear"):
            self.assertEqual(new["paired_by_family"][family]["improved"], 8)
            for row in new["cases"]:
                if row["family"] == family and row["method"] == "preview_v1":
                    self.assertLessEqual(row["max_masked_rgb_error"], 1)
        old = self.current["development_48"]["paired_by_family"]["gradient"]
        self.assertEqual(old["improved"], 8)
        self.assertEqual(old["preview_mean_masked_mae"], 0)

    def test_hidden_detail_errors_and_regressions_cannot_be_hidden(self):
        new = self.current["new_identities_96"]
        hidden = new["paired_by_family"]["hidden_detail"]
        self.assertEqual(hidden["interpolated"], 8)
        self.assertEqual(hidden["interpolation_candidates_error_above_one"], 8)
        self.assertEqual(hidden["worse"], 4)
        self.assertGreater(hidden["preview_mean_masked_mae"], 0.39)
        self.assertEqual(new["summary"]["preview_v1"]["abstained"], 8)
        self.assertEqual(new["summary"]["preview_v1"]["interpolated"], 40)


if __name__ == "__main__":
    unittest.main()
