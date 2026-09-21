import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import pytest

pytest.importorskip("PIL")


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def corpus(root):
    raw = b"source data"
    (root / "source.bin").write_bytes(raw)
    (root / "sample.bin").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {"schema": "bellium-independent-compression-corpus/1", "sources": [
        {"id": "one", "path": "source.bin", "sha256": digest, "split": "development",
         "license": "CC0", "license_evidence": "fixture"}], "samples": [
        {"id": "sample", "path": "sample.bin", "sha256": digest, "bytes": len(raw),
         "source_id": "one", "split": "development", "kind": "stream"}]}
    return manifest


def save(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_fingerprints_reject_changed_corpus(tmp_path):
    manifest = corpus(tmp_path)
    save(tmp_path, manifest)
    evaluator = module("evaluate_compression_corpus")
    assert len(evaluator.load_corpus(tmp_path)["samples"]) == 1
    (tmp_path / "source.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        evaluator.load_corpus(tmp_path)


def test_source_cannot_cross_the_split(tmp_path):
    manifest = corpus(tmp_path)
    manifest["samples"][0]["split"] = "holdout"
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="crosses"):
        module("evaluate_compression_corpus").load_corpus(tmp_path)


def test_renaming_a_source_cannot_bypass_split_isolation(tmp_path):
    manifest = corpus(tmp_path)
    alias = {**manifest["sources"][0], "id": "alias", "split": "holdout"}
    manifest["sources"].append(alias)
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="identical source"):
        module("evaluate_compression_corpus").load_corpus(tmp_path)


def test_corpus_rejects_path_escape_and_missing_rights(tmp_path):
    manifest = corpus(tmp_path)
    manifest["samples"][0]["path"] = "../outside.bin"
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="contained"):
        module("evaluate_compression_corpus").load_corpus(tmp_path)
    manifest = corpus(tmp_path)
    manifest["sources"][0]["license"] = ""
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="license"):
        module("evaluate_compression_corpus").load_corpus(tmp_path)


def test_ply_adaptation_is_explicit_and_rejects_wrong_sizes():
    names = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
             "scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
             "f_rest_0"]
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
              + "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    ply = header + struct.pack("<15f", 1, 2, 3, 1, 0, 0, 0,
                              math.log(2), math.log(2), math.log(2), 0, 0, 0, 0, 10)
    prepare = module("prepare_compression_corpus")
    result, count = prepare.canonical_splats(ply, 0, count=1)
    assert count == 1
    assert len(result) == 56
    assert struct.unpack("<14f", result) == pytest.approx(
        (1, 2, 3, 0, 0, 0, 1, 2, 2, 2, .5, .5, .5, .5))
    with pytest.raises(ValueError, match="size"):
        prepare.canonical_splats(ply[:-1], 0, count=1)


def test_candidate_cannot_pass_when_larger_than_the_fixed_baseline():
    def measure(size):
        return {"packet_bytes": size, "encode_ms_median": 1, "decode_ms_median": 1}
    row = {"best_fixed": "left", "methods": {"left": measure(100), "micro-nn": measure(101)}}
    assert not module("evaluate_compression_corpus").spatial_gate([row])["development_passed"]


def test_complete_evaluation_passes_image_options_and_records_selection(tmp_path):
    manifest = corpus(tmp_path)
    manifest["samples"][0].update(kind="image", width=11, height=1, mode="L")
    heldout = b"dlrow olleh"
    (tmp_path / "holdout.bin").write_bytes(heldout)
    fingerprint = hashlib.sha256(heldout).hexdigest()
    manifest["sources"].append({"id": "holdout", "path": "holdout.bin", "sha256": fingerprint,
                                "split": "holdout", "license": "CC0", "license_evidence": "fixture"})
    manifest["samples"].append({"id": "holdout", "path": "holdout.bin", "sha256": fingerprint,
                                "source_id": "holdout", "split": "holdout", "kind": "image",
                                "width": 11, "height": 1, "mode": "L", "bytes": 11})
    manifest["limitations"] = ["test fixture"]
    save(tmp_path, manifest)
    output = tmp_path / "report.json"
    module("evaluate_compression_corpus").evaluate(tmp_path, output, 1)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["all_exact"]
    assert report["sample_count"] == 2
    assert sum(report["auto_selections"].values()) == 2
