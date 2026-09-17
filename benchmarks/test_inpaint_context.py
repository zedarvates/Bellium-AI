"""Replay committed quality evidence; time and allocation figures remain local."""
import json
from pathlib import Path
import unittest

from benchmarks.inpaint_context import evaluate, fixtures, quality, sha


class InpaintEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = evaluate()
        root = Path(__file__).resolve().parents[1]
        cls.recorded = json.loads((root / "docs/evidence/inpaint-context-v1.json").read_text(encoding="utf-8"))

    def test_report_replays_fixed_inputs_sources_and_outputs(self):
        for field in ("schema", "baseline_revision", "dataset_sha256", "source_sha256_lf", "parameters",
                      "context_only_parameters", "fixture_identities", "paired_by_family"):
            self.assertEqual(self.current[field], self.recorded[field], field)
        for current, recorded in zip(self.current["cases"], self.recorded["cases"]):
            self.assertEqual({k: v for k, v in current.items() if k != "latency_ms"},
                             {k: v for k, v in recorded.items() if k != "latency_ms"})
        self.assertEqual(len(self.current["cases"]), len(self.recorded["cases"]))

    def test_abstention_is_separate_from_quality(self):
        for row in self.current["cases"]:
            if not row["completed"]:
                self.assertIsNone(row["quality"])
                self.assertEqual(row["filled_pixels"], 0)
            self.assertTrue(row["outside_unchanged"])
            self.assertTrue(row["source_unchanged"])
        # The observed gradient regression must remain visible in the frozen comparison.
        self.assertEqual(self.current["paired_by_family"]["gradient"]["worse"], 8)

    def test_reference_metric_is_zero_and_damage_is_detected(self):
        case = fixtures()[0]
        self.assertEqual(quality(case["reference"], case["reference"], case["mask"])["masked_rgb_mae"], 0)
        self.assertGreater(quality(case["reference"], case["damaged"], case["mask"])["masked_rgb_mae"], 0)


if __name__ == "__main__":
    unittest.main()
