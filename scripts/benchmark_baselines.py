"""Bounded synthetic benchmarks. Scores measure rule fidelity, not field quality."""
import argparse
import importlib.util
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bellium.knn.color_cutout import cutout  # noqa: E402
from bellium.micro_nn.mlp import load_mlp, predict_mlp  # noqa: E402


def benchmark():
    root = Path(__file__).resolve().parents[1]
    report = {"scope": "synthetic-only; no production quality or RAM certification",
              "python": sys.version, "platform": platform.platform(), "cutout": [], "models": {}}
    for size in (32, 64, 128):
        image = [[(230, 40, 30) if size//4 <= r < 3*size//4 and size//4 <= c < 3*size//4
                  else (10, 20, 200) for c in range(size)] for r in range(size)]
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = cutout(image)
            times.append(time.perf_counter() - start)
            assert not result.abstained
        report["cutout"].append({"size": size, "median_seconds": statistics.median(times), "runs": times})
    for name in ("inpaint_router", "tool_router"):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"train_{name}.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        model = load_mlp(root / "models" / "micro_nn" / name.replace("_", "-") / "v0.json")
        samples = trainer.make_samples(1000, random.Random(20260915))
        matrix = [[0] * len(model["labels"]) for _ in model["labels"]]
        times = []
        abstained = 0
        for vector, label in samples:
            start = time.perf_counter()
            probabilities = predict_mlp(model, vector)
            times.append(time.perf_counter() - start)
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            matrix[label][predicted] += 1
            abstained += max(probabilities) < 0.55
        report["models"][name] = {
            "samples": len(samples), "seed": 20260915, "confusion_true_rows": matrix,
            "accuracy": sum(matrix[i][i] for i in range(len(matrix))) / len(samples),
            "abstentions": abstained, "median_inference_seconds": statistics.median(times),
            "label_rule_accuracy_by_construction": 1.0,
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark(), indent=2), encoding="utf-8")
    print(args.output)
