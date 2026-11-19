"""Exercise installed Bellium resources without relying on a source checkout."""
import importlib
import json
import sys
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import bellium
from bellium.contracts import AuthorityMode
from bellium.knn.patch_inpaint import inpaint
from bellium.micro_nn.features import feature_names
from bellium.micro_nn.legacy import classify_legacy, list_legacy_models
from bellium.registry.catalog import list_specialists
from bellium.resources import model_path


def check_small_learning():
    """Exercise the new APIs through the installed package, with no source imports."""
    from bellium.knn import PreparedMemory, rerank_memory
    from bellium.micro_nn.evaluate import classification_metrics
    from bellium.micro_nn.mlp import load_mlp, save_mlp, validate_mlp
    from bellium.micro_nn.train import train_classifier, train_regressor
    from bellium.micro_nn.calibration import fit_temperature, predict_calibrated, probability_metrics
    from bellium.specialists.tool_router import classify_tool_need, route_tool

    query = {"domain": "bellium", "terms": "How can I find the image cutout for this background?"}
    assert rerank_memory(query, memory=PreparedMemory()).abstained
    phrases = rerank_memory(query, memory=PreparedMemory(profile="fr-en"))
    assert not phrases.abstained and phrases.output["top_id"] == "mem_cutout"
    assert phrases.output["lexical_profile"] == "fr-en"
    records = [{"id": name, "domain": "fr", "title": name,
                "terms": "mémoire réseau", "evidence": "attested"} for name in ("a", "b")]
    prepared = PreparedMemory(records)
    unicode_query = {"domain": "fr", "terms": "me\u0301moire re\u0301seau"}
    before = rerank_memory(unicode_query, memory=prepared)
    assert before.output["neighbors"][0]["similarity"] == 1.0
    records[0]["terms"] = "changed"
    assert rerank_memory(unicode_query, memory=prepared) == before
    features = {"has_code": 0.5, "has_files": 0.0, "has_error": 0.0,
                "wants_mutation": 0.0, "wants_external_state": 0.0, "criticality": 0.2}
    result = classify_tool_need({"features": features})
    assert result.abstained and result.output["reason"] == "outside_training_domain"
    veto = route_tool({"features": {**features, "wants_mutation": 0.5}})
    assert veto.output["tool"] == "escalate" and veto.output["micro_nn"] is None

    samples = [([0.0], 0), ([0.1], 0), ([0.9], 1), ([1.0], 1)]
    original = [(list(vector), label) for vector, label in samples]
    controls = {"epochs": 25, "patience": 3, "min_delta": 1000.0}
    classifier = train_classifier(samples, layer_sizes=[1, 3, 2],
                                  validation_samples=[([0.2], 0), ([0.8], 1)], **controls)
    assert samples == original and classifier["training"]["stopped_early"]
    assert classifier == train_classifier(samples, layer_sizes=[1, 3, 2],
                                         validation_samples=[([0.2], 0), ([0.8], 1)], **controls)
    metrics = classification_metrics(classifier, [([0.2], 0), ([0.8], 1)])
    assert metrics["class_support"] == [1, 1] and metrics["samples"] == 2
    assert metrics["accepted"] + metrics["abstained"] == 2
    regressor = train_regressor([([0.0], 1.0), ([1.0], 3.0)], layer_sizes=[1, 3, 1],
                                validation_samples=[([0.3], 1.6), ([0.7], 2.4)], **controls)
    assert regressor["training"]["stopped_early"]
    with TemporaryDirectory(prefix="bellium-installed-check-") as temporary:
        for name, model in (("classifier", classifier), ("regressor", regressor)):
            validate_mlp(model)
            path = Path(temporary) / f"{name}.json"
            save_mlp(path, model)
            assert load_mlp(path) == model
    toy = {"layers": [1, 2], "weights": [[4.0, -4.0]], "biases": [[0.0, 0.0]],
           "activations": ["softmax"]}
    temperature = fit_temperature(toy, [([1.0], 0)] * 40 + [([1.0], 1)] * 10)
    assert abs(predict_calibrated(toy, [1.0], temperature)[0] - 0.8) < 1e-5
    assert probability_metrics([[1000.0, -1000.0]], [1])["nll"] == 2000.0
    return {"prepared_memory": True, "fr_en_profile": True, "unicode_snapshot": True,
            "domain_abstention_and_veto": True, "validation_trainers": True,
            "classification_metrics": True, "model_roundtrip": True,
            "native_temperature_calibration": True}


def main():
    if "site-packages" not in str(Path(bellium.__file__).resolve()):
        raise RuntimeError("this check must run against an installed wheel")
    models = list(model_path().rglob("*.json"))
    if not models:
        raise RuntimeError("model resources were not bundled")
    for path in models:
        json.loads(path.read_text(encoding="utf-8"))
    for name in list_legacy_models():
        result = classify_legacy(name, dict.fromkeys(feature_names(name), 0.0))
        assert result.authority_mode is AuthorityMode.OBSERVE
    loaders = {
        "phoneme": "load_inventory", "pronunciation": "load_pronunciations",
        "grapheme_phoneme": "load_mappings", "cognate": "load_cognates",
        "prosody": "load_profiles", "panel_crop": "load_layouts",
        "speech_bubble": "load_bubbles", "tool_router": "load_exemplars",
        "visual_anomaly": "load_exemplars", "memory_rerank": "load_memory",
    }
    for module, loader in loaders.items():
        assert getattr(importlib.import_module("bellium.knn." + module), loader)()
    # Locate any newer k-NN default memories too; packaging must not depend on a
    # manually maintained list of model filenames.
    package = Path(bellium.__file__).resolve().parent
    for path in sorted((package / "knn").glob("*.py")):
        module = importlib.import_module("bellium.knn." + path.stem)
        for name, value in vars(module).items():
            if name.startswith("DEFAULT_") and isinstance(value, Path):
                assert value.is_file(), f"missing installed resource: {value}"
                assert package in value.resolve().parents
    result = inpaint([[(200, 200, 200)] * 16 for _ in range(16)], [[int(6 <= r < 9 and 6 <= c < 9) for c in range(16)] for r in range(16)])
    assert not result.abstained and result.output["filled"] == 9
    from bellium.specialists.tool_router import route_tool
    assert route_tool({"signals": {"mentions_secret": True}}).output["tool"] == "escalate"
    from bellium.knn.pronunciation import compare_pronunciation
    assert not compare_pronunciation({"language_id": "fixture-lab", "form": "pata", "ipa": ["p", "p", "a", "t", "a"]}).output["certified"]
    assert all(not spec.decision_eligible for spec in list_specialists())
    small_learning_checks = check_small_learning()
    for arguments, expected in ((["--help"], "usage:"),
                                (["filter", "--help"], "grayscale"),
                                (["route", "--tags", "inpaint", "--zero-tokens"], "bellium/inpaint_knn")):
        command = subprocess.run([sys.executable, "-I", "-m", "bellium.cli", *arguments],
                                 capture_output=True, text=True)
        assert command.returncode == 0, command.stderr
        assert expected in command.stdout
    if "--image" in sys.argv:
        from PIL import Image
        from bellium.cutout import extract_foreground
        assert extract_foreground(Image.new("RGBA", (20, 20), (255, 0, 0, 0))).image.getpixel((10, 10))[3] == 0
    print(json.dumps({"installed_package": bellium.__file__, "json_resources": len(models),
                      "legacy_models": len(list_legacy_models()), "registry_entries": len(list_specialists()),
                      "image_extra_checked": "--image" in sys.argv,
                      "small_learning_checks": small_learning_checks,
                      "cli_help_and_routing_checked": True, "passed": True}))


if __name__ == "__main__":
    main()
