"""Exercise installed Bellium resources without relying on a source checkout."""
import importlib
import json
import sys
from pathlib import Path

import bellium
from bellium.contracts import AuthorityMode
from bellium.knn.patch_inpaint import inpaint
from bellium.micro_nn.features import feature_names
from bellium.micro_nn.legacy import classify_legacy, list_legacy_models
from bellium.registry.catalog import list_specialists
from bellium.resources import model_path


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
    if "--image" in sys.argv:
        from PIL import Image
        from bellium.cutout import extract_foreground
        assert extract_foreground(Image.new("RGBA", (20, 20), (255, 0, 0, 0))).image.getpixel((10, 10))[3] == 0
    print(json.dumps({"installed_package": bellium.__file__, "json_resources": len(models),
                      "legacy_models": len(list_legacy_models()), "registry_entries": len(list_specialists()),
                      "image_extra_checked": "--image" in sys.argv, "passed": True}))


if __name__ == "__main__":
    main()
