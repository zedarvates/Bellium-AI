"""Locate bundled models in an installed wheel or an editable source checkout."""
from pathlib import Path


def model_path(*parts: str) -> Path:
    package = Path(__file__).resolve().parent
    bundled = package / "_data" / "models"
    root = bundled if bundled.is_dir() else package.parent / "models"
    return root.joinpath(*parts)
