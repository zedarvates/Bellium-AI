from pathlib import Path
import subprocess
import sys

import pytest


BLOCK_PILLOW = """
import importlib.abc
import sys
class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise ModuleNotFoundError("Pillow unavailable for this test", name='PIL')
sys.meta_path.insert(0, NoPillow())
from bellium.cli import main
sys.exit(main(sys.argv[1:]))
"""


def _run(*args):
    return subprocess.run([sys.executable, "-c", BLOCK_PILLOW, *args],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)


@pytest.mark.parametrize("args", [("--help",), ("filter", "--help"), ("route", "--help")])
def test_help_does_not_require_optional_pillow(args):
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("args,expected", [
    (("route", "--tags", "cutout", "alpha"), "bellium/cutout"),
    (("route", "--tags", "inpaint", "--zero-tokens"), "bellium/inpaint_knn"),
])
def test_non_image_routing_remains_available_without_pillow(args, expected):
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize("args", [
    ("cutout", "missing.png"),
    ("filter", "grayscale", "missing.png", "-o", "unused-output.png"),
    ("texture", "missing.png"),
    ("inspect-frame", "missing.png"),
])
def test_image_commands_explain_missing_extra_without_traceback(args):
    result = _run(*args)
    assert result.returncode == 2
    assert "bellium-ai[image]" in result.stderr
    assert "Traceback" not in result.stderr
