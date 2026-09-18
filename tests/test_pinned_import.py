import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def test_import_reads_pinned_blobs_and_never_overwrites_a_version(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_legacy_botte.py"
    spec = importlib.util.spec_from_file_location("pinned_import", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.PIPE)
    git("init", "-q")
    for name in module.FILES:
        path = source / "skills" / "botte_nn" / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": name}), encoding="utf-8")
    (source / "LICENSE").write_text("test license", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Bellium test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD").decode().strip()
    first = module.FILES[0]
    original = git("cat-file", "blob", f"{revision}:skills/botte_nn/models/{first}")
    (source / "skills" / "botte_nn" / "models" / first).write_text("dirty data", encoding="utf-8")
    destination = tmp_path / "imported"
    module.import_models(source, revision, destination)
    assert (destination / first).read_bytes() == original
    assert json.loads((destination / "PROVENANCE.json").read_text())["source_revision"] == revision
    with pytest.raises(FileExistsError):
        module.import_models(source, revision, destination)
    assert (destination / first).read_bytes() == original
