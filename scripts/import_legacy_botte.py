#!/usr/bin/env python3
"""Import exact Git blobs into a new versioned directory; never copy dirty weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = [
    "anomaly_detector.json", "binary_router.json", "cloud_escalation_predictor.json",
    "compressibility_predictor.json", "context_pruning_predictor.json", "effort_classifier.json",
    "error_classifier.json", "response_length_predictor.json", "semantic_cache_hit_predictor.json",
    "skip_agent_predictor.json", "tool_call_predictor.json",
]


def import_models(source_repo: Path, revision: str, destination: Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full 40-character commit SHA")
    source_repo = source_repo.resolve()
    def blob(path):
        return subprocess.check_output(
            ["git", "-C", str(source_repo), "cat-file", "blob", f"{revision}:{path}"],
            stderr=subprocess.PIPE,
        )
    # Resolve and read every blob before creating any output.
    resolved = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", "--verify", revision + "^{commit}"],
        text=True, stderr=subprocess.PIPE,
    ).strip()
    data = {name: blob("skills/botte_nn/models/" + name) for name in FILES}
    for payload in data.values():
        json.loads(payload)
    license_bytes = blob("LICENSE")
    destination = (destination or REPO / "models" / "micro_nn" / ("legacy-botte-" + resolved[:12])).resolve()
    if destination.exists():
        raise FileExistsError("import destination already exists; choose a new version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".bellium-import-", dir=destination.parent)).resolve()
    try:
        models = []
        for name, payload in data.items():
            (stage / name).write_bytes(payload)
            models.append({"file": name, "sha256": hashlib.sha256(payload).hexdigest(),
                           "bytes": len(payload), "source_path": "skills/botte_nn/models/" + name})
        (stage / "LICENSE.MIT").write_bytes(license_bytes)
        provenance = {
            "schema": "bellium.legacy-botte-provenance/v1",
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_repo": "zedarvates/botte-secrete", "source_revision": resolved,
            "license": "MIT", "authority_mode": "observe", "decision_eligible": False,
            "models": models, "notes": ["Read from pinned Git blobs; import grants no authority."],
        }
        (stage / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        if stage.parent != destination.parent or destination.exists():
            raise ValueError("destination changed during import")
        os.rename(stage, destination)
    finally:
        # Only the temporary directory created immediately above may be removed.
        if stage.exists() and stage.parent == destination.parent and stage.name.startswith(".bellium-import-"):
            shutil.rmtree(stage)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(import_models(args.source_repo, args.revision, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
