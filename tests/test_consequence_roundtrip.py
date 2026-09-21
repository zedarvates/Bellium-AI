"""Normalized precedents keep the same semantics across disk and injected inputs."""
import json

import pytest

from bellium.knn.consequence import SCHEMA, load_precedents, retrieve_precedent
from bellium.specialists.consequence import classify_consequence, predict_consequence


def _state():
    return {"reversible": True, "scope_ratio": 0.15, "touches_protected": False,
            "has_backup": True, "dry_run_available": True, "blast_radius": 0.1,
            "idempotent": True, "recent_failures": 0.05}


def test_normalized_disk_and_injected_precedents_agree(tmp_path):
    records = [{"id": f"sample-{i}", "family": "filesystem", "label": "safe",
                "verified": True, "source": " [REDACTED] ", "features": _state()} for i in range(3)]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"schema": SCHEMA, "items": records}), encoding="utf-8")
    loaded = load_precedents(path)
    assert all(item["source"] == "[REDACTED]" for item in loaded)
    assert all(item["features"]["has_backup"] == 1.0 for item in loaded)
    query = {"family": "filesystem", "features": _state()}
    direct = retrieve_precedent(query, memory=records)
    from_disk = retrieve_precedent(query, memory=loaded)
    assert direct == from_disk
    assert not direct.abstained


def test_supported_boolean_flags_work_in_all_tiers():
    neural = classify_consequence(_state())
    hybrid = predict_consequence({"family": "filesystem", "state": _state(),
                                  "constraints": {"irreversible": False, "has_backup": True,
                                                  "dry_run_available": True}})
    assert neural.output["label"] == "safe"
    assert hybrid.output["status"] == "suggest"
    assert hybrid.output["requires_confirmation"] is True


@pytest.mark.parametrize("value", ["false", "junk", [], {}])
def test_flag_junk_is_rejected_consistently(value):
    state = _state()
    state["has_backup"] = value
    with pytest.raises(ValueError):
        classify_consequence(state)
    with pytest.raises(ValueError):
        retrieve_precedent({"family": "filesystem", "features": state})
