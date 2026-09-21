"""Bounded JSONL reads and appends for local, single-writer advisory memories."""
import json
from pathlib import Path

MAX_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024


def read_tail(path: Path, max_entries: int) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            start = max(0, handle.tell() - MAX_BYTES)
            handle.seek(start)
            data = handle.read(MAX_BYTES)
    except FileNotFoundError:
        return []
    if start:
        # The first record may start before the bounded window. Never parse a fragment.
        data = data.partition(b"\n")[2]
    return data.decode("utf-8", errors="replace").splitlines()[-max_entries:]


def append_record(path: Path, record: dict) -> None:
    line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(line) > MAX_RECORD_BYTES:
        raise ValueError("asset record exceeds the size budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        if handle.tell() + len(line) > MAX_BYTES:
            raise ValueError("asset ledger is full; rotate it before recording more outcomes")
        handle.write(line)
