"""Append-only, tamper-evident audit log.

Every prediction is written as one JSON line recording *what* the model said
and *everything needed to reproduce it*: model id and revision, prompt version
and hash, dataset hash, image hash, timing and code version.

Each record also stores the hash of the previous record (a simple hash chain,
like a mini ledger). If anyone edits or deletes a line later, ``verify`` will
report exactly where the chain breaks. In regulated settings you would also
ship these records to write-once storage; the chain makes tampering visible
even in a plain file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unversioned"


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["record_hash"] if last else GENESIS

    def append(self, record: dict[str, Any]) -> dict:
        rec = dict(record)
        rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        rec["prev_hash"] = self._prev
        rec["record_hash"] = hashlib.sha256(_canonical(rec).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._prev = rec["record_hash"]
        return rec


def verify(path: str | Path) -> tuple[bool, str]:
    """Re-compute the chain. Returns (ok, message)."""
    prev = GENESIS
    n = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            stored = rec.pop("record_hash", None)
            if rec.get("prev_hash") != prev:
                return False, f"line {i}: previous-hash link broken (a record was removed or reordered)"
            if hashlib.sha256(_canonical(rec).encode()).hexdigest() != stored:
                return False, f"line {i}: record contents were modified"
            prev = stored
            n += 1
    return True, f"OK -- {n} records, chain intact"
