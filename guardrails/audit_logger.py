"""
CRRA Lab C4 — Audit Logger

Appends one JSON object per line to logs/audit_trail.jsonl and prints each entry.
The file is opened in append mode on every write, never "w": re-running the
pipeline adds to the trail instead of replacing it. Delete the file by hand if
you want a clean demo.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, path: Path = DEFAULT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Ties together every entry written by one run of the program
        self.run_id = uuid.uuid4().hex[:8]

    def log(self, contract_id: str, actor: str, action: str, detail: str, **data) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "contract_id": contract_id,
            "actor": actor,
            "action": action,
            "detail": detail,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        print(f"[AUDIT] {actor}: {action} — {detail}")
        return entry
