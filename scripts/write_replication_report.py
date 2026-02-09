"""Write a machine-readable replication report from two snapshot indices.

Intended to be run by third parties to produce an auditable, signable report:
- references the original snapshot_id + suite
- includes their rerun snapshot_id + suite
- records hash mismatches for overlapping artifact paths

stdlib-only by design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class Index:
    path: Path
    doc: dict[str, Any]

    @property
    def suite(self) -> str:
        return str(self.doc.get("suite", ""))

    @property
    def snapshot_id(self) -> str:
        return str(self.doc.get("snapshot_id", ""))

    @property
    def artifacts(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for a in self.doc.get("artifacts", []) or []:
            if isinstance(a, dict) and isinstance(a.get("path"), str):
                out[a["path"]] = a
        return out

    @property
    def sha256(self) -> str:
        return _sha256_bytes(json.dumps(self.doc, sort_keys=True).encode("utf-8"))


def _load_index(p: Path) -> Index:
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema_version") != "nextstat.snapshot_index.v1":
        raise SystemExit(f"unexpected snapshot_index schema_version in {p}")
    return Index(path=p, doc=doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-index", required=True, help="path to snapshot_index.json for the original snapshot")
    ap.add_argument("--replica-index", required=True, help="path to snapshot_index.json for the rerun (replication)")
    ap.add_argument("--out", required=True, help="output replication report JSON path")
    ap.add_argument("--notes", default="", help="optional: free-form notes (one line)")
    args = ap.parse_args()

    original = _load_index(Path(args.original_index).resolve())
    replica = _load_index(Path(args.replica_index).resolve())

    overlap = sorted(set(original.artifacts.keys()) & set(replica.artifacts.keys()))
    mismatches: list[dict[str, Any]] = []
    for path in overlap:
        oa = original.artifacts[path]
        ra = replica.artifacts[path]
        if oa.get("sha256") != ra.get("sha256"):
            mismatches.append(
                {
                    "path": path,
                    "original_sha256": oa.get("sha256"),
                    "replica_sha256": ra.get("sha256"),
                    "original_bytes": oa.get("bytes"),
                    "replica_bytes": ra.get("bytes"),
                }
            )

    doc: dict[str, Any] = {
        "schema_version": "nextstat.replication_report.v1",
        "generated_at": _utc_now_iso(),
        "original": {
            "snapshot_id": original.snapshot_id,
            "suite": original.suite,
            "snapshot_index_sha256": original.sha256,
        },
        "replica": {
            "snapshot_id": replica.snapshot_id,
            "suite": replica.suite,
            "snapshot_index_sha256": replica.sha256,
        },
        "comparison": {"overlap_paths": len(overlap), "mismatches": mismatches, "ok": len(mismatches) == 0},
        "notes": args.notes.strip(),
        "replica_snapshot_index": replica.doc,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
