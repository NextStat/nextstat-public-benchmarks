#!/usr/bin/env python3
"""Update the snapshot registry from a freshly produced snapshot directory.

Registry goal:
- machine-readable discovery of published artifacts
- stable, commit-backed log for the website/docs to consume

This script is intentionally stdlib-only so it can run in CI without extra deps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _run_url() -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""


def _index_sha(index: dict[str, Any], *, name: str) -> str:
    for a in index.get("artifacts") or []:
        if not isinstance(a, dict):
            continue
        if a.get("path") == name and isinstance(a.get("sha256"), str):
            return str(a["sha256"])
    raise SystemExit(f"snapshot_index.json missing sha256 for {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir", required=True, help="Snapshot directory containing snapshot_index.json.")
    ap.add_argument(
        "--registry",
        default="manifests/registry/snapshot_registry.json",
        help="Registry JSON path to update (created if missing).",
    )
    args = ap.parse_args()

    snap_dir = Path(args.snapshot_dir).resolve()
    index_path = snap_dir / "snapshot_index.json"
    if not index_path.exists():
        raise SystemExit(f"missing snapshot index: {index_path}")

    idx = _load_json(index_path)
    if idx.get("schema_version") != "nextstat.snapshot_index.v1":
        raise SystemExit("snapshot_index.json has unexpected schema_version")

    snapshot_id = str(idx.get("snapshot_id") or "").strip()
    suite = str(idx.get("suite") or "").strip()
    generated_at = str(idx.get("generated_at") or "").strip()
    if not snapshot_id or not suite or not generated_at:
        raise SystemExit("snapshot_index.json missing required fields (snapshot_id/suite/generated_at)")

    entry: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "suite": suite,
        "generated_at": generated_at,
        "git": idx.get("git") if isinstance(idx.get("git"), dict) else {},
        "workflow": idx.get("workflow") if isinstance(idx.get("workflow"), dict) else {},
        "links": {"github_run_url": _run_url()},
        "artifacts": {
            "baseline_manifest_sha256": _index_sha(idx, name="baseline_manifest.json"),
            "snapshot_index_sha256": _sha256_file(index_path),
        },
    }

    reg_path = Path(args.registry).resolve()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    if reg_path.exists():
        reg = _load_json(reg_path)
    else:
        reg = {"schema_version": "nextstat.snapshot_registry.v1", "updated_at": "", "snapshots": []}

    if reg.get("schema_version") != "nextstat.snapshot_registry.v1":
        raise SystemExit("registry has unexpected schema_version")

    snapshots = reg.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []

    # Replace existing entry with same snapshot_id, otherwise append.
    out: list[dict[str, Any]] = []
    replaced = False
    for e in snapshots:
        if isinstance(e, dict) and str(e.get("snapshot_id") or "") == snapshot_id:
            out.append(entry)
            replaced = True
        else:
            out.append(e if isinstance(e, dict) else {})
    if not replaced:
        out.append(entry)

    # Sort newest-first by generated_at (lex order works for ISO timestamps).
    out = [e for e in out if e.get("snapshot_id")]
    out.sort(key=lambda e: str(e.get("generated_at") or ""), reverse=True)

    reg["updated_at"] = _utc_now_iso()
    reg["snapshots"] = out
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

