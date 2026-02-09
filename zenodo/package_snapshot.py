#!/usr/bin/env python3
"""Package a snapshot directory for Zenodo/DOI publication (seed tool).

Creates:
- `<snapshot_id>.tar.gz` archive
- `<snapshot_id>.tar.gz.sha256` checksum file
- `zenodo_deposition.json` metadata derived from `zenodo/zenodo.json` + baseline manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot-dir",
        dest="snapshot_dir",
        default="",
        help="Snapshot directory containing baseline_manifest.json",
    )
    ap.add_argument(
        "--snapshot",
        dest="snapshot_dir",
        default="",
        help="Alias for --snapshot-dir (deprecated).",
    )
    ap.add_argument("--out-dir", default="zenodo/out", help="Output directory for archive + metadata")
    ap.add_argument("--template", default="zenodo/zenodo.json", help="Zenodo JSON template path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not str(args.snapshot_dir).strip():
        raise SystemExit("missing required argument: --snapshot-dir (or deprecated alias: --snapshot)")
    snap_dir = Path(args.snapshot_dir).resolve()
    out_dir_in = Path(args.out_dir)
    out_dir = (out_dir_in if out_dir_in.is_absolute() else (repo_root / out_dir_in)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = snap_dir / "baseline_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing baseline_manifest.json in {snap_dir}")

    index_path = snap_dir / "snapshot_index.json"
    if not index_path.exists():
        raise SystemExit(f"missing snapshot_index.json in {snap_dir} (run scripts/publish_snapshot.py first)")

    manifest = load_json(manifest_path)
    snapshot_id = str(manifest.get("snapshot_id") or snap_dir.name)
    harness_sha = str((manifest.get("harness") or {}).get("git_commit", "unknown"))
    nextstat_version = str((manifest.get("nextstat") or {}).get("version", "unknown"))
    nextstat_wheel_sha256 = str((manifest.get("nextstat") or {}).get("wheel_sha256", ""))
    nextstat_source = (manifest.get("nextstat") or {}).get("source") or {}
    nextstat_source_repo = str(nextstat_source.get("repo", "") or "")
    nextstat_source_ref = str(nextstat_source.get("ref", "") or "")
    nextstat_source_sha = str(nextstat_source.get("git_commit", "") or "")

    archive_name = f"{snapshot_id}.tar.gz"
    archive_path = out_dir / archive_name
    if archive_path.exists():
        raise SystemExit(f"archive already exists: {archive_path}")

    # Archive the snapshot dir contents under a top-level folder named `snapshot_id/`.
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(snap_dir, arcname=snapshot_id)

    digest = sha256_file(archive_path)
    (out_dir / f"{archive_name}.sha256").write_text(f"{digest}  {archive_name}\n")

    template_in = Path(args.template)
    template_path = (template_in if template_in.is_absolute() else (repo_root / template_in)).resolve()
    meta = load_json(template_path) if template_path.exists() else {}
    desc = meta.get("description", "")
    parts = [
        f"{desc}\n\n",
        f"Snapshot id: {snapshot_id}\n",
        f"Harness commit: {harness_sha}\n",
        f"NextStat version: {nextstat_version}\n",
    ]
    if nextstat_wheel_sha256:
        parts.append(f"NextStat wheel sha256: {nextstat_wheel_sha256}\n")
    if nextstat_source_repo or nextstat_source_ref or nextstat_source_sha:
        parts.append("NextStat source:\n")
        if nextstat_source_repo:
            parts.append(f"  - repo: {nextstat_source_repo}\n")
        if nextstat_source_ref:
            parts.append(f"  - ref: {nextstat_source_ref}\n")
        if nextstat_source_sha:
            parts.append(f"  - git_commit: {nextstat_source_sha}\n")
    parts.append(f"Archive sha256: {digest}\n")
    desc2 = "".join(parts)
    meta["title"] = f"NextStat Public Benchmarks: {snapshot_id}"
    meta["description"] = desc2
    (out_dir / "zenodo_deposition.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(str(archive_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
