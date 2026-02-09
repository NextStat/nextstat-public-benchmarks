#!/usr/bin/env python3
"""Create a rerun snapshot and a draft signed replication report (seed tool).

This does not perform cryptographic signing by default; it produces:
- rerun snapshot dir
- snapshot_comparison.json (machine diff)
- signed_report_draft.md (human template filled with what we can infer)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", required=True, help="Published snapshot dir (contains baseline_manifest.json)")
    ap.add_argument("--rerun-id", default="rerun-local", help="Snapshot id for the rerun")
    ap.add_argument("--out-root", default="tmp/replication_bundles", help="Output root for bundles")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit-repeat", type=int, default=3)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    published_dir = Path(args.published).resolve()
    published_manifest = load_json(published_dir / "baseline_manifest.json")

    out_root_in = Path(args.out_root)
    bundle_root = (out_root_in if out_root_in.is_absolute() else (repo_root / out_root_in)).resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = bundle_root / f"replication-{args.rerun_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1) Create rerun snapshot.
    rerun_snapshots_root = bundle_dir / "rerun_snapshots"
    rerun_snapshots_root.mkdir(parents=True, exist_ok=True)
    publish_py = repo_root / "scripts/publish_snapshot.py"
    cmd = [
        sys.executable,
        str(publish_py),
        "--snapshot-id",
        str(args.rerun_id),
        "--out-root",
        str(rerun_snapshots_root),
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.fit:
        cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
    subprocess.check_call(cmd, cwd=str(repo_root))

    rerun_dir = rerun_snapshots_root / str(args.rerun_id)
    rerun_manifest = load_json(rerun_dir / "baseline_manifest.json")

    # 2) Machine comparison.
    compare_py = repo_root / "replication/compare_snapshots.py"
    comparison_path = bundle_dir / "snapshot_comparison.json"
    subprocess.check_call(
        [
            sys.executable,
            str(compare_py),
            "--a",
            str(published_dir),
            "--b",
            str(rerun_dir),
            "--out",
            str(comparison_path),
        ],
        cwd=str(repo_root),
    )

    # 3) Draft signed report.
    tmpl = (repo_root / "replication/signed_report_template.md").read_text()
    txt = tmpl
    txt = txt.replace("- Snapshot id:", f"- Snapshot id: {published_manifest.get('snapshot_id','')}")
    txt = txt.replace("- Harness repo:", f"- Harness repo: {((published_manifest.get('harness') or {}).get('repo',''))}")
    txt = txt.replace(
        "- Harness commit SHA:", f"- Harness commit SHA: {((published_manifest.get('harness') or {}).get('git_commit',''))}"
    )
    txt = txt.replace(
        "- NextStat version / wheel hash (if provided):",
        f"- NextStat version / wheel hash (if provided): {((published_manifest.get('nextstat') or {}).get('version',''))} / {((published_manifest.get('nextstat') or {}).get('wheel_sha256',''))}",
    )
    txt = txt.replace(
        "- `rerun_baseline_manifest.json`:",
        f"- `rerun_baseline_manifest.json`: {rerun_dir / 'baseline_manifest.json'}",
    )
    txt += "\n## Seed Tool Outputs\n\n"
    txt += f"- Published snapshot dir: `{published_dir}`\n"
    txt += f"- Rerun snapshot dir: `{rerun_dir}`\n"
    txt += f"- Snapshot comparison JSON: `{comparison_path}`\n"
    txt += "\n"
    (bundle_dir / "signed_report_draft.md").write_text(txt)

    print(str(bundle_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
