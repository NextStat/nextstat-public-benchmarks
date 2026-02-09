"""Write a small snapshot index JSON for published artifacts.

This is intentionally stdlib-only so it can run in CI without extra deps.

The index is meant for *discovery* and *auditability*:
- suite name + snapshot id
- git/workflow context (when available)
      - list of artifact files with size + sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Artifact:
    relpath: str
    bytes: int
    sha256: str


def _iter_artifacts(root: Path, *, exclude_dirs: set[str]) -> Iterable[Artifact]:
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        top = rel.split("/", 1)[0] if rel else rel
        if top in exclude_dirs:
            continue
        st = p.stat()
        yield Artifact(relpath=rel, bytes=st.st_size, sha256=_sha256_file(p))


def _env_get(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out


def _default_snapshot_id() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    sha = os.environ.get("GITHUB_SHA")
    if run_id and attempt and sha:
        return f"gha-{run_id}-{attempt}-{sha[:12]}"
    if sha:
        return f"local-{sha[:12]}"
    return "local-unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="suite name (e.g. hep, pharma, or a composite label)")
    ap.add_argument("--artifacts-dir", required=True, help="directory containing artifacts to index")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--snapshot-id",
        default="",
        help="optional: snapshot id (default: derived from GitHub env or git SHA)",
    )
    args = ap.parse_args()

    root = Path(args.artifacts_dir).resolve()
    if not root.exists():
        raise SystemExit(f"artifacts dir not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"artifacts dir is not a directory: {root}")

    snapshot_id = args.snapshot_id.strip() or _default_snapshot_id()

    # Exclude implementation detail dirs that may exist inside artifact bundles.
    exclude_dirs = {"mplconfig", ".gnupg", ".replication", "__pycache__"}
    artifacts = list(_iter_artifacts(root, exclude_dirs=exclude_dirs))

    doc: dict[str, Any] = {
        "schema_version": "nextstat.snapshot_index.v1",
        "generated_at": _utc_now_iso(),
        "snapshot_id": snapshot_id,
        "suite": args.suite,
        "git": {
            "sha": os.environ.get("GITHUB_SHA") or "",
            "ref": os.environ.get("GITHUB_REF") or "",
            "repository": os.environ.get("GITHUB_REPOSITORY") or "",
        },
        "workflow": _env_get(
            [
                "GITHUB_WORKFLOW",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "RUNNER_OS",
                "RUNNER_ARCH",
            ]
        ),
        "artifacts": [{"path": a.relpath, "bytes": a.bytes, "sha256": a.sha256} for a in artifacts],
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
