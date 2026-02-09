#!/usr/bin/env python3
"""Export the seed public benchmarks harness into a standalone repo directory.

This is a convenience wrapper around "copy the seed folder into a new repo",
excluding local outputs/caches (snapshots, tmp, __pycache__, etc.).

Example (from the NextStat monorepo root):

  python benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
    --out /path/to/nextstat-public-benchmarks

Then:
  cd /path/to/nextstat-public-benchmarks
  git init
  git add -A
  git commit -m "Initial public benchmarks harness"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _ignore_for_export(src_root: Path):
    def _fn(dirpath: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        d = Path(dirpath)
        rel = d.resolve().relative_to(src_root.resolve())

        for name in names:
            # macOS AppleDouble / resource fork files often appear when exporting
            # from a macOS filesystem. They must never end up in the standalone repo.
            if name.startswith("._"):
                ignored.add(name)
                continue
            if name in {".DS_Store", "__pycache__"}:
                ignored.add(name)
                continue
            if name in {".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                ignored.add(name)
                continue
            if name.startswith(".venv"):
                ignored.add(name)
                continue
            if name.endswith(".pyc"):
                ignored.add(name)
                continue

            # Root-level local output/caches.
            if rel == Path(".") and name in {"out", "tmp"}:
                ignored.add(name)
                continue

            # Snapshot outputs are intentionally excluded from a seed export.
            if rel == Path("manifests") and name == "snapshots":
                ignored.add(name)
                continue

            # Zenodo packager output folders (local): `out`, `out_*`.
            if rel == Path("zenodo") and name.startswith("out"):
                ignored.add(name)
                continue

        return ignored

    return _fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for the exported standalone repo.")
    ap.add_argument(
        "--with-github-workflows",
        action="store_true",
        help="Also copy `ci/*.yml` into `.github/workflows/` in the exported repo.",
    )
    args = ap.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    out_dir = Path(str(args.out)).expanduser().resolve()

    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output dir: {out_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, out_dir, ignore=_ignore_for_export(src_root), dirs_exist_ok=False)

    if args.with_github_workflows:
        ci_dir = out_dir / "ci"
        wf_dir = out_dir / ".github" / "workflows"
        wf_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(ci_dir.glob("*.yml")):
            shutil.copy2(p, wf_dir / p.name)

    print(f"exported seed repo: {out_dir}")
    print("next steps:")
    print(f"  cd {out_dir}")
    print("  git init && git add -A && git commit -m \"Initial public benchmarks harness\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
