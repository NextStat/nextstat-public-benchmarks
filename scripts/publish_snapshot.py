#!/usr/bin/env python3
"""Publish a benchmark snapshot locally (seed script).

Creates an immutable snapshot directory containing:
- suite outputs (e.g. `hep/hep_suite.json` + per-case results)
- a baseline manifest (`baseline_manifest.json`)
- schema validation (fails fast on invalid artifacts)

This is meant to be runnable by outsiders in the standalone benchmarks repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path


def git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_json(instance_path: Path, schema_path: Path) -> None:
    import jsonschema  # type: ignore

    schema = load_json(schema_path)
    inst = load_json(instance_path)
    jsonschema.validate(inst, schema)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", default="", help="Snapshot id (default: auto-generated).")
    ap.add_argument("--out-root", default="manifests/snapshots", help="Root directory for snapshots.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--suite",
        default="public-benchmarks-seed",
        help="Suite label recorded in snapshot_index.json (for discovery).",
    )
    ap.add_argument(
        "--nextstat-wheel",
        default="",
        help="Optional: path to the measured NextStat wheel (hashed into baseline_manifest.json).",
    )
    ap.add_argument("--hep", action="store_true", help="Run HEP suite.")
    ap.add_argument("--pharma", action="store_true", help="Run pharma suite.")
    ap.add_argument("--bayesian", action="store_true", help="Run Bayesian suite.")
    ap.add_argument(
        "--bayesian-backends",
        default="nextstat",
        help="Bayesian suite backends (comma-separated): nextstat,cmdstanpy,pymc (optional).",
    )
    ap.add_argument("--bayesian-n-chains", type=int, default=4)
    ap.add_argument("--bayesian-warmup", type=int, default=500)
    ap.add_argument("--bayesian-samples", type=int, default=1000)
    ap.add_argument("--bayesian-seed", type=int, default=0)
    ap.add_argument("--bayesian-max-treedepth", type=int, default=10)
    ap.add_argument("--bayesian-target-accept", type=float, default=0.8)
    ap.add_argument("--bayesian-init-jitter-rel", type=float, default=0.10)
    ap.add_argument("--ml", action="store_true", help="Run ML suite.")
    ap.add_argument(
        "--ml-backends",
        default="numpy,jax_jit_cpu,jax_jit_gpu",
        help="ML suite backends (comma-separated): numpy,jax_jit_cpu,jax_jit_gpu. Drop jax_jit_gpu for CPU-only runners.",
    )
    ap.add_argument("--fit", action="store_true", help="Also benchmark MLE fits where supported.")
    ap.add_argument("--fit-repeat", type=int, default=3)
    ap.add_argument(
        "--extra-pythonpath",
        action="append",
        default=[],
        help="Optional: prepend an extra path to PYTHONPATH for suite subprocesses (repeatable).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    snapshot_id = str(args.snapshot_id).strip()
    if not snapshot_id:
        ts = time.strftime("%Y%m%d-%H%M%S")
        snapshot_id = f"snapshot-{ts}-{git_short_sha(repo_root)}"

    snap_dir = out_root / snapshot_id
    if snap_dir.exists():
        raise SystemExit(f"snapshot dir already exists: {snap_dir}")
    snap_dir.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    extra_pp = [str(p).strip() for p in (args.extra_pythonpath or []) if str(p).strip()]
    if extra_pp:
        # Preserve order: user-provided extra paths first.
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = ":".join(extra_pp + ([existing] if existing else []))

    # Optional: copy the measured NextStat wheel into the snapshot directory so DOI/public
    # snapshots can be fully self-contained.
    nextstat_wheel_path = str(args.nextstat_wheel).strip()
    if nextstat_wheel_path:
        wheel_src = Path(nextstat_wheel_path).resolve()
        if not wheel_src.exists():
            raise SystemExit(f"nextstat wheel not found: {wheel_src}")
        wheel_dst = snap_dir / "nextstat_wheel.whl"
        if wheel_dst.exists():
            if wheel_dst.resolve() != wheel_src:
                raise SystemExit(f"snapshot already contains nextstat_wheel.whl: {wheel_dst}")
        else:
            shutil.copy2(wheel_src, wheel_dst)
        nextstat_wheel_path = str(wheel_dst)

    results: list[Path] = []

    # Defaults: keep the seed publisher lightweight (HEP + pharma) unless a suite is explicitly requested.
    run_any = bool(args.hep or args.pharma or args.bayesian or args.ml)
    run_hep = bool(args.hep or (not run_any))
    run_pharma = bool(args.pharma or (not run_any))
    run_bayesian = bool(args.bayesian)
    run_ml = bool(args.ml)

    if run_hep:
        hep_out = snap_dir / "hep"
        cmd = [
            sys.executable,
            "suites/hep/suite.py",
            "--out-dir",
            str(hep_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(hep_out / "hep_suite.json")
        # Generate a small README snippet for humans (does not replace machine JSONs).
        subprocess.check_call(
            [
                sys.executable,
                "suites/hep/report.py",
                "--suite",
                str(hep_out / "hep_suite.json"),
                "--format",
                "markdown",
                "--out",
                str(snap_dir / "README_snippet.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_pharma:
        pharma_out = snap_dir / "pharma"
        cmd = [
            sys.executable,
            "suites/pharma/suite.py",
            "--out-dir",
            str(pharma_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(pharma_out / "pharma_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/pharma/report.py",
                "--suite",
                str(pharma_out / "pharma_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_pharma.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_bayesian:
        bayes_out = snap_dir / "bayesian"
        cmd = [
            sys.executable,
            "suites/bayesian/suite.py",
            "--out-dir",
            str(bayes_out),
            "--backends",
            str(args.bayesian_backends),
            "--n-chains",
            str(int(args.bayesian_n_chains)),
            "--warmup",
            str(int(args.bayesian_warmup)),
            "--samples",
            str(int(args.bayesian_samples)),
            "--seed",
            str(int(args.bayesian_seed)),
            "--max-treedepth",
            str(int(args.bayesian_max_treedepth)),
            "--target-accept",
            str(float(args.bayesian_target_accept)),
            "--init-jitter-rel",
            str(float(args.bayesian_init_jitter_rel)),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(bayes_out / "bayesian_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/bayesian/report.py",
                "--suite",
                str(bayes_out / "bayesian_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_bayesian.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_ml:
        ml_out = snap_dir / "ml"
        cmd = [
            sys.executable,
            "suites/ml/suite.py",
            "--out-dir",
            str(ml_out),
            "--backends",
            str(args.ml_backends),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(ml_out / "ml_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/ml/report.py",
                "--suite",
                str(ml_out / "ml_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_ml.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    # Validate suite + case schemas.
    schema_case = repo_root / "manifests/schema/benchmark_result_v1.schema.json"
    schema_suite = repo_root / "manifests/schema/benchmark_suite_result_v1.schema.json"
    schema_pharma_case = repo_root / "manifests/schema/pharma_benchmark_result_v1.schema.json"
    schema_pharma_suite = repo_root / "manifests/schema/pharma_benchmark_suite_result_v1.schema.json"
    schema_bayes_case = repo_root / "manifests/schema/bayesian_benchmark_result_v1.schema.json"
    schema_bayes_suite = repo_root / "manifests/schema/bayesian_benchmark_suite_result_v1.schema.json"
    schema_ml_case = repo_root / "manifests/schema/ml_benchmark_result_v1.schema.json"
    schema_ml_suite = repo_root / "manifests/schema/ml_benchmark_suite_result_v1.schema.json"
    for r in results:
        obj = load_json(r)
        sv = str(obj.get("schema_version", ""))
        if sv == "nextstat.benchmark_suite_result.v1":
            validate_json(r, schema_suite)
            for e in obj.get("cases", []):
                validate_json((r.parent / e["path"]).resolve(), schema_case)
        elif sv == "nextstat.pharma_benchmark_suite_result.v1":
            validate_json(r, schema_pharma_suite)
            for e in obj.get("cases", []):
                validate_json((r.parent / e["path"]).resolve(), schema_pharma_case)
        elif sv == "nextstat.bayesian_benchmark_suite_result.v1":
            validate_json(r, schema_bayes_suite)
            for e in obj.get("cases", []):
                validate_json((r.parent / e["path"]).resolve(), schema_bayes_case)
        elif sv == "nextstat.ml_benchmark_suite_result.v1":
            validate_json(r, schema_ml_suite)
            for e in obj.get("cases", []):
                validate_json((r.parent / e["path"]).resolve(), schema_ml_case)
        elif sv == "nextstat.benchmark_result.v1":
            validate_json(r, schema_case)
        elif sv == "nextstat.pharma_benchmark_result.v1":
            validate_json(r, schema_pharma_case)
        elif sv == "nextstat.bayesian_benchmark_result.v1":
            validate_json(r, schema_bayes_case)
        elif sv == "nextstat.ml_benchmark_result.v1":
            validate_json(r, schema_ml_case)

    # Write baseline manifest referencing produced results; dataset list will be derived from results.
    baseline_path = snap_dir / "baseline_manifest.json"
    cmd = [
        sys.executable,
        "scripts/write_baseline_manifest.py",
        "--snapshot-id",
        snapshot_id,
        "--out",
        str(baseline_path),
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if nextstat_wheel_path:
        cmd.extend(["--nextstat-wheel", nextstat_wheel_path])
    for r in results:
        cmd.extend(["--result", str(r)])
    subprocess.check_call(cmd, cwd=str(repo_root), env=env)

    validate_json(baseline_path, repo_root / "manifests/schema/baseline_manifest_v1.schema.json")

    # Write an index for the full snapshot artifact set (hash inventory).
    index_path = snap_dir / "snapshot_index.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/write_snapshot_index.py",
            "--suite",
            str(args.suite),
            "--artifacts-dir",
            str(snap_dir),
            "--out",
            str(index_path),
            "--snapshot-id",
            snapshot_id,
        ],
        cwd=str(repo_root),
        env=env,
    )
    validate_json(index_path, repo_root / "manifests/schema/snapshot_index_v1.schema.json")

    print(f"snapshot ready: {snap_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
