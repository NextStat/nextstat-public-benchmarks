#!/usr/bin/env python3
"""Bayesian suite runner (seed).

Runs a small set of NUTS baselines intended to cover different geometry regimes:

1) Simple HistFactory (few params)
2) Logistic regression (GLM-style)
3) Hierarchical random intercepts (non-centered)

Writes:
- per-case JSON files (bayesian_benchmark_result_v1)
- a suite index JSON (bayesian_benchmark_suite_result_v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--backends",
        default="nextstat",
        help="Comma-separated list of backends: nextstat,cmdstanpy,pymc (optional).",
    )
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--init-jitter-rel", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    cases = [
        {"case_id": "histfactory_simple_8p", "model": "histfactory_simple"},
        {"case_id": "glm_logistic_regression", "model": "glm_logistic"},
        {"case_id": "hier_random_intercept_non_centered", "model": "hier_random_intercept"},
    ]

    index_cases = []
    n_ok = 0
    n_warn = 0
    n_failed = 0

    worst_case = "none"
    worst_score = float("inf")

    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    allowed = {"nextstat", "cmdstanpy", "pymc"}
    for b in backends:
        if b not in allowed:
            raise SystemExit(f"unknown backend: {b} (allowed: {sorted(allowed)})")

    for c in cases:
        case_id = c["case_id"]
        for backend in backends:
            out_name = f"{case_id}.json" if backends == ["nextstat"] else f"{case_id}__{backend}.json"
            out_path = cases_dir / out_name
            cmd = [
                sys.executable,
                str(run_py),
                "--case",
                case_id,
                "--model",
                c["model"],
                "--backend",
                backend,
                "--out",
                str(out_path),
                "--n-chains",
                str(int(args.n_chains)),
                "--warmup",
                str(int(args.warmup)),
                "--samples",
                str(int(args.samples)),
                "--seed",
                str(int(args.seed)),
                "--max-treedepth",
                str(int(args.max_treedepth)),
                "--target-accept",
                str(float(args.target_accept)),
                "--init-jitter-rel",
                str(float(args.init_jitter_rel)),
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            p = subprocess.run(cmd)

            # Always include artifact if it exists (even if non-zero), but reflect status.
            try:
                obj = json.loads(out_path.read_text())
            except Exception:
                obj = {}

            status = str(obj.get("status") or "failed")
            if status == "ok":
                n_ok += 1
            elif status == "warn":
                n_warn += 1
            else:
                n_failed += 1

            wall = _maybe_float(obj.get("timing", {}).get("wall_time_s"))
            min_ess_bulk = _maybe_float(obj.get("diagnostics_summary", {}).get("min_ess_bulk"))
            min_ess_tail = _maybe_float(obj.get("diagnostics_summary", {}).get("min_ess_tail"))
            max_r_hat = _maybe_float(obj.get("diagnostics_summary", {}).get("max_r_hat"))
            min_ess_bulk_per_sec = _maybe_float(obj.get("timing", {}).get("ess_bulk_per_sec", {}).get("min"))

            # Worst-case heuristic: failed always worst; otherwise lowest ESS/sec when present.
            score = -1.0 if status == "failed" else float(min_ess_bulk_per_sec) if min_ess_bulk_per_sec is not None else float("inf")
            if score < worst_score:
                worst_score = score
                worst_case = f"{case_id}::{backend}"

            sha = sha256_file(out_path) if out_path.exists() else "0" * 64
            index_cases.append(
                {
                    "case": case_id,
                    "backend": backend,
                    "path": os.path.relpath(out_path, out_dir),
                    "sha256": sha,
                    "status": status,
                    "wall_time_s": wall,
                    "min_ess_bulk": min_ess_bulk,
                    "min_ess_tail": min_ess_tail,
                    "max_r_hat": max_r_hat,
                    "min_ess_bulk_per_sec": min_ess_bulk_per_sec,
                }
            )

            # If the runner itself failed hard, bubble up.
            if int(p.returncode) != 0 and status == "failed":
                # Continue collecting the rest for debugging, but preserve suite failure.
                pass

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
    }

    index = {
        "schema_version": "nextstat.bayesian_benchmark_suite_result.v1",
        "suite": "bayesian",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_failed": n_failed,
            "worst_case": worst_case,
        },
    }

    index_path = out_dir / "bayesian_suite.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
