#!/usr/bin/env python3
"""Pharma suite runner (seed).

Runs multiple generated PK/NLME cases and writes:
- per-case JSON files (pharma_benchmark_result_v1)
- a suite index JSON (pharma_benchmark_suite_result_v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit-repeat", type=int, default=3)
    ap.add_argument("--pk-nobs", default="8,32,128", help="Comma-separated obs counts for PK.")
    ap.add_argument("--nlme-nsub", default="4,16", help="Comma-separated subject counts for NLME.")
    ap.add_argument("--nlme-nobs", type=int, default=6, help="Obs per subject for NLME.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    pk_sizes = [int(x.strip()) for x in str(args.pk_nobs).split(",") if x.strip()]
    nlme_sizes = [int(x.strip()) for x in str(args.nlme_nsub).split(",") if x.strip()]

    run_py = Path(__file__).resolve().parent / "run.py"
    repo_root = Path(__file__).resolve().parents[2]

    suite_cases: list[dict] = []
    for n in pk_sizes:
        suite_cases.append(
            {
                "case_id": f"pk_1c_oral_nobs_{n}",
                "args": ["--model", "pk", "--n-obs", str(n)],
            }
        )
    for n in nlme_sizes:
        suite_cases.append(
            {
                "case_id": f"nlme_1c_oral_nsub_{n}",
                "args": ["--model", "nlme", "--n-subjects", str(n), "--n-obs-per-subject", str(int(args.nlme_nobs))],
            }
        )

    index_cases = []
    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable,
            str(run_py),
            "--case",
            case_id,
            "--out",
            str(out_path),
            "--seed",
            str(int(args.seed)),
            *c["args"],
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd)

        obj = json.loads(out_path.read_text())
        sha = sha256_file(out_path)
        model = obj.get("model", {})
        timing = obj.get("timing", {}).get("nll_time_s_per_call", {})
        fit = obj.get("fit") or {}
        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "n_subjects": int(model.get("n_subjects", 0)),
                "n_obs": int(model.get("n_obs", 0)),
                "n_params": int(model.get("n_params", 0)),
                "nll_time_s_per_call_nextstat": float(timing.get("nextstat", 0.0)),
                "fit_status": str(fit.get("status", "")) if isinstance(fit, dict) else "",
                "fit_time_s_nextstat": float((fit.get("time_s", {}) or {}).get("nextstat", 0.0))
                if isinstance(fit, dict)
                else 0.0,
            }
        )

    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "nextstat_version": nextstat.__version__}
    index = {
        "schema_version": "nextstat.pharma_benchmark_suite_result.v1",
        "suite": "pharma",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases)},
    }
    (out_dir / "pharma_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

