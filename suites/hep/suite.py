#!/usr/bin/env python3
"""HEP suite runner (seed).

Runs multiple NLL parity+timing cases and writes:
- per-case JSON files (benchmark_result_v1)
- a suite index JSON (benchmark_suite_result_v1)
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

import pyhf

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_workspace(path: Path) -> dict:
    return json.loads(path.read_text())


def make_synthetic_shapesys_workspace(n_bins: int) -> dict:
    signal = {
        "name": "signal",
        "data": [5.0] * n_bins,
        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    }
    bkg = {
        "name": "background",
        "data": [50.0] * n_bins,
        "modifiers": [{"name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0] * n_bins}],
    }
    return {
        "channels": [{"name": "c", "samples": [signal, bkg]}],
        "observations": [{"name": "c", "data": [53.0] * n_bins}],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


def sha256_json_obj(obj: dict) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def run_case(
    case_id: str,
    *,
    workspace_path: Path,
    measurement_name: str,
    deterministic: bool,
    out_path: Path,
    dataset_id: str,
    dataset_sha256: str | None,
    workspace_obj: dict | None = None,
    fit: bool = False,
    fit_repeat: int = 3,
) -> int:
    run_py = Path(__file__).resolve().parent / "run.py"

    # For generated/synthetic cases, we write a temporary workspace file; for file-based datasets,
    # we pass the dataset file path directly to preserve portability.
    tmp_ws: Path | None = None
    ws_arg_path = workspace_path
    if workspace_obj is not None:
        tmp_ws = out_path.with_suffix(".workspace.json")
        tmp_ws.write_text(json.dumps(workspace_obj, indent=2, sort_keys=True) + "\n")
        ws_arg_path = tmp_ws

    args = [
        sys.executable,
        str(run_py),
        "--case",
        case_id,
        "--workspace",
        str(ws_arg_path),
        "--measurement-name",
        measurement_name,
        "--out",
        str(out_path),
        "--dataset-id",
        dataset_id,
    ]
    if dataset_sha256:
        args.extend(["--dataset-sha256", dataset_sha256])
    if deterministic:
        args.append("--deterministic")
    if fit:
        args.append("--fit")
        args.extend(["--fit-repeat", str(int(fit_repeat))])

    try:
        p = subprocess.run(args)
        return int(p.returncode)
    finally:
        if tmp_ws is not None:
            try:
                tmp_ws.unlink()
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--sizes",
        default="2,16,64,256",
        help="Comma-separated synthetic bin counts (shapesys) to include.",
    )
    ap.add_argument(
        "--cases",
        default="simple,complex,synthetic",
        help="Comma-separated case groups: simple,complex,synthetic.",
    )
    ap.add_argument("--fit", action="store_true", help="Also benchmark full MLE fits.")
    ap.add_argument("--fit-repeat", type=int, default=3, help="Fit timing repeats (wall-clock).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    deterministic = bool(args.deterministic)
    fit = bool(args.fit)
    fit_repeat = int(args.fit_repeat)

    case_groups = [x.strip() for x in args.cases.split(",") if x.strip()]
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    suite_cases: list[dict] = []
    repo_root = Path(__file__).resolve().parents[2]
    ds_dir = Path(__file__).resolve().parent / "datasets"
    if "simple" in case_groups:
        ws_path = ds_dir / "simple_workspace.json"
        suite_cases.append(
            {
                "case_id": "simple_workspace_nll",
                "workspace_path": ws_path,
                "workspace_obj": None,
                "measurement": "GaussExample",
                "dataset_id": os.path.relpath(ws_path, repo_root),
                "dataset_sha256": None,
            }
        )
    if "complex" in case_groups:
        complex_path = ds_dir / "complex_workspace.json"
        if complex_path.exists():
            suite_cases.append(
                {
                    "case_id": "complex_workspace_nll",
                    "workspace_path": complex_path,
                    "workspace_obj": None,
                    "measurement": "measurement",
                    "dataset_id": os.path.relpath(complex_path, repo_root),
                    "dataset_sha256": None,
                }
            )
    if "synthetic" in case_groups:
        for n in sizes:
            ws = make_synthetic_shapesys_workspace(n)
            suite_cases.append(
                {
                    "case_id": f"synthetic_shapesys_{n}",
                    "workspace_path": ds_dir / "synthetic.workspace.json",
                    "workspace_obj": ws,
                    "measurement": "m",
                    "dataset_id": f"generated:synthetic_shapesys_{n}",
                    "dataset_sha256": sha256_json_obj(ws),
                }
            )

    index_cases = []
    n_ok = 0
    worst_abs = 0.0
    worst_case = "none"

    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        rc = run_case(
            case_id,
            workspace_path=c["workspace_path"],
            workspace_obj=c["workspace_obj"],
            measurement_name=c["measurement"],
            deterministic=deterministic,
            out_path=out_path,
            dataset_id=c["dataset_id"],
            dataset_sha256=c["dataset_sha256"],
            fit=fit,
            fit_repeat=fit_repeat,
        )
        if rc != 0:
            # Still include it in the index if it exists, but fail overall.
            pass
        obj = json.loads(out_path.read_text())
        sha = sha256_file(out_path)
        parity_ok = bool(obj.get("parity", {}).get("ok", False))
        abs_diff = float(obj.get("parity", {}).get("abs_diff", 0.0))
        speedup = float(obj.get("timing", {}).get("speedup_pyhf_over_nextstat", 0.0))
        if parity_ok:
            n_ok += 1
        if abs_diff >= worst_abs:
            worst_abs = abs_diff
            worst_case = case_id
        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "parity_ok": parity_ok,
                "abs_diff": abs_diff,
                "speedup_pyhf_over_nextstat": speedup,
            }
        )

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pyhf_version": pyhf.__version__,
        "nextstat_version": nextstat.__version__,
    }

    index = {
        "schema_version": "nextstat.benchmark_suite_result.v1",
        "suite": "hep",
        "deterministic": deterministic,
        "meta": meta,
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "worst_abs_diff": worst_abs,
            "worst_case": worst_case,
        },
    }

    index_path = out_dir / "hep_suite.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    return 0 if n_ok == len(index_cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
