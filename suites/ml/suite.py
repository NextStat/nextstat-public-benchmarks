#!/usr/bin/env python3
"""ML suite runner (seed).

Goal: publishable measurements for "compile vs execution" style ML workloads.

Seed scope:
- measures cold-start distributions using fresh Python processes
- measures warm-call throughput in a single process
- optional JAX backend (skipped/warn if not installed)

Artifacts:
- per-case JSON (`ml_benchmark_result_v1`)
- suite index JSON (`ml_benchmark_suite_result_v1`)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_obj(obj: dict) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return float(xs[0])
    # Linear interpolation on index.
    k = (len(xs) - 1) * float(p)
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    a = xs[i]
    b = xs[j]
    t = k - i
    return a * (1.0 - t) + b * t


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "median": _pctl(values, 0.5),
        "p95": _pctl(values, 0.95),
    }


def _try_import(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        v = getattr(mod, "__version__", "unknown")
        return True, str(v)
    except Exception:
        return False, ""


def _rm_tree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def _prepend_cuda_bin_for_ptxas(env: dict[str, str]) -> None:
    """Ensure a sufficiently new `ptxas` is visible for JAX GPU compilation.

    Some hosts ship an older `ptxas` in `/usr/bin` that cannot assemble PTX
    produced by newer JAX/XLA releases. Prefer CUDA toolkit bins if present.
    """
    candidates = [
        "/usr/local/cuda/bin",
        "/usr/local/cuda-12.6/bin",
        "/usr/local/cuda-12/bin",
    ]
    for c in candidates:
        p = Path(c) / "ptxas"
        if p.exists():
            env["PATH"] = f"{c}:{env.get('PATH','')}"
            return


def run_worker(
    *,
    run_py: Path,
    out_path: Path,
    env: dict[str, str],
    backend: str,
    n: int,
    dtype: str,
    warm_iters: int,
) -> dict:
    cmd = [
        sys.executable,
        str(run_py),
        "--backend",
        ("jax_jit" if backend.startswith("jax_jit") else backend),
        "--workload",
        "matmul",
        "--n",
        str(int(n)),
        "--dtype",
        dtype,
        "--warm-iters",
        str(int(warm_iters)),
        "--platform",
        ("gpu" if backend == "jax_jit_gpu" else "cpu"),
        "--out",
        str(out_path),
    ]
    p = subprocess.run(cmd, env=env)
    try:
        obj = json.loads(out_path.read_text())
    except Exception:
        obj = {"status": "failed", "reason": "no_output"}
    obj["_returncode"] = int(p.returncode)
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--cold-runs", type=int, default=5, help="Fresh-process runs per case.")
    ap.add_argument("--warm-iters", type=int, default=30, help="Warm-call iterations.")
    ap.add_argument(
        "--cases",
        default="matmul_512_f32,matmul_1024_f32",
        help="Comma-separated case ids (seed): matmul_512_f32,matmul_1024_f32",
    )
    ap.add_argument(
        "--backends",
        default="numpy,jax_jit_cpu,jax_jit_gpu",
        help="Comma-separated backends: numpy,jax_jit_cpu,jax_jit_gpu",
    )
    ap.add_argument(
        "--cache-policy",
        default="process_cold",
        help="Seed policy: process_cold (fresh process). Disk cache control is best-effort.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    backends = [x.strip() for x in str(args.backends).split(",") if x.strip()]
    cases_raw = [x.strip() for x in str(args.cases).split(",") if x.strip()]

    # Determine optional dependency availability once (best-effort; fresh processes may differ).
    has_jax, jax_v = _try_import("jax")

    suite_entries: list[dict] = []
    n_ok = 0
    n_warn = 0
    n_failed = 0

    worst_case = "none"
    worst_score = float("inf")

    for case_id in cases_raw:
        if not case_id.startswith("matmul_"):
            raise SystemExit(f"unknown seed case: {case_id!r}")
        parts = case_id.split("_")
        if len(parts) != 3:
            raise SystemExit(f"invalid case id: {case_id!r}")
        n = int(parts[1])
        dtype = {"f32": "float32", "f64": "float64"}.get(parts[2], "")
        if not dtype:
            raise SystemExit(f"invalid dtype suffix in case: {case_id!r}")

        for backend in backends:
            backend = backend.strip()
            full_case = f"{backend}_{case_id}"
            out_path = cases_dir / f"{full_case}.json"

            config_obj = {
                "backend": backend,
                "workload": "matmul",
                "n": n,
                "dtype": dtype,
                "cold_runs": int(args.cold_runs),
                "warm_iters": int(args.warm_iters),
                "cache_policy": str(args.cache_policy),
            }
            dataset = {"id": f"generated:ml:{full_case}", "sha256": sha256_json_obj(config_obj)}

            # JAX backend is optional: if missing, write a warn result and continue.
            if backend.startswith("jax_jit") and not has_jax:
                obj = {
                    "schema_version": "nextstat.ml_benchmark_result.v1",
                    "suite": "ml",
                    "case": full_case,
                    "deterministic": bool(args.deterministic),
                    "status": "warn",
                    "reason": "missing_dependency: jax",
                    "meta": {
                        "python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy_version": _try_import("numpy")[1],
                        "jax_version": "",
                    },
                    "dataset": dataset,
                    "config": config_obj,
                    "device": None,
                    "timing": {"cold": {"runs": []}, "warm": {"calls_s": []}},
                }
                out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
                suite_entries.append(
                    {
                        "case": full_case,
                        "path": os.path.relpath(out_path, out_dir),
                        "sha256": sha256_file(out_path),
                        "status": "warn",
                        "cold_ttfr_median_s": 0.0,
                        "warm_call_median_s": 0.0,
                    }
                )
                n_warn += 1
                continue

            # Fresh-process cold runs.
            cold_runs: list[dict] = []
            env = os.environ.copy()

            # Best-effort cache isolation for JAX: use a per-suite directory.
            cache_dir = out_dir / "_cache" / backend / case_id
            env["JAX_PLATFORM_NAME"] = "cpu"
            if backend == "jax_jit_gpu":
                env["JAX_PLATFORM_NAME"] = "gpu"
                _prepend_cuda_bin_for_ptxas(env)
            env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
            if str(args.cache_policy).strip().lower() == "process_cold":
                # Each run is a fresh process; additionally wipe the disk cache directory
                # to reduce cross-run reuse when possible.
                pass

            any_failed = False
            for i in range(int(args.cold_runs)):
                if backend.startswith("jax_jit") and str(args.cache_policy).strip().lower() == "process_cold":
                    _rm_tree(cache_dir)
                tmp = cases_dir / f".tmp_{full_case}_cold_{i}.json"
                r = run_worker(
                    run_py=run_py,
                    out_path=tmp,
                    env=env,
                    backend=backend,
                    n=n,
                    dtype=dtype,
                    warm_iters=0,
                )
                cold_runs.append(r)
                if str(r.get("status")) != "ok":
                    any_failed = True

            # Warm run (single process): compile + warm-call distribution.
            if backend.startswith("jax_jit"):
                # Keep the cache dir for warm run (best-effort).
                cache_dir.mkdir(parents=True, exist_ok=True)
            warm_tmp = cases_dir / f".tmp_{full_case}_warm.json"
            warm_obj = run_worker(
                run_py=run_py,
                out_path=warm_tmp,
                env=env,
                backend=backend,
                n=n,
                dtype=dtype,
                warm_iters=int(args.warm_iters),
            )
            if str(warm_obj.get("status")) != "ok":
                any_failed = True

            runtime = warm_obj.get("meta", {}).get("runtime", {}) if isinstance(warm_obj, dict) else {}
            device = None
            if isinstance(runtime, dict):
                plat = runtime.get("jax_device_platform") or runtime.get("jax_default_backend") or ""
                kind = runtime.get("jax_device_kind") or ""
                cnt = runtime.get("jax_device_count")
                if isinstance(plat, str) and plat:
                    device = {"platform": str(plat), "kind": str(kind), "count": int(cnt or 0)}

            import_s = [float(r.get("timing", {}).get("import_s", 0.0) or 0.0) for r in cold_runs]
            first_s = [float(r.get("timing", {}).get("first_call_s", 0.0) or 0.0) for r in cold_runs]
            total_s = [float(r.get("timing", {}).get("total_s", 0.0) or 0.0) for r in cold_runs]
            ttfr_s = [a + b for a, b in zip(import_s, first_s)]

            warm_calls = warm_obj.get("timing", {}).get("warm_calls_s") or []
            warm_calls_f = [float(x) for x in warm_calls if isinstance(x, (int, float))]

            status = "ok"
            reason = ""
            if any_failed:
                # For GPU-intended cases, treat "no GPU backend" as a warning rather than a failure.
                # This keeps CPU-only environments runnable while still reporting the absence.
                warm_reason = str(warm_obj.get("reason") or "")
                if backend == "jax_jit_gpu" and "requested platform=gpu" in warm_reason:
                    status = "warn"
                    reason = "gpu_unavailable"
                else:
                    status = "failed"
                    reason = "worker_failed"

            meta = {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy_version": _try_import("numpy")[1],
                "jax_version": jax_v if backend.startswith("jax_jit") else "",
            }

            obj = {
                "schema_version": "nextstat.ml_benchmark_result.v1",
                "suite": "ml",
                "case": full_case,
                "deterministic": bool(args.deterministic),
                "status": status,
                "reason": reason if reason else None,
                "meta": meta,
                "dataset": dataset,
                "config": config_obj,
                "device": device,
                "timing": {
                    "cold": {
                        "n_runs": int(args.cold_runs),
                        "import_s": _summary(import_s),
                        "first_call_s": _summary(first_s),
                        "ttfr_s": _summary(ttfr_s),
                        "total_s": _summary(total_s),
                        "runs": [
                            {
                                "import_s": float(r.get("timing", {}).get("import_s", 0.0) or 0.0),
                                "first_call_s": float(
                                    r.get("timing", {}).get("first_call_s", 0.0) or 0.0
                                ),
                                "second_call_s": float(
                                    r.get("timing", {}).get("second_call_s", 0.0) or 0.0
                                ),
                                "total_s": float(r.get("timing", {}).get("total_s", 0.0) or 0.0),
                                "status": str(r.get("status") or "failed"),
                                "reason": r.get("reason") or "",
                            }
                            for r in cold_runs
                        ],
                    },
                    "warm": {
                        "n_iters": int(args.warm_iters),
                        "call_s": _summary(warm_calls_f),
                        "calls_s": warm_calls_f,
                        "status": str(warm_obj.get("status") or "failed"),
                        "reason": warm_obj.get("reason") or "",
                    },
                },
            }

            out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

            cold_ttfr_med = float(obj["timing"]["cold"]["ttfr_s"]["median"])
            warm_call_med = float(obj["timing"]["warm"]["call_s"]["median"])

            suite_entries.append(
                {
                    "case": full_case,
                    "path": os.path.relpath(out_path, out_dir),
                    "sha256": sha256_file(out_path),
                    "status": status,
                    "cold_ttfr_median_s": cold_ttfr_med,
                    "warm_call_median_s": warm_call_med,
                }
            )

            if status == "ok":
                n_ok += 1
            elif status == "warn":
                n_warn += 1
            else:
                n_failed += 1

            # Worst-case heuristic: highest median TTFR (failed is always worst).
            score = float("inf") if status == "failed" else cold_ttfr_med
            if score < worst_score:
                worst_score = score
                worst_case = full_case

    suite = {
        "schema_version": "nextstat.ml_benchmark_suite_result.v1",
        "suite": "ml",
        "deterministic": bool(args.deterministic),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "cases": suite_entries,
        "summary": {
            "n_cases": len(suite_entries),
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_failed": n_failed,
            "worst_case": worst_case,
        },
    }
    (out_dir / "ml_suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")

    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
