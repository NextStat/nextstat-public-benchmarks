#!/usr/bin/env python3
"""ML suite worker run (seed).

This script runs a *single* measurement in the current process and writes a JSON
artifact. The suite runner (`suite.py`) executes this worker in multiple fresh
processes to measure cold-start / TTFR distributions.

Design goal (seed):
- Minimal dependencies (stdlib + optional numpy/jax)
- Measures *components* (import/setup/first-call/second-call/warm calls)
- Explicitly blocks for readiness on async backends (JAX)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class RunTiming:
    import_s: float
    setup_s: float
    first_call_s: float
    second_call_s: float
    warm_calls_s: list[float]
    total_s: float


def _now() -> float:
    return time.perf_counter()


def _dtype_normalize(dtype: str) -> str:
    d = str(dtype).strip().lower()
    if d in ("f32", "float32"):
        return "float32"
    if d in ("f64", "float64"):
        return "float64"
    raise SystemExit(f"unsupported dtype: {dtype!r} (use float32|float64)")


def _load_numpy_backend(*, n: int, dtype: str) -> tuple[str, Callable[[], None]]:
    np = importlib.import_module("numpy")
    dt = getattr(np, dtype)
    a = np.ones((n, n), dtype=dt)
    b = np.ones((n, n), dtype=dt)

    def step() -> None:
        _ = a @ b

    return f"numpy=={np.__version__}", step


def _load_jax_backend(*, n: int, dtype: str, platform_name: str) -> tuple[str, Callable[[], Any]]:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")

    # Best-effort: pin platform to avoid surprising GPU dispatch in CI.
    # (If the user wants GPU, they should run on a GPU runner and publish that manifest.)
    try:
        jax.config.update("jax_platform_name", platform_name)
    except Exception:
        pass

    dt = getattr(jnp, dtype)
    a = jnp.ones((n, n), dtype=dt)
    b = jnp.ones((n, n), dtype=dt)

    @jax.jit
    def f(x, y):
        return x @ y

    def step() -> Any:
        out = f(a, b)
        # Ensure the timing includes device execution, not just dispatch.
        try:
            out = out.block_until_ready()
        except Exception:
            pass
        return out

    return f"jax=={getattr(jax, '__version__', 'unknown')}", step


def run_one(
    *,
    backend: str,
    workload: str,
    n: int,
    dtype: str,
    warm_iters: int,
    platform_name: str,
) -> tuple[RunTiming, dict[str, Any]]:
    if workload != "matmul":
        raise SystemExit(f"unsupported workload: {workload!r} (seed supports: matmul)")

    t0 = _now()

    backend = str(backend).strip().lower()
    runtime: dict[str, Any] = {}
    if backend == "numpy":
        version, step = _load_numpy_backend(n=n, dtype=dtype)
    elif backend == "jax_jit":
        version, step = _load_jax_backend(n=n, dtype=dtype, platform_name=platform_name)
        try:
            jax = importlib.import_module("jax")
            runtime["jax_default_backend"] = str(getattr(jax, "default_backend")())
            devs = list(getattr(jax, "devices")())
            runtime["jax_device_count"] = int(len(devs))
            if devs:
                d0 = devs[0]
                runtime["jax_device_platform"] = str(getattr(d0, "platform", ""))
                runtime["jax_device_kind"] = str(getattr(d0, "device_kind", ""))
        except Exception:
            pass

        # Correctness guard: if the user requested GPU profiling, ensure we actually ran on GPU.
        requested = str(platform_name).strip().lower()
        if requested in {"gpu"}:
            actual = str(runtime.get("jax_default_backend") or runtime.get("jax_device_platform") or "").lower()
            if actual and actual != "gpu":
                raise RuntimeError(f"requested platform=gpu but jax backend is {actual!r}")
    else:
        raise SystemExit(f"unsupported backend: {backend!r} (use: numpy|jax_jit)")

    t_import_end = _now()

    # Setup: one un-timed call to initialize any runtime state *except* compilation.
    # For JAX JIT, compilation is part of the first call; for NumPy this is a no-op.
    t_setup_start = _now()
    t_setup_end = _now()

    t_first_start = _now()
    step()
    t_first_end = _now()

    t_second_start = _now()
    step()
    t_second_end = _now()

    warm_calls: list[float] = []
    for _ in range(int(warm_iters)):
        tw0 = _now()
        step()
        tw1 = _now()
        warm_calls.append(tw1 - tw0)

    t1 = _now()

    timing = RunTiming(
        import_s=t_import_end - t0,
        setup_s=t_setup_end - t_setup_start,
        first_call_s=t_first_end - t_first_start,
        second_call_s=t_second_end - t_second_start,
        warm_calls_s=warm_calls,
        total_s=t1 - t0,
    )
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backend_version": version,
        "backend": backend,
        "workload": workload,
        "runtime": runtime,
    }
    return timing, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="Backend: numpy|jax_jit")
    ap.add_argument("--workload", default="matmul", help="Workload: matmul (seed)")
    ap.add_argument("--n", type=int, default=512, help="Matrix size for matmul (N x N).")
    ap.add_argument("--dtype", default="float32", help="float32|float64")
    ap.add_argument("--warm-iters", type=int, default=30, help="Warm-call iterations.")
    ap.add_argument(
        "--platform",
        default="cpu",
        help="Backend platform hint (seed): cpu (default).",
    )
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_normalize(args.dtype)
    backend = str(args.backend).strip()

    # Optional cache control is handled by the suite runner, which sets env vars
    # (and optionally deletes per-run cache dirs) before launching this worker.
    # Worker records best-effort hints only.
    cache_hint = {
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platform_name": os.environ.get("JAX_PLATFORM_NAME", ""),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
    }

    try:
        timing, meta = run_one(
            backend=backend,
            workload=str(args.workload),
            n=int(args.n),
            dtype=dtype,
            warm_iters=int(args.warm_iters),
            platform_name=str(args.platform),
        )
        doc: dict[str, Any] = {
            "status": "ok",
            "timing": asdict(timing),
            "meta": meta,
            "cache_hint": cache_hint,
        }
        out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as e:
        doc = {
            "status": "failed",
            "reason": f"{type(e).__name__}: {e}",
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "backend": backend,
            },
            "cache_hint": cache_hint,
        }
        out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
