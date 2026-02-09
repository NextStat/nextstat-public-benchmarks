#!/usr/bin/env python3
"""Minimal HEP benchmark seed: NLL parity + timing (pyhf vs NextStat).

This is intentionally small and self-contained so outsiders can rerun it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import timeit
from array import array
from pathlib import Path
from typing import Any, Callable

import pyhf

import nextstat


NLL_SANITY_ATOL = 1e-8
NLL_SANITY_RTOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pyhf_model_and_data(workspace: dict[str, Any], measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def pyhf_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        out[dst_index[name]] = float(value)
    return out


def bench_time_per_call(fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 5) -> float:
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    return min(times)


def bench_time_per_call_raw(
    fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 5
) -> tuple[int, list[float]]:
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    return number, times


def bench_wall_time_raw(fn: Callable[[], Any], *, repeat: int = 3) -> list[float]:
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="simple_workspace_nll", help="Case id for reporting.")
    parser.add_argument(
        "--workspace",
        default=str(Path(__file__).resolve().parent / "datasets/simple_workspace.json"),
        help="Path to a pyhf workspace JSON.",
    )
    parser.add_argument(
        "--measurement-name",
        default="GaussExample",
        help="pyhf measurement name to load.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic output.")
    parser.add_argument("--target-s", type=float, default=0.25, help="Target timing window (seconds).")
    parser.add_argument("--repeat", type=int, default=5, help="Number of timing repeats.")
    parser.add_argument("--fit", action="store_true", help="Also benchmark full MLE fits.")
    parser.add_argument("--fit-repeat", type=int, default=3, help="Fit timing repeats (wall-clock).")
    parser.add_argument("--dataset-id", default="", help="Optional dataset id for reporting (stable, portable).")
    parser.add_argument("--dataset-sha256", default="", help="Optional dataset sha256 override.")
    args = parser.parse_args()

    ws_path = Path(args.workspace).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workspace: dict[str, Any] = json.loads(ws_path.read_text())
    pyhf_model, pyhf_data = pyhf_model_and_data(workspace, args.measurement_name)
    pyhf_params = list(pyhf_model.config.suggested_init())

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_params_list = map_params_by_name(
        pyhf_model.config.par_names,
        pyhf_params,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )
    # Buffer-protocol input to avoid per-element Python overhead for large vectors.
    ns_params = array("d", ns_params_list)

    pyhf_val = pyhf_nll(pyhf_model, pyhf_data, pyhf_params)
    ns_val = float(ns_model.nll(ns_params))
    abs_diff = abs(ns_val - pyhf_val)
    rel_diff = abs_diff / max(abs(ns_val), abs(pyhf_val), 1.0)
    ok = not (abs_diff > NLL_SANITY_ATOL and rel_diff > NLL_SANITY_RTOL)

    def f_pyhf():
        return pyhf_nll(pyhf_model, pyhf_data, pyhf_params)

    def f_ns():
        return ns_model.nll(ns_params)

    # Warmup (avoid one-time allocations dominating timing).
    for _ in range(10):
        f_pyhf()
        f_ns()

    target_s = float(args.target_s)
    repeat = int(args.repeat)
    number, pyhf_times = bench_time_per_call_raw(f_pyhf, target_s=target_s, repeat=repeat)
    number2, ns_times = bench_time_per_call_raw(f_ns, target_s=target_s, repeat=repeat)
    # Keep a single shared `number` in the raw metadata. If they differ, record the larger
    # and keep the arrays as-is (this is fine for reporting).
    number = max(number, number2)
    pyhf_t = min(pyhf_times)
    ns_t = min(ns_times)
    speedup = pyhf_t / ns_t if ns_t > 0 else float("inf")

    n_main_bins = sum(len(obs["data"]) for obs in workspace.get("observations", []))

    dataset_id_in = str(args.dataset_id).strip()
    dataset_id = dataset_id_in or str(args.workspace)
    dataset_sha = str(args.dataset_sha256).strip() or sha256_file(ws_path)

    doc: dict[str, Any] = {
        "schema_version": "nextstat.benchmark_result.v1",
        "suite": "hep",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyhf_version": pyhf.__version__,
            "nextstat_version": nextstat.__version__,
        },
        "dataset": {
            "id": dataset_id,
            **({} if dataset_id_in else {"path": str(args.workspace)}),
            "sha256": dataset_sha,
        },
        "model": {"n_main_bins": int(n_main_bins), "n_params": int(ns_model.n_params())},
        "parity": {
            "ok": bool(ok),
            "nll_pyhf": float(pyhf_val),
            "nll_nextstat": float(ns_val),
            "abs_diff": float(abs_diff),
            "rel_diff": float(rel_diff),
            "atol": float(NLL_SANITY_ATOL),
            "rtol": float(NLL_SANITY_RTOL),
        },
        "timing": {
            "nll_time_s_per_call": {"pyhf": float(pyhf_t), "nextstat": float(ns_t)},
            "speedup_pyhf_over_nextstat": float(speedup),
            "raw": {
                "number": int(number),
                "repeat": int(repeat),
                "target_s": float(target_s),
                "policy": "min",
                "per_call_s": {"pyhf": [float(x) for x in pyhf_times], "nextstat": [float(x) for x in ns_times]},
            },
        },
    }

    if args.fit:
        fit_repeat = int(args.fit_repeat)
        poi_name = getattr(pyhf_model.config, "poi_name", None)
        poi_index = getattr(pyhf_model.config, "poi_index", None)
        if poi_index is None:
            doc["fit"] = {"status": "skipped", "reason": "pyhf model has no POI"}
        else:
            try:
                def fit_pyhf():
                    pyhf.infer.mle.fit(pyhf_data, pyhf_model)

                def fit_nextstat():
                    mle = nextstat.MaximumLikelihoodEstimator()
                    mle.fit(ns_model)

                # Warmup once each (avoid one-time allocations dominating timing).
                fit_pyhf()
                fit_nextstat()

                pyhf_fit_times = bench_wall_time_raw(fit_pyhf, repeat=fit_repeat)
                ns_fit_times = bench_wall_time_raw(fit_nextstat, repeat=fit_repeat)
                pyhf_fit = min(pyhf_fit_times) if pyhf_fit_times else 0.0
                ns_fit = min(ns_fit_times) if ns_fit_times else 0.0
                fit_speedup = pyhf_fit / ns_fit if ns_fit > 0 else float("inf")

                # Basic fit metadata (one run each; independent of timing repeats).
                pyhf_best = pyhf.infer.mle.fit(pyhf_data, pyhf_model)
                pyhf_nll_at_pyhf = pyhf_nll(pyhf_model, pyhf_data, pyhf_best)
                pyhf_poi_hat = float(pyhf_best[int(poi_index)])

                ns_res = nextstat.MaximumLikelihoodEstimator().fit(ns_model)
                ns_nll_at_ns = float(getattr(ns_res, "nll", float("nan")))
                ns_names = list(ns_model.parameter_names())
                ns_poi_hat = None
                if poi_name and poi_name in ns_names:
                    ns_poi_hat = float(ns_res.bestfit[ns_names.index(poi_name)])

                # Compare NLL values at a shared point (mapped by parameter names).
                ns_at_pyhf = map_params_by_name(
                    pyhf_model.config.par_names,
                    pyhf_best,
                    ns_names,
                    ns_model.suggested_init(),
                )
                ns_nll_at_pyhf = float(ns_model.nll(array("d", ns_at_pyhf)))
                abs_diff_pyhf_point = abs(ns_nll_at_pyhf - pyhf_nll_at_pyhf)

                pyhf_at_ns = map_params_by_name(
                    ns_names,
                    ns_res.bestfit,
                    pyhf_model.config.par_names,
                    pyhf_model.config.suggested_init(),
                )
                pyhf_nll_at_ns = pyhf_nll(pyhf_model, pyhf_data, pyhf_at_ns)
                abs_diff_ns_point = abs(pyhf_nll_at_ns - ns_nll_at_ns)

                doc["fit"] = {
                    "status": "ok",
                    "time_s": {"pyhf": float(pyhf_fit), "nextstat": float(ns_fit)},
                    "speedup_pyhf_over_nextstat": float(fit_speedup),
                    "raw": {
                        "repeat": int(fit_repeat),
                        "policy": "min",
                        "per_fit_s": {
                            "pyhf": [float(x) for x in pyhf_fit_times],
                            "nextstat": [float(x) for x in ns_fit_times],
                        },
                    },
                    "meta": {
                        **({} if poi_name is None else {"poi_name": str(poi_name)}),
                        "poi_hat_pyhf": float(pyhf_poi_hat),
                        **({} if ns_poi_hat is None else {"poi_hat_nextstat": float(ns_poi_hat)}),
                        "nll_pyhf": float(pyhf_nll_at_pyhf),
                        "nll_nextstat": float(ns_nll_at_ns),
                        "nextstat_success": bool(getattr(ns_res, "success", False)),
                        "nextstat_converged": bool(getattr(ns_res, "converged", False)),
                        "nextstat_n_iter": int(getattr(ns_res, "n_iter", 0)),
                        "nextstat_n_evaluations": int(getattr(ns_res, "n_evaluations", 0)),
                        "nextstat_termination_reason": str(getattr(ns_res, "termination_reason", "")),
                    },
                    "parity": {
                        "abs_diff_nll_at_pyhf_bestfit": float(abs_diff_pyhf_point),
                        "abs_diff_nll_at_nextstat_bestfit": float(abs_diff_ns_point),
                    },
                }
            except Exception as e:
                doc["fit"] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    if not ok:
        # Write the artifact anyway (for debugging) but return non-zero.
        out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return 2

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
