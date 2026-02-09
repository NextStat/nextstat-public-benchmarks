#!/usr/bin/env python3
"""Minimal pharma benchmark seed: PK/NLME timing (NextStat only).

This intentionally starts with a portable synthetic dataset generator so outsiders can rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Callable

import nextstat


def sha256_json_obj(obj: dict[str, Any]) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


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


def generate_pk_1c_oral(*, n_obs: int, seed: int, dose: float, sigma: float) -> dict[str, Any]:
    rng = random.Random(int(seed))
    # Geometric sampling times (hours) in a realistic window, excluding t=0.
    n_obs = int(n_obs)
    t0, t1 = 0.25, 24.0
    if n_obs <= 1:
        times = [t0]
    else:
        ratio = (t1 / t0) ** (1.0 / (n_obs - 1))
        times = [t0 * (ratio**i) for i in range(n_obs)]
    y0 = [0.0] * n_obs
    m0 = nextstat.OneCompartmentOralPkModel(times, y0, dose=float(dose), sigma=float(sigma))
    true_params = [1.2, 12.0, 0.9]
    mu = m0.predict(true_params)
    # Ensure non-negative concentrations (NextStat input validation requires y >= 0).
    y = [float(max(0.0, mu_i + rng.gauss(0.0, float(sigma)))) for mu_i in mu]
    return {"kind": "pk_1c_oral", "times": times, "y": y, "dose": float(dose), "sigma": float(sigma), "seed": int(seed)}


def generate_nlme_1c_oral(
    *, n_subjects: int, n_obs_per_subject: int, seed: int, dose: float, sigma: float
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    n_obs_per_subject = int(n_obs_per_subject)
    t0, t1 = 0.25, 24.0
    if n_obs_per_subject <= 1:
        times_one = [t0]
    else:
        ratio = (t1 / t0) ** (1.0 / (n_obs_per_subject - 1))
        times_one = [t0 * (ratio**i) for i in range(n_obs_per_subject)]
    true_pop = [1.2, 12.0, 0.9]
    # Crude random effects: multiplicative noise on CL and V.
    times: list[float] = []
    y: list[float] = []
    subject_idx: list[int] = []
    for s in range(int(n_subjects)):
        subj = list(true_pop)
        subj[1] *= float(1.0 + rng.gauss(0.0, 0.15))
        subj[2] *= float(1.0 + rng.gauss(0.0, 0.10))
        m0 = nextstat.OneCompartmentOralPkModel(times_one, [0.0] * len(times_one), dose=float(dose), sigma=float(sigma))
        mu = m0.predict(subj)
        for t, mu_i in zip(times_one, mu):
            times.append(float(t))
            y.append(float(max(0.0, mu_i + rng.gauss(0.0, float(sigma)))))
            subject_idx.append(int(s))
    return {
        "kind": "nlme_pk_1c_oral",
        "times": times,
        "y": y,
        "subject_idx": subject_idx,
        "n_subjects": int(n_subjects),
        "n_obs_per_subject": int(n_obs_per_subject),
        "dose": float(dose),
        "sigma": float(sigma),
        "seed": int(seed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="pk_1c_oral", help="Case id for reporting.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--target-s", type=float, default=0.25)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--fit", action="store_true", help="Also time an MLE fit (wall-clock).")
    ap.add_argument("--fit-repeat", type=int, default=3)

    ap.add_argument("--model", choices=["pk", "nlme"], default="pk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dose", type=float, default=100.0)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--n-obs", type=int, default=8, help="For pk: number of observations.")
    ap.add_argument("--n-subjects", type=int, default=8, help="For nlme: number of subjects.")
    ap.add_argument("--n-obs-per-subject", type=int, default=6, help="For nlme: observations per subject.")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "nlme":
        spec = generate_nlme_1c_oral(
            n_subjects=int(args.n_subjects),
            n_obs_per_subject=int(args.n_obs_per_subject),
            seed=int(args.seed),
            dose=float(args.dose),
            sigma=float(args.sigma),
        )
        model = nextstat.OneCompartmentOralPkNlmeModel(
            spec["times"],
            spec["y"],
            spec["subject_idx"],
            int(spec["n_subjects"]),
            dose=float(spec["dose"]),
            sigma=float(spec["sigma"]),
        )
    else:
        spec = generate_pk_1c_oral(
            n_obs=int(args.n_obs),
            seed=int(args.seed),
            dose=float(args.dose),
            sigma=float(args.sigma),
        )
        model = nextstat.OneCompartmentOralPkModel(
            spec["times"],
            spec["y"],
            dose=float(spec["dose"]),
            sigma=float(spec["sigma"]),
        )

    params = list(model.suggested_init())

    def f_nll():
        return model.nll(params)

    for _ in range(10):
        f_nll()

    number, times = bench_time_per_call_raw(f_nll, target_s=float(args.target_s), repeat=int(args.repeat))
    t = min(times)

    # Dataset id must be stable and *disambiguate* distinct generated datasets.
    # This id is included in baseline manifests and is used for third-party replication.
    if spec["kind"] == "nlme_pk_1c_oral":
        dataset_id = (
            f"generated:{spec['kind']}:n_sub{int(spec['n_subjects'])}:"
            f"n_obs{int(spec['n_obs_per_subject'])}:seed{int(args.seed)}"
        )
    else:
        dataset_id = f"generated:{spec['kind']}:n_obs{int(args.n_obs)}:seed{int(args.seed)}"
    dataset_sha = sha256_json_obj(spec)

    doc: dict[str, Any] = {
        "schema_version": "nextstat.pharma_benchmark_result.v1",
        "suite": "pharma",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
        },
        "dataset": {"id": dataset_id, "sha256": dataset_sha, "spec": spec},
        "model": {
            "kind": str(spec["kind"]),
            "n_subjects": int(spec.get("n_subjects", 1)),
            "n_obs": int(len(spec["times"])),
            "n_params": int(model.n_params()),
        },
        "timing": {
            "nll_time_s_per_call": {"nextstat": float(t)},
            "raw": {
                "number": int(number),
                "repeat": int(args.repeat),
                "target_s": float(args.target_s),
                "policy": "min",
                "per_call_s": {"nextstat": [float(x) for x in times]},
            },
        },
    }

    if args.fit:
        fit_repeat = int(args.fit_repeat)
        try:
            def fit_once():
                mle = nextstat.MaximumLikelihoodEstimator()
                mle.fit(model)

            # Warmup.
            fit_once()
            fit_times = bench_wall_time_raw(fit_once, repeat=fit_repeat)
            fit_t = min(fit_times) if fit_times else 0.0

            res = nextstat.MaximumLikelihoodEstimator().fit(model)
            doc["fit"] = {
                "status": "ok",
                "time_s": {"nextstat": float(fit_t)},
                "raw": {"repeat": int(fit_repeat), "policy": "min", "per_fit_s": {"nextstat": [float(x) for x in fit_times]}},
                "meta": {
                    "nextstat_nll": float(getattr(res, "nll", float("nan"))),
                    "nextstat_success": bool(getattr(res, "success", False)),
                    "nextstat_converged": bool(getattr(res, "converged", False)),
                    "nextstat_n_iter": int(getattr(res, "n_iter", 0)),
                    "nextstat_n_evaluations": int(getattr(res, "n_evaluations", 0)),
                    "nextstat_termination_reason": str(getattr(res, "termination_reason", "")),
                },
            }
        except Exception as e:
            doc["fit"] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
