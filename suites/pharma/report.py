#!/usr/bin/env python3
"""Small report generator for the pharma suite results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_rows_markdown(rows: list[tuple]) -> str:
    lines = []
    lines.append("| case | subjects | obs | params | NLL time/call (NextStat) | fit | fit time (NextStat) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for case, nsub, nobs, npar, nll_t, fit_status, fit_t in rows:
        fit_t_txt = "-" if fit_status != "ok" else f"{fit_t:.3f}s"
        lines.append(
            f"| `{case}` | {nsub} | {nobs} | {npar} | {nll_t:.6f}s | {fit_status} | {fit_t_txt} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="out/pharma/pharma_suite.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_dir = suite_path.parent
    suite = json.loads(suite_path.read_text())

    rows = []
    for e in suite.get("cases", []):
        inst = json.loads((out_dir / e["path"]).read_text())
        model = inst.get("model", {})
        timing = (inst.get("timing", {}) or {}).get("nll_time_s_per_call", {}) or {}
        fit = inst.get("fit") or {}
        fit_status = "-"
        fit_t = 0.0
        if isinstance(fit, dict):
            fit_status = str(fit.get("status", "-"))
            if fit_status == "ok":
                fit_t = float((fit.get("time_s", {}) or {}).get("nextstat", 0.0))

        rows.append(
            (
                inst.get("case", e.get("case")),
                int(model.get("n_subjects", 0)),
                int(model.get("n_obs", 0)),
                int(model.get("n_params", 0)),
                float(timing.get("nextstat", 0.0)),
                fit_status,
                fit_t,
            )
        )

    txt = format_rows_markdown(rows)
    if str(args.out).strip():
        Path(args.out).resolve().write_text(txt)
    else:
        print(txt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

