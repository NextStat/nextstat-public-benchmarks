#!/usr/bin/env python3
"""Small report generator for the HEP suite results.

Reads a `benchmark_suite_result_v1` index and prints a compact summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_rows_text(rows: list[tuple]) -> str:
    lines = []
    lines.append(
        f"{'case':<28} {'bins':>6} {'pars':>6}  {'parity':>6}  {'|dNLL|':>12}  {'NLL speedup':>11}  {'fit':>6}  {'fit speedup':>11}"
    )
    lines.append("-" * 104)
    for case, bins, pars, parity, abs_diff, nll_speedup, fit_status, fit_speedup in rows:
        lines.append(
            f"{str(case):<28} {bins:>6} {pars:>6}  {parity:>6}  {abs_diff:>12.3e}  {nll_speedup:>10.2f}x  {fit_status:>6}  {fit_speedup:>11}"
        )
    return "\n".join(lines) + "\n"


def format_rows_markdown(rows: list[tuple]) -> str:
    lines = []
    lines.append("| case | bins | pars | parity | \\|dNLL\\| | NLL speedup | fit | fit speedup |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case, bins, pars, parity, abs_diff, nll_speedup, fit_status, fit_speedup in rows:
        lines.append(
            f"| `{case}` | {bins} | {pars} | {parity} | {abs_diff:.3e} | {nll_speedup:.2f}x | {fit_status} | {fit_speedup} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="out/hep/hep_suite.json", help="Path to hep_suite.json")
    ap.add_argument("--format", default="text", choices=["text", "markdown"])
    ap.add_argument("--out", default="", help="Optional output file path.")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_dir = suite_path.parent
    suite = json.loads(suite_path.read_text())

    rows = []
    for e in suite.get("cases", []):
        case_path = out_dir / e["path"]
        inst = json.loads(case_path.read_text())
        model = inst.get("model", {})
        fit = inst.get("fit")
        fit_status = "-"
        fit_speedup = "-"
        if isinstance(fit, dict):
            fit_status = str(fit.get("status", "-"))
            if fit_status == "ok":
                fit_speedup = f"{float(fit.get('speedup_pyhf_over_nextstat', 0.0)):.2f}x"

        rows.append(
            (
                inst.get("case", e.get("case")),
                int(model.get("n_main_bins", 0)),
                int(model.get("n_params", 0)),
                "ok" if inst.get("parity", {}).get("ok") else "fail",
                float(inst.get("parity", {}).get("abs_diff", 0.0)),
                float(inst.get("timing", {}).get("speedup_pyhf_over_nextstat", 0.0)),
                fit_status,
                fit_speedup,
            )
        )

    if args.format == "markdown":
        txt = format_rows_markdown(rows)
    else:
        txt = format_rows_text(rows)

    if str(args.out).strip():
        Path(args.out).resolve().write_text(txt)
    else:
        print(txt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
