#!/usr/bin/env python3
"""Render a small human README snippet for the Bayesian suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x, *, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if v != v:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to bayesian_suite.json")
    ap.add_argument("--out", required=True, help="Output Markdown path")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = json.loads(suite_path.read_text())
    cases = obj.get("cases") if isinstance(obj.get("cases"), list) else []

    lines = []
    lines.append("# Bayesian suite (NUTS diagnostics + ESS/sec)")
    lines.append("")
    lines.append("This snapshot reports sampler health metrics (rank-normalized R-hat, Geyer ESS, E-BFMI) and a simple ESS/sec proxy computed as `min(ESS_bulk)/wall_time` (includes warmup).")
    lines.append("")
    lines.append("| Case | Backend | Status | Wall time (s) | min ESS_bulk | min ESS_tail | max R-hat | min ESS_bulk/sec |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(c.get("case") or "unknown"),
                    str(c.get("backend") or "unknown"),
                    str(c.get("status") or "unknown"),
                    _fmt(c.get("wall_time_s")),
                    _fmt(c.get("min_ess_bulk")),
                    _fmt(c.get("min_ess_tail")),
                    _fmt(c.get("max_r_hat")),
                    _fmt(c.get("min_ess_bulk_per_sec")),
                ]
            )
            + " |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
