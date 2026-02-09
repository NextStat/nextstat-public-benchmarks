#!/usr/bin/env python3
"""Render a small human-readable snippet for the ML suite (seed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_s(x: float) -> str:
    if x <= 0:
        return "—"
    if x < 1e-3:
        return f"{x*1e6:.1f} µs"
    if x < 1:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to ml_suite.json")
    ap.add_argument("--out", required=True, help="Output markdown path")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suite = load_json(suite_path)
    base = suite_path.parent

    rows = []
    for e in suite.get("cases", []):
        rel = e.get("path")
        if not isinstance(rel, str):
            continue
        case = load_json((base / rel).resolve())
        cid = str(case.get("case", ""))
        status = str(case.get("status", ""))
        dev = case.get("device") or {}
        dev_plat = str(dev.get("platform", "") or "")
        cold_med = float(case.get("timing", {}).get("cold", {}).get("ttfr_s", {}).get("median", 0.0) or 0.0)
        warm_med = float(case.get("timing", {}).get("warm", {}).get("call_s", {}).get("median", 0.0) or 0.0)
        suffix = f" ({dev_plat})" if dev_plat else ""
        rows.append((cid + suffix, status, cold_med, warm_med))

    lines = []
    lines.append("## ML (Compile vs Execution) — Seed Snapshot\n")
    lines.append(
        "Cold-start is measured as TTFR = import + first call in a fresh process (median over runs). "
        "Warm throughput is the median per-call time over repeated warm calls.\n"
    )
    lines.append("\n| Case | Status | Cold TTFR (median) | Warm call (median) |\n|---|---:|---:|---:|\n")
    for cid, status, cold_med, warm_med in rows:
        lines.append(f"| `{cid}` | `{status}` | {fmt_s(cold_med)} | {fmt_s(warm_med)} |\n")
    lines.append("\n")
    lines.append(
        "Notes:\n"
        "- `jax_jit_*` cases are optional in the seed harness; if JAX is not installed, they appear as `warn`.\n"
        "- Cache policy is best-effort and pinned in the case `config`.\n"
    )

    out_path.write_text("".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
