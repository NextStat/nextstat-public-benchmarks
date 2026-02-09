#!/usr/bin/env python3
"""ROOT baseline runner (reference path).

This is a "reference path" hook for RooFit/RooStats-based validation.

For now it validates environment availability (PyROOT import + version) and records
the dataset identity. Full RooFit/RooStats benchmark wiring is tracked separately.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="root_env_smoke")
    ap.add_argument("--workspace", default="", help="Optional pyhf workspace JSON path (for provenance).")
    ap.add_argument("--measurement-name", default="", help="Optional pyhf measurement name (for provenance).")
    ap.add_argument("--dataset-id", default="", help="Optional stable dataset id.")
    ap.add_argument("--dataset-sha256", default="", help="Optional dataset sha256.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = {
        "schema_version": "nextstat.hep_root_baseline_result.v1",
        "baseline": "root",
        "suite": "hep",
        "case": str(args.case),
        "status": "skipped",
        "reason": "PyROOT not available",
        "meta": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    ws = str(args.workspace).strip()
    meas = str(args.measurement_name).strip()
    ds_id = str(args.dataset_id).strip()
    ds_sha = str(args.dataset_sha256).strip()
    if ws or meas or ds_id or ds_sha:
        doc["dataset"] = {}
        if ws:
            doc["dataset"]["path"] = ws
        if meas:
            doc["dataset"]["measurement_name"] = meas
        if ds_id:
            doc["dataset"]["id"] = ds_id
        if ds_sha:
            doc["dataset"]["sha256"] = ds_sha

    try:
        import ROOT  # type: ignore

        v = ""
        try:
            v = str(ROOT.gROOT.GetVersion())
        except Exception:
            v = ""
        doc["status"] = "ok"
        doc["reason"] = ""
        if v:
            doc["meta"]["root_version"] = v
        doc["meta"]["note"] = "RooFit/RooStats evaluation not yet implemented in this baseline runner"
    except Exception as e:
        doc["status"] = "skipped"
        doc["reason"] = f"{type(e).__name__}: {e}"

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
