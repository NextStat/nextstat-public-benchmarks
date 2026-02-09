#!/usr/bin/env python3
"""ROOT baseline runner (template).

For now this only validates environment availability (PyROOT import + version).
Full RooFit/RooStats benchmark wiring is tracked separately.
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
    except Exception as e:
        doc["status"] = "skipped"
        doc["reason"] = f"{type(e).__name__}: {e}"

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

