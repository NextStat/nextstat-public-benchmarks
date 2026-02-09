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


def _workspace_supported(ws: dict) -> tuple[bool, str]:
    # Minimal "reference path" support: one POI normfactor "mu" on the signal sample, no other modifiers.
    try:
        channels = ws.get("channels") or []
        observations = ws.get("observations") or []
        measurements = ws.get("measurements") or []
        if not channels or not observations or not measurements:
            return False, "workspace missing channels/observations/measurements"
        poi = measurements[0].get("config", {}).get("poi")
        if poi != "mu":
            return False, f"unsupported poi: {poi!r} (expected 'mu')"
        # All samples must have only an optional normfactor(mu) modifier; no systematics.
        for ch in channels:
            for s in ch.get("samples") or []:
                mods = s.get("modifiers") or []
                for m in mods:
                    t = m.get("type")
                    n = m.get("name")
                    if t == "normfactor" and n == "mu":
                        continue
                    return False, f"unsupported modifier: {t}:{n}"
        return True, ""
    except Exception as e:
        return False, f"workspace parse error: {type(e).__name__}: {e}"


def _extract_counts(ws: dict) -> tuple[list[float], list[float], list[float]]:
    # Returns (signal_yields, background_yields, observed_counts) for the first channel.
    ch = (ws.get("channels") or [])[0]
    obs = (ws.get("observations") or [])[0]
    obs_counts = [float(x) for x in (obs.get("data") or [])]

    sig = None
    bkg = None
    for s in ch.get("samples") or []:
        if s.get("name") == "signal":
            sig = [float(x) for x in (s.get("data") or [])]
        if s.get("name") == "background":
            bkg = [float(x) for x in (s.get("data") or [])]
    if sig is None or bkg is None:
        raise RuntimeError("expected samples named 'signal' and 'background'")
    if not (len(sig) == len(bkg) == len(obs_counts)):
        raise RuntimeError("inconsistent bin counts across signal/background/observations")
    return sig, bkg, obs_counts


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
        doc["meta"]["note"] = "RooFit reference path (minimal: normfactor-only workspaces)"

        ws_path = str(args.workspace).strip()
        if ws_path:
            ws_obj = json.loads(Path(ws_path).read_text())
        else:
            # If no workspace is provided, treat this as a pure env smoke test.
            ws_obj = None

        if ws_obj is None:
            return 0

        ok, why = _workspace_supported(ws_obj)
        if not ok:
            doc["status"] = "skipped"
            doc["reason"] = f"unsupported_workspace:{why}"
            out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            return 0

        signal, background, observed = _extract_counts(ws_obj)

        # Build a minimal RooFit NLL:
        #   NLL(mu) = sum_i [nu_i(mu) - n_i * log nu_i(mu)]   (dropping constants log(n!))
        #   nu_i(mu) = mu*s_i + b_i
        mu = ROOT.RooRealVar("mu", "mu", 1.0, 0.0, 20.0)

        terms = ROOT.RooArgList()
        for i, (s_i, b_i, n_i) in enumerate(zip(signal, background, observed)):
            s = ROOT.RooConstVar(f"s_{i}", f"s_{i}", float(s_i))
            b = ROOT.RooConstVar(f"b_{i}", f"b_{i}", float(b_i))
            n = ROOT.RooConstVar(f"n_{i}", f"n_{i}", float(n_i))
            nu = ROOT.RooFormulaVar(f"nu_{i}", "@0*@1+@2", ROOT.RooArgList(mu, s, b))
            t = ROOT.RooFormulaVar(
                f"nll_{i}",
                "@0-@1*log(@0)",
                ROOT.RooArgList(nu, n),
            )
            terms.add(t)

        nll = ROOT.RooAddition("nll", "nll", terms)

        # Fit free mu (bounded).
        mu.setVal(1.0)
        mu.setConstant(False)
        minim = ROOT.RooMinimizer(nll)
        minim.setPrintLevel(-1)
        minim.setStrategy(1)
        rc = int(minim.minimize("Minuit2", "migrad"))

        muhat = float(mu.getVal())
        nll_free = float(nll.getVal())

        # Conditional mu=0.
        mu.setVal(0.0)
        mu.setConstant(True)
        nll_mu0 = float(nll.getVal())

        q0 = max(0.0, 2.0 * (nll_mu0 - nll_free))
        z0 = float(q0) ** 0.5

        doc["status"] = "ok" if rc == 0 else "failed"
        doc["reason"] = "" if rc == 0 else f"root_minimizer_failed:rc={rc}"
        doc["fit"] = {"muhat": muhat, "nll": nll_free, "minimizer_rc": rc}
        doc["profile"] = {"poi0": 0.0, "nll_mu0": nll_mu0, "q0": q0, "z0": z0}
    except Exception as e:
        doc["status"] = "skipped"
        doc["reason"] = f"{type(e).__name__}: {e}"

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
