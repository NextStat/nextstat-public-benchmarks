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
    # Minimal "reference path" support:
    # - POI `mu` as a normfactor on the `signal` sample
    # - optional `shapesys` modifiers (treated as per-bin Gaussian-constrained additive deformations)
    # - single channel (for now)
    try:
        channels = ws.get("channels") or []
        observations = ws.get("observations") or []
        measurements = ws.get("measurements") or []
        if not channels or not observations or not measurements:
            return False, "workspace missing channels/observations/measurements"
        if len(channels) != 1:
            return False, "only single-channel workspaces are supported (reference baseline)"
        poi = measurements[0].get("config", {}).get("poi")
        if poi != "mu":
            return False, f"unsupported poi: {poi!r} (expected 'mu')"
        # All samples must have only:
        # - normfactor(mu) on the signal sample
        # - shapesys on any sample (per-bin)
        ch = channels[0]
        for s in ch.get("samples") or []:
            sample_name = s.get("name")
            mods = s.get("modifiers") or []
            for m in mods:
                t = m.get("type")
                n = m.get("name")
                if t == "normfactor" and n == "mu" and sample_name == "signal":
                    continue
                if t == "shapesys":
                    data = m.get("data")
                    if not isinstance(data, list):
                        return False, f"shapesys modifier {n!r} must have list data"
                    continue
                return False, f"unsupported modifier: {t}:{n}"
        return True, ""
    except Exception as e:
        return False, f"workspace parse error: {type(e).__name__}: {e}"


def _extract_channel(ws: dict) -> tuple[dict, list[float]]:
    ch = (ws.get("channels") or [])[0]
    ch_name = ch.get("name")
    obs_counts: list[float] | None = None
    for o in ws.get("observations") or []:
        if o.get("name") == ch_name:
            obs_counts = [float(x) for x in (o.get("data") or [])]
            break
    if obs_counts is None:
        # Fallback: first observation.
        obs = (ws.get("observations") or [])[0]
        obs_counts = [float(x) for x in (obs.get("data") or [])]
    return ch, obs_counts


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

        ch, observed = _extract_channel(ws_obj)
        samples = ch.get("samples") or []
        if not samples:
            raise RuntimeError("workspace has no samples")

        n_bins = len(observed)
        if n_bins == 0:
            raise RuntimeError("workspace has zero bins")

        # POI.
        mu = ROOT.RooRealVar("mu", "mu", 1.0, 0.0, 20.0)

        # Per-bin nuisance parameters for shapesys (Gaussian(0,1) penalty; additive sigma*alpha).
        shape_alphas: list = []
        shape_penalties = ROOT.RooArgList()

        # Build per-bin expected nu_i = sum_a nu_{a,i}.
        nu_terms: list = []
        for i in range(n_bins):
            # Start with 0.0 as RooConstVar to sum into.
            nu_i_terms = ROOT.RooArgList()
            for s in samples:
                s_name = str(s.get("name") or "sample")
                nom = [float(x) for x in (s.get("data") or [])]
                if len(nom) != n_bins:
                    raise RuntimeError(f"sample {s_name!r} has {len(nom)} bins, expected {n_bins}")

                # Base yield for this sample and bin.
                base = ROOT.RooConstVar(f"nom_{s_name}_{i}", f"nom_{s_name}_{i}", float(nom[i]))
                yield_expr = base

                # Apply shapesys modifiers (additive sigma_i * alpha_i).
                for m in s.get("modifiers") or []:
                    if m.get("type") != "shapesys":
                        continue
                    mod_name = str(m.get("name") or "shapesys")
                    sigmas = m.get("data")
                    if not isinstance(sigmas, list) or len(sigmas) != n_bins:
                        raise RuntimeError(
                            f"shapesys {mod_name!r} on sample {s_name!r} has invalid data length"
                        )
                    sigma_i = float(sigmas[i])
                    if sigma_i == 0.0:
                        continue

                    # Bounds chosen to keep yields positive:
                    #   base + alpha*sigma > eps  => alpha > -(base-eps)/sigma
                    eps = 1e-9
                    lo = (-(float(nom[i]) - eps) / sigma_i) if sigma_i > 0 else -10.0
                    lo = max(lo, -50.0)
                    hi = 50.0
                    alpha = ROOT.RooRealVar(
                        f"alpha_{s_name}_{mod_name}_{i}",
                        f"alpha_{s_name}_{mod_name}_{i}",
                        0.0,
                        float(lo),
                        float(hi),
                    )
                    shape_alphas.append(alpha)
                    sigma = ROOT.RooConstVar(
                        f"sigma_{s_name}_{mod_name}_{i}",
                        f"sigma_{s_name}_{mod_name}_{i}",
                        float(sigma_i),
                    )
                    delta = ROOT.RooFormulaVar(
                        f"delta_{s_name}_{mod_name}_{i}",
                        "@0*@1",
                        ROOT.RooArgList(alpha, sigma),
                    )
                    yield_expr = ROOT.RooFormulaVar(
                        f"yield_{s_name}_{mod_name}_{i}",
                        "@0+@1",
                        ROOT.RooArgList(yield_expr, delta),
                    )
                    # Gaussian penalty: 0.5*alpha^2 (dropping constant).
                    pen = ROOT.RooFormulaVar(
                        f"pen_{s_name}_{mod_name}_{i}",
                        "0.5*(@0*@0)",
                        ROOT.RooArgList(alpha),
                    )
                    shape_penalties.add(pen)

                # Apply mu normfactor to the signal sample if present.
                has_mu = any(
                    (mm.get("type") == "normfactor" and mm.get("name") == "mu")
                    for mm in (s.get("modifiers") or [])
                )
                if has_mu:
                    if s_name != "signal":
                        raise RuntimeError("normfactor(mu) only supported on sample named 'signal'")
                    yield_expr = ROOT.RooFormulaVar(
                        f"yield_{s_name}_{i}_mu",
                        "@0*@1",
                        ROOT.RooArgList(mu, yield_expr),
                    )

                nu_i_terms.add(yield_expr)

            # Sum samples -> nu_i
            nu_i = ROOT.RooAddition(f"nu_{i}", f"nu_{i}", nu_i_terms)
            nu_terms.append(nu_i)

        # NLL terms: sum_i [nu_i - n_i*log(nu_i)] + sum penalties.
        terms = ROOT.RooArgList()
        for i, n_i in enumerate(observed):
            n = ROOT.RooConstVar(f"n_{i}", f"n_{i}", float(n_i))
            t = ROOT.RooFormulaVar(
                f"nll_bin_{i}",
                "@0-@1*log(@0)",
                ROOT.RooArgList(nu_terms[i], n),
            )
            terms.add(t)
        for j in range(shape_penalties.getSize()):
            terms.add(shape_penalties.at(j))

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
        doc["fit"] = {
            "muhat": muhat,
            "nll": nll_free,
            "minimizer_rc": rc,
        }
        doc["profile"] = {"poi0": 0.0, "nll_mu0": nll_mu0, "q0": q0, "z0": z0}
        doc["meta"]["n_nuisance_shapesys"] = int(len(shape_alphas))
    except Exception as e:
        doc["status"] = "skipped"
        doc["reason"] = f"{type(e).__name__}: {e}"

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
