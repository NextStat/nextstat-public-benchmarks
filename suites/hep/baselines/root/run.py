#!/usr/bin/env python3
"""ROOT baseline runner (reference path).

This is a "reference path" hook for RooFit/RooStats-based validation.

Goal: provide an independent ROOT/RooFit implementation of a small subset of the
HistFactory model that the public HEP suite uses (pyhf JSON workspaces).

Supported modifier types (aligned with pyhf/HistFactory semantics):
- normfactor (e.g. POI `mu`)                         [unconstrained]
- lumi (multiplicative, constrained by Normal)       [from measurement config]
- normsys (multiplicative, constrained by Normal)    [interp: code4]
- histosys (additive, constrained by Normal)         [interp: code4p]
- staterror (per-bin multiplicative, Normal)         [sigma derived from abs errors]
- shapesys (per-bin multiplicative, Poisson)         [tau = nom^2 / unc^2]
- shapefactor (per-bin multiplicative, unconstrained)

The baseline computes:
- free fit (mu floating) -> muhat, NLL
- conditional fit (mu=0) -> NLL(mu=0)
- profiled q0 and Z0

If PyROOT is unavailable or the workspace contains unsupported structures, the
baseline is skipped (best-effort).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


_ALLOWED_MODIFIER_TYPES = {
    "normfactor",
    "lumi",
    "normsys",
    "histosys",
    "staterror",
    "shapesys",
    "shapefactor",
}


def _workspace_supported(ws: dict) -> tuple[bool, str]:
    try:
        channels = ws.get("channels") or []
        observations = ws.get("observations") or []
        measurements = ws.get("measurements") or []
        if not channels or not observations or not measurements:
            return False, "workspace missing channels/observations/measurements"

        poi = measurements[0].get("config", {}).get("poi")
        if poi != "mu":
            return False, f"unsupported poi: {poi!r} (expected 'mu')"

        for ch in channels:
            for s in ch.get("samples") or []:
                mods = s.get("modifiers") or []
                for m in mods:
                    t = m.get("type")
                    n = m.get("name")
                    if t not in _ALLOWED_MODIFIER_TYPES:
                        return False, f"unsupported modifier: {t}:{n}"
                    if t in ("histosys", "normsys") and not isinstance(m.get("data"), dict):
                        return False, f"{t} modifier {n!r} must have object data"
                    if t in ("staterror", "shapesys") and not isinstance(m.get("data"), list):
                        return False, f"{t} modifier {n!r} must have list data"
        return True, ""
    except Exception as e:
        return False, f"workspace parse error: {type(e).__name__}: {e}"


def _obs_for_channel(ws: dict, channel_name: str) -> list[float]:
    for o in ws.get("observations") or []:
        if o.get("name") == channel_name:
            return [float(x) for x in (o.get("data") or [])]
    raise RuntimeError(f"missing observation for channel {channel_name!r}")


def _measurement_config(ws: dict, measurement_name: str) -> dict:
    if measurement_name:
        for m in ws.get("measurements") or []:
            if m.get("name") == measurement_name:
                return m.get("config") or {}
    meas = (ws.get("measurements") or [])[0]
    return meas.get("config") or {}


def _code4_coefficients_scalar(
    delta_dn: float, delta_up: float, *, alpha0: float = 1.0
) -> tuple[float, float, float, float, float, float]:
    # Scalar equivalent of pyhf.interpolators.code4 coefficient computation.
    import math

    a0 = float(alpha0)
    if a0 <= 0:
        raise ValueError("alpha0 must be positive")
    if delta_dn <= 0 or delta_up <= 0:
        raise ValueError("code4 deltas must be positive")

    Ainv = [
        [15.0 / (16 * a0), -15.0 / (16 * a0), -7.0 / 16.0, -7.0 / 16.0, 1.0 / 16 * a0, -1.0 / 16.0 * a0],
        [3.0 / (2 * (a0**2)), 3.0 / (2 * (a0**2)), -9.0 / (16 * a0), 9.0 / (16 * a0), 1.0 / 16, 1.0 / 16],
        [-5.0 / (8 * (a0**3)), 5.0 / (8 * (a0**3)), 5.0 / (8 * (a0**2)), 5.0 / (8 * (a0**2)), -1.0 / (8 * a0), 1.0 / (8 * a0)],
        [3.0 / (-2 * (a0**4)), 3.0 / (-2 * (a0**4)), -7.0 / (-8 * (a0**3)), 7.0 / (-8 * (a0**3)), -1.0 / (8 * (a0**2)), -1.0 / (8 * (a0**2))],
        [3.0 / (16 * (a0**5)), -3.0 / (16 * (a0**5)), -3.0 / (16 * (a0**4)), -3.0 / (16 * (a0**4)), 1.0 / (16 * (a0**3)), -1.0 / (16 * (a0**3))],
        [1.0 / (2 * (a0**6)), 1.0 / (2 * (a0**6)), -5.0 / (16 * (a0**5)), 5.0 / (16 * (a0**5)), 1.0 / (16 * (a0**4)), 1.0 / (16 * (a0**4))],
    ]

    du_a0 = delta_up**a0
    dd_a0 = delta_dn**a0
    b = [
        du_a0 - 1.0,
        dd_a0 - 1.0,
        math.log(delta_up) * du_a0,
        -math.log(delta_dn) * dd_a0,
        (math.log(delta_up) ** 2) * du_a0,
        (math.log(delta_dn) ** 2) * dd_a0,
    ]
    coeffs = []
    for row in Ainv:
        coeffs.append(sum(r * bb for r, bb in zip(row, b)))
    return (
        float(coeffs[0]),
        float(coeffs[1]),
        float(coeffs[2]),
        float(coeffs[3]),
        float(coeffs[4]),
        float(coeffs[5]),
    )


def _rf_code4_factor(ROOT, name: str, alpha, *, lo: float, hi: float) -> object:
    lo = float(lo)
    hi = float(hi)
    c1, c2, c3, c4, c5, c6 = _code4_coefficients_scalar(lo, hi, alpha0=1.0)
    lo_c = ROOT.RooConstVar(f"{name}_lo", f"{name}_lo", lo)
    hi_c = ROOT.RooConstVar(f"{name}_hi", f"{name}_hi", hi)
    expr = (
        "(@0>=1 ? pow(@1,@0) : "
        "(@0<=-1 ? pow(@2, -@0) : "
        f"(1 + ({c1})*@0 + ({c2})*pow(@0,2) + ({c3})*pow(@0,3) + ({c4})*pow(@0,4) + ({c5})*pow(@0,5) + ({c6})*pow(@0,6))))"
    )
    return ROOT.RooFormulaVar(name, name, expr, ROOT.RooArgList(alpha, hi_c, lo_c))


def _rf_code4p_delta(ROOT, name: str, alpha, *, lo: float, nom: float, hi: float) -> object:
    lo = float(lo)
    nom = float(nom)
    hi = float(hi)
    delta_up = hi - nom
    delta_dn = nom - lo
    S = 0.5 * (delta_up + delta_dn)
    A = 0.0625 * (delta_up - delta_dn)
    du_c = ROOT.RooConstVar(f"{name}_du", f"{name}_du", float(delta_up))
    dd_c = ROOT.RooConstVar(f"{name}_dd", f"{name}_dd", float(delta_dn))
    S_c = ROOT.RooConstVar(f"{name}_S", f"{name}_S", float(S))
    A_c = ROOT.RooConstVar(f"{name}_A", f"{name}_A", float(A))
    expr = (
        "(@0>1 ? (@1*@0) : "
        "(@0<-1 ? (@2*@0) : "
        "(@0*@3 + @4*(15*pow(@0,2) - 10*pow(@0,4) + 3*pow(@0,6)))))"
    )
    return ROOT.RooFormulaVar(name, name, expr, ROOT.RooArgList(alpha, du_c, dd_c, S_c, A_c))


def _rf_sum(ROOT, name: str, terms: list[object]) -> object:
    if not terms:
        return ROOT.RooConstVar(f"{name}_zero", f"{name}_zero", 0.0)
    acc = terms[0]
    for j, t in enumerate(terms[1:], start=1):
        acc = ROOT.RooFormulaVar(f"{name}_add_{j}", "@0+@1", ROOT.RooArgList(acc, t))
    return acc


def _rf_prod(ROOT, name: str, factors: list[object]) -> object:
    if not factors:
        return ROOT.RooConstVar(f"{name}_one", f"{name}_one", 1.0)
    acc = factors[0]
    for j, f in enumerate(factors[1:], start=1):
        acc = ROOT.RooFormulaVar(f"{name}_mul_{j}", "@0*@1", ROOT.RooArgList(acc, f))
    return acc


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
        doc["meta"]["note"] = "RooFit reference path (HistFactory subset: code4/code4p + constraints)"

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

        channels = ws_obj.get("channels") or []
        if not channels:
            raise RuntimeError("workspace has no channels")

        obs_by_channel: dict[str, list[float]] = {}
        n_total_bins = 0
        for ch in channels:
            ch_name = str(ch.get("name") or "")
            if not ch_name:
                raise RuntimeError("channel missing name")
            obs = _obs_for_channel(ws_obj, ch_name)
            obs_by_channel[ch_name] = obs
            n_total_bins += len(obs)

        if n_total_bins > 128:
            doc["status"] = "skipped"
            doc["reason"] = f"too_many_bins:{n_total_bins}"
            out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            return 0

        meas_cfg = _measurement_config(ws_obj, str(args.measurement_name).strip())

        mu = ROOT.RooRealVar("mu", "mu", 1.0, 0.0, 10.0)

        lumi = None
        lumi_aux = None
        lumi_sigma = None
        for p in meas_cfg.get("parameters") or []:
            if p.get("name") != "lumi":
                continue
            init = float((p.get("inits") or [1.0])[0])
            b0, b1 = (p.get("bounds") or [[0.0, 10.0]])[0]
            aux = float((p.get("auxdata") or [init])[0])
            sig = float((p.get("sigmas") or [1.0])[0])
            lumi = ROOT.RooRealVar("lumi", "lumi", init, float(b0), float(b1))
            lumi_aux = ROOT.RooConstVar("lumi_aux", "lumi_aux", aux)
            lumi_sigma = ROOT.RooConstVar("lumi_sigma", "lumi_sigma", sig)
            break

        alpha_normsys: dict[str, object] = {}
        alpha_histosys: dict[str, object] = {}
        staterror_gamma: dict[tuple[str, str, int], object] = {}
        staterror_sigma: dict[tuple[str, str, int], float] = {}
        shapesys_gamma: dict[tuple[str, str, str, int], object] = {}
        shapesys_tau: dict[tuple[str, str, str, int], float] = {}
        shapefactor_params: dict[tuple[str, int], object] = {}
        shapefactor_max_bins: dict[str, int] = {}

        uses_lumi = False
        for ch in channels:
            ch_name = str(ch.get("name"))
            n_bins = len(obs_by_channel[ch_name])
            for s in ch.get("samples") or []:
                s_name = str(s.get("name") or "sample")
                nom = [float(x) for x in (s.get("data") or [])]
                if len(nom) != n_bins:
                    raise RuntimeError(
                        f"sample {s_name!r} in {ch_name!r} has {len(nom)} bins, expected {n_bins}"
                    )
                for m in s.get("modifiers") or []:
                    mtype = str(m.get("type") or "")
                    mname = str(m.get("name") or "")
                    if mtype == "lumi":
                        uses_lumi = True
                    elif mtype == "normsys":
                        if mname not in alpha_normsys:
                            alpha_normsys[mname] = ROOT.RooRealVar(
                                f"alpha_{mname}", f"alpha_{mname}", 0.0, -5.0, 5.0
                            )
                    elif mtype == "histosys":
                        if mname not in alpha_histosys:
                            alpha_histosys[mname] = ROOT.RooRealVar(
                                f"alpha_{mname}", f"alpha_{mname}", 0.0, -5.0, 5.0
                            )
                    elif mtype == "staterror":
                        uncrt = m.get("data") or []
                        if not isinstance(uncrt, list) or len(uncrt) != n_bins:
                            raise RuntimeError(
                                f"staterror {mname!r} in {ch_name!r} has invalid data length"
                            )
                        for i in range(n_bins):
                            key = (mname, ch_name, i)
                            if key in staterror_gamma:
                                continue
                            sigma = (float(uncrt[i]) / float(nom[i])) if nom[i] > 0 else 0.0
                            staterror_sigma[key] = float(sigma)
                            g = ROOT.RooRealVar(
                                f"gamma_stat_{mname}_{ch_name}_{i}",
                                f"gamma_stat_{mname}_{ch_name}_{i}",
                                1.0,
                                1e-10,
                                10.0,
                            )
                            if sigma == 0.0:
                                g.setConstant(True)
                            staterror_gamma[key] = g
                    elif mtype == "shapesys":
                        uncrt = m.get("data") or []
                        if not isinstance(uncrt, list) or len(uncrt) != n_bins:
                            raise RuntimeError(
                                f"shapesys {mname!r} on {s_name!r} in {ch_name!r} has invalid data length"
                            )
                        for i in range(n_bins):
                            key = (mname, ch_name, s_name, i)
                            if key in shapesys_gamma:
                                continue
                            nom_i = float(nom[i])
                            unc_i = float(uncrt[i])
                            valid = (nom_i > 0.0) and (unc_i > 0.0)
                            tau = (nom_i * nom_i / (unc_i * unc_i)) if valid else 1.0
                            shapesys_tau[key] = float(tau)
                            g = ROOT.RooRealVar(
                                f"gamma_shape_{mname}_{ch_name}_{s_name}_{i}",
                                f"gamma_shape_{mname}_{ch_name}_{s_name}_{i}",
                                1.0,
                                1e-10,
                                10.0,
                            )
                            if not valid:
                                g.setConstant(True)
                            shapesys_gamma[key] = g
                    elif mtype == "shapefactor":
                        shapefactor_max_bins[mname] = max(
                            shapefactor_max_bins.get(mname, 0), n_bins
                        )

        if uses_lumi and lumi is None:
            doc["status"] = "skipped"
            doc["reason"] = "unsupported_workspace:lumi_used_but_missing_measurement_config"
            out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            return 0

        for mname, n_bins in shapefactor_max_bins.items():
            for i in range(int(n_bins)):
                shapefactor_params[(mname, i)] = ROOT.RooRealVar(
                    f"shapef_{mname}_{i}",
                    f"shapef_{mname}_{i}",
                    1.0,
                    0.0,
                    10.0,
                )

        nll_terms: list[object] = []

        for name, alpha in sorted(alpha_normsys.items()):
            nll_terms.append(
                ROOT.RooFormulaVar(f"pen_normsys_{name}", "0.5*(@0*@0)", ROOT.RooArgList(alpha))
            )
        for name, alpha in sorted(alpha_histosys.items()):
            nll_terms.append(
                ROOT.RooFormulaVar(f"pen_histosys_{name}", "0.5*(@0*@0)", ROOT.RooArgList(alpha))
            )

        if lumi is not None and lumi_aux is not None and lumi_sigma is not None:
            nll_terms.append(
                ROOT.RooFormulaVar(
                    "pen_lumi",
                    "0.5*pow((@0-@1)/@2,2)",
                    ROOT.RooArgList(lumi, lumi_aux, lumi_sigma),
                )
            )

        for (mname, ch_name, i), gamma in sorted(staterror_gamma.items()):
            sigma = float(staterror_sigma[(mname, ch_name, i)])
            if sigma == 0.0:
                continue
            sigma_c = ROOT.RooConstVar(
                f"sigma_stat_{mname}_{ch_name}_{i}",
                f"sigma_stat_{mname}_{ch_name}_{i}",
                sigma,
            )
            nll_terms.append(
                ROOT.RooFormulaVar(
                    f"pen_stat_{mname}_{ch_name}_{i}",
                    "0.5*pow((@0-1)/@1,2)",
                    ROOT.RooArgList(gamma, sigma_c),
                )
            )

        for (mname, ch_name, s_name, i), gamma in sorted(shapesys_gamma.items()):
            tau = float(shapesys_tau[(mname, ch_name, s_name, i)])
            if tau <= 0.0:
                continue
            tau_c = ROOT.RooConstVar(
                f"tau_shape_{mname}_{ch_name}_{s_name}_{i}",
                f"tau_shape_{mname}_{ch_name}_{s_name}_{i}",
                tau,
            )
            nll_terms.append(
                ROOT.RooFormulaVar(
                    f"pen_shape_{mname}_{ch_name}_{s_name}_{i}",
                    "@1*(@0 - log(@0))",
                    ROOT.RooArgList(gamma, tau_c),
                )
            )

        for ch in channels:
            ch_name = str(ch.get("name"))
            observed = obs_by_channel[ch_name]
            n_bins = len(observed)
            samples = ch.get("samples") or []
            if not samples:
                raise RuntimeError(f"channel {ch_name!r} has no samples")

            for i in range(n_bins):
                nu_i_terms: list[object] = []
                for s in samples:
                    s_name = str(s.get("name") or "sample")
                    nom = [float(x) for x in (s.get("data") or [])]

                    y = ROOT.RooConstVar(
                        f"nom_{ch_name}_{s_name}_{i}",
                        f"nom_{ch_name}_{s_name}_{i}",
                        float(nom[i]),
                    )

                    for m in s.get("modifiers") or []:
                        if m.get("type") != "histosys":
                            continue
                        mname = str(m.get("name") or "histosys")
                        alpha = alpha_histosys.get(mname)
                        if alpha is None:
                            raise RuntimeError(
                                f"internal: missing histosys param for {mname!r}"
                            )
                        data = m.get("data") or {}
                        lo_data = data.get("lo_data") or []
                        hi_data = data.get("hi_data") or []
                        if (
                            not isinstance(lo_data, list)
                            or not isinstance(hi_data, list)
                            or len(lo_data) != n_bins
                            or len(hi_data) != n_bins
                        ):
                            raise RuntimeError(
                                f"histosys {mname!r} in {ch_name!r} has invalid hi/lo_data length"
                            )
                        delta = _rf_code4p_delta(
                            ROOT,
                            f"delta_hist_{mname}_{ch_name}_{s_name}_{i}",
                            alpha,
                            lo=float(lo_data[i]),
                            nom=float(nom[i]),
                            hi=float(hi_data[i]),
                        )
                        y = ROOT.RooFormulaVar(
                            f"y_hist_{mname}_{ch_name}_{s_name}_{i}",
                            "@0+@1",
                            ROOT.RooArgList(y, delta),
                        )

                    factors: list[object] = [y]

                    for m in s.get("modifiers") or []:
                        mtype = str(m.get("type") or "")
                        mname = str(m.get("name") or "")
                        if mtype == "normfactor":
                            if mname == "mu":
                                factors.append(mu)
                            else:
                                key = f"nf_{mname}"
                                if not hasattr(main, "_nf_cache"):
                                    setattr(main, "_nf_cache", {})
                                cache = getattr(main, "_nf_cache")
                                if key not in cache:
                                    cache[key] = ROOT.RooRealVar(mname, mname, 1.0, 0.0, 10.0)
                                factors.append(cache[key])
                        elif mtype == "lumi":
                            if lumi is None:
                                raise RuntimeError(
                                    "lumi modifier present but lumi param missing"
                                )
                            factors.append(lumi)
                        elif mtype == "normsys":
                            alpha = alpha_normsys.get(mname)
                            if alpha is None:
                                raise RuntimeError(
                                    f"internal: missing normsys param for {mname!r}"
                                )
                            data = m.get("data") or {}
                            lo = float(data.get("lo"))
                            hi = float(data.get("hi"))
                            f = _rf_code4_factor(
                                ROOT,
                                f"f_norm_{mname}_{ch_name}_{s_name}_{i}",
                                alpha,
                                lo=lo,
                                hi=hi,
                            )
                            factors.append(f)
                        elif mtype == "staterror":
                            g = staterror_gamma.get((mname, ch_name, i))
                            if g is None:
                                raise RuntimeError(
                                    f"internal: missing staterror param for {mname!r} in {ch_name!r} bin {i}"
                                )
                            factors.append(g)
                        elif mtype == "shapesys":
                            g = shapesys_gamma.get((mname, ch_name, s_name, i))
                            if g is None:
                                raise RuntimeError(
                                    f"internal: missing shapesys param for {mname!r} on {s_name!r} in {ch_name!r} bin {i}"
                                )
                            factors.append(g)
                        elif mtype == "shapefactor":
                            p = shapefactor_params.get((mname, i))
                            if p is None:
                                raise RuntimeError(
                                    f"internal: missing shapefactor param for {mname!r} bin {i}"
                                )
                            factors.append(p)
                        elif mtype == "histosys":
                            continue

                    y_final = _rf_prod(ROOT, f"y_{ch_name}_{s_name}_{i}", factors)
                    nu_i_terms.append(y_final)

                nu_i = _rf_sum(ROOT, f"nu_{ch_name}_{i}", nu_i_terms)
                n_const = ROOT.RooConstVar(
                    f"n_{ch_name}_{i}", f"n_{ch_name}_{i}", float(observed[i])
                )
                nll_terms.append(
                    ROOT.RooFormulaVar(
                        f"nll_bin_{ch_name}_{i}",
                        "@0-@1*log(@0)",
                        ROOT.RooArgList(nu_i, n_const),
                    )
                )

        nll = _rf_sum(ROOT, "nll", nll_terms)

        def fit_and_capture(*, poi_fixed: bool, poi_value: float) -> tuple[int, float, dict[str, float]]:
            mu.setVal(float(poi_value))
            mu.setConstant(bool(poi_fixed))

            minim = ROOT.RooMinimizer(nll)
            minim.setPrintLevel(-1)
            minim.setStrategy(1)
            rc = int(minim.minimize("Minuit2", "migrad"))
            nll_val = float(nll.getVal())

            params: dict[str, float] = {"mu": float(mu.getVal())}
            if lumi is not None:
                params["lumi"] = float(lumi.getVal())
            for name, a in alpha_normsys.items():
                params[f"alpha_{name}"] = float(a.getVal())
            for name, a in alpha_histosys.items():
                params[f"alpha_{name}"] = float(a.getVal())
            for (mname, ch_name, i), g in staterror_gamma.items():
                params[f"gamma_stat_{mname}_{ch_name}_{i}"] = float(g.getVal())
            for (mname, ch_name, s_name, i), g in shapesys_gamma.items():
                params[f"gamma_shape_{mname}_{ch_name}_{s_name}_{i}"] = float(g.getVal())
            for (mname, i), p in shapefactor_params.items():
                params[f"shapef_{mname}_{i}"] = float(p.getVal())
            return rc, nll_val, params

        rc_free, nll_free, params_free = fit_and_capture(poi_fixed=False, poi_value=1.0)
        rc_mu0, nll_mu0, params_mu0 = fit_and_capture(poi_fixed=True, poi_value=0.0)

        muhat = float(params_free.get("mu", float(mu.getVal())))
        q0 = max(0.0, 2.0 * (nll_mu0 - nll_free))
        z0 = float(q0) ** 0.5

        doc["status"] = "ok" if (rc_free == 0 and rc_mu0 == 0) else "failed"
        doc["reason"] = "" if doc["status"] == "ok" else f"root_minimizer_failed:rc_free={rc_free},rc_mu0={rc_mu0}"
        doc["fit"] = {
            "muhat": muhat,
            "nll": nll_free,
            "minimizer_rc": rc_free,
            "minimizer": "Minuit2:migrad",
            "strategy": 1,
        }
        doc["fit_conditional"] = {
            "poi0": 0.0,
            "nll": nll_mu0,
            "minimizer_rc": rc_mu0,
            "minimizer": "Minuit2:migrad",
            "strategy": 1,
        }
        doc["profile"] = {"poi0": 0.0, "nll_mu0": nll_mu0, "q0": q0, "z0": z0}
        doc["params"] = params_free
        doc["params_mu0"] = params_mu0
        doc["meta"]["n_channels"] = int(len(channels))
        doc["meta"]["n_total_bins"] = int(n_total_bins)
        doc["meta"]["n_nuisance_normsys"] = int(len(alpha_normsys))
        doc["meta"]["n_nuisance_histosys"] = int(len(alpha_histosys))
        doc["meta"]["n_nuisance_staterror"] = int(len(staterror_gamma))
        doc["meta"]["n_nuisance_shapesys"] = int(len(shapesys_gamma))
        doc["meta"]["n_nuisance_shapefactor"] = int(len(shapefactor_params))
    except Exception as e:
        doc["status"] = "skipped"
        doc["reason"] = f"{type(e).__name__}: {e}"

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
