#!/usr/bin/env python3
"""Compare two benchmark snapshot directories (seed tool).

This is intended to support third-party replication by producing a structured diff between:
- published snapshot artifact set
- rerun snapshot artifact set
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


@dataclass
class Snapshot:
    root: Path
    manifest: dict[str, Any]

    @staticmethod
    def load(root: Path) -> "Snapshot":
        root = root.resolve()
        manifest = load_json(root / "baseline_manifest.json")
        return Snapshot(root=root, manifest=manifest)


def as_dataset_set(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for d in manifest.get("datasets", []):
        try:
            out.add((str(d["id"]), str(d["sha256"])))
        except Exception:
            pass
    return out


def index_results(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in manifest.get("results", []):
        try:
            out[str(r["suite"])] = {"path": str(r["path"]), "sha256": str(r["sha256"])}
        except Exception:
            pass
    return out


def compare_hep_suite(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = load_json(a_path)
    b = load_json(b_path)
    out: dict[str, Any] = {"suite": "hep", "case_diffs": [], "ok": True}

    def case_map(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
        base = a_path.parent if obj is a else b_path.parent
        out2 = {}
        for e in obj.get("cases", []):
            cid = str(e.get("case", ""))
            if not cid:
                continue
            inst = load_json((base / str(e["path"])).resolve())
            out2[cid] = inst
        return out2

    am = case_map(a)
    bm = case_map(b)
    keys = sorted(set(am.keys()) | set(bm.keys()))
    for k in keys:
        if k not in am or k not in bm:
            out["case_diffs"].append({"case": k, "status": "missing_in_one"})
            out["ok"] = False
            continue
        ap = am[k].get("parity", {})
        bp = bm[k].get("parity", {})
        a_ok = bool(ap.get("ok", False))
        b_ok = bool(bp.get("ok", False))
        if not (a_ok and b_ok):
            out["ok"] = False
        out["case_diffs"].append(
            {
                "case": k,
                "parity_ok": {"a": a_ok, "b": b_ok},
                "abs_diff": {"a": float(ap.get("abs_diff", 0.0)), "b": float(bp.get("abs_diff", 0.0))},
                "nll_speedup": {
                    "a": float(am[k].get("timing", {}).get("speedup_pyhf_over_nextstat", 0.0)),
                    "b": float(bm[k].get("timing", {}).get("speedup_pyhf_over_nextstat", 0.0)),
                },
                "fit_status": {
                    "a": str((am[k].get("fit") or {}).get("status", "")) if isinstance(am[k].get("fit"), dict) else "",
                    "b": str((bm[k].get("fit") or {}).get("status", "")) if isinstance(bm[k].get("fit"), dict) else "",
                },
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Snapshot dir A (contains baseline_manifest.json).")
    ap.add_argument("--b", required=True, help="Snapshot dir B (contains baseline_manifest.json).")
    ap.add_argument("--out", default="", help="Optional output JSON path.")
    args = ap.parse_args()

    sa = Snapshot.load(Path(args.a))
    sb = Snapshot.load(Path(args.b))

    ra = index_results(sa.manifest)
    rb = index_results(sb.manifest)

    datasets_a = as_dataset_set(sa.manifest)
    datasets_b = as_dataset_set(sb.manifest)

    report: dict[str, Any] = {
        "schema_version": "nextstat.snapshot_comparison.v1",
        "a": {"root": str(sa.root), "snapshot_id": sa.manifest.get("snapshot_id", "")},
        "b": {"root": str(sb.root), "snapshot_id": sb.manifest.get("snapshot_id", "")},
        "datasets": {
            "same": sorted(list(datasets_a & datasets_b)),
            "only_a": sorted(list(datasets_a - datasets_b)),
            "only_b": sorted(list(datasets_b - datasets_a)),
        },
        "results": {"only_a": sorted(list(set(ra.keys()) - set(rb.keys()))), "only_b": sorted(list(set(rb.keys()) - set(ra.keys())))},
        "suites": [],
        "ok": True,
    }

    if report["datasets"]["only_a"] or report["datasets"]["only_b"]:
        report["ok"] = False

    # Verify result file hashes and do suite-specific comparisons where we understand the format.
    for suite in sorted(set(ra.keys()) & set(rb.keys())):
        a_rel = ra[suite]["path"]
        b_rel = rb[suite]["path"]
        a_path = (sa.root / a_rel).resolve()
        b_path = (sb.root / b_rel).resolve()

        a_sha = sha256_file(a_path) if a_path.exists() else ""
        b_sha = sha256_file(b_path) if b_path.exists() else ""
        suite_entry: dict[str, Any] = {
            "suite": suite,
            "paths": {"a": a_rel, "b": b_rel},
            "sha256_ok": {"a": (a_sha == ra[suite]["sha256"]), "b": (b_sha == rb[suite]["sha256"])},
        }
        if not suite_entry["sha256_ok"]["a"] or not suite_entry["sha256_ok"]["b"]:
            report["ok"] = False

        if suite == "hep":
            suite_entry["comparison"] = compare_hep_suite(a_path, b_path)
            if not suite_entry["comparison"].get("ok", False):
                report["ok"] = False

        report["suites"].append(suite_entry)

    txt = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if str(args.out).strip():
        Path(args.out).resolve().write_text(txt)
    else:
        print(txt, end="")

    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

