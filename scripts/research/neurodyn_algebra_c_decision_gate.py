#!/usr/bin/env python3
"""Apply the Algebra-C continuous associator fidelity decision contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.algebra_c.decision_gate.v1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", required=True, type=Path)
    ap.add_argument("--target-audit", required=True, type=Path)
    ap.add_argument("--fidelity-summary", required=True, type=Path, help="fidelity_summary.tsv")
    ap.add_argument("--external-summary", required=True, type=Path, help="GRU overall_metrics.tsv")
    ap.add_argument("--null-summary", action="append", default=[], type=Path, help="null fidelity_summary.tsv")
    ap.add_argument(
        "--trajectory-null-summary",
        action="append",
        default=[],
        type=Path,
        help="trajectory-preserving continuous-target permutation null fidelity_summary.tsv",
    )
    ap.add_argument("--o-condition", default="octonion")
    ap.add_argument("--a8-condition", default="a8")
    ap.add_argument("--h-condition", default="h")
    ap.add_argument("--projection-condition", default="projection")
    ap.add_argument("--o-model", default="O-SSM")
    ap.add_argument("--a8-model", default="A8-SSM")
    ap.add_argument("--h-model", default="H-SSM")
    ap.add_argument("--projection-model", default="O-SSM")
    ap.add_argument("--generic-model", default="gru")
    ap.add_argument("--higher-capacity-generic-model", default="gru_wide")
    ap.add_argument("--spearman-margin", type=float, default=0.10)
    ap.add_argument("--r2-margin", type=float, default=0.05)
    ap.add_argument("--null-count-required", type=int, default=99)
    ap.add_argument("--trajectory-null-count-required", type=int, default=20)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_fidelity(rows: list[dict[str, str]], condition: str, model: str) -> dict[str, float] | None:
    for row in rows:
        if row.get("condition") == condition and row.get("model") == model:
            return {
                "spearman": float(row["spearman_mean"]),
                "r2": float(row["r2_mean"]),
                "sign_auc": float(row["sign_auc_mean"]) if row.get("sign_auc_mean") not in {"", None} else 0.0,
            }
    return None


def find_external(rows: list[dict[str, str]], model: str) -> dict[str, float] | None:
    for row in rows:
        if row.get("model") == model:
            return {
                "spearman": float(row["spearman_mean"]),
                "r2": float(row["r2_mean"]),
                "sign_auc": float(row["sign_auc_mean"]) if row.get("sign_auc_mean") not in {"", None} else 0.0,
                "parameter_count": float(row.get("parameter_count", 0.0)),
            }
    return None


def envelope(true_value: float, null_values: list[float]) -> dict[str, Any]:
    return {
        "true": true_value,
        "null_count": len(null_values),
        "null_min": min(null_values) if null_values else None,
        "null_max": max(null_values) if null_values else None,
        "null_mean": statistics.fmean(null_values) if null_values else None,
        "null_ge_true": sum(1 for value in null_values if value >= true_value),
        "plus_one_p_ge_true": (1 + sum(1 for value in null_values if value >= true_value)) / (len(null_values) + 1)
        if null_values
        else None,
    }


def main() -> int:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    target_audit = json.loads(args.target_audit.read_text(encoding="utf-8"))
    fidelity_rows = read_tsv(args.fidelity_summary)
    external_rows = read_tsv(args.external_summary)
    failures: list[str] = []
    if target_audit.get("decision") != "ALGEBRA_C_TARGET_AUDIT_PASS":
        failures.append("target audit did not pass")

    o = find_fidelity(fidelity_rows, args.o_condition, args.o_model)
    a8 = find_fidelity(fidelity_rows, args.a8_condition, args.a8_model)
    h = find_fidelity(fidelity_rows, args.h_condition, args.h_model)
    projection = find_fidelity(fidelity_rows, args.projection_condition, args.projection_model)
    generic = find_external(external_rows, args.generic_model)
    higher_capacity_generic = find_external(external_rows, args.higher_capacity_generic_model)
    metrics = {
        "o": o,
        "a8": a8,
        "h": h,
        "projection": projection,
        "generic": generic,
        "higher_capacity_generic": higher_capacity_generic,
    }
    for name, value in metrics.items():
        if name == "higher_capacity_generic":
            continue
        if value is None:
            failures.append(f"missing required surface: {name}")
    warnings: list[str] = []
    if higher_capacity_generic is None:
        warnings.append(
            "WARN_UNDERCONTROLLED: higher-capacity generic control missing; run external baseline model=gru_wide"
        )
    null_spearman: list[float] = []
    null_r2: list[float] = []
    for path in args.null_summary:
        null_rows = read_tsv(path)
        row = find_fidelity(null_rows, args.o_condition, args.o_model)
        if row is None:
            failures.append(f"missing O-SSM null metric in {path}")
            continue
        null_spearman.append(row["spearman"])
        null_r2.append(row["r2"])
    trajectory_null_spearman: list[float] = []
    trajectory_null_r2: list[float] = []
    for path in args.trajectory_null_summary:
        null_rows = read_tsv(path)
        row = find_fidelity(null_rows, args.o_condition, args.o_model)
        if row is None:
            failures.append(f"missing O-SSM trajectory-preserving null metric in {path}")
            continue
        trajectory_null_spearman.append(row["spearman"])
        trajectory_null_r2.append(row["r2"])

    envelopes: dict[str, Any] = {}
    if o is not None:
        control_values = [value for key, value in metrics.items() if key != "o" and value is not None]
        if o["spearman"] <= 0.0:
            failures.append("O-SSM Spearman is not positive")
        if control_values:
            best_control_spearman = max(value["spearman"] for value in control_values)
            best_control_r2 = max(value["r2"] for value in control_values)
            if o["spearman"] - best_control_spearman < args.spearman_margin:
                failures.append("O-SSM Spearman margin over best control is insufficient")
            if o["r2"] - best_control_r2 < args.r2_margin:
                failures.append("O-SSM R2 margin over best control is insufficient")
        if projection is not None and o["spearman"] - projection["spearman"] < args.spearman_margin:
            failures.append("associative projection did not collapse below O-SSM margin")
        envelopes = {
            "spearman": envelope(o["spearman"], null_spearman),
            "r2": envelope(o["r2"], null_r2),
            "trajectory_preserving_spearman": envelope(o["spearman"], trajectory_null_spearman),
            "trajectory_preserving_r2": envelope(o["r2"], trajectory_null_r2),
        }
        if len(trajectory_null_spearman) < args.trajectory_null_count_required:
            failures.append(
                "trajectory-preserving continuous-target retrain nulls required; "
                f"found {len(trajectory_null_spearman)}"
            )
        if len(null_spearman) < args.null_count_required:
            failures.append(f"99 retrain nulls required; found {len(null_spearman)}")
        if len(trajectory_null_spearman) >= args.trajectory_null_count_required:
            if envelopes["trajectory_preserving_spearman"]["null_ge_true"] != 0:
                failures.append("trajectory-preserving null Spearman envelope reaches/exceeds true O-SSM")
            if envelopes["trajectory_preserving_r2"]["null_ge_true"] != 0:
                failures.append("trajectory-preserving null R2 envelope reaches/exceeds true O-SSM")
        if len(null_spearman) >= args.null_count_required:
            if envelopes["spearman"]["null_ge_true"] != 0:
                failures.append("null Spearman envelope reaches/exceeds true O-SSM")
            if envelopes["r2"]["null_ge_true"] != 0:
                failures.append("null R2 envelope reaches/exceeds true O-SSM")

    if failures:
        decision = "ALGEBRA_C_BLOCKED_OR_NEGATIVE"
    elif warnings:
        decision = "ALGEBRA_C_WARN_UNDERCONTROLLED"
    else:
        decision = "ALGEBRA_C_CONTINUOUS_ASSOCIATOR_FIDELITY_CANDIDATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "claim_boundary": (
            "Synthetic generator-matched inductive-bias result only; no clinical, biomarker, "
            "biological-mechanism, MDD/ADHD, or broad O-SSM superiority claim."
        ),
        "inputs": {
            "prereg": str(args.prereg),
            "prereg_sha256": sha256_file(args.prereg),
            "target_audit": str(args.target_audit),
            "target_audit_sha256": sha256_file(args.target_audit),
            "fidelity_summary": str(args.fidelity_summary),
            "fidelity_summary_sha256": sha256_file(args.fidelity_summary),
            "external_summary": str(args.external_summary),
            "external_summary_sha256": sha256_file(args.external_summary),
            "null_summaries": [str(path) for path in args.null_summary],
        },
        "thresholds": {
            "spearman_margin": args.spearman_margin,
            "r2_margin": args.r2_margin,
            "null_count_required": args.null_count_required,
            "trajectory_null_count_required": args.trajectory_null_count_required,
        },
        "metrics": metrics,
        "null_envelopes": envelopes,
        "warnings": warnings,
        "failures": failures,
    }
    (out / "algebra_c_decision_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
