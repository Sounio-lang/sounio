#!/usr/bin/env python3
"""Decision gate for ABIDE dynamic-FC switch-event artifacts.

This gate combines the target-builder audit with switch-event model metrics.
It emits a bounded follow-up verdict only. It does not authorize clinical,
diagnostic, biomarker, mechanism, ASD-detection, or O-SSM superiority claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.abide_dynamic_fc_switching_decision.v1"
CLAIM_BOUNDARY = (
    "Dynamic-FC switching decision gate only; no diagnostic, biomarker, "
    "mechanism, ASD-detection, clinical-decision, or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-audit-json", required=True, type=Path)
    parser.add_argument("--gate-json", required=True, type=Path)
    parser.add_argument("--gate-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-decision-subjects", type=int, default=50)
    parser.add_argument("--min-decision-sites", type=int, default=5)
    parser.add_argument("--min-decision-null-permutations", type=int, default=20)
    parser.add_argument("--min-ap-gap", type=float, default=0.03)
    parser.add_argument("--min-proper-score-gap", type=float, default=0.002)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--require-split-policy", default="leave_one_site_out")
    parser.add_argument("--require-full-sounio-ossm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def by_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        if model:
            out[model] = row
    return out


def score(row: dict[str, str], field: str, default: float = 0.0) -> float:
    return parse_float(row.get(field, ""), default=default)


def lower_is_better_gain(ossm: dict[str, str], controls: list[dict[str, str]], field: str) -> tuple[float, str]:
    if not controls:
        return 0.0, "missing"
    best = min(controls, key=lambda row: score(row, field, default=float("inf")))
    return score(best, field, default=float("inf")) - score(ossm, field, default=float("inf")), best.get("model", "")


def higher_is_better_gain(ossm: dict[str, str], controls: list[dict[str, str]], field: str) -> tuple[float, str]:
    if not controls:
        return 0.0, "missing"
    best = max(controls, key=lambda row: score(row, field, default=float("-inf")))
    return score(ossm, field, default=float("-inf")) - score(best, field, default=float("-inf")), best.get("model", "")


def low_power_reasons(target: dict[str, Any], gate: dict[str, Any], summary: dict[str, dict[str, str]], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    subject_count = int(parse_float(target.get("subject_count", 0)))
    site_count = int(parse_float(target.get("site_count", 0)))
    split_policy = str(target.get("parameters", {}).get("split_policy", ""))
    if subject_count < args.min_decision_subjects:
        reasons.append(f"subject_count {subject_count} < min_decision_subjects {args.min_decision_subjects}")
    if site_count < args.min_decision_sites:
        reasons.append(f"site_count {site_count} < min_decision_sites {args.min_decision_sites}")
    if split_policy != args.require_split_policy:
        reasons.append(f"split_policy {split_policy!r} != required {args.require_split_policy!r}")
    null_counts = [
        score(row, "null_permutations_mean", default=0.0)
        for row in summary.values()
        if row.get("null_permutations_mean") not in {"", None}
    ]
    max_nulls = max(null_counts) if null_counts else 0.0
    if max_nulls < args.min_decision_null_permutations:
        reasons.append(
            f"max null_permutations_mean {max_nulls:g} < min_decision_null_permutations {args.min_decision_null_permutations}"
        )
    if gate.get("verdict") not in {
        "O_SSM_RESERVOIR_GATE_EXECUTED_NO_PROMOTION",
        "TRAINED_O_SSM_GATE_EXECUTED_NO_PROMOTION",
        "FULL_O_SSM_GATE_EXECUTED",
    }:
        reasons.append(f"model gate verdict is {gate.get('verdict')!r}")
    return reasons


def missing_control_reasons(summary: dict[str, dict[str, str]]) -> list[str]:
    required = {
        "base_rate",
        "persistence",
        "logistic",
        "gru_reservoir",
        "hssm_reservoir",
        "ossm_reservoir",
        "trained_hssm",
        "trained_ossm",
    }
    return [f"missing required model surface: {model}" for model in sorted(required - set(summary))]


def decide(target: dict[str, Any], gate: dict[str, Any], summary_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[str, list[str], dict[str, Any]]:
    summary = by_model(summary_rows)
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {}

    if target.get("status") != "pass":
        return "BLOCKED_TARGET_AUDIT_FAILED", list(target.get("failures", [])), diagnostics

    missing = missing_control_reasons(summary)
    if missing:
        return "UNDERCONTROLLED_MISSING_SURFACES", missing, diagnostics

    ossm_model = "trained_ossm" if "trained_ossm" in summary else "ossm_reservoir"
    ossm = summary[ossm_model]
    controls = [row for model, row in summary.items() if model != ossm_model]
    ap_gain, best_ap_control = higher_is_better_gain(ossm, controls, "average_precision_mean")
    brier_gain, best_brier_control = lower_is_better_gain(ossm, controls, "brier_mean")
    log_loss_gain, best_log_loss_control = lower_is_better_gain(ossm, controls, "log_loss_mean")
    null_p = score(ossm, "null_average_precision_p_ge_mean", default=1.0)
    diagnostics.update(
        {
            "ossm_model_surface": ossm_model,
            "ap_gain_over_best_control": ap_gain,
            "best_ap_control": best_ap_control,
            "brier_gain_over_best_control": brier_gain,
            "best_brier_control": best_brier_control,
            "log_loss_gain_over_best_control": log_loss_gain,
            "best_log_loss_control": best_log_loss_control,
            "ossm_null_average_precision_p_ge": null_p,
        }
    )

    low_power = low_power_reasons(target, gate, summary, args)
    if low_power:
        reasons.extend(low_power)
        return "UNDERCONTROLLED_LOW_POWER_OR_SMOKE_SPLIT", reasons, diagnostics

    if args.require_full_sounio_ossm and "full_sounio_ossm" not in summary:
        reasons.append("full_sounio_ossm surface missing")
        return "UNDERCONTROLLED_RESERVOIR_ONLY", reasons, diagnostics

    reasons.extend(gate.get("reasons", []))
    if "reservoir_o_ssm_surface_only_not_full_trained_sio_model" in reasons:
        return "NO_PROMOTION_RESERVOIR_ONLY", reasons, diagnostics
    if "trained_python_o_ssm_surface_not_full_sounio_model" in reasons:
        return "NO_PROMOTION_PYTHON_TRAINED_ONLY", reasons, diagnostics

    proper_score_pass = brier_gain >= args.min_proper_score_gap or log_loss_gain >= args.min_proper_score_gap
    if ap_gain >= args.min_ap_gap and proper_score_pass and null_p <= args.alpha:
        return "EXPLORATORY_O_SSM_SWITCHING_SIGNAL", ["passes exploratory metric thresholds"], diagnostics
    if ap_gain < args.min_ap_gap:
        reasons.append(f"AP gain {ap_gain:.6f} < min_ap_gap {args.min_ap_gap:.6f}")
    if not proper_score_pass:
        reasons.append(
            "O-SSM does not improve Brier or log-loss over the best control by "
            f"{args.min_proper_score_gap:.6f}"
        )
    if null_p > args.alpha:
        reasons.append(f"O-SSM null AP p_ge {null_p:.6f} > alpha {args.alpha:.6f}")
    return "NEGATIVE_OR_UNDERCONTROLLED_NO_ROBUST_O_SSM_SIGNAL", reasons, diagnostics


def markdown(payload: dict[str, Any]) -> str:
    diagnostics = payload["diagnostics"]
    lines = [
        "# ABIDE Dynamic-FC Switching Decision",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- Overall verdict: `{payload['overall_verdict']}`",
        f"- Target audit status: `{payload['target_status']}`",
        f"- Model gate verdict: `{payload['gate_verdict']}`",
        "",
        "## Diagnostics",
        "",
        f"- O-SSM surface: `{diagnostics.get('ossm_model_surface', 'missing')}`",
        f"- AP gain over best control: `{diagnostics.get('ap_gain_over_best_control', '')}`",
        f"- Brier gain over best control: `{diagnostics.get('brier_gain_over_best_control', '')}`",
        f"- Log-loss gain over best control: `{diagnostics.get('log_loss_gain_over_best_control', '')}`",
        f"- O-SSM null AP p>=: `{diagnostics.get('ossm_null_average_precision_p_ge', '')}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in payload["reasons"])
    if not payload["reasons"]:
        lines.append("- no blocking reason recorded")
    lines.extend(
        [
            "",
            "Interpretation: this gate decides whether the dynamic-FC switching artifacts can support a follow-up decision.",
            "A positive exploratory verdict is not a clinical, biomarker, mechanism, ASD-detection, or O-SSM superiority claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = read_json(args.target_audit_json)
    gate = read_json(args.gate_json)
    summary_rows = read_tsv(args.gate_summary)
    verdict, reasons, diagnostics = decide(target, gate, summary_rows, args)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "target_audit_json": str(args.target_audit_json.resolve()),
        "gate_json": str(args.gate_json.resolve()),
        "gate_summary": str(args.gate_summary.resolve()),
        "target_status": target.get("status", "missing"),
        "target_failures": target.get("failures", []),
        "gate_verdict": gate.get("verdict", "missing"),
        "gate_reasons": gate.get("reasons", []),
        "parameters": {
            "min_decision_subjects": args.min_decision_subjects,
            "min_decision_sites": args.min_decision_sites,
            "min_decision_null_permutations": args.min_decision_null_permutations,
            "min_ap_gap": args.min_ap_gap,
            "min_proper_score_gap": args.min_proper_score_gap,
            "alpha": args.alpha,
            "require_split_policy": args.require_split_policy,
            "require_full_sounio_ossm": args.require_full_sounio_ossm,
        },
        "target_summary": {
            "subject_count": target.get("subject_count"),
            "site_count": target.get("site_count"),
            "split_policy": target.get("parameters", {}).get("split_policy"),
            "primary_switch_event_frac": target.get("primary_test_switch_event_frac"),
            "primary_window_switch_corr": target.get("primary_switching_rate_window_count_corr"),
        },
        "model_summary": summary_rows,
        "diagnostics": diagnostics,
        "overall_verdict": verdict,
        "reasons": reasons,
    }
    (args.output_dir / "dynamic_fc_switching_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dynamic_fc_switching_decision.md").write_text(markdown(payload), encoding="utf-8")
    print(f"verdict={verdict}")
    print(f"outputs={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
