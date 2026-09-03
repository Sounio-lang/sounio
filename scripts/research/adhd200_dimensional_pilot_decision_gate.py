#!/usr/bin/env python3
"""Decision gate for ADHD-200 dimensional O-SSM pilot artifacts.

The gate compares O-SSM hidden-state dimensional probes against associative,
generic recurrent, covariate, and static-input controls. It emits an honest
pilot verdict only; it does not promote diagnostic, biomarker, mechanism, or
O-SSM superiority claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.adhd200_dimensional_pilot_decision.v1"
CLAIM_BOUNDARY = (
    "Pilot decision gate only. No diagnostic, biomarker, treatment-response, "
    "biological-mechanism, or O-SSM superiority claim."
)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def row_score(row: dict[str, str]) -> float:
    return parse_float(row.get("spearman_mean", "0"))


def row_null_p(row: dict[str, str]) -> float:
    return parse_float(row.get("null_spearman_p_ge_mean", "1"), default=1.0)


def row_null_count(row: dict[str, str]) -> float:
    return parse_float(row.get("null_permutations_mean", "0"), default=0.0)


def best(rows: list[dict[str, str]], phenotype: str, surface: str | None = None, models: set[str] | None = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("phenotype") == phenotype]
    if surface is not None:
        candidates = [row for row in candidates if row.get("surface") == surface]
    if models is not None:
        candidates = [row for row in candidates if row.get("model") in models]
    if not candidates:
        return None
    return max(candidates, key=row_score)


def summarize_phenotype(
    phenotype: str,
    rows: list[dict[str, str]],
    min_gap: float,
    alpha: float,
) -> dict[str, Any]:
    o_hidden = best(rows, phenotype, surface="hidden", models={"O-SSM"})
    h_hidden = best(rows, phenotype, surface="hidden", models={"H-SSM"})
    generic = best(
        rows,
        phenotype,
        models={"gru_reservoir", "gru_reservoir_wide", "trained_rnn", "trained_rnn_wide"},
    )
    generic_hidden = best(rows, phenotype, surface="hidden", models={"gru_reservoir", "gru_reservoir_wide"})
    generic_trained = best(rows, phenotype, surface="trained_recurrent_prediction", models={"trained_rnn", "trained_rnn_wide"})
    covariates = best(rows, phenotype, surface="covariates")
    static = best(rows, phenotype, surface="static_input_summary")

    controls = [row for row in [h_hidden, generic, covariates, static] if row is not None]
    best_control = max(controls, key=row_score) if controls else None
    result: dict[str, Any] = {
        "phenotype": phenotype,
        "o_ssm_hidden": o_hidden,
        "h_ssm_hidden": h_hidden,
        "generic_best": generic,
        "generic_hidden_best": generic_hidden,
        "generic_trained_best": generic_trained,
        "covariates_best": covariates,
        "static_input_best": static,
        "best_control": best_control,
        "min_gap": min_gap,
        "alpha": alpha,
    }
    if o_hidden is None:
        result["verdict"] = "BLOCKED_MISSING_O_SSM_HIDDEN"
        result["reason"] = "No O-SSM hidden-state summary row was found."
        return result
    if h_hidden is None:
        result["verdict"] = "BLOCKED_MISSING_ASSOCIATIVE_CONTROL"
        result["reason"] = "No H-SSM hidden-state control row was found."
        return result
    if generic is None:
        result["verdict"] = "UNDERCONTROLLED_MISSING_GENERIC_CONTROL"
        result["reason"] = "No generic recurrent baseline row was found."
        return result

    o_score = row_score(o_hidden)
    o_null_p = row_null_p(o_hidden)
    best_control_score = row_score(best_control) if best_control is not None else float("-inf")
    gap = o_score - best_control_score
    result["o_score"] = o_score
    result["o_null_p"] = o_null_p
    result["best_control_score"] = best_control_score
    result["o_minus_best_control"] = gap

    if best_control is not None and best_control.get("surface") in {"covariates", "static_input_summary"} and row_score(best_control) >= o_score - min_gap:
        result["verdict"] = "NEGATIVE_NUISANCE_OR_STATIC_COMPETITIVE"
        result["reason"] = "Covariate or static-input surface matches or exceeds the O-SSM hidden score within the preregistered margin."
    elif generic is not None and row_score(generic) >= o_score - min_gap:
        result["verdict"] = "NEGATIVE_GENERIC_RECURRENT_COMPETITIVE"
        result["reason"] = "Generic recurrent baseline matches or exceeds the O-SSM hidden score within the preregistered margin."
    elif h_hidden is not None and row_score(h_hidden) >= o_score - min_gap:
        result["verdict"] = "NEGATIVE_ASSOCIATIVE_CONTROL_COMPETITIVE"
        result["reason"] = "Associative H-SSM hidden state matches or exceeds the O-SSM hidden score within the preregistered margin."
    elif gap >= min_gap and o_null_p <= alpha:
        result["verdict"] = "EXPLORATORY_O_SSM_SIGNAL"
        result["reason"] = "O-SSM hidden state exceeds the available controls and passes the pilot null threshold."
    else:
        result["verdict"] = "NO_ROBUST_PILOT_SIGNAL"
        result["reason"] = "O-SSM hidden state does not clear both the control margin and null threshold."
    return result


def overall_verdict(phenotype_rows: list[dict[str, Any]]) -> str:
    verdicts = [row["verdict"] for row in phenotype_rows]
    if any(value.startswith("BLOCKED") for value in verdicts):
        return "BLOCKED"
    if any(value.startswith("UNDERCONTROLLED") for value in verdicts):
        return "UNDERCONTROLLED"
    if any(value == "EXPLORATORY_O_SSM_SIGNAL" for value in verdicts):
        if all(value == "EXPLORATORY_O_SSM_SIGNAL" for value in verdicts):
            return "PILOT_EXPLORATORY_O_SSM_SIGNAL_ALL_PRIMARY"
        return "PILOT_MIXED_EXPLORATORY"
    if all(value.startswith("NEGATIVE") for value in verdicts):
        return "PILOT_NEGATIVE_CONTROLS_SUFFICE"
    return "PILOT_NO_ROBUST_SIGNAL"


def low_power_reasons(readiness: dict[str, Any], rows: list[dict[str, str]], min_subjects: int, min_null_permutations: int) -> list[str]:
    reasons = []
    row_count = int(parse_float(readiness.get("row_count", 0), default=0.0))
    if row_count < min_subjects:
        reasons.append(f"row_count {row_count} < min_decision_subjects {min_subjects}")
    null_counts = [row_null_count(row) for row in rows if row.get("null_permutations_mean") not in {None, ""}]
    min_observed_nulls = min(null_counts) if null_counts else 0.0
    if min_observed_nulls < min_null_permutations:
        reasons.append(
            f"min null_permutations_mean {min_observed_nulls:g} < min_decision_null_permutations {min_null_permutations}"
        )
    return reasons


def downgrade_low_power(phenotype_rows: list[dict[str, Any]], reasons: list[str]) -> None:
    reason_text = "; ".join(reasons)
    for row in phenotype_rows:
        if row["verdict"].startswith("BLOCKED") or row["verdict"].startswith("UNDERCONTROLLED"):
            continue
        row["pilot_metric_verdict"] = row["verdict"]
        row["verdict"] = "UNDERCONTROLLED_LOW_POWER"
        row["reason"] = (
            "Pilot scores were computed, but the run is underpowered for a negative or positive decision: "
            + reason_text
            + "."
        )


def compact_row(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    model = row.get("model", "")
    surface = row.get("surface", "")
    score = row_score(row)
    null_p = row_null_p(row)
    return f"{model}/{surface} spearman={score:.6f} null_p_ge={null_p:.6f}"


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ADHD-200 Dimensional Pilot Decision",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        f"Overall verdict: `{payload['overall_verdict']}`",
        "",
        "| phenotype | verdict | O-SSM hidden | best control | reason |",
        "|---|---|---|---|---|",
    ]
    for row in payload["phenotype_decisions"]:
        lines.append(
            "| {phenotype} | {verdict} | {ossm} | {control} | {reason} |".format(
                phenotype=row["phenotype"],
                verdict=row["verdict"],
                ossm=compact_row(row.get("o_ssm_hidden")),
                control=compact_row(row.get("best_control")),
                reason=row["reason"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this gate decides only whether the pilot artifacts are worth deeper follow-up.",
            "A positive pilot verdict remains exploratory until a larger independently reviewed cohort, trained generic baselines, leakage audits, and external review all pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", required=True, type=Path)
    parser.add_argument("--state-summary", required=True, type=Path)
    parser.add_argument("--generic-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--primary-phenotypes", default="inattention,hyperactivity_impulsivity,adhd_total")
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-decision-subjects", type=int, default=50)
    parser.add_argument("--min-decision-null-permutations", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness = read_json(args.readiness_json)
    state_rows = read_tsv(args.state_summary)
    generic_rows = read_tsv(args.generic_summary)
    rows = state_rows + generic_rows
    phenotypes = [item.strip() for item in args.primary_phenotypes.split(",") if item.strip()]
    phenotype_decisions = [summarize_phenotype(name, rows, args.min_gap, args.alpha) for name in phenotypes]
    power_reasons = low_power_reasons(readiness, rows, args.min_decision_subjects, args.min_decision_null_permutations)
    if power_reasons:
        downgrade_low_power(phenotype_decisions, power_reasons)
    if readiness.get("status") != "pass":
        overall = "BLOCKED_READINESS_GATE_FAILED"
    elif power_reasons:
        overall = "UNDERCONTROLLED_LOW_POWER"
    else:
        overall = overall_verdict(phenotype_decisions)

    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "readiness_json": str(args.readiness_json.resolve()),
        "state_summary": str(args.state_summary.resolve()),
        "generic_summary": str(args.generic_summary.resolve()),
        "readiness_status": readiness.get("status", "missing"),
        "readiness_failures": readiness.get("failures", []),
        "primary_phenotypes": phenotypes,
        "min_gap": args.min_gap,
        "alpha": args.alpha,
        "min_decision_subjects": args.min_decision_subjects,
        "min_decision_null_permutations": args.min_decision_null_permutations,
        "low_power_reasons": power_reasons,
        "overall_verdict": overall,
        "phenotype_decisions": phenotype_decisions,
    }
    (output_dir / "adhd_dimensional_pilot_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "adhd_dimensional_pilot_decision.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
