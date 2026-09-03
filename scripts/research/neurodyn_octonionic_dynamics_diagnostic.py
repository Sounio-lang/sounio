#!/usr/bin/env python3
"""Diagnose octonionic state-space activity from a verified NeuroDyn O-SSM run.

The goal is not to make a clinical claim. It is to keep the O-SSM evaluation
honest as an octonionic state-space model by separating:

* replayable, nonzero octonionic state
* associator/hidden-state activity
* saturated classifier readout artifacts
* label/fold summaries that need scale before interpretation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.octonionic_dynamics_diagnostic.v1"
CLAIM_BOUNDARY = (
    "Diagnostic only. Nonzero octonionic state and associator activity are "
    "necessary but not sufficient for a clinical, mechanistic, or biomarker claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True, help="subject_roi_assoc_features.tsv from trained associator.")
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prob-saturation-threshold", type=float, default=0.99)
    parser.add_argument("--logit-saturation-threshold", type=float, default=11.999)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def values(ckpt: dict[str, Any], name: str) -> list[float]:
    return [float(value) for value in ckpt.get("arrays", {}).get(name, {}).get("values", [])]


def fixed_to_float(values_in: list[float]) -> list[float]:
    return [value / 100_000_000.0 for value in values_in]


def vector_summary(raw: list[float], scale: float = 1.0) -> dict[str, Any]:
    vals = [value * scale for value in raw]
    if not vals:
        return {
            "count": 0,
            "nonzero_count": 0,
            "mean": 0.0,
            "mean_abs": 0.0,
            "l1": 0.0,
            "l2": 0.0,
            "linf": 0.0,
            "positive_count": 0,
            "negative_count": 0,
        }
    abs_vals = [abs(value) for value in vals]
    return {
        "count": len(vals),
        "nonzero_count": sum(1 for value in abs_vals if value > 1.0e-12),
        "mean": sum(vals) / len(vals),
        "mean_abs": sum(abs_vals) / len(abs_vals),
        "l1": sum(abs_vals),
        "l2": math.sqrt(sum(value * value for value in vals)),
        "linf": max(abs_vals),
        "positive_count": sum(1 for value in vals if value > 1.0e-12),
        "negative_count": sum(1 for value in vals if value < -1.0e-12),
    }


def read_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "subject_index": int(row["subject_index"]),
                    "subject_id": row["subject_id"],
                    "label": int(row["label"]),
                    "label_name": row["label_name"],
                    "site": row["site"],
                    "roi_index": int(row["roi_index"]),
                    "base_prob": float(row["base_prob"]),
                    "masked_prob": float(row["masked_prob"]),
                    "base_assoc": float(row["base_assoc"]),
                    "masked_assoc": float(row["masked_assoc"]),
                    "assoc_delta_abs": float(row["assoc_delta_abs"]),
                    "prob_delta_abs": float(row["prob_delta_abs"]),
                    "base_logit": float(row["base_logit"]),
                    "masked_logit": float(row["masked_logit"]),
                    "logit_delta_abs": float(row["logit_delta_abs"]),
                    "hidden_delta_abs": float(row["hidden_delta_abs"]),
                }
            )
    return rows


def mean(values_in: list[float]) -> float:
    return sum(values_in) / len(values_in) if values_in else 0.0


def stdev(values_in: list[float]) -> float:
    return statistics.pstdev(values_in) if len(values_in) > 1 else 0.0


def summarize_group(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=str):
        chunk = grouped[key]
        out.append(
            {
                group_key: key,
                "feature_rows": len(chunk),
                "subject_count": len({row["subject_index"] for row in chunk}),
                "base_prob_mean": mean([row["base_prob"] for row in chunk]),
                "base_assoc_mean": mean([row["base_assoc"] for row in chunk]),
                "assoc_delta_abs_mean": mean([row["assoc_delta_abs"] for row in chunk]),
                "prob_delta_abs_mean": mean([row["prob_delta_abs"] for row in chunk]),
                "logit_delta_abs_mean": mean([row["logit_delta_abs"] for row in chunk]),
                "hidden_delta_abs_mean": mean([row["hidden_delta_abs"] for row in chunk]),
            }
        )
    return out


def subject_level(rows: list[dict[str, Any]], prob_threshold: float, logit_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject_index"]].append(row)
    out: list[dict[str, Any]] = []
    for subject_index in sorted(grouped):
        chunk = grouped[subject_index]
        first = chunk[0]
        base_prob = first["base_prob"]
        base_logit = first["base_logit"]
        out.append(
            {
                "subject_index": subject_index,
                "subject_id": first["subject_id"],
                "label": first["label"],
                "label_name": first["label_name"],
                "site": first["site"],
                "roi_count": len(chunk),
                "base_prob": base_prob,
                "base_assoc": first["base_assoc"],
                "base_logit": base_logit,
                "prob_saturated": int(base_prob >= prob_threshold or base_prob <= 1.0 - prob_threshold),
                "logit_saturated": int(abs(base_logit) >= logit_threshold),
                "assoc_delta_abs_mean": mean([row["assoc_delta_abs"] for row in chunk]),
                "assoc_delta_abs_std": stdev([row["assoc_delta_abs"] for row in chunk]),
                "hidden_delta_abs_mean": mean([row["hidden_delta_abs"] for row in chunk]),
                "hidden_delta_abs_std": stdev([row["hidden_delta_abs"] for row in chunk]),
                "prob_delta_abs_mean": mean([row["prob_delta_abs"] for row in chunk]),
                "logit_delta_abs_mean": mean([row["logit_delta_abs"] for row in chunk]),
            }
        )
    return out


def top_subjects(rows: list[dict[str, Any]], field: str, limit: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row[field]), reverse=True)[:limit]


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Octonionic Dynamics Diagnostic",
        "",
        f"- schema: `{summary['schema']}`",
        f"- run_id: `{summary['run_id']}`",
        f"- checkpoint_replay_status: `{summary['checkpoint_verification'].get('status')}`",
        f"- checkpoint_replay_delta_pp: `{summary['checkpoint_verification'].get('delta_pp')}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## State Activity",
        "",
        f"- A_nonzero: `{summary['state']['A']['nonzero_count']}` / `{summary['state']['A']['count']}`",
        f"- A_l2: `{summary['state']['A']['l2']}`",
        f"- LAST_H_nonzero: `{summary['state']['LAST_H_fixed']['nonzero_count']}` / `{summary['state']['LAST_H_fixed']['count']}`",
        f"- LAST_ASSOC_ACC_nonzero: `{summary['state']['LAST_ASSOC_ACC_fixed']['nonzero_count']}` / `{summary['state']['LAST_ASSOC_ACC_fixed']['count']}`",
        "",
        "## Readout Saturation",
        "",
        f"- subject_count: `{summary['subject_count']}`",
        f"- prob_saturated_subjects: `{summary['saturation']['prob_saturated_subjects']}`",
        f"- logit_saturated_subjects: `{summary['saturation']['logit_saturated_subjects']}`",
        f"- mean_base_prob: `{summary['saturation']['mean_base_prob']}`",
        f"- mean_base_assoc: `{summary['saturation']['mean_base_assoc']}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    features_path = Path(args.features).resolve()
    run_summary_path = Path(args.run_summary).resolve()
    output_dir = Path(args.output_dir).resolve()

    checkpoint = read_json(checkpoint_path)
    run_summary = read_json(run_summary_path)
    rows = read_feature_rows(features_path)
    subjects = subject_level(rows, args.prob_saturation_threshold, args.logit_saturation_threshold)

    state = {
        "A": vector_summary(values(checkpoint, "A")),
        "A_FIXED_as_float": vector_summary(fixed_to_float(values(checkpoint, "A_FIXED"))),
        "LAST_H_fixed": vector_summary(fixed_to_float(values(checkpoint, "LAST_H"))),
        "LAST_ASSOC_ACC_fixed": vector_summary(fixed_to_float(values(checkpoint, "LAST_ASSOC_ACC"))),
        "W_fixed": vector_summary(fixed_to_float(values(checkpoint, "W"))),
        "PROJ_W_fixed": vector_summary(fixed_to_float(values(checkpoint, "PROJ_W"))),
    }
    prob_saturated = sum(row["prob_saturated"] for row in subjects)
    logit_saturated = sum(row["logit_saturated"] for row in subjects)
    interpretation = (
        "The checkpoint contains nonzero replayable octonionic state and nonzero associator activity. "
        "However, the current subject-level readout is saturated for most or all subjects, so probability/logit "
        "perturbation metrics are weak evidence for octonionic structure. Treat hidden/associator diagnostics as "
        "necessary instrumentation for the scaled run, not as a standalone positive claim."
    )

    summary = {
        "schema": SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "features_path": str(features_path),
        "run_summary_path": str(run_summary_path),
        "run_id": checkpoint.get("run_id"),
        "checkpoint_verification": checkpoint.get("verification", {}),
        "extractor_verification": run_summary.get("extractor_verification", {}),
        "state": state,
        "feature_rows": len(rows),
        "subject_count": len(subjects),
        "roi_count": len({row["roi_index"] for row in rows}),
        "saturation": {
            "prob_threshold": args.prob_saturation_threshold,
            "logit_threshold": args.logit_saturation_threshold,
            "prob_saturated_subjects": prob_saturated,
            "logit_saturated_subjects": logit_saturated,
            "prob_saturated_fraction": prob_saturated / len(subjects) if subjects else 0.0,
            "logit_saturated_fraction": logit_saturated / len(subjects) if subjects else 0.0,
            "mean_base_prob": mean([row["base_prob"] for row in subjects]),
            "mean_base_assoc": mean([row["base_assoc"] for row in subjects]),
            "mean_hidden_delta_abs": mean([row["hidden_delta_abs_mean"] for row in subjects]),
            "mean_assoc_delta_abs": mean([row["assoc_delta_abs_mean"] for row in subjects]),
        },
        "by_label": summarize_group(rows, "label_name"),
        "by_site": summarize_group(rows, "site"),
        "top_subjects_by_hidden_delta_abs_mean": top_subjects(subjects, "hidden_delta_abs_mean"),
        "top_subjects_by_assoc_delta_abs_mean": top_subjects(subjects, "assoc_delta_abs_mean"),
        "interpretation": interpretation,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "octonionic_dynamics_diagnostic.json", summary)
    write_tsv(output_dir / "subject_octonionic_dynamics.tsv", subjects)
    write_tsv(output_dir / "label_octonionic_dynamics.tsv", summary["by_label"])
    write_tsv(output_dir / "site_octonionic_dynamics.tsv", summary["by_site"])
    (output_dir / "octonionic_dynamics_diagnostic.md").write_text(markdown(summary), encoding="utf-8")

    print(f"summary={output_dir / 'octonionic_dynamics_diagnostic.json'}")
    print(f"subjects={output_dir / 'subject_octonionic_dynamics.tsv'}")
    print(f"claim_boundary={CLAIM_BOUNDARY}")
    print(f"prob_saturated_subjects={prob_saturated}/{len(subjects)}")
    print(f"logit_saturated_subjects={logit_saturated}/{len(subjects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
