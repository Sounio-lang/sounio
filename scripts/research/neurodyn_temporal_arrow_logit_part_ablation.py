#!/usr/bin/env python3
"""Post-hoc ablations for temporal-arrow Brain O-SSM logit part traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SCHEMA = "neurodyn.temporal_arrow_logit_part_ablation.v1"
CLAIM_BOUNDARY = (
    "Temporal-arrow logit-channel diagnostic only. This post-hoc readout "
    "ablation is not a clinical, diagnostic, mechanistic, or O-SSM superiority "
    "claim; trained ablations are still required for causal readout claims."
)

PARTS = ("bias", "hidden", "assoc", "mean", "delta", "flat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input as CONDITION=path/to/temporal_arrow_logit_parts_rows.tsv. May repeat.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
    for row in rows:
        row["seed"] = int(row["seed"])
        row["label"] = int(row["label"])
        for part in (*PARTS, "total", "logit"):
            row[part] = float(row[part])
    return rows


def parse_inputs(raw_inputs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in raw_inputs:
        if "=" not in raw:
            raise SystemExit(f"input must be CONDITION=path: {raw}")
        condition, path_raw = raw.split("=", 1)
        condition = condition.strip()
        path = Path(path_raw).resolve()
        if not condition:
            raise SystemExit(f"empty condition for input: {raw}")
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        parsed.append((condition, path))
    return parsed


def score_full(row: dict[str, Any]) -> float:
    return float(row["logit"])


def score_total_unclipped(row: dict[str, Any]) -> float:
    return float(row["total"])


def score_no_flat(row: dict[str, Any]) -> float:
    return row["bias"] + row["hidden"] + row["assoc"] + row["mean"] + row["delta"]


def score_no_delta(row: dict[str, Any]) -> float:
    return row["bias"] + row["hidden"] + row["assoc"] + row["mean"] + row["flat"]


def score_no_delta_flat(row: dict[str, Any]) -> float:
    return row["bias"] + row["hidden"] + row["assoc"] + row["mean"]


def score_state_only(row: dict[str, Any]) -> float:
    return row["bias"] + row["hidden"] + row["assoc"]


def score_hidden_only(row: dict[str, Any]) -> float:
    return row["bias"] + row["hidden"]


def score_assoc_only(row: dict[str, Any]) -> float:
    return row["bias"] + row["assoc"]


def score_summary_only(row: dict[str, Any]) -> float:
    return row["bias"] + row["mean"] + row["delta"]


def score_delta_flat_only(row: dict[str, Any]) -> float:
    return row["bias"] + row["delta"] + row["flat"]


ABLATIONS: dict[str, Callable[[dict[str, Any]], float]] = {
    "full_clipped": score_full,
    "full_unclipped": score_total_unclipped,
    "no_flat": score_no_flat,
    "no_delta": score_no_delta,
    "no_delta_flat": score_no_delta_flat,
    "state_only": score_state_only,
    "hidden_only": score_hidden_only,
    "assoc_only": score_assoc_only,
    "summary_only": score_summary_only,
    "delta_flat_only": score_delta_flat_only,
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def balanced_accuracy(labels: list[int], preds: list[int]) -> float:
    positives = [idx for idx, label in enumerate(labels) if label == 1]
    negatives = [idx for idx, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    tpr = sum(1 for idx in positives if preds[idx] == 1) / len(positives)
    tnr = sum(1 for idx in negatives if preds[idx] == 0) / len(negatives)
    return 100.0 * (tpr + tnr) / 2.0


def accuracy(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    return 100.0 * sum(1 for label, pred in zip(labels, preds, strict=True) if label == pred) / len(labels)


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["model"], row["seed"])].append(row)

    for (condition, model, seed), chunk in sorted(grouped.items()):
        labels = [int(row["label"]) for row in chunk]
        for ablation, score_fn in ABLATIONS.items():
            scores = [score_fn(row) for row in chunk]
            preds = [1 if score >= 0.0 else 0 for score in scores]
            detail.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "ablation": ablation,
                    "rows": len(chunk),
                    "accuracy": accuracy(labels, preds),
                    "balanced_accuracy": balanced_accuracy(labels, preds),
                    "positive_score_frac": mean([1.0 if score >= 0.0 else 0.0 for score in scores]),
                    "score_mean": mean(scores),
                    "score_std": pstdev(scores),
                }
            )

    summary_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        summary_groups[(row["condition"], row["model"], row["ablation"])].append(row)

    summary: list[dict[str, Any]] = []
    for (condition, model, ablation), chunk in sorted(summary_groups.items()):
        ba_values = [float(row["balanced_accuracy"]) for row in chunk]
        acc_values = [float(row["accuracy"]) for row in chunk]
        pos_values = [float(row["positive_score_frac"]) for row in chunk]
        score_values = [float(row["score_mean"]) for row in chunk]
        summary.append(
            {
                "condition": condition,
                "model": model,
                "ablation": ablation,
                "seed_count": len(chunk),
                "balanced_accuracy_mean": mean(ba_values),
                "balanced_accuracy_std": pstdev(ba_values),
                "accuracy_mean": mean(acc_values),
                "accuracy_std": pstdev(acc_values),
                "positive_score_frac_mean": mean(pos_values),
                "score_mean_mean": mean(score_values),
            }
        )
    return summary, detail


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hashes(output_dir: Path) -> None:
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Temporal-Arrow Logit Part Ablation",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "## Summary",
        "",
        "| condition | model | ablation | balanced accuracy mean | ba std | positive score frac |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {condition} | {model} | {ablation} | {ba:.6f} | {ba_std:.6f} | {pos:.6f} |".format(
                condition=row["condition"],
                model=row["model"],
                ablation=row["ablation"],
                ba=row["balanced_accuracy_mean"],
                ba_std=row["balanced_accuracy_std"],
                pos=row["positive_score_frac_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = parse_inputs(args.input)
    all_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for condition, path in input_paths:
        rows = read_tsv(path)
        for row in rows:
            row["condition"] = condition
        all_rows.extend(rows)
        sources[condition] = str(path)

    summary, detail = summarize(all_rows)
    by_key = {(row["condition"], row["model"], row["ablation"]): row for row in summary}
    interpretation = (
        "If full_clipped remains high while no_delta_flat, state_only, or assoc_only collapse, "
        "the temporal-arrow separability is being carried mainly by direct summary/readout terms. "
        "If state_only remains high, the corrected O-SSM hidden dynamics deserve a trained "
        "readout ablation and a stricter non-clinical temporal-dynamics benchmark."
    )
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "sources": sources,
        "row_count": len(all_rows),
        "summary": summary,
        "detail": detail,
        "interpretation": interpretation,
        "state_signal_candidates": [
            row
            for key, row in sorted(by_key.items())
            if key[2] in {"no_delta_flat", "state_only", "assoc_only"}
            and float(row["balanced_accuracy_mean"]) >= 60.0
        ],
    }

    write_json(output_dir / "temporal_arrow_logit_part_ablation.json", payload)
    write_tsv(output_dir / "temporal_arrow_logit_part_ablation_summary.tsv", summary)
    write_tsv(output_dir / "temporal_arrow_logit_part_ablation_detail.tsv", detail)
    (output_dir / "temporal_arrow_logit_part_ablation.md").write_text(markdown(payload), encoding="utf-8")
    write_hashes(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
