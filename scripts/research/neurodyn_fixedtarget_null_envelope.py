#!/usr/bin/env python3
"""Aggregate true-vs-null fixed-target associator smoke metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.fixedtarget_null_envelope.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-run", required=True, type=Path)
    parser.add_argument("--null-run", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv_one(path: Path, predicate: dict[str, str] | None = None) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if predicate:
        for row in rows:
            if all(row.get(key) == value for key, value in predicate.items()):
                return row
        raise SystemExit(f"no row matching {predicate} in {path}")
    if len(rows) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def run_id(path: Path) -> str:
    return path.name


def load_run(path: Path, condition: str) -> dict[str, Any]:
    overall = read_tsv_one(path / "results" / "overall_metrics.tsv", {"model": "O-SSM"})
    hidden = read_tsv_one(
        path / "hidden_state_separability" / "hidden_state_separability_summary.tsv",
        {"model": "O-SSM"},
    )
    paired = read_tsv_one(
        path / "paired_hidden_contrast" / "paired_hidden_contrast_summary.tsv",
        {"model": "O-SSM"},
    )
    replay = read_tsv_one(
        path / "trained_hidden_readout" / "trained_hidden_readout_probe_summary.tsv",
        {"model": "O-SSM"},
    )
    return {
        "condition": condition,
        "run_id": run_id(path),
        "path": str(path),
        "o_balanced_accuracy_pct": f(overall, "balanced_accuracy_pct_mean"),
        "o_auroc_pct": f(overall, "auroc_pct_mean"),
        "o_brier": f(overall, "brier_mean"),
        "o_ece_pct": f(overall, "ece_pct_mean"),
        "o_hidden_centroid_ba_pct": f(hidden, "nearest_centroid_balanced_accuracy_mean"),
        "o_hidden_centroid_margin": f(hidden, "centroid_margin_mean"),
        "o_paired_hidden_direction_pct": f(paired, "leave_site_pair_direction_accuracy_mean"),
        "o_hidden_pair_distance_mean": f(paired, "hidden_pair_distance_mean"),
        "o_trained_hidden_readout_ba_pct": f(replay, "trained_hidden_readout_balanced_accuracy_mean"),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_envelope(true_value: float, null_values: list[float]) -> dict[str, Any]:
    return {
        "true": true_value,
        "null_count": len(null_values),
        "null_mean": statistics.fmean(null_values) if null_values else 0.0,
        "null_std": statistics.pstdev(null_values) if len(null_values) > 1 else 0.0,
        "null_min": min(null_values) if null_values else 0.0,
        "null_max": max(null_values) if null_values else 0.0,
        "null_ge_true_count": sum(1 for value in null_values if value >= true_value),
        "empirical_p_ge_true_plus_one": (1 + sum(1 for value in null_values if value >= true_value)) / (len(null_values) + 1),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: dict[str, Any]) -> str:
    metric_names = {
        "o_balanced_accuracy_pct": "O-SSM BA",
        "o_auroc_pct": "O-SSM AUROC",
        "o_hidden_centroid_ba_pct": "O hidden centroid BA",
        "o_paired_hidden_direction_pct": "O paired hidden direction",
    }
    lines = [
        "# Fixed-Target Null Envelope",
        "",
        "Claim boundary: synthetic non-clinical null-envelope diagnostic only. No clinical, biomarker, biological mechanism, solved-associator, scaled-transfer, or broad O-SSM superiority claim.",
        "",
        f"- True run: `{summary['true_run']}`",
        f"- Null runs: `{summary['null_count']}`",
        "",
        "## Envelope",
        "",
        "| metric | true | null max | null mean | null >= true | plus-one p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in metric_names.items():
        env = summary["metrics"][key]
        lines.append(
            f"| {label} | {env['true']:.6f} | {env['null_max']:.6f} | {env['null_mean']:.6f} | "
            f"{env['null_ge_true_count']}/{env['null_count']} | {env['empirical_p_ge_true_plus_one']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
            summary["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    true_row = load_run(args.true_run, "true")
    null_rows = [load_run(path, "null") for path in args.null_run]
    all_rows = [true_row, *null_rows]
    metric_keys = [
        "o_balanced_accuracy_pct",
        "o_auroc_pct",
        "o_hidden_centroid_ba_pct",
        "o_paired_hidden_direction_pct",
    ]
    metrics = {
        key: metric_envelope(true_row[key], [row[key] for row in null_rows])
        for key in metric_keys
    }
    ba_ok = metrics["o_balanced_accuracy_pct"]["null_ge_true_count"] == 0
    auroc_ok = metrics["o_auroc_pct"]["null_ge_true_count"] == 0
    centroid_ok = metrics["o_hidden_centroid_ba_pct"]["null_ge_true_count"] == 0
    paired_ok = metrics["o_paired_hidden_direction_pct"]["null_ge_true_count"] == 0
    if ba_ok and auroc_ok and centroid_ok and paired_ok:
        decision = "FIXEDTARGET_MAIN_READOUT_NULL_ENVELOPE_PASSES_MICRO_GATE"
        interpretation = (
            "The true fixed-target micro-run exceeds the five-null envelope on global and hidden-state metrics. "
            "This supports only a bounded synthetic micro-assay result and still requires larger controlled variants."
        )
    elif ba_ok and (centroid_ok or paired_ok):
        decision = "FIXEDTARGET_MAIN_READOUT_GLOBAL_PARTIAL_HIDDEN_NOT_NULL_SAFE"
        interpretation = (
            "The true run exceeds the null envelope on balanced accuracy and at least one hidden-state probe, "
            "but one or more nulls match or exceed another hidden/ranking metric. Do not promote; the next step is assay/objective repair or more stringent hidden target alignment."
        )
    else:
        decision = "FIXEDTARGET_MAIN_READOUT_NOT_NULL_SAFE"
        interpretation = (
            "At least one pair-label null matches or exceeds the true run on the required global or hidden-state gates. "
            "The micro-positive result is not null-safe and must not be promoted."
        )

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic non-clinical null-envelope diagnostic only.",
        "true_run": str(args.true_run),
        "null_runs": [str(path) for path in args.null_run],
        "null_count": len(null_rows),
        "metrics": metrics,
        "decision": decision,
        "interpretation": interpretation,
    }
    write_tsv(out / "fixedtarget_null_envelope_runs.tsv", all_rows)
    write_tsv(
        out / "fixedtarget_null_envelope_metrics.tsv",
        [
            {"metric": key, **value}
            for key, value in metrics.items()
        ],
    )
    (out / "fixedtarget_null_envelope.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "fixedtarget_null_envelope.md").write_text(markdown(summary), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
