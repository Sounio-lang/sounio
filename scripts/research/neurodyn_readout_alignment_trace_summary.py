#!/usr/bin/env python3
"""Summarize O-SSM READOUT_ALIGN_TRACE rows from brain_ossm_abide output."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.readout_alignment_trace_summary.v1"


FIELDS = [
    "tag",
    "model",
    "seed",
    "seed_b",
    "holdout_key",
    "holdout_site",
    "phase",
    "align_epochs",
    "align_lr_scale_bp",
    "train_n",
    "train_balanced_accuracy_pct",
    "train_accuracy_pct",
    "train_brier",
    "train_margin",
    "holdout_n",
    "holdout_balanced_accuracy_pct",
    "holdout_accuracy_pct",
    "holdout_brier",
    "holdout_margin",
]


def bp(value: str) -> float:
    return int(value) / 1_000_000.0


def parse_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("READOUT_ALIGN_TRACE\t"):
            idx += 1
            continue
        parts = [part for part in line.split("\t") if part]
        line_no = idx + 1
        next_idx = idx + 1
        while len(parts) < len(FIELDS) and next_idx < len(lines) and lines[next_idx].startswith("\t"):
            parts.extend(part for part in lines[next_idx].split("\t") if part)
            next_idx += 1
        if len(parts) != len(FIELDS):
            raise SystemExit(f"malformed READOUT_ALIGN_TRACE at line {line_no}: expected {len(FIELDS)} fields, got {len(parts)}")
        raw = dict(zip(FIELDS, parts, strict=True))
        rows.append(
            {
                "model": raw["model"],
                "seed": int(raw["seed"]),
                "seed_b": int(raw["seed_b"]),
                "holdout_key": int(raw["holdout_key"]),
                "holdout_site": raw["holdout_site"],
                "phase": raw["phase"],
                "align_epochs": int(raw["align_epochs"]),
                "align_lr_scale": bp(raw["align_lr_scale_bp"]),
                "train_n": int(raw["train_n"]),
                "train_balanced_accuracy_pct": bp(raw["train_balanced_accuracy_pct"]),
                "train_accuracy_pct": bp(raw["train_accuracy_pct"]),
                "train_brier": bp(raw["train_brier"]),
                "train_margin": bp(raw["train_margin"]),
                "holdout_n": int(raw["holdout_n"]),
                "holdout_balanced_accuracy_pct": bp(raw["holdout_balanced_accuracy_pct"]),
                "holdout_accuracy_pct": bp(raw["holdout_accuracy_pct"]),
                "holdout_brier": bp(raw["holdout_brier"]),
                "holdout_margin": bp(raw["holdout_margin"]),
            }
        )
        idx = next_idx
    if not rows:
        raise SystemExit("no READOUT_ALIGN_TRACE rows found")
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def summarize_phase(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for phase in ["pre", "post"]:
        phase_rows = [row for row in rows if row["phase"] == phase]
        if not phase_rows:
            continue
        out.append(
            {
                "phase": phase,
                "fold_count": len(phase_rows),
                "train_balanced_accuracy_pct_mean": mean([row["train_balanced_accuracy_pct"] for row in phase_rows]),
                "train_balanced_accuracy_pct_std": pstdev([row["train_balanced_accuracy_pct"] for row in phase_rows]),
                "train_brier_mean": mean([row["train_brier"] for row in phase_rows]),
                "train_margin_mean": mean([row["train_margin"] for row in phase_rows]),
                "holdout_balanced_accuracy_pct_mean": mean([row["holdout_balanced_accuracy_pct"] for row in phase_rows]),
                "holdout_balanced_accuracy_pct_std": pstdev([row["holdout_balanced_accuracy_pct"] for row in phase_rows]),
                "holdout_brier_mean": mean([row["holdout_brier"] for row in phase_rows]),
                "holdout_margin_mean": mean([row["holdout_margin"] for row in phase_rows]),
            }
        )
    return out


def summarize_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fold: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_fold[(row["seed"], row["holdout_key"], row["holdout_site"])][row["phase"]] = row
    deltas: list[dict[str, Any]] = []
    for (seed, holdout_key, holdout_site), phases in sorted(by_fold.items()):
        if "pre" not in phases or "post" not in phases:
            continue
        pre = phases["pre"]
        post = phases["post"]
        deltas.append(
            {
                "seed": seed,
                "holdout_key": holdout_key,
                "holdout_site": holdout_site,
                "train_balanced_accuracy_delta": post["train_balanced_accuracy_pct"] - pre["train_balanced_accuracy_pct"],
                "train_brier_delta": post["train_brier"] - pre["train_brier"],
                "train_margin_delta": post["train_margin"] - pre["train_margin"],
                "holdout_balanced_accuracy_delta": post["holdout_balanced_accuracy_pct"] - pre["holdout_balanced_accuracy_pct"],
                "holdout_brier_delta": post["holdout_brier"] - pre["holdout_brier"],
                "holdout_margin_delta": post["holdout_margin"] - pre["holdout_margin"],
            }
        )
    if not deltas:
        raise SystemExit("no matched pre/post fold traces found")
    return deltas


def summarize_delta_overall(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fold_count": len(deltas),
            "train_balanced_accuracy_delta_mean": mean([row["train_balanced_accuracy_delta"] for row in deltas]),
            "train_brier_delta_mean": mean([row["train_brier_delta"] for row in deltas]),
            "train_margin_delta_mean": mean([row["train_margin_delta"] for row in deltas]),
            "holdout_balanced_accuracy_delta_mean": mean([row["holdout_balanced_accuracy_delta"] for row in deltas]),
            "holdout_brier_delta_mean": mean([row["holdout_brier_delta"] for row in deltas]),
            "holdout_margin_delta_mean": mean([row["holdout_margin_delta"] for row in deltas]),
        }
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-output", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "readout_alignment_trace_summary.tsv"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")

    rows = parse_rows(args.raw_output)
    phase_summary = summarize_phase(rows)
    deltas = summarize_delta(rows)
    delta_summary = summarize_delta_overall(deltas)

    write_tsv(summary_path, phase_summary)
    write_tsv(args.output_dir / "readout_alignment_trace_delta_summary.tsv", delta_summary)
    write_tsv(args.output_dir / "readout_alignment_trace_delta_detail.tsv", deltas)

    payload = {
        "schema": SCHEMA,
        "raw_output": str(args.raw_output),
        "trace_rows": len(rows),
        "phase_summary": phase_summary,
        "delta_summary": delta_summary,
    }
    (args.output_dir / "readout_alignment_trace_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
