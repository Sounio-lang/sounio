#!/usr/bin/env python3
"""Parse raw `brain_ossm_abide.sio` output into structured campaign artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from abide_campaign_lib import summarize_prediction_rows, write_json, write_tsv


OVERALL_RE = re.compile(
    r"^\s+(?P<model>[OH]-SSM)\s+balanced accuracy:\s+(?P<mean>-?\d+(?:\.\d+)?)\s+±\s+(?P<std>-?\d+(?:\.\d+)?)$"
)
GAP_RE = re.compile(r"^\s+Gap \(O-H\):\s+(?P<gap>-?\d+(?:\.\d+)?)")


def parse_raw_output(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"gap_pp": None, "raw_overall": []}

    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("PRED"):
            parts = [part for part in line.split("\t") if part]
            next_idx = idx + 1
            while len(parts) < 8 and next_idx < len(lines) and lines[next_idx].startswith("\t"):
                parts.extend(part for part in lines[next_idx].split("\t") if part)
                next_idx += 1
            if len(parts) != 8:
                raise ValueError(f"malformed prediction line: {line}")
            _, model, seed, site, label, prob, pred, assoc = parts
            prediction_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "site": site,
                    "label": int(label),
                    "prob": float(prob),
                    "pred": int(pred),
                    "assoc": None if assoc == "NA" else float(assoc),
                }
            )
            idx = next_idx
            continue

        idx += 1
        overall_match = OVERALL_RE.match(line)
        if overall_match:
            summary["raw_overall"].append(
                {
                    "model": overall_match.group("model"),
                    "balanced_accuracy_pct_mean": float(overall_match.group("mean")),
                    "balanced_accuracy_pct_std": float(overall_match.group("std")),
                }
            )
            continue

        gap_match = GAP_RE.match(line)
        if gap_match:
            summary["gap_pp"] = float(gap_match.group("gap"))

    return prediction_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Brain O-SSM ABIDE raw output")
    parser.add_argument("--input", required=True, help="Raw benchmark stdout file")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/TSV outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows, raw_summary = parse_raw_output(input_path)
    if not prediction_rows:
        raise SystemExit("no PRED lines found in raw benchmark output; structured parse requires the new trace format")

    summaries = summarize_prediction_rows(prediction_rows)
    payload = {
        "schema": "brain_ossm.abide.structured.v1",
        "source": str(input_path),
        "raw_summary": raw_summary,
        "overall": summaries["overall"],
    }
    write_json(output_dir / "overall_metrics.json", payload)
    write_tsv(
        output_dir / "overall_metrics.tsv",
        summaries["overall"],
        [
            "model",
            "seed_count",
            "site_count",
            "subjects",
            "accuracy_pct_mean",
            "accuracy_pct_std",
            "balanced_accuracy_pct_mean",
            "balanced_accuracy_pct_std",
            "macro_f1_pct_mean",
            "macro_f1_pct_std",
            "auroc_pct_mean",
            "auroc_pct_std",
            "brier_mean",
            "brier_std",
            "ece_pct_mean",
            "ece_pct_std",
            "assoc_mean_mean",
            "assoc_mean_std",
        ],
    )
    write_tsv(
        output_dir / "per_seed_metrics.tsv",
        summaries["per_seed"],
        [
            "model",
            "seed",
            "site_count",
            "subjects",
            "accuracy_pct",
            "balanced_accuracy_pct",
            "macro_f1_pct",
            "auroc_pct",
            "brier",
            "ece_pct",
            "assoc_mean",
        ],
    )
    write_tsv(
        output_dir / "per_site_metrics.tsv",
        summaries["per_site"],
        [
            "model",
            "site",
            "seed_count",
            "subjects_mean",
            "accuracy_pct_mean",
            "accuracy_pct_std",
            "balanced_accuracy_pct_mean",
            "balanced_accuracy_pct_std",
            "macro_f1_pct_mean",
            "macro_f1_pct_std",
            "auroc_pct_mean",
            "auroc_pct_std",
            "brier_mean",
            "brier_std",
            "ece_pct_mean",
            "ece_pct_std",
            "assoc_mean_mean",
            "assoc_mean_std",
        ],
    )
    write_tsv(
        output_dir / "prediction_rows.tsv",
        prediction_rows,
        ["model", "seed", "site", "label", "prob", "pred", "assoc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
