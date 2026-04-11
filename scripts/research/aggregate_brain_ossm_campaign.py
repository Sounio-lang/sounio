#!/usr/bin/env python3
"""Aggregate structured Sounio + external baseline outputs into one campaign table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from abide_campaign_lib import write_json, write_tsv


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _normalize_row(row: dict[str, object], source: str) -> dict[str, object]:
    out = dict(row)
    out["source"] = source
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Brain O-SSM campaign outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sounio-dir", help="Structured output dir from parse_brain_ossm_abide_output.py")
    parser.add_argument("--external-dir", help="Structured output dir from abide_external_baselines.py")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_rows: list[dict[str, object]] = []
    per_site_rows: list[dict[str, object]] = []

    if args.sounio_dir:
        sounio_dir = Path(args.sounio_dir)
        sounio_overall = _load_json(sounio_dir / "overall_metrics.json")
        for row in sounio_overall["overall"]:
            leaderboard_rows.append(_normalize_row(row, "sounio"))
        for row in _load_tsv(sounio_dir / "per_site_metrics.tsv"):
            per_site_rows.append(_normalize_row(row, "sounio"))

    if args.external_dir:
        external_dir = Path(args.external_dir)
        external_overall = _load_json(external_dir / "overall_metrics.json")
        for row in external_overall["overall"]:
            leaderboard_rows.append(_normalize_row(row, "external"))
        for row in _load_tsv(external_dir / "per_site_metrics.tsv"):
            per_site_rows.append(_normalize_row(row, "external"))

    leaderboard_rows.sort(
        key=lambda row: float(row.get("balanced_accuracy_pct_mean", 0.0)),
        reverse=True,
    )
    per_site_rows.sort(
        key=lambda row: (
            str(row.get("site", "")),
            -float(row.get("balanced_accuracy_pct_mean", 0.0)),
            str(row.get("model", "")),
        )
    )

    summary = {
        "schema": "brain_ossm.abide.campaign.v1",
        "leaderboard_rows": len(leaderboard_rows),
        "per_site_rows": len(per_site_rows),
        "best_model_by_balanced_accuracy": leaderboard_rows[0]["model"] if leaderboard_rows else None,
    }
    write_json(output_dir / "campaign_summary.json", summary)
    write_tsv(
        output_dir / "leaderboard.tsv",
        leaderboard_rows,
        [
            "source",
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
        output_dir / "per_site_leaderboard.tsv",
        per_site_rows,
        [
            "source",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
