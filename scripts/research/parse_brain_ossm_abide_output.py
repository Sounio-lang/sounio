#!/usr/bin/env python3
"""
Parse examples/brain_ossm_abide.sio output into machine-readable artifacts.

This normalizes the multiline PRED trace blocks emitted by the Sounio runner and
reuses the shared ABIDE campaign summarization helpers so the native benchmark
produces the same overall/per-seed/per-site tables as the external baselines.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from abide_campaign_lib import (
    load_manifest,
    summarize_prediction_rows,
    write_json,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse brain_ossm_abide output")
    parser.add_argument("output_path", help="Path to captured brain_ossm_abide stdout")
    parser.add_argument(
        "--manifest",
        default="",
        help="Path to the manifest used for the run; defaults to abide_roi_manifest.tsv next to the output",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write overall_metrics/per_seed/per_site/prediction_rows artifacts",
    )
    return parser.parse_args()


def resolve_manifest(output_path: Path, explicit_manifest: str) -> Path | None:
    if explicit_manifest:
        candidate = Path(explicit_manifest)
        return candidate if candidate.exists() else None

    sibling = output_path.parent / "abide_roi_manifest.tsv"
    if sibling.exists():
        return sibling

    return None


def parse_prediction_rows(output_path: Path) -> list[dict[str, object]]:
    lines = output_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("PRED\t"):
            i += 1
            continue

        if i + 3 >= len(lines):
            raise ValueError(f"truncated PRED block near line {i + 1}")

        head = line.split("\t")
        site_line = lines[i + 1].lstrip("\t").split("\t")
        prob_line = lines[i + 2].lstrip("\t").split("\t")
        assoc_line = lines[i + 3].lstrip("\t").split("\t")

        if len(head) != 3 or len(site_line) != 2 or len(prob_line) != 2 or len(assoc_line) != 1:
            raise ValueError(f"malformed PRED block near line {i + 1}")

        rows.append(
            {
                "model": head[1],
                "seed": int(head[2]),
                "site": site_line[0],
                "label": int(site_line[1]),
                "prob": float(prob_line[0]),
                "pred": int(prob_line[1]),
                "assoc": float(assoc_line[0]),
            }
        )
        i += 4

    if not rows:
        raise ValueError(f"no PRED rows found in {output_path}")

    return rows


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path)
    if not output_path.exists():
        raise SystemExit(f"output file not found: {output_path}")

    output_dir = Path(args.output_dir) if args.output_dir else output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = parse_prediction_rows(output_path)
    summaries = summarize_prediction_rows(prediction_rows)

    manifest_path = resolve_manifest(output_path, args.manifest)
    manifest_payload: dict[str, object] | None = None
    if manifest_path is not None:
        manifest_meta, _ = load_manifest(manifest_path)
        manifest_payload = asdict(manifest_meta)

    payload = {
        "schema": "brain_ossm.abide.sounio.v1",
        "source_output": str(output_path),
        "manifest": manifest_payload,
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
    write_tsv(
        output_dir / "per_site_seed_metrics.tsv",
        summaries["per_site_seed"],
        [
            "model",
            "seed",
            "site",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
