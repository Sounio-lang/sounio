#!/usr/bin/env python3
"""Audit temporal-arrow label/readout polarity from prediction receipts.

This is deliberately clinical-claim-free.  The goal is to distinguish absent
temporal signal from anti-aligned readout polarity in paired forward/reverse
benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.temporal_arrow_polarity_audit.v1"
CLAIM_BOUNDARY = (
    "Temporal-arrow diagnostic only. No clinical, diagnostic, mechanistic, "
    "or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="prediction_rows.tsv from O-SSM/H-SSM.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint-replay", default="")
    parser.add_argument("--external-overall", default="", help="Optional external baseline overall_metrics.tsv.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def subject_direction(subject_id: str) -> str:
    if subject_id.endswith("__forward"):
        return "forward"
    if subject_id.endswith("__reverse"):
        return "reverse"
    return "unknown"


def summarize_predictions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in model_rows:
            by_seed[row["seed"]].append(row)

        raw_acc: list[float] = []
        inv_acc: list[float] = []
        forward_raw: list[float] = []
        reverse_raw: list[float] = []
        prob_forward: list[float] = []
        prob_reverse: list[float] = []
        confusion: Counter[tuple[int, int]] = Counter()
        per_site_seed: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

        for row in model_rows:
            per_site_seed[(row["seed"], row["site"])].append(row)
            y = int(row["label"])
            pred = int(row["pred"])
            confusion[(y, pred)] += 1
            direction = subject_direction(row.get("subject_id") or "")
            if direction == "forward":
                prob_forward.append(float(row["prob"]))
            elif direction == "reverse":
                prob_reverse.append(float(row["prob"]))

        for seed_rows in by_seed.values():
            raw_acc.append(100.0 * mean([1.0 if int(row["pred"]) == int(row["label"]) else 0.0 for row in seed_rows]))
            inv_acc.append(100.0 * mean([1.0 if (1 - int(row["pred"])) == int(row["label"]) else 0.0 for row in seed_rows]))
            fw = [row for row in seed_rows if subject_direction(row.get("subject_id") or "") == "forward"]
            rv = [row for row in seed_rows if subject_direction(row.get("subject_id") or "") == "reverse"]
            forward_raw.append(100.0 * mean([1.0 if int(row["pred"]) == int(row["label"]) else 0.0 for row in fw]))
            reverse_raw.append(100.0 * mean([1.0 if int(row["pred"]) == int(row["label"]) else 0.0 for row in rv]))

        site_raw_values: list[float] = []
        site_inv_values: list[float] = []
        for site_rows in per_site_seed.values():
            site_raw_values.append(100.0 * mean([1.0 if int(row["pred"]) == int(row["label"]) else 0.0 for row in site_rows]))
            site_inv_values.append(100.0 * mean([1.0 if (1 - int(row["pred"])) == int(row["label"]) else 0.0 for row in site_rows]))

        out.append(
            {
                "model": model,
                "prediction_rows": len(model_rows),
                "seed_count": len(by_seed),
                "raw_accuracy_pct_mean": mean(raw_acc),
                "raw_accuracy_pct_std": pstdev(raw_acc),
                "inverted_accuracy_pct_mean": mean(inv_acc),
                "inverted_accuracy_pct_std": pstdev(inv_acc),
                "raw_minus_inverted_pct": mean(raw_acc) - mean(inv_acc),
                "forward_raw_accuracy_pct_mean": mean(forward_raw),
                "reverse_raw_accuracy_pct_mean": mean(reverse_raw),
                "forward_prob_mean": mean(prob_forward),
                "reverse_prob_mean": mean(prob_reverse),
                "site_seed_raw_accuracy_pct_min": min(site_raw_values) if site_raw_values else 0.0,
                "site_seed_inverted_accuracy_pct_max": max(site_inv_values) if site_inv_values else 0.0,
                "confusion_y0_p0": confusion[(0, 0)],
                "confusion_y0_p1": confusion[(0, 1)],
                "confusion_y1_p0": confusion[(1, 0)],
                "confusion_y1_p1": confusion[(1, 1)],
            }
        )
    return out


def write_hashes(output_dir: Path) -> None:
    rows: list[str] = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.name}\n")
    (output_dir / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Temporal-Arrow Polarity Audit",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
    ]
    if summary.get("run_id"):
        lines.append(f"- RUN_ID: `{summary['run_id']}`")
    if summary.get("checkpoint_replay"):
        lines.append(f"- checkpoint replay: {summary['checkpoint_replay']}")
    lines.extend(
        [
            "",
            "## Polarity",
            "",
            "| model | raw acc % | inverted acc % | forward prob mean | reverse prob mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["polarity"]:
        lines.append(
            "| {model} | {raw:.6f} | {inv:.6f} | {pf:.6f} | {pr:.6f} |".format(
                model=row["model"],
                raw=row["raw_accuracy_pct_mean"],
                inv=row["inverted_accuracy_pct_mean"],
                pf=row["forward_prob_mean"],
                pr=row["reverse_prob_mean"],
            )
        )
    if summary["external_baselines"]:
        lines.extend(["", "## External Baselines", "", "| model | balanced accuracy % | AUROC % |", "|---|---:|---:|"])
        for row in summary["external_baselines"]:
            lines.append(
                f"| {row['model']} | {float(row['balanced_accuracy_pct_mean']):.6f} | {float(row['auroc_pct_mean']):.6f} |"
            )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = Path(args.predictions).resolve()
    rows = read_tsv(predictions_path)
    if not rows:
        raise SystemExit(f"no prediction rows: {predictions_path}")
    polarity = summarize_predictions(rows)
    external_baselines = read_tsv(Path(args.external_overall).resolve()) if args.external_overall else []
    best_raw = max(float(row["raw_accuracy_pct_mean"]) for row in polarity)
    best_inv = max(float(row["inverted_accuracy_pct_mean"]) for row in polarity)
    if best_inv > best_raw + 20.0:
        decision = "anti_aligned_temporal_signal"
        interpretation = (
            "Predictions are strongly anti-aligned with manifest labels. This is evidence "
            "against absent temporal signal, but it is a readout/label-polarity blocker "
            "until checked with swapped labels and native state decoding."
        )
    else:
        decision = "no_strong_polarity_inversion"
        interpretation = (
            "The raw and inverted scores do not show a large polarity gap. Treat any "
            "temporal-arrow signal as ordinary benchmark performance until deeper state "
            "diagnostics are available."
        )

    summary = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "checkpoint_replay": args.checkpoint_replay,
        "predictions": str(predictions_path),
        "external_overall": str(Path(args.external_overall).resolve()) if args.external_overall else "",
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
        "polarity": polarity,
        "external_baselines": external_baselines,
        "interpretation": interpretation,
    }
    write_json(output_dir / "temporal_arrow_polarity_audit.json", summary)
    write_tsv(output_dir / "temporal_arrow_polarity_audit.tsv", polarity)
    (output_dir / "temporal_arrow_polarity_audit.md").write_text(markdown(summary), encoding="utf-8")
    write_hashes(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
