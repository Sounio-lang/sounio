#!/usr/bin/env python3
"""Analyze Brain O-SSM STATE_TRACE hidden-state separability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.hidden_state_separability.v1"
CLAIM_BOUNDARY = (
    "Hidden-state separability diagnostic only. This is not a clinical, "
    "mechanistic, biomarker, or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_state_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("STATE_TRACE"):
            idx += 1
            continue
        line_no = idx + 1
        parts = [part for part in line.split("\t") if part]
        next_idx = idx + 1
        while len(parts) < 22 and next_idx < len(lines) and lines[next_idx].startswith("\t"):
            parts.extend(part for part in lines[next_idx].split("\t") if part)
            next_idx += 1
        if len(parts) != 22:
            raise SystemExit(f"malformed STATE_TRACE at line {line_no}: expected 22 fields, got {len(parts)}")
        _, model, seed, subject_index, site, label, *raw_values = parts
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "subject_index": int(subject_index),
                "site": site,
                "label": int(label),
                "h": [int(value) / 1_000_000.0 for value in raw_values],
            }
        )
        idx = next_idx
    if not rows:
        raise SystemExit("no STATE_TRACE rows found; run with trace_hidden_state=1")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(vec[idx] for vec in vectors) / len(vectors) for idx in range(dim)]


def dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


def balanced_accuracy(labels: list[int], preds: list[int]) -> float:
    pos = [idx for idx, label in enumerate(labels) if label == 1]
    neg = [idx for idx, label in enumerate(labels) if label == 0]
    if not pos or not neg:
        return 0.0
    tpr = sum(1 for idx in pos if preds[idx] == 1) / len(pos)
    tnr = sum(1 for idx in neg if preds[idx] == 0) / len(neg)
    return 100.0 * (tpr + tnr) / 2.0


def norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["seed"])].append(row)

    for (model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        labels_all: list[int] = []
        preds_all: list[int] = []
        for site in sites:
            train = [row for row in chunk if row["site"] != site]
            test = [row for row in chunk if row["site"] == site]
            pos_centroid = centroid([row["h"] for row in train if row["label"] == 1])
            neg_centroid = centroid([row["h"] for row in train if row["label"] == 0])
            if not pos_centroid or not neg_centroid:
                continue
            for row in test:
                pred = 1 if dist2(row["h"], pos_centroid) <= dist2(row["h"], neg_centroid) else 0
                labels_all.append(int(row["label"]))
                preds_all.append(pred)
        pos_centroid_all = centroid([row["h"] for row in chunk if row["label"] == 1])
        neg_centroid_all = centroid([row["h"] for row in chunk if row["label"] == 0])
        margin = norm([a - b for a, b in zip(pos_centroid_all, neg_centroid_all, strict=True)])
        within: list[float] = []
        for row in chunk:
            center = pos_centroid_all if row["label"] == 1 else neg_centroid_all
            within.append(math.sqrt(dist2(row["h"], center)))
        detail.append(
            {
                "model": model,
                "seed": seed,
                "site_count": len(sites),
                "rows": len(chunk),
                "nearest_centroid_balanced_accuracy": balanced_accuracy(labels_all, preds_all),
                "centroid_margin": margin,
                "within_class_radius_mean": mean(within),
                "margin_radius_ratio": margin / mean(within) if mean(within) > 1e-12 else 0.0,
            }
        )

    summary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        summary_groups[row["model"]].append(row)

    summary: list[dict[str, Any]] = []
    for model, chunk in sorted(summary_groups.items()):
        ba = [float(row["nearest_centroid_balanced_accuracy"]) for row in chunk]
        margin = [float(row["centroid_margin"]) for row in chunk]
        ratio = [float(row["margin_radius_ratio"]) for row in chunk]
        summary.append(
            {
                "model": model,
                "seed_count": len(chunk),
                "nearest_centroid_balanced_accuracy_mean": mean(ba),
                "nearest_centroid_balanced_accuracy_std": pstdev(ba),
                "centroid_margin_mean": mean(margin),
                "margin_radius_ratio_mean": mean(ratio),
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
        "# NeuroDyn Hidden-State Separability Diagnostic",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "| model | seeds | nearest-centroid BA mean | BA std | centroid margin mean | margin/radius mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {model} | {seed_count} | {ba:.6f} | {ba_std:.6f} | {margin:.6f} | {ratio:.6f} |".format(
                model=row["model"],
                seed_count=row["seed_count"],
                ba=row["nearest_centroid_balanced_accuracy_mean"],
                ba_std=row["nearest_centroid_balanced_accuracy_std"],
                margin=row["centroid_margin_mean"],
                ratio=row["margin_radius_ratio_mean"],
            )
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_state_rows(Path(args.raw_output))
    summary, detail = summarize(rows)
    interpretation = (
        "Nearest-centroid balanced accuracy on traced final hidden states asks whether "
        "the state representation itself contains class/order information before the "
        "trained classifier readout. Chance-level values imply the current recurrent "
        "state/update path is not separating this assay; above-chance values with a "
        "chance classifier would point to readout/training failure instead."
    )
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "raw_output": str(Path(args.raw_output).resolve()),
        "state_trace_rows": len(rows),
        "summary": summary,
        "detail": detail,
        "interpretation": interpretation,
    }
    write_json(output_dir / "hidden_state_separability.json", payload)
    write_tsv(output_dir / "hidden_state_separability_summary.tsv", summary)
    write_tsv(output_dir / "hidden_state_separability_detail.tsv", detail)
    (output_dir / "hidden_state_separability.md").write_text(markdown(payload), encoding="utf-8")
    write_hashes(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
