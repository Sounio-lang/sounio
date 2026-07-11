#!/usr/bin/env python3
"""Aggregate true-label and permutation-null NeuroDyn benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.permutation_null_summary.v1"
CLAIM_BOUNDARY = (
    "Synthetic non-clinical null-summary diagnostic only. No clinical, "
    "biomarker, mechanistic, or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-run", required=True, type=Path)
    parser.add_argument("--null-run", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def model_row(run: Path, model: str) -> dict[str, str]:
    rows = read_tsv(run / "results" / "overall_metrics.tsv")
    for row in rows:
        if row["model"] == model:
            return row
    raise SystemExit(f"model {model} not found in {run}")


def read_seed(run: Path) -> int | None:
    for candidate in (run / "pair_label_permutation_summary.json", run.parent / "pair_label_permutation_summary.json"):
        if candidate.exists():
            return int(json.loads(candidate.read_text(encoding="utf-8"))["seed"])
    name = run.name
    for token in name.split("_"):
        if token.startswith("202607") and token.isdigit():
            return int(token)
    match = re.search(r"(202607\d+)", name)
    if match:
        return int(match.group(1))
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) * (value - m) for value in values) / len(values))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Pair-Label Permutation Null Summary",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "| condition | seed | O-SSM BA | H-SSM BA | O-SSM AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {condition} | {seed} | {o_ba:.6f} | {h_ba:.6f} | {o_auroc:.6f} |".format(
                condition=row["condition"],
                seed=row["seed"] if row["seed"] is not None else -1,
                o_ba=row["o_balanced_accuracy_pct"],
                h_ba=row["h_balanced_accuracy_pct"],
                o_auroc=row["o_auroc_pct"],
            )
        )
    s = payload["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- true O-SSM BA: `{s['true_o_balanced_accuracy_pct']:.6f}`",
            f"- null O-SSM BA mean: `{s['null_o_balanced_accuracy_pct_mean']:.6f}`",
            f"- null O-SSM BA std: `{s['null_o_balanced_accuracy_pct_std']:.6f}`",
            f"- null O-SSM BA max: `{s['null_o_balanced_accuracy_pct_max']:.6f}`",
            f"- true-minus-null-mean gap: `{s['true_minus_null_mean_pp']:.6f}` pp",
            f"- empirical p_ge_true: `{s['empirical_p_ge_true']:.6f}`",
            "",
            "The empirical p-value uses plus-one smoothing: "
            "`(1 + count(null >= true)) / (1 + null_count)`. This is a small "
            "synthetic null sample, not a final inferential claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for condition, run in [("true", args.true_run), *[("null", path) for path in args.null_run]]:
        o = model_row(run, "O-SSM")
        h = model_row(run, "H-SSM")
        rows.append(
            {
                "condition": condition,
                "seed": None if condition == "true" else read_seed(run),
                "run": str(run),
                "o_balanced_accuracy_pct": float(o["balanced_accuracy_pct_mean"]),
                "h_balanced_accuracy_pct": float(h["balanced_accuracy_pct_mean"]),
                "o_auroc_pct": float(o["auroc_pct_mean"]),
                "h_auroc_pct": float(h["auroc_pct_mean"]),
                "o_seed_count": int(o["seed_count"]),
                "site_count": int(o["site_count"]),
                "subjects": int(o["subjects"]),
            }
        )

    true_row = [row for row in rows if row["condition"] == "true"][0]
    null_rows = [row for row in rows if row["condition"] == "null"]
    null_o = [float(row["o_balanced_accuracy_pct"]) for row in null_rows]
    null_o_auroc = [float(row["o_auroc_pct"]) for row in null_rows]
    true_o = float(true_row["o_balanced_accuracy_pct"])
    true_o_auroc = float(true_row["o_auroc_pct"])
    exceed = sum(1 for value in null_o if value >= true_o)
    exceed_auroc = sum(1 for value in null_o_auroc if value >= true_o_auroc)
    summary = {
        "true_o_balanced_accuracy_pct": true_o,
        "true_h_balanced_accuracy_pct": float(true_row["h_balanced_accuracy_pct"]),
        "true_o_auroc_pct": true_o_auroc,
        "true_h_auroc_pct": float(true_row["h_auroc_pct"]),
        "null_count": len(null_rows),
        "null_o_balanced_accuracy_pct_mean": mean(null_o),
        "null_o_balanced_accuracy_pct_std": pstdev(null_o),
        "null_o_balanced_accuracy_pct_min": min(null_o) if null_o else 0.0,
        "null_o_balanced_accuracy_pct_max": max(null_o) if null_o else 0.0,
        "null_o_auroc_pct_mean": mean(null_o_auroc),
        "null_o_auroc_pct_std": pstdev(null_o_auroc),
        "null_o_auroc_pct_min": min(null_o_auroc) if null_o_auroc else 0.0,
        "null_o_auroc_pct_max": max(null_o_auroc) if null_o_auroc else 0.0,
        "true_minus_null_mean_pp": true_o - mean(null_o),
        "true_auroc_minus_null_mean_pp": true_o_auroc - mean(null_o_auroc),
        "null_ge_true_count": exceed,
        "null_auroc_ge_true_count": exceed_auroc,
        "empirical_p_ge_true": (1.0 + exceed) / (1.0 + len(null_o)) if null_o else 1.0,
        "empirical_auroc_p_ge_true": (1.0 + exceed_auroc) / (1.0 + len(null_o_auroc)) if null_o_auroc else 1.0,
    }
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "rows": rows,
        "summary": summary,
    }
    (args.output_dir / "permutation_null_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(args.output_dir / "permutation_null_rows.tsv", rows)
    write_tsv(args.output_dir / "permutation_null_summary.tsv", [summary])
    (args.output_dir / "permutation_null_summary.md").write_text(markdown(payload), encoding="utf-8")
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in args.output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
