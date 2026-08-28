#!/usr/bin/env python3
"""Summarize ORIENT_HEAD_TRACE rows from Brain O-SSM synthetic runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def bp_to_pct(v: str) -> float:
    return int(v) / 10000.0


def bp_to_unit(v: str) -> float:
    return int(v) / 1_000_000.0


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_rows(raw_output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: list[str] | None = None

    def flush(parts: list[str] | None) -> None:
        if not parts or len(parts) != 12:
            return
        rows.append(
            {
                "model": parts[1],
                "seed_a": int(parts[2]),
                "seed_b": int(parts[3]),
                "holdout_key": int(parts[4]),
                "holdout_site": parts[5],
                "train_pairs": int(parts[6]),
                "train_orientation_accuracy_pct": bp_to_pct(parts[7]),
                "train_orientation_margin": bp_to_unit(parts[8]),
                "holdout_pairs": int(parts[9]),
                "holdout_orientation_accuracy_pct": bp_to_pct(parts[10]),
                "holdout_orientation_margin": bp_to_unit(parts[11]),
            }
        )

    with raw_output.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("ORIENT_HEAD_TRACE\t"):
                flush(current)
                current = line.split("\t")
                continue
            if current is not None and line.startswith("\t"):
                current.extend(line.lstrip("\t").split("\t"))
                continue
            flush(current)
            current = None
    flush(current)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-output", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_rows(args.raw_output)
    if not rows:
        raise SystemExit("no ORIENT_HEAD_TRACE rows found")

    summary = {
        "schema": "neurodyn.orientation_head_trace_summary.v1",
        "raw_output": str(args.raw_output),
        "raw_output_sha256": sha256_file(args.raw_output),
        "row_count": len(rows),
        "seed_fold_count": len(rows),
        "train_orientation_accuracy_pct_mean": mean([float(r["train_orientation_accuracy_pct"]) for r in rows]),
        "train_orientation_accuracy_pct_std": stdev([float(r["train_orientation_accuracy_pct"]) for r in rows]),
        "train_orientation_margin_mean": mean([float(r["train_orientation_margin"]) for r in rows]),
        "train_orientation_margin_std": stdev([float(r["train_orientation_margin"]) for r in rows]),
        "holdout_orientation_accuracy_pct_mean": mean([float(r["holdout_orientation_accuracy_pct"]) for r in rows]),
        "holdout_orientation_accuracy_pct_std": stdev([float(r["holdout_orientation_accuracy_pct"]) for r in rows]),
        "holdout_orientation_margin_mean": mean([float(r["holdout_orientation_margin"]) for r in rows]),
        "holdout_orientation_margin_std": stdev([float(r["holdout_orientation_margin"]) for r in rows]),
    }

    detail_path = args.output_dir / "orientation_head_trace_detail.tsv"
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_dir / "orientation_head_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Orientation Head Trace Summary",
        "",
        "Claim boundary: synthetic orientation-head diagnostic only. This is not clinical, biomarker, mechanistic, null-validated, or broad O-SSM evidence.",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- train orientation accuracy mean: `{summary['train_orientation_accuracy_pct_mean']:.6f}`",
        f"- train orientation margin mean: `{summary['train_orientation_margin_mean']:.6f}`",
        f"- holdout orientation accuracy mean: `{summary['holdout_orientation_accuracy_pct_mean']:.6f}`",
        f"- holdout orientation margin mean: `{summary['holdout_orientation_margin_mean']:.6f}`",
        "",
    ]
    (args.output_dir / "orientation_head_trace_summary.md").write_text("\n".join(md), encoding="utf-8")

    sums = []
    for path in (summary_path, detail_path, args.output_dir / "orientation_head_trace_summary.md"):
        sums.append(f"{sha256_file(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
