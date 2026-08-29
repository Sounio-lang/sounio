#!/usr/bin/env python3
"""Summarize temporal-arrow logit contribution traces from Brain O-SSM output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from abide_campaign_lib import load_manifest


SCHEMA = "neurodyn.temporal_arrow_logit_parts_audit.v1"
PART_FIELDS = ["bias", "hidden", "assoc", "mean", "delta", "flat", "total", "logit"]
CLAIM_BOUNDARY = (
    "Temporal-arrow diagnostic only. Channel-level readout tracing is not a "
    "clinical, diagnostic, mechanistic, or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--condition", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def direction(subject_id: str) -> str:
    if subject_id.endswith("__forward"):
        return "forward"
    if subject_id.endswith("__reverse"):
        return "reverse"
    return "unknown"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def collect_parts(lines: list[str], records: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("LOGIT_PARTS"):
            idx += 1
            continue
        parts = [part for part in line.split("\t") if part]
        next_idx = idx + 1
        while len(parts) < 14 and next_idx < len(lines) and lines[next_idx].startswith("\t"):
            parts.extend(part for part in lines[next_idx].split("\t") if part)
            next_idx += 1
        if len(parts) != 14:
            raise ValueError(f"malformed LOGIT_PARTS line at {idx + 1}: {line}")
        _, model, seed, subject_index_raw, site, label, *raw_values = parts
        subject_index = int(subject_index_raw)
        subject_id = records[subject_index].subject_id if 0 <= subject_index < len(records) else ""
        values = {field: int(raw) / 1_000_000.0 for field, raw in zip(PART_FIELDS, raw_values, strict=True)}
        dominant = max(PART_FIELDS[:-2], key=lambda field: abs(values[field]))
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "subject_index": subject_index,
                "subject_id": subject_id,
                "direction": direction(subject_id),
                "site": site,
                "label": int(label),
                "dominant_part": dominant,
                **values,
            }
        )
        idx = next_idx
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["direction"])].append(row)
    out: list[dict[str, Any]] = []
    for (model, dir_name) in sorted(grouped):
        chunk = grouped[(model, dir_name)]
        dominant_counts = Counter(str(row["dominant_part"]) for row in chunk)
        top_part, top_count = dominant_counts.most_common(1)[0]
        summary: dict[str, Any] = {
            "condition": condition,
            "model": model,
            "direction": dir_name,
            "rows": len(chunk),
            "seed_count": len({row["seed"] for row in chunk}),
            "label_mean": mean([float(row["label"]) for row in chunk]),
            "dominant_part_mode": top_part,
            "dominant_part_mode_frac": top_count / len(chunk),
            "positive_logit_frac": mean([1.0 if float(row["logit"]) > 0 else 0.0 for row in chunk]),
        }
        for field in PART_FIELDS:
            values = [float(row[field]) for row in chunk]
            summary[f"{field}_mean"] = mean(values)
            summary[f"{field}_std"] = pstdev(values)
        out.append(summary)
    return out


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


def write_hashes(output_dir: Path) -> None:
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Temporal-Arrow Logit Parts Audit",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        f"- condition: `{payload['condition']}`",
        f"- RUN_ID: `{payload['run_id']}`",
        f"- row_count: `{payload['row_count']}`",
        "",
        "## Direction Summary",
        "",
        "| model | direction | logit mean | hidden mean | assoc mean | mean-term | delta mean | flat mean | dominant mode | positive logit frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {model} | {direction} | {logit:.6f} | {hidden:.6f} | {assoc:.6f} | {mean:.6f} | {delta:.6f} | {flat:.6f} | {dom} | {pos:.6f} |".format(
                model=row["model"],
                direction=row["direction"],
                logit=row["logit_mean"],
                hidden=row["hidden_mean"],
                assoc=row["assoc_mean"],
                mean=row["mean_mean"],
                delta=row["delta_mean"],
                flat=row["flat_mean"],
                dom=row["dominant_part_mode"],
                pos=row["positive_logit_frac"],
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

    _, records = load_manifest(args.manifest)
    rows = collect_parts(read_lines(Path(args.raw_output)), records)
    if not rows:
        raise SystemExit("no LOGIT_PARTS rows found; rebuild Brain O-SSM with logit trace instrumentation")
    summary = summarize(rows, args.condition)
    dominant_modes = sorted({row["dominant_part_mode"] for row in summary})
    interpretation = (
        "The trace identifies which readout term dominates signed temporal-arrow logits. "
        "Use this as a localization diagnostic for the observed anti-polarity, not as a "
        "claim that the channel is mechanistic."
    )
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_id": args.run_id,
        "condition": args.condition,
        "raw_output": str(Path(args.raw_output).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "row_count": len(rows),
        "dominant_modes": dominant_modes,
        "summary": summary,
        "interpretation": interpretation,
    }
    write_json(output_dir / "temporal_arrow_logit_parts_audit.json", payload)
    write_tsv(output_dir / "temporal_arrow_logit_parts_summary.tsv", summary)
    write_tsv(output_dir / "temporal_arrow_logit_parts_rows.tsv", rows)
    (output_dir / "temporal_arrow_logit_parts_audit.md").write_text(markdown(payload), encoding="utf-8")
    write_hashes(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
