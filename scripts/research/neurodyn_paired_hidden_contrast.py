#!/usr/bin/env python3
"""Paired oriented-vs-swapped hidden-state contrast audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from neurodyn_hidden_state_separability import parse_state_rows


SCHEMA = "neurodyn.paired_hidden_contrast.v1"


def read_manifest(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"manifest has no data rows: {path}")
    return list(csv.DictReader(lines, delimiter="\t"))


def pair_id(subject_id: str) -> str:
    if "__" in subject_id:
        return subject_id.split("__", 1)[0]
    return subject_id


def variant(subject_id: str, label: int) -> str:
    if subject_id.endswith("__oriented"):
        return "oriented"
    if subject_id.endswith("__swapped"):
        return "swapped"
    return "oriented" if label == 1 else "swapped"


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def dist(a: list[float], b: list[float]) -> float:
    return norm([x - y for x, y in zip(a, b)])


def mean_vec(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-output", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "paired_hidden_contrast_summary.tsv"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")

    manifest_rows = read_manifest(args.manifest)
    subject_meta = {
        idx: {
            "subject_id": row["subject_id"],
            "pair_id": pair_id(row["subject_id"]),
            "variant": variant(row["subject_id"], int(row["label"])),
            "label": int(row["label"]),
            "site": row["site"],
        }
        for idx, row in enumerate(manifest_rows)
    }
    state_rows = parse_state_rows(args.raw_output)
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_rows:
        meta = subject_meta.get(row["subject_index"])
        if meta is None:
            continue
        key = (row["model"], row["seed"])
        grouped[key][f"{meta['pair_id']}::{meta['variant']}"] = {
            **row,
            **meta,
        }

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (model, seed), chunk in sorted(grouped.items()):
        pair_ids = sorted({value["pair_id"] for value in chunk.values()})
        pair_diffs: list[dict[str, Any]] = []
        for pid in pair_ids:
            o = chunk.get(f"{pid}::oriented")
            s = chunk.get(f"{pid}::swapped")
            if o is None or s is None:
                continue
            diff = [a - b for a, b in zip(o["h"], s["h"])]
            pair_diffs.append(
                {
                    "model": model,
                    "seed": seed,
                    "pair_id": pid,
                    "site": o["site"],
                    "hidden_pair_distance": dist(o["h"], s["h"]),
                    "diff": diff,
                }
            )
        if not pair_diffs:
            continue
        sites = sorted({row["site"] for row in pair_diffs})
        correct = 0
        total = 0
        for site in sites:
            train = [row for row in pair_diffs if row["site"] != site]
            test = [row for row in pair_diffs if row["site"] == site]
            direction = mean_vec([row["diff"] for row in train])
            if not direction:
                continue
            for row in test:
                if dot(row["diff"], direction) > 0.0:
                    correct += 1
                total += 1
        diffs = [row["diff"] for row in pair_diffs]
        mean_direction = mean_vec(diffs)
        distances = [float(row["hidden_pair_distance"]) for row in pair_diffs]
        direction_norm = norm(mean_direction)
        detail_rows.extend(
            {
                "model": row["model"],
                "seed": row["seed"],
                "pair_id": row["pair_id"],
                "site": row["site"],
                "hidden_pair_distance": row["hidden_pair_distance"],
                "diff_alignment_with_seed_mean": dot(row["diff"], mean_direction) / (norm(row["diff"]) * direction_norm)
                if norm(row["diff"]) > 0 and direction_norm > 0
                else 0.0,
            }
            for row in pair_diffs
        )
        summary_rows.append(
            {
                "model": model,
                "seed": seed,
                "pairs": len(pair_diffs),
                "hidden_pair_distance_mean": mean(distances),
                "hidden_pair_distance_std": pstdev(distances),
                "mean_direction_norm": direction_norm,
                "direction_norm_over_pair_distance": direction_norm / mean(distances) if mean(distances) > 0 else 0.0,
                "leave_site_pair_direction_accuracy": 100.0 * correct / total if total else 0.0,
            }
        )

    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        model_groups[row["model"]].append(row)
    model_summary = []
    for model, rows in sorted(model_groups.items()):
        acc = [float(r["leave_site_pair_direction_accuracy"]) for r in rows]
        dist_mean = [float(r["hidden_pair_distance_mean"]) for r in rows]
        ratio = [float(r["direction_norm_over_pair_distance"]) for r in rows]
        model_summary.append(
            {
                "model": model,
                "seed_count": len(rows),
                "leave_site_pair_direction_accuracy_mean": mean(acc),
                "leave_site_pair_direction_accuracy_std": pstdev(acc),
                "hidden_pair_distance_mean": mean(dist_mean),
                "direction_norm_over_pair_distance_mean": mean(ratio),
            }
        )

    def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as f:
            if not rows:
                return
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(args.output_dir / "paired_hidden_contrast_summary.tsv", model_summary)
    write_tsv(args.output_dir / "paired_hidden_contrast_seed_detail.tsv", summary_rows)
    write_tsv(args.output_dir / "paired_hidden_contrast_pair_detail.tsv", detail_rows)
    payload = {
        "schema": SCHEMA,
        "raw_output": str(args.raw_output),
        "manifest": str(args.manifest),
        "state_trace_rows": len(state_rows),
        "model_summary": model_summary,
    }
    (args.output_dir / "paired_hidden_contrast.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
