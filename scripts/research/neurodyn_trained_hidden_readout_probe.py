#!/usr/bin/env python3
"""Probe STATE_TRACE rows with the trained hidden readout vector from CKPT blocks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from neurodyn_hidden_state_separability import parse_state_rows


SCHEMA = "neurodyn.trained_hidden_readout_probe.v1"


def parse_ckpts(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    ckpts: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        if lines[idx] != "CKPT_BEGIN":
            idx += 1
            continue
        meta: dict[str, str] = {}
        arrays: dict[str, list[int]] = {}
        idx += 1
        while idx < len(lines) and lines[idx] != "CKPT_END":
            line = lines[idx]
            if line.startswith("CKPT_META\t"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    meta[parts[1]] = parts[2]
                idx += 1
                continue
            if line.startswith("CKPT_ARRAY\t"):
                parts = line.split("\t")
                if len(parts) < 4:
                    raise SystemExit(f"malformed CKPT_ARRAY at line {idx + 1}: {line}")
                name = parts[1]
                count = int(parts[3])
                values: list[int] = []
                idx += 1
                while idx < len(lines) and len(values) < count:
                    raw = lines[idx].strip()
                    if raw and name == "W":
                        values.append(int(raw))
                    idx += 1
                if name == "W" and len(values) != count:
                    raise SystemExit(f"short CKPT_ARRAY {name}: expected {count}, got {len(values)}")
                if name == "W":
                    arrays[name] = values
                continue
            idx += 1
        if "W" in arrays and meta.get("model") == "O-SSM":
            w = arrays["W"]
            ckpts.append(
                {
                    "seed": int(meta["seed_a"]),
                    "holdout_site": meta["holdout_site"],
                    "holdout_key": meta.get("holdout_key", ""),
                    "balanced_accuracy_pct": float(meta.get("balanced_accuracy_pct", "0")),
                    "hidden_w": [value / 100_000_000.0 for value in w[:16]],
                    "bias": w[570] / 100_000_000.0 if len(w) > 570 else 0.0,
                }
            )
        idx += 1
    if not ckpts:
        raise SystemExit("no O-SSM CKPT W arrays found")
    return ckpts


def parse_readout_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("READOUT_TRACE\t"):
            idx += 1
            continue
        parts = [part for part in line.split("\t") if part]
        line_no = idx + 1
        next_idx = idx + 1
        while len(parts) < 26 and next_idx < len(lines) and lines[next_idx].startswith("\t"):
            parts.extend(part for part in lines[next_idx].split("\t") if part)
            next_idx += 1
        if len(parts) != 26:
            raise SystemExit(f"malformed READOUT_TRACE at line {line_no}: expected 26 fields, got {len(parts)}")
        (
            _,
            model,
            seed,
            seed_b,
            holdout_key,
            holdout_site,
            bal_bp,
            acc_bp,
            test_n,
            bias,
            *raw_w,
        ) = parts
        if model != "O-SSM":
            continue
        traces.append(
            {
                "seed": int(seed),
                "seed_b": int(seed_b),
                "holdout_key": holdout_key,
                "holdout_site": holdout_site,
                "balanced_accuracy_pct": int(bal_bp) / 1_000_000.0,
                "accuracy_pct": int(acc_bp) / 1_000_000.0,
                "test_n": int(test_n),
                "hidden_w": [int(value) / 100_000_000.0 for value in raw_w],
                "bias": int(bias) / 100_000_000.0,
                "source": "READOUT_TRACE",
            }
        )
        idx = next_idx
    return traces


def balanced_accuracy(labels: list[int], preds: list[int]) -> float:
    pos = [i for i, label in enumerate(labels) if label == 1]
    neg = [i for i, label in enumerate(labels) if label == 0]
    if not pos or not neg:
        return 0.0
    tpr = sum(1 for i in pos if preds[i] == 1) / len(pos)
    tnr = sum(1 for i in neg if preds[i] == 0) / len(neg)
    return 100.0 * (tpr + tnr) / 2.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-output", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "trained_hidden_readout_probe_summary.tsv"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")

    ckpts = parse_readout_traces(args.raw_output)
    source = "READOUT_TRACE"
    if not ckpts:
        ckpts = parse_ckpts(args.raw_output)
        source = "CKPT_ARRAY_W"
    state_rows = parse_state_rows(args.raw_output)
    by_seed_site: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        if row["model"] == "O-SSM":
            by_seed_site[(row["seed"], row["site"])].append(row)

    detail = []
    for ckpt in ckpts:
        rows = by_seed_site.get((ckpt["seed"], ckpt["holdout_site"]), [])
        if not rows:
            continue
        labels: list[int] = []
        preds: list[int] = []
        scores: list[float] = []
        for row in rows:
            score = ckpt["bias"] + sum(h * w for h, w in zip(row["h"], ckpt["hidden_w"], strict=True))
            labels.append(int(row["label"]))
            preds.append(1 if score >= 0.0 else 0)
            scores.append(score)
        detail.append(
            {
                "seed": ckpt["seed"],
                "holdout_site": ckpt["holdout_site"],
                "rows": len(rows),
                "ckpt_balanced_accuracy_pct": ckpt["balanced_accuracy_pct"],
                "trained_hidden_readout_balanced_accuracy": balanced_accuracy(labels, preds),
                "score_mean": mean(scores),
                "score_std": pstdev(scores),
                "positive_score_frac": mean([1.0 if score >= 0.0 else 0.0 for score in scores]),
            }
        )

    if not detail:
        raise SystemExit("no checkpoint/state rows matched by seed and holdout site")

    ba = [float(row["trained_hidden_readout_balanced_accuracy"]) for row in detail]
    ckpt_ba = [float(row["ckpt_balanced_accuracy_pct"]) for row in detail]
    summary = [
        {
            "model": "O-SSM",
            "fold_count": len(detail),
            "trained_hidden_readout_balanced_accuracy_mean": mean(ba),
            "trained_hidden_readout_balanced_accuracy_std": pstdev(ba),
            "ckpt_balanced_accuracy_pct_mean": mean(ckpt_ba),
            "ckpt_balanced_accuracy_pct_std": pstdev(ckpt_ba),
        }
    ]

    def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(summary_path, summary)
    write_tsv(args.output_dir / "trained_hidden_readout_probe_detail.tsv", detail)
    payload = {
        "schema": SCHEMA,
        "raw_output": str(args.raw_output),
        "readout_source": source,
        "checkpoint_count": len(ckpts),
        "matched_fold_count": len(detail),
        "summary": summary,
    }
    (args.output_dir / "trained_hidden_readout_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
