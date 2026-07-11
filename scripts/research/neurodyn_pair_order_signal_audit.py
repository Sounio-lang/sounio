#!/usr/bin/env python3
"""Audit paired temporal-order signals in NeuroDyn synthetic manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def read_manifest(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"manifest has no data rows: {path}")
    return list(csv.DictReader(lines, delimiter="\t"))


def pair_id(subject_id: str) -> str:
    if "__" in subject_id:
        return subject_id.split("__", 1)[0]
    return subject_id


def variant(subject_id: str, label: str) -> str:
    if subject_id.endswith("__oriented"):
        return "oriented"
    if subject_id.endswith("__swapped"):
        return "swapped"
    return f"label_{label}"


def feature_matrix(row: dict[str, str], seq_len: int, input_dim: int) -> list[list[float]]:
    xs: list[list[float]] = []
    for step in range(seq_len):
        cur = []
        for dim in range(input_dim):
            cur.append(float(row[f"f{step * input_dim + dim}"]))
        xs.append(cur)
    return xs


def wedge_signal(xs: list[list[float]]) -> list[float]:
    input_dim = len(xs[0])
    seq_len = len(xs)
    out = []
    for dim in range(input_dim):
        nxt = (dim + 1) % input_dim
        acc = 0.0
        for step in range(seq_len - 1):
            acc += xs[step][dim] * xs[step + 1][nxt] - xs[step][nxt] * xs[step + 1][dim]
        out.append(acc / seq_len)
    return out


def l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a: list[float], b: list[float]) -> float:
    den = l2(a) * l2(b)
    if den <= 0:
        return 0.0
    return dot(a, b) / den


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--input-dim", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pair_order_signal_summary.json"
    pairs_path = args.output_dir / "pair_order_signal_pairs.tsv"
    if not args.overwrite and (summary_path.exists() or pairs_path.exists()):
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")

    rows = read_manifest(args.manifest)
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        sid = row["subject_id"]
        lab = row["label"]
        xs = feature_matrix(row, args.seq_len, args.input_dim)
        sig = wedge_signal(xs)
        grouped[pair_id(sid)][variant(sid, lab)] = {
            "subject_id": sid,
            "label": int(lab),
            "site": row["site"],
            "signal": sig,
            "norm": l2(sig),
        }

    pair_rows = []
    for pid in sorted(grouped):
        item = grouped[pid]
        if "oriented" not in item or "swapped" not in item:
            continue
        oriented = item["oriented"]
        swapped = item["swapped"]
        osig = oriented["signal"]  # type: ignore[assignment]
        ssig = swapped["signal"]  # type: ignore[assignment]
        assert isinstance(osig, list)
        assert isinstance(ssig, list)
        diff = [a - b for a, b in zip(osig, ssig)]
        pair_rows.append(
            {
                "pair_id": pid,
                "site": oriented["site"],
                "oriented_subject": oriented["subject_id"],
                "swapped_subject": swapped["subject_id"],
                "oriented_norm": float(oriented["norm"]),
                "swapped_norm": float(swapped["norm"]),
                "diff_norm": l2(diff),
                "cosine": cosine(osig, ssig),
                "dot": dot(osig, ssig),
                **{f"diff_d{i}": diff[i] for i in range(args.input_dim)},
            }
        )

    if not pair_rows:
        raise SystemExit("no complete oriented/swapped pairs found")

    diff_norms = [float(r["diff_norm"]) for r in pair_rows]
    cosines = [float(r["cosine"]) for r in pair_rows]
    oriented_norms = [float(r["oriented_norm"]) for r in pair_rows]
    swapped_norms = [float(r["swapped_norm"]) for r in pair_rows]
    summary = {
        "schema": "neurodyn_pair_order_signal_audit.v1",
        "manifest": str(args.manifest),
        "pairs": len(pair_rows),
        "seq_len": args.seq_len,
        "input_dim": args.input_dim,
        "diff_norm_mean": mean(diff_norms),
        "diff_norm_std": pstdev(diff_norms),
        "diff_norm_min": min(diff_norms),
        "diff_norm_max": max(diff_norms),
        "cosine_mean": mean(cosines),
        "cosine_std": pstdev(cosines),
        "oriented_norm_mean": mean(oriented_norms),
        "swapped_norm_mean": mean(swapped_norms),
    }

    with pairs_path.open("w", newline="") as f:
        fieldnames = list(pair_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(pair_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
