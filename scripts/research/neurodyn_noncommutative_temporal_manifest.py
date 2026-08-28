#!/usr/bin/env python3
"""Build a synthetic noncommutative temporal-composition manifest.

The assay is deliberately non-clinical. It creates paired sequences from
oriented octonion/Fano triples. Each pair has the same endpoints, same mean,
same energy, and same unordered vector multiset; only the internal temporal
order differs. It is meant to test whether the Brain O-SSM path can use
order/composition after direct delta/flat readouts are disabled.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "brain_ossm.neurodyn_noncommutative_temporal_prepare.v1"
CLAIM_BOUNDARY = (
    "Synthetic non-clinical temporal-composition diagnostic only. No clinical, "
    "biomarker, mechanistic, or O-SSM superiority claim."
)

# Oriented Fano lines for the common octonion convention over e1..e7.
FANO_LINES = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs", type=int, default=280)
    parser.add_argument("--sites", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--magnitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--line-mode",
        choices=("fano_cycle", "fixed_pair"),
        default="fano_cycle",
        help="Use all oriented Fano lines, or a single fixed e1/e2/e3 sanity pair.",
    )
    parser.add_argument(
        "--anchor",
        choices=("zero", "unit_real"),
        default="zero",
        help="Common first/last vector. unit_real avoids zero-step state erasure in Brain O-SSM.",
    )
    parser.add_argument(
        "--site-mode",
        choices=("round_robin", "fano_line"),
        default="round_robin",
        help=(
            "Pseudo-site assignment. round_robin preserves the original mixed-line "
            "holdouts; fano_line makes leave-one-site-out CV a leave-one-Fano-line-out "
            "transfer assay."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def basis(index: int, sign: float = 1.0) -> list[float]:
    if index < 0 or index > 7:
        raise ValueError(index)
    out = [0.0] * 8
    out[index] = sign
    return out


def zero() -> list[float]:
    return [0.0] * 8


def neg(vec: list[float]) -> list[float]:
    return [-value for value in vec]


def add_noise(vec: list[float], rng: random.Random, noise_std: float) -> list[float]:
    if noise_std <= 0.0:
        return list(vec)
    return [value + rng.gauss(0.0, noise_std) for value in vec]


def vector_key(vec: list[float]) -> tuple[int, ...]:
    return tuple(int(round(value * 1000000.0)) for value in vec)


def make_core(line: tuple[int, int, int], positive: bool, variant: int, anchor: str, magnitude: float) -> list[list[float]]:
    a, b, c = line
    va = basis(a, magnitude)
    vb = basis(b, magnitude)
    vc = basis(c, magnitude)
    if anchor == "zero":
        endpoint = zero()
    elif anchor == "unit_real":
        endpoint = basis(0)
    else:
        raise ValueError(anchor)
    # Positive uses the oriented Fano order. Negative swaps the first two
    # elements. Both retain the same signed multiset after the balancing tail.
    if positive:
        first = [va, vb, vc]
    else:
        first = [vb, va, vc]
    if variant % 2 == 0:
        tail = [neg(first[0]), neg(first[1]), neg(first[2])]
    else:
        tail = [neg(first[2]), neg(first[1]), neg(first[0])]
    return [endpoint, *first, *tail, endpoint]


def stretch_to_seq(core: list[list[float]], seq_len: int, rng: random.Random, noise_std: float) -> list[list[float]]:
    if seq_len < len(core):
        raise ValueError(f"seq_len={seq_len} < core_len={len(core)}")
    out: list[list[float]] = []
    for idx in range(seq_len):
        source = (idx * len(core)) // seq_len
        out.append(add_noise(core[source], rng, noise_std))
    return out


def flatten(seq: list[list[float]]) -> list[float]:
    return [value for row in seq for value in row]


def stats(seq: list[list[float]]) -> dict[str, Any]:
    dim = len(seq[0])
    mean = [sum(row[d] for row in seq) / len(seq) for d in range(dim)]
    delta = [seq[-1][d] - seq[0][d] for d in range(dim)]
    energy = sum(value * value for row in seq for value in row)
    return {
        "mean": mean,
        "delta": delta,
        "energy": energy,
        "start": seq[0],
        "end": seq[-1],
        "multiset": Counter(vector_key(row) for row in seq),
    }


def max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def paired_invariant_rows(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos, neg_record in pairs:
        pos_stats = stats(pos["sequence_noiseless"])
        neg_stats = stats(neg_record["sequence_noiseless"])
        rows.append(
            {
                "pair_id": pos["pair_id"],
                "line": pos["line"],
                "variant": pos["variant"],
                "mean_max_abs_diff": max_abs([a - b for a, b in zip(pos_stats["mean"], neg_stats["mean"], strict=True)]),
                "delta_max_abs_diff": max_abs([a - b for a, b in zip(pos_stats["delta"], neg_stats["delta"], strict=True)]),
                "start_max_abs_diff": max_abs([a - b for a, b in zip(pos_stats["start"], neg_stats["start"], strict=True)]),
                "end_max_abs_diff": max_abs([a - b for a, b in zip(pos_stats["end"], neg_stats["end"], strict=True)]),
                "energy_abs_diff": abs(pos_stats["energy"] - neg_stats["energy"]),
                "same_unordered_multiset": int(pos_stats["multiset"] == neg_stats["multiset"]),
            }
        )
    return rows


def write_tsv(path: Path, records: list[dict[str, Any]], seq_len: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# schema=brain_ossm.neurodyn.synthetic.v1\n"
            "# dataset_id=synthetic_noncommutative_temporal\n"
            f"# seq_len={seq_len}\n"
            "# input_dim=8\n"
            "# feature_layout=flat\n"
            "# label_space=fano_oriented_order_vs_swapped_order\n"
            "# label_positive_name=FANO_ORIENTED_ORDER\n"
            "# label_negative_name=SWAPPED_ORDER\n"
            "# split_policy=pseudo_site_holdout_pair_balanced\n"
            "# feature_recipe=paired_fano_triple_same_mean_delta_energy_multiset\n"
            "# task=noncommutative_temporal_composition\n"
            f"# subject_count={len(records)}\n"
            f"# site_count={len({record['site'] for record in records})}\n"
            f"# claim_boundary={CLAIM_BOUNDARY}\n"
        )
        header = ["subject_id", "label", "site"] + [f"f{i}" for i in range(seq_len * 8)]
        handle.write("\t".join(header) + "\n")
        for record in records:
            fields = [record["subject_id"], str(record["label"]), record["site"]]
            fields.extend(f"{value:.10f}" for value in flatten(record["sequence"]))
            handle.write("\t".join(fields) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pairs < args.sites:
        raise SystemExit("--pairs must be >= --sites")
    if args.sites < 2:
        raise SystemExit("--sites must be >= 2")

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    pair_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair_id in range(args.pairs):
        if args.line_mode == "fixed_pair":
            line = FANO_LINES[0]
            variant = pair_id
            line_idx = 0
        else:
            line_idx = pair_id % len(FANO_LINES)
            line = FANO_LINES[line_idx]
            variant = pair_id // len(FANO_LINES)
        if args.site_mode == "fano_line":
            if args.line_mode != "fano_cycle":
                raise SystemExit("--site-mode=fano_line requires --line-mode=fano_cycle")
            if args.sites != len(FANO_LINES):
                raise SystemExit(f"--site-mode=fano_line requires --sites={len(FANO_LINES)}")
            site = f"fano_line_{line_idx:02d}_{'-'.join(str(item) for item in line)}"
        else:
            site = f"pseudo_site_{pair_id % args.sites:02d}"
        pair_seed = rng.randrange(1 << 30)
        pos_rng = random.Random(pair_seed)
        neg_rng = random.Random(pair_seed)
        pos_core = make_core(line, True, variant, args.anchor, args.magnitude)
        neg_core = make_core(line, False, variant, args.anchor, args.magnitude)
        pos_record = {
            "subject_id": f"synthetic_pair_{pair_id:04d}__oriented",
            "pair_id": pair_id,
            "label": 1,
            "site": site,
            "line": "-".join(str(item) for item in line),
            "variant": variant,
            "sequence_noiseless": stretch_to_seq(pos_core, args.seq_len, random.Random(0), 0.0),
            "sequence": stretch_to_seq(pos_core, args.seq_len, pos_rng, args.noise_std),
        }
        neg_record = {
            "subject_id": f"synthetic_pair_{pair_id:04d}__swapped",
            "pair_id": pair_id,
            "label": 0,
            "site": site,
            "line": "-".join(str(item) for item in line),
            "variant": variant,
            "sequence_noiseless": stretch_to_seq(neg_core, args.seq_len, random.Random(0), 0.0),
            "sequence": stretch_to_seq(neg_core, args.seq_len, neg_rng, args.noise_std),
        }
        records.extend([pos_record, neg_record])
        pair_records.append((pos_record, neg_record))

    manifest = output_dir / "noncommutative_temporal_manifest.tsv"
    write_tsv(manifest, records, args.seq_len)
    invariant_rows = paired_invariant_rows(pair_records)
    write_csv(output_dir / "paired_invariant_audit.csv", invariant_rows)
    by_site: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        by_site[record["site"]][int(record["label"])] += 1
    summary = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "manifest": str(manifest),
        "pairs": args.pairs,
        "records": len(records),
        "sites": args.sites,
        "seq_len": args.seq_len,
        "input_dim": 8,
        "noise_std": args.noise_std,
        "magnitude": args.magnitude,
        "seed": args.seed,
        "line_mode": args.line_mode,
        "site_mode": args.site_mode,
        "anchor": args.anchor,
        "claim_boundary": CLAIM_BOUNDARY,
        "fano_lines": FANO_LINES,
        "label_counts": {
            "0": sum(1 for record in records if record["label"] == 0),
            "1": sum(1 for record in records if record["label"] == 1),
        },
        "site_label_counts": {
            site: {"0": counts[0], "1": counts[1]} for site, counts in sorted(by_site.items())
        },
        "invariant_max": {
            "mean_max_abs_diff": max(row["mean_max_abs_diff"] for row in invariant_rows),
            "delta_max_abs_diff": max(row["delta_max_abs_diff"] for row in invariant_rows),
            "start_max_abs_diff": max(row["start_max_abs_diff"] for row in invariant_rows),
            "end_max_abs_diff": max(row["end_max_abs_diff"] for row in invariant_rows),
            "energy_abs_diff": max(row["energy_abs_diff"] for row in invariant_rows),
            "same_unordered_multiset_min": min(row["same_unordered_multiset"] for row in invariant_rows),
        },
    }
    write_json(output_dir / "noncommutative_temporal_prepare_summary.json", summary)
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(f"manifest={manifest}")
    print(f"records={len(records)}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
