#!/usr/bin/env python3
"""Build a synthetic octonionic associator trajectory manifest.

This generator avoids the quaternionic/Fano-line trap: triples that lie on one
Fano line are associative, so they are weak tests of an octonionic state-space
model. Here every pair uses an imaginary-unit triple with nonzero associator.
The paired labels keep the same endpoints, mean, energy, and unordered vector
multiset; only the temporal order flips the associator sign.
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


SCHEMA = "neurodyn.octonionic_associator_manifest.v1"
UNIT_TOL = 1.0e-9
CONTINUOUS_TOL = 1.0e-7
CLAIM_BOUNDARY = (
    "Synthetic non-clinical octonionic associator trajectory assay only. "
    "No clinical, biomarker, biological mechanism, solved-transfer, global "
    "classifier, or broad O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=56)
    ap.add_argument("--sites", type=int, default=7)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--magnitude", type=float, default=3.0)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026071010)
    ap.add_argument("--anchor", choices=("zero", "unit_real"), default="unit_real")
    ap.add_argument("--site-mode", choices=("assoc_dim", "round_robin"), default="assoc_dim")
    ap.add_argument("--triple-selection", choices=("ordered", "seeded_shuffle"), default="ordered")
    ap.add_argument(
        "--triple-source",
        choices=("unit", "continuous", "scaled_unit"),
        default="unit",
        help="Use unit triples, scaled unit triples, or continuous imaginary vectors around fixed-target unit triples.",
    )
    ap.add_argument(
        "--scale-jitter",
        type=float,
        default=0.12,
        help="Uniform relative scale jitter for --triple-source=scaled_unit.",
    )
    ap.add_argument(
        "--continuous-jitter",
        type=float,
        default=0.18,
        help="Gaussian jitter scale for --triple-source=continuous before renormalizing each imaginary vector.",
    )
    ap.add_argument(
        "--continuous-max-attempts",
        type=int,
        default=200000,
        help="Maximum rejection-sampling attempts for continuous fixed-target triples.",
    )
    ap.add_argument(
        "--target-assoc-dim",
        type=int,
        default=-1,
        help="Restrict pairs to a fixed dominant associator component 0..7; -1 keeps all components.",
    )
    ap.add_argument(
        "--target-assoc-sign",
        type=int,
        choices=(-1, 0, 1),
        default=0,
        help="Dominant associator sign for the positive-order member when --target-assoc-dim is set; 0 keeps both signs.",
    )
    ap.add_argument(
        "--exclude-assoc-dims",
        default="",
        help="Comma-separated associator components to exclude before pair selection, e.g. 0.",
    )
    ap.add_argument(
        "--assoc-sign-filter",
        type=int,
        choices=(-1, 0, 1),
        default=0,
        help="Keep only triples whose positive-order dominant associator sign is -1 or 1; 0 keeps both signs.",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def basis(index: int, value: float = 1.0) -> list[float]:
    out = [0.0] * 8
    out[index] = value
    return out


def neg(vec: list[float]) -> list[float]:
    return [-value for value in vec]


def add_vec(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


def sub_vec(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def scale_vec(vec: list[float], scale: float) -> list[float]:
    return [scale * value for value in vec]


def oct_mul(a: list[float], b: list[float]) -> list[float]:
    a0, a1, a2, a3, a4, a5, a6, a7 = a
    b0, b1, b2, b3, b4, b5, b6, b7 = b
    return [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6,
        a0 * b2 + a2 * b0 - a1 * b3 + a3 * b1 + a4 * b6 - a6 * b4 + a5 * b7 - a7 * b5,
        a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1 + a4 * b7 - a7 * b4 - a5 * b6 + a6 * b5,
        a0 * b4 + a4 * b0 - a1 * b5 + a5 * b1 - a2 * b6 + a6 * b2 - a3 * b7 + a7 * b3,
        a0 * b5 + a5 * b0 + a1 * b4 - a4 * b1 - a2 * b7 + a7 * b2 + a3 * b6 - a6 * b3,
        a0 * b6 + a6 * b0 + a1 * b7 - a7 * b1 + a2 * b4 - a4 * b2 - a3 * b5 + a5 * b3,
        a0 * b7 + a7 * b0 - a1 * b6 + a6 * b1 + a2 * b5 - a5 * b2 + a3 * b4 - a4 * b3,
    ]


def associator(a: list[float], b: list[float], c: list[float]) -> list[float]:
    left = oct_mul(oct_mul(a, b), c)
    right = oct_mul(a, oct_mul(b, c))
    return [x - y for x, y in zip(left, right, strict=True)]


def norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def dominant_component(vec: list[float]) -> tuple[int, float]:
    idx, value = max(enumerate(vec), key=lambda item: abs(item[1]))
    return idx, value


def nonassociative_triples(magnitude: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) != 3:
                    continue
                av = associator(basis(a, magnitude), basis(b, magnitude), basis(c, magnitude))
                assoc_norm = norm(av)
                if assoc_norm <= 1.0e-9:
                    continue
                swapped = associator(basis(b, magnitude), basis(a, magnitude), basis(c, magnitude))
                dim, value = dominant_component(av)
                swapped_dim, swapped_value = dominant_component(swapped)
                if dim != swapped_dim or abs(value + swapped_value) > 1.0e-9:
                    continue
                rows.append(
                    {
                        "triple": (a, b, c),
                        "assoc_vector": av,
                        "assoc_norm": assoc_norm,
                        "assoc_dim": dim,
                        "assoc_sign": 1 if value > 0.0 else -1,
                        "assoc_value": value,
                    }
                )
    rows.sort(key=lambda row: (row["assoc_dim"], -row["assoc_sign"], row["triple"]))
    return rows


def imaginary_norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec[1:]))


def normalize_imaginary(vec: list[float], magnitude: float) -> list[float]:
    out = list(vec)
    out[0] = 0.0
    n = imaginary_norm(out)
    if n <= 1.0e-12:
        raise ValueError("cannot normalize zero imaginary vector")
    scale = magnitude / n
    return [0.0, *[value * scale for value in out[1:]]]


def jitter_basis(index: int, magnitude: float, rng: random.Random, jitter: float) -> list[float]:
    vec = basis(index, magnitude)
    if jitter <= 0.0:
        return vec
    for dim in range(1, 8):
        vec[dim] += rng.gauss(0.0, jitter * magnitude)
    return normalize_imaginary(vec, magnitude)


def continuous_triples(
    base_triples: list[dict[str, Any]],
    pairs: int,
    magnitude: float,
    rng: random.Random,
    target_assoc_dim: int,
    target_assoc_sign: int,
    jitter: float,
    max_attempts: int,
) -> list[dict[str, Any]]:
    if target_assoc_dim < 0 or target_assoc_dim > 7:
        raise SystemExit("--triple-source=continuous currently requires --target-assoc-dim 0..7")
    if not base_triples:
        raise SystemExit("--triple-source=continuous has no fixed-target base triples after filtering")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(rows) < pairs and attempts < max_attempts:
        attempts += 1
        base = rng.choice(base_triples)
        a_idx, b_idx, c_idx = base["triple"]
        va = jitter_basis(a_idx, magnitude, rng, jitter)
        vb = jitter_basis(b_idx, magnitude, rng, jitter)
        vc = jitter_basis(c_idx, magnitude, rng, jitter)
        av = associator(va, vb, vc)
        assoc_norm = norm(av)
        if assoc_norm <= 1.0e-9:
            continue
        swapped = associator(vb, va, vc)
        dim, value = dominant_component(av)
        swapped_dim, swapped_value = dominant_component(swapped)
        sign = 1 if value > 0.0 else -1
        if dim != target_assoc_dim or (target_assoc_sign != 0 and sign != target_assoc_sign):
            continue
        if swapped_dim != dim or abs(value + swapped_value) > CONTINUOUS_TOL:
            continue
        # Avoid exact duplicate continuous samples after rounding; this keeps
        # scaled packages from silently replaying the same jitter realization.
        key = tuple(vector_key(va) + vector_key(vb) + vector_key(vc))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "triple": base["triple"],
                "vectors": (va, vb, vc),
                "assoc_vector": av,
                "assoc_norm": assoc_norm,
                "assoc_dim": dim,
                "assoc_sign": sign,
                "assoc_value": value,
                "base_triple": base["triple"],
                "continuous_attempt": attempts,
            }
        )
    if len(rows) < pairs:
        raise SystemExit(
            "not enough continuous fixed-target triples: "
            f"target_dim={target_assoc_dim} target_sign={target_assoc_sign} "
            f"generated={len(rows)} requested_pairs={pairs} attempts={attempts}"
        )
    return rows


def endpoint(anchor: str) -> list[float]:
    if anchor == "zero":
        return [0.0] * 8
    if anchor == "unit_real":
        return basis(0)
    raise ValueError(anchor)


def make_core(triple: tuple[int, int, int], label: int, anchor: str, magnitude: float) -> list[list[float]]:
    a, b, c = triple
    va = basis(a, magnitude)
    vb = basis(b, magnitude)
    vc = basis(c, magnitude)
    return make_core_from_vectors(va, vb, vc, label, anchor)


def make_core_from_vectors(
    va: list[float],
    vb: list[float],
    vc: list[float],
    label: int,
    anchor: str,
) -> list[list[float]]:
    first = [va, vb, vc] if label == 1 else [vb, va, vc]
    tail = [neg(first[0]), neg(first[1]), neg(first[2])]
    return [endpoint(anchor), *first, *tail, endpoint(anchor)]


def add_noise(vec: list[float], rng: random.Random, noise_std: float) -> list[float]:
    if noise_std <= 0.0:
        return list(vec)
    return [value + rng.gauss(0.0, noise_std) for value in vec]


def stretch(core: list[list[float]], seq_len: int, rng: random.Random, noise_std: float) -> list[list[float]]:
    if seq_len < len(core):
        raise ValueError(f"seq_len={seq_len} < core_len={len(core)}")
    return [add_noise(core[(idx * len(core)) // seq_len], rng, noise_std) for idx in range(seq_len)]


def stretched_core_indices(seq_len: int, core_len: int = 8) -> dict[int, int]:
    indices: dict[int, int] = {}
    for idx in range(seq_len):
        core_idx = (idx * core_len) // seq_len
        indices.setdefault(core_idx, idx)
    missing = [core_idx for core_idx in (1, 2, 3) if core_idx not in indices]
    if missing:
        raise ValueError(f"seq_len={seq_len} does not expose associator event core indices: {missing}")
    return indices


def realized_associator_target(
    sequence: list[list[float]],
    component: int,
    event_indices: dict[int, int],
) -> dict[str, Any]:
    av = associator(
        sequence[event_indices[1]],
        sequence[event_indices[2]],
        sequence[event_indices[3]],
    )
    dim, dominant_value = dominant_component(av)
    if component < 0:
        target_component = dim
    elif component < len(av):
        target_component = component
    else:
        raise ValueError(f"target component out of range: {component}")
    value = av[target_component]
    return {
        "target_component": target_component,
        "target_scalar": value,
        "target_sign": 1 if value >= 0.0 else -1,
        "dominant_component": dim,
        "dominant_value": dominant_value,
        "assoc_norm": norm(av),
        "assoc_vector": av,
    }


def flatten(seq: list[list[float]]) -> list[float]:
    return [value for row in seq for value in row]


def vector_key(vec: list[float]) -> tuple[int, ...]:
    return tuple(int(round(value * 1_000_000.0)) for value in vec)


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


def pair_invariants(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos, neg_record in pairs:
        ps = stats(pos["sequence_noiseless"])
        ns = stats(neg_record["sequence_noiseless"])
        rows.append(
            {
                "pair_id": pos["pair_id"],
                "triple": pos["triple"],
                "assoc_dim": pos["assoc_dim"],
                "positive_assoc_value": pos["assoc_value"],
                "negative_assoc_value": neg_record["assoc_value"],
                "assoc_sum_abs": abs(pos["assoc_value"] + neg_record["assoc_value"]),
                "mean_max_abs_diff": max_abs([a - b for a, b in zip(ps["mean"], ns["mean"], strict=True)]),
                "delta_max_abs_diff": max_abs([a - b for a, b in zip(ps["delta"], ns["delta"], strict=True)]),
                "start_max_abs_diff": max_abs([a - b for a, b in zip(ps["start"], ns["start"], strict=True)]),
                "end_max_abs_diff": max_abs([a - b for a, b in zip(ps["end"], ns["end"], strict=True)]),
                "energy_abs_diff": abs(ps["energy"] - ns["energy"]),
                "same_unordered_multiset": int(ps["multiset"] == ns["multiset"]),
            }
        )
    return rows


def write_manifest(path: Path, records: list[dict[str, Any]], seq_len: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# schema=neurodyn.octonionic_associator_manifest.v1\n"
            "# dataset_id=synthetic_octonionic_associator_trajectory\n"
            f"# seq_len={seq_len}\n"
            "# input_dim=8\n"
            "# feature_layout=flat\n"
            "# label_space=associator_positive_vs_swapped_sign\n"
            "# label_positive_name=ASSOCIATOR_POSITIVE_ORDER\n"
            "# label_negative_name=ASSOCIATOR_SWAPPED_SIGN\n"
            "# split_policy=pseudo_site_holdout_pair_balanced\n"
            "# feature_recipe=paired_nonassociative_triple_same_mean_delta_energy_multiset\n"
            "# task=octonionic_nonassociative_temporal_trajectory\n"
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


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def target_support_summary(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["target_scalar"]) for row in target_rows]
    rounded = [round(value, 10) for value in values]
    counts = Counter(rounded)
    tied_rows = sum(count for count in counts.values() if count > 1)
    sign_counts = Counter(int(row["target_sign"]) for row in target_rows)
    by_site: dict[str, Counter[int]] = defaultdict(Counter)
    for row in target_rows:
        by_site[str(row["site"])][int(row["target_sign"])] += 1
    return {
        "target_row_count": len(target_rows),
        "distinct_target_values": len(counts),
        "tie_fraction": tied_rows / len(target_rows) if target_rows else 1.0,
        "target_min": min(values) if values else None,
        "target_max": max(values) if values else None,
        "target_sign_counts": {str(key): sign_counts[key] for key in sorted(sign_counts)},
        "site_target_sign_counts": {
            site: {str(key): counts[key] for key in sorted(counts)}
            for site, counts in sorted(by_site.items())
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Octonionic Associator Trajectory Manifest",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "## Package",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Pairs: `{summary['pairs']}`",
        f"- Records: `{summary['records']}`",
        f"- Sites: `{summary['sites']}`",
        f"- Sequence length: `{summary['seq_len']}`",
        f"- Magnitude: `{summary['magnitude']}`",
        f"- Noise std: `{summary['noise_std']}`",
        f"- Anchor: `{summary['anchor']}`",
        "",
        "## Algebraic Gate",
        "",
        f"- Nonassociative triples available: `{summary['nonassociative_triples_available']}`",
        f"- Target associator dim: `{summary['target_assoc_dim']}`",
        f"- Target associator sign: `{summary['target_assoc_sign']}`",
        f"- Associator norm min: `{summary['associator_norm_min']}`",
        f"- Associator norm max: `{summary['associator_norm_max']}`",
        f"- Pair sign-flip max residual: `{summary['invariant_max']['assoc_sum_abs']}`",
        "",
        "## Pair Invariants",
        "",
    ]
    for key, value in summary["invariant_max"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
            "This package is ready for one hidden-state trace smoke only. Do not expand to nulls or clinical data unless the trained model passes global, hidden-state, checkpoint replay, and associator-specific gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.site_mode == "assoc_dim" and args.sites != 7:
        raise SystemExit("--site-mode=assoc_dim requires --sites=7")
    if args.target_assoc_dim != -1:
        if args.target_assoc_dim < 0 or args.target_assoc_dim > 7:
            raise SystemExit("--target-assoc-dim must be -1 or an octonion associator component 0..7")
        if args.site_mode == "assoc_dim":
            raise SystemExit("--target-assoc-dim requires --site-mode=round_robin so held-out sites do not equal the target component")
    if args.pairs < args.sites:
        raise SystemExit("--pairs must be >= --sites")

    triples = nonassociative_triples(args.magnitude)
    if not triples:
        raise SystemExit("no nonassociative triples found")
    total_nonassociative_triples = len(triples)
    excluded_dims = {int(part.strip()) for part in args.exclude_assoc_dims.split(",") if part.strip()}
    if excluded_dims:
        triples = [row for row in triples if int(row["assoc_dim"]) not in excluded_dims]
        if not triples:
            raise SystemExit(f"--exclude-assoc-dims removed all triples: {sorted(excluded_dims)}")
    if args.assoc_sign_filter != 0:
        triples = [row for row in triples if int(row["assoc_sign"]) == args.assoc_sign_filter]
        if not triples:
            raise SystemExit(f"--assoc-sign-filter={args.assoc_sign_filter} removed all triples")
    if args.triple_source in {"continuous", "scaled_unit"} and args.target_assoc_dim == -1:
        raise SystemExit(f"--triple-source={args.triple_source} requires --target-assoc-dim")
    if args.target_assoc_dim != -1:
        triples = [
            row
            for row in triples
            if int(row["assoc_dim"]) == args.target_assoc_dim
            and (args.target_assoc_sign == 0 or int(row["assoc_sign"]) == args.target_assoc_sign)
        ]
        if args.triple_source == "unit" and len(triples) < args.pairs:
            raise SystemExit(
                "not enough fixed-target nonassociative triples: "
                f"target_dim={args.target_assoc_dim} target_sign={args.target_assoc_sign} "
                f"available={len(triples)} requested_pairs={args.pairs}"
            )
    rng = random.Random(args.seed)
    if args.triple_selection == "seeded_shuffle":
        rng.shuffle(triples)
    if args.triple_source == "continuous":
        triples = continuous_triples(
            triples,
            args.pairs,
            args.magnitude,
            rng,
            args.target_assoc_dim,
            args.target_assoc_sign,
            args.continuous_jitter,
            args.continuous_max_attempts,
        )
    elif args.triple_source == "scaled_unit":
        base_triples = list(triples)
        triples = []
        if not base_triples:
            raise SystemExit("--triple-source=scaled_unit has no fixed-target base triples after filtering")
        for pair_id in range(args.pairs):
            base = base_triples[pair_id % len(base_triples)]
            rel = 1.0
            if args.scale_jitter > 0.0:
                rel += rng.uniform(-args.scale_jitter, args.scale_jitter)
            if rel <= 0.0:
                raise SystemExit("--scale-jitter produced non-positive pair scale")
            pair_magnitude = args.magnitude * rel
            a, b, c = base["triple"]
            va = basis(a, pair_magnitude)
            vb = basis(b, pair_magnitude)
            vc = basis(c, pair_magnitude)
            av = associator(va, vb, vc)
            dim, value = dominant_component(av)
            triples.append(
                {
                    **base,
                    "vectors": (va, vb, vc),
                    "assoc_vector": av,
                    "assoc_norm": norm(av),
                    "assoc_dim": dim,
                    "assoc_sign": 1 if value > 0.0 else -1,
                    "assoc_value": value,
                    "base_triple": base["triple"],
                    "scale": pair_magnitude,
                }
            )
    triples_by_dim: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        triples_by_dim[int(triple["assoc_dim"])].append(triple)
    if args.triple_selection == "seeded_shuffle":
        for dim_rows in triples_by_dim.values():
            rng.shuffle(dim_rows)
    dims = sorted(triples_by_dim)
    if args.site_mode == "assoc_dim" and len(dims) != args.sites:
        raise SystemExit(f"--site-mode=assoc_dim expected {args.sites} associator dims, got {len(dims)}")
    records: list[dict[str, Any]] = []
    pair_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    triple_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    event_indices = stretched_core_indices(args.seq_len)

    for pair_id in range(args.pairs):
        if args.site_mode == "assoc_dim":
            dim = dims[pair_id % len(dims)]
            dim_rows = triples_by_dim[dim]
            triple_meta = dim_rows[(pair_id // len(dims)) % len(dim_rows)]
        else:
            triple_meta = triples[pair_id % len(triples)]
        triple = tuple(triple_meta["triple"])
        va, vb, vc = (
            tuple(triple_meta["vectors"])
            if args.triple_source in {"continuous", "scaled_unit"}
            else (basis(triple[0], args.magnitude), basis(triple[1], args.magnitude), basis(triple[2], args.magnitude))
        )
        assoc_dim = int(triple_meta["assoc_dim"])
        if args.site_mode == "assoc_dim":
            site = f"assoc_dim_{assoc_dim:02d}"
        else:
            site = f"pseudo_site_{pair_id % args.sites:02d}"
        pair_seed = rng.randrange(1 << 30)
        pos_core = make_core_from_vectors(list(va), list(vb), list(vc), 1, args.anchor)
        neg_core = make_core_from_vectors(list(va), list(vb), list(vc), 0, args.anchor)
        pos_assoc = associator(pos_core[1], pos_core[2], pos_core[3])
        neg_assoc = associator(neg_core[1], neg_core[2], neg_core[3])
        pos_dim, pos_value = dominant_component(pos_assoc)
        neg_dim, neg_value = dominant_component(neg_assoc)
        tolerance = CONTINUOUS_TOL if args.triple_source == "continuous" else UNIT_TOL
        if pos_dim != neg_dim or abs(pos_value + neg_value) > tolerance:
            raise SystemExit(f"associator sign-flip invariant failed for pair {pair_id}: {triple}")
        pos_sequence_noiseless = stretch(pos_core, args.seq_len, random.Random(0), 0.0)
        neg_sequence_noiseless = stretch(neg_core, args.seq_len, random.Random(0), 0.0)
        pos_sequence = stretch(pos_core, args.seq_len, random.Random(pair_seed), args.noise_std)
        neg_sequence = stretch(neg_core, args.seq_len, random.Random(pair_seed), args.noise_std)
        target_component = args.target_assoc_dim if args.target_assoc_dim != -1 else pos_dim
        pos_target = realized_associator_target(pos_sequence, target_component, event_indices)
        neg_target = realized_associator_target(neg_sequence, target_component, event_indices)
        pos_record = {
            "subject_id": f"assoc_pair_{pair_id:04d}__positive",
            "pair_id": pair_id,
            "label": 1,
            "site": site,
            "triple": "-".join(str(item) for item in triple),
            "assoc_dim": pos_dim,
            "assoc_value": pos_value,
            "target_scalar": pos_target["target_scalar"],
            "target_sign": pos_target["target_sign"],
            "target_component": pos_target["target_component"],
            "sequence_noiseless": pos_sequence_noiseless,
            "sequence": pos_sequence,
        }
        neg_record = {
            "subject_id": f"assoc_pair_{pair_id:04d}__negative",
            "pair_id": pair_id,
            "label": 0,
            "site": site,
            "triple": "-".join(str(item) for item in triple),
            "assoc_dim": neg_dim,
            "assoc_value": neg_value,
            "target_scalar": neg_target["target_scalar"],
            "target_sign": neg_target["target_sign"],
            "target_component": neg_target["target_component"],
            "sequence_noiseless": neg_sequence_noiseless,
            "sequence": neg_sequence,
        }
        records.extend([pos_record, neg_record])
        pair_records.append((pos_record, neg_record))
        for orientation, record, target in (
            ("positive", pos_record, pos_target),
            ("negative", neg_record, neg_target),
        ):
            target_rows.append(
                {
                    "subject_id": record["subject_id"],
                    "pair_id": pair_id,
                    "orientation": orientation,
                    "site": site,
                    "label": record["label"],
                    "target_component": target["target_component"],
                    "target_scalar": f"{target['target_scalar']:.12f}",
                    "target_sign": target["target_sign"],
                    "dominant_component": target["dominant_component"],
                    "dominant_value": f"{target['dominant_value']:.12f}",
                    "assoc_norm": f"{target['assoc_norm']:.12f}",
                    "assoc_vector": ",".join(f"{value:.12f}" for value in target["assoc_vector"]),
                    "target_source": "realized_stretched_sequence",
                }
            )
        triple_rows.append(
            {
                "pair_id": pair_id,
                "triple": pos_record["triple"],
                "triple_source": args.triple_source,
                "base_triple": "-".join(str(item) for item in triple_meta.get("base_triple", triple)),
                "continuous_attempt": triple_meta.get("continuous_attempt", 0),
                "scale": triple_meta.get("scale", args.magnitude),
                "site": site,
                "assoc_dim": pos_dim,
                "positive_assoc_value": pos_value,
                "negative_assoc_value": neg_value,
                "positive_assoc_norm": norm(pos_assoc),
                "negative_assoc_norm": norm(neg_assoc),
            }
        )

    manifest = out / "octonionic_associator_manifest.tsv"
    write_manifest(manifest, records, args.seq_len)
    invariant_rows = pair_invariants(pair_records)
    write_csv(out / "paired_associator_invariant_audit.csv", invariant_rows)
    write_tsv(out / "associator_triples.tsv", triple_rows)
    target_path = out / "associator_targets.tsv"
    write_tsv(target_path, target_rows)
    support_summary = target_support_summary(target_rows)
    by_site: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        by_site[record["site"]][int(record["label"])] += 1
    invariant_max = {
        "assoc_sum_abs": max(row["assoc_sum_abs"] for row in invariant_rows),
        "mean_max_abs_diff": max(row["mean_max_abs_diff"] for row in invariant_rows),
        "delta_max_abs_diff": max(row["delta_max_abs_diff"] for row in invariant_rows),
        "start_max_abs_diff": max(row["start_max_abs_diff"] for row in invariant_rows),
        "end_max_abs_diff": max(row["end_max_abs_diff"] for row in invariant_rows),
        "energy_abs_diff": max(row["energy_abs_diff"] for row in invariant_rows),
        "same_unordered_multiset_min": min(row["same_unordered_multiset"] for row in invariant_rows),
    }
    for key, value in invariant_max.items():
        if key == "same_unordered_multiset_min":
            if int(value) != 1:
                raise SystemExit("paired unordered multiset invariant failed")
        elif float(value) > (CONTINUOUS_TOL if args.triple_source == "continuous" else UNIT_TOL):
            raise SystemExit(f"paired invariant failed for {key}: {value}")
    assoc_norms = [row["positive_assoc_norm"] for row in triple_rows] + [row["negative_assoc_norm"] for row in triple_rows]
    summary = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(out),
        "manifest": str(manifest),
        "pairs": args.pairs,
        "records": len(records),
        "sites": args.sites,
        "seq_len": args.seq_len,
        "input_dim": 8,
        "magnitude": args.magnitude,
        "noise_std": args.noise_std,
        "seed": args.seed,
        "anchor": args.anchor,
        "site_mode": args.site_mode,
        "triple_selection": args.triple_selection,
        "triple_source": args.triple_source,
        "continuous_jitter": args.continuous_jitter if args.triple_source == "continuous" else 0.0,
        "continuous_max_attempts": args.continuous_max_attempts if args.triple_source == "continuous" else 0,
        "scale_jitter": args.scale_jitter if args.triple_source == "scaled_unit" else 0.0,
        "target_assoc_dim": args.target_assoc_dim,
        "target_assoc_sign": args.target_assoc_sign if args.target_assoc_dim != -1 else 0,
        "associator_targets": str(target_path),
        "target_source": "realized_stretched_sequence",
        "target_support": support_summary,
        "excluded_assoc_dims": sorted(excluded_dims),
        "assoc_sign_filter": args.assoc_sign_filter,
        "nonassociative_triples_available": total_nonassociative_triples,
        "selected_triples_available": len(triples),
        "associator_norm_min": min(assoc_norms),
        "associator_norm_max": max(assoc_norms),
        "label_counts": {
            "0": sum(1 for record in records if record["label"] == 0),
            "1": sum(1 for record in records if record["label"] == 1),
        },
        "site_label_counts": {
            site: {"0": counts[0], "1": counts[1]} for site, counts in sorted(by_site.items())
        },
        "invariant_max": invariant_max,
        "decision": "OCTONIONIC_ASSOCIATOR_TRAJECTORY_MANIFEST_READY_FOR_SINGLE_TRACE_SMOKE",
    }
    (out / "octonionic_associator_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "octonionic_associator_manifest_summary.md").write_text(markdown(summary), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
