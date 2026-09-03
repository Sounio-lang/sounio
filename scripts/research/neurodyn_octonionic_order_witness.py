#!/usr/bin/env python3
"""Deterministic order-sensitivity witness for the Brain O-SSM update path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.octonionic_order_witness.v1"
CLAIM_BOUNDARY = (
    "Deterministic synthetic update-path diagnostic only. This is not a "
    "clinical, biomarker, mechanistic, or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: list[float]) -> list[float]:
    n = norm(v)
    if n <= 1e-12:
        return [1.0] + [0.0] * 7
    return [x / n for x in v]


def dist(a: list[float], b: list[float]) -> float:
    return norm([x - y for x, y in zip(a, b, strict=True)])


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


def basis(index: int, sign: float = 1.0) -> list[float]:
    out = [0.0] * 8
    out[index] = sign
    return out


def md2_squash(v: float) -> float:
    if v >= 0.0:
        return v / (1.0 + v)
    return v / (1.0 + (-v))


def mandelbrot_d2_feature(seed: float, delta: float) -> float:
    z = clip(0.55 * seed + 0.20 * delta, -2.0, 2.0)
    c = clip(0.35 * seed + 0.25 * delta, -2.0, 2.0)
    dz = 1.0 + 0.10 * delta
    ddz = 0.0
    for _ in range(3):
        ddz = 2.0 * (dz * dz + z * ddz)
        dz = 2.0 * z * dz + 1.0 + 0.15 * delta
        z = clip(z * z + c, -4.0, 4.0)
        dz = clip(dz, -6.0, 6.0)
        ddz = clip(ddz, -8.0, 8.0)
    return clip(md2_squash(ddz), -1.5, 1.5)


def project_identity(x: list[float], prev_raw: list[float] | None) -> tuple[list[float], list[float]]:
    delta = [0.0] * 8 if prev_raw is None else [x[i] - prev_raw[i] for i in range(8)]
    return list(x), delta


def project_mandel_hybrid(x: list[float], prev_raw: list[float] | None) -> tuple[list[float], list[float]]:
    delta = [0.0] * 8 if prev_raw is None else [x[i] - prev_raw[i] for i in range(8)]
    out: list[float] = []
    for d in range(8):
        base = x[d]
        md2 = mandelbrot_d2_feature(base, delta[d])
        # Mirrors the benchmark's preset-ish projection defaults: structured and
        # delta terms are active; learned correlation starts close to identity.
        struct_gate = clip(0.55 + 0.45 * 0.65, -1.5, 1.5)
        delta_gate = clip(0.75, -1.5, 1.5)
        corr = base
        s = base + 0.35 * struct_gate * md2 + 0.18 * delta_gate * delta[d] + 0.12 * 0.20 * (corr - base)
        out.append(clip(s, -4.0, 4.0))
    return out, delta


def brain_g2_preset_a() -> list[float]:
    return normalize([1.0, 0.62, 0.62, 0.0, 0.62, 0.0, 0.0, 0.0])


def brain_step(h: list[float], x: list[float], a: list[float], clip_state: bool = True) -> tuple[list[float], float]:
    ah = oct_mul(a, h)
    left = oct_mul(ah, x)
    hx = oct_mul(h, x)
    right = oct_mul(a, hx)
    assoc = dist(left, right)
    next_h = [left[i] + 0.2 * right[i] for i in range(8)]
    if clip_state:
        next_h = [clip(v, -5.0, 5.0) for v in next_h]
    return next_h, assoc


def stable_sigmoid_step(h: list[float], x: list[float]) -> tuple[list[float], float]:
    a = [0.82, 0.06, 0.03, 0.02, 0.03, 0.02, 0.01, 0.01]
    b = [0.44, 0.09, 0.06, 0.04, 0.05, 0.03, 0.02, 0.02]
    ah = oct_mul(a, h)
    bx = oct_mul(b, x)
    raw = [ah[i] + bx[i] + 0.18 * h[i] for i in range(8)]
    next_h = [1.0 / (1.0 + math.exp(-clip(v, -15.0, 15.0))) for v in raw]
    assoc = dist(oct_mul(oct_mul(h, x), next_h), oct_mul(h, oct_mul(x, next_h)))
    return next_h, assoc


def fixed_pair_core(oriented: bool, anchor: str) -> list[list[float]]:
    e1, e2, e3 = basis(1), basis(2), basis(3)
    if anchor == "zero":
        endpoint = [0.0] * 8
    elif anchor == "unit_real":
        endpoint = basis(0)
    else:
        raise ValueError(anchor)
    first = [e1, e2, e3] if oriented else [e2, e1, e3]
    tail = [[-v for v in first[0]], [-v for v in first[1]], [-v for v in first[2]]]
    return [endpoint, *first, *tail, endpoint]


def stretch(core: list[list[float]], seq_len: int = 32) -> list[list[float]]:
    return [core[(idx * len(core)) // seq_len] for idx in range(seq_len)]


def run_sequence(
    seq: list[list[float]],
    projection: str,
    update: str,
    clip_state: bool = True,
) -> dict[str, Any]:
    h = basis(0)
    a = brain_g2_preset_a()
    prev_raw: list[float] | None = None
    assoc_values: list[float] = []
    states: list[list[float]] = []
    for raw in seq:
        if projection == "identity":
            x, _ = project_identity(raw, prev_raw)
        elif projection == "mandel_hybrid":
            x, _ = project_mandel_hybrid(raw, prev_raw)
        else:
            raise ValueError(projection)
        if update == "brain":
            h, assoc = brain_step(h, x, a, clip_state=clip_state)
        elif update == "stable_sigmoid":
            h, assoc = stable_sigmoid_step(h, x)
        else:
            raise ValueError(update)
        assoc_values.append(assoc)
        states.append(list(h))
        prev_raw = raw
    return {
        "final_h": h,
        "final_norm": norm(h),
        "assoc_mean": sum(assoc_values) / len(assoc_values),
        "assoc_max": max(assoc_values),
        "states": states,
    }


def compare_regime(anchor: str, projection: str, update: str, clip_state: bool = True) -> dict[str, Any]:
    pos = run_sequence(stretch(fixed_pair_core(True, anchor)), projection, update, clip_state)
    neg = run_sequence(stretch(fixed_pair_core(False, anchor)), projection, update, clip_state)
    step_distances = [dist(a, b) for a, b in zip(pos["states"], neg["states"], strict=True)]
    return {
        "anchor": anchor,
        "projection": projection,
        "update": update,
        "clip_state": clip_state,
        "final_distance": dist(pos["final_h"], neg["final_h"]),
        "max_step_distance": max(step_distances),
        "mean_step_distance": sum(step_distances) / len(step_distances),
        "positive_final_norm": pos["final_norm"],
        "negative_final_norm": neg["final_norm"],
        "positive_assoc_mean": pos["assoc_mean"],
        "negative_assoc_mean": neg["assoc_mean"],
        "positive_assoc_max": pos["assoc_max"],
        "negative_assoc_max": neg["assoc_max"],
        "positive_final_h": pos["final_h"],
        "negative_final_h": neg["final_h"],
    }


def write_hashes(output_dir: Path) -> None:
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Octonionic Order Witness",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "| anchor | projection | update | clip | final distance | max step distance | pos assoc mean | neg assoc mean |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["regimes"]:
        lines.append(
            "| {anchor} | {projection} | {update} | {clip_state} | {final_distance:.9f} | {max_step_distance:.9f} | {positive_assoc_mean:.9f} | {negative_assoc_mean:.9f} |".format(
                **row
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

    regimes = [
        compare_regime("zero", "identity", "brain", True),
        compare_regime("zero", "mandel_hybrid", "brain", True),
        compare_regime("zero", "identity", "stable_sigmoid", True),
        compare_regime("unit_real", "identity", "brain", True),
        compare_regime("unit_real", "mandel_hybrid", "brain", True),
        compare_regime("unit_real", "identity", "stable_sigmoid", True),
    ]
    interpretation = (
        "The deterministic update-path witness checks whether order reversal changes final "
        "state before any classifier. If identity+brain is order-sensitive but "
        "mandel_hybrid+brain is weak, the projection/encoding path is suspect. If both are "
        "order-sensitive while trained hidden traces are chance, optimization/readout-to-state "
        "coupling is suspect. If identity+brain is not order-sensitive, the recurrent update "
        "itself needs redesign."
    )
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "assay": "fixed_pair_e1_e2_e3_oriented_vs_swapped",
        "regimes": regimes,
        "interpretation": interpretation,
    }
    (output_dir / "octonionic_order_witness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "octonionic_order_witness.md").write_text(markdown(payload), encoding="utf-8")
    write_hashes(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "regimes": regimes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
