#!/usr/bin/env python3
"""Moonshot A: 168/ZD-labeled transport modulation prototype."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from abide_epistemic_orc_slice import (  # noqa: E402
    N,
    build_correlation,
    kappa_band,
    kappa_band_jacobian,
    pack_edge,
    pick_edges,
    read_subject,
    reference_sinkhorn_diag,
    reference_sinkhorn_jacobian,
    synthetic_subject,
    uv_var_from_jacobian,
)


@dataclass(frozen=True)
class PrimSed:
    lo: int
    hi: int
    neg: bool

    def key(self) -> tuple[int, int, int]:
        return (self.lo, self.hi, 1 if self.neg else 0)


def valid_prims() -> list[PrimSed]:
    return [PrimSed(lo, hi, neg) for lo in range(1, 8) for hi in range(9, 16) for neg in (False, True) if (lo ^ hi) != 8]


def zd_classes() -> list[tuple[PrimSed, PrimSed]]:
    prims = valid_prims()
    pairs: list[tuple[PrimSed, PrimSed]] = []
    for fiber in range(9, 16):
        group = [p for p in prims if (p.lo ^ p.hi) == fiber]
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pairs.append((a, b))
    if len(pairs) < 168:
        raise ValueError("168 runtime slots require at least 168 candidate fiber-group pairs")
    pairs = pairs[:168]
    return pairs


def class_mask(pair: tuple[PrimSed, PrimSed]) -> np.ndarray:
    mask = np.zeros(N, dtype=np.float64)
    for i in {pair[0].lo, pair[0].hi, pair[1].lo, pair[1].hi}:
        mask[i] = 1.0
    return mask


def modulate_k_log(k_log: np.ndarray, mask: np.ndarray, strength: float, mode: str) -> np.ndarray:
    if mode == "subspace":
        penalty = mask[:, None] * mask[None, :]
    elif mode == "cross":
        penalty = np.abs(mask[:, None] - mask[None, :])
    elif mode == "separable":
        penalty = (mask[:, None] + mask[None, :]) * 0.5
    else:
        raise ValueError(mode)
    return k_log - strength * penalty


@dataclass
class BaseEdge:
    subject: str
    corr: np.ndarray
    edge: tuple[int, int]
    lam: float
    marginal_var: float
    kernel_var: float


def band_for_edge(edge: Any, *, z: float, covariance_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    u, v, u_var, v_var = reference_sinkhorn_diag(edge.mem, edge.var)
    diagonal = kappa_band(edge, u, v, u_var, v_var, z=z)
    bands: dict[str, Any] = {"diagonal": diagonal}
    active = diagonal
    if covariance_mode in {"jacobian", "both"}:
        ju, jv, jac = reference_sinkhorn_jacobian(edge.mem)
        jacobian = kappa_band_jacobian(edge, ju, jv, jac, z=z)
        bands["jacobian"] = jacobian
        active = jacobian
    return active, bands


def row_for_class(idx: int, pair: tuple[PrimSed, PrimSed], base: BaseEdge, *, strength: float, mode: str, z: float, covariance_mode: str) -> dict[str, Any]:
    mask = class_mask(pair)
    edge = pack_edge(base.subject, base.corr, base.edge, lam=base.lam, marginal_var=base.marginal_var, kernel_var=base.kernel_var)
    edge.k_log[:, :] = modulate_k_log(edge.k_log, mask, strength, mode)
    edge.mem[2 * N : 2 * N + N * N] = edge.k_log.astype(np.float32).reshape(-1)
    active, bands = band_for_edge(edge, z=z, covariance_mode=covariance_mode)
    p0, p1 = pair
    ratio = 0.0
    if "jacobian" in bands and float(bands["diagonal"]["w1_sigma"]) > 0:
        ratio = float(bands["jacobian"]["w1_sigma"]) / float(bands["diagonal"]["w1_sigma"])
    return {
        **active,
        "class_index": idx,
        "is_padding": False,
        "covariance_mode": covariance_mode,
        "fiber_label": int(p0.lo ^ p0.hi),
        "primitives": [{"lo": p0.lo, "hi": p0.hi, "sign": -1 if p0.neg else 1}, {"lo": p1.lo, "hi": p1.hi, "sign": -1 if p1.neg else 1}],
        "forgotten_indices": np.nonzero(mask)[0].astype(int).tolist(),
        "bands": bands,
        "jacobian_vs_diagonal": {"w1_sigma_ratio": ratio, "uv_var_sum_ratio": 1.0},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-seed", type=int, default=7)
    ap.add_argument("--edge-rank", type=int, default=0)
    ap.add_argument("--classes", type=int, default=168)
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--mode", choices=("subspace", "cross", "separable"), default="subspace")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--marginal-var", type=float, default=0.0)
    ap.add_argument("--kernel-var", type=float, default=1e-4)
    ap.add_argument("--covariance-mode", choices=("diagonal", "jacobian", "both"), default="diagonal")
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--out")
    args = ap.parse_args()
    pairs = zd_classes()
    subject, ts = synthetic_subject(args.synthetic_seed) if args.synthetic else read_subject(args.subject)
    corr = build_correlation(ts)
    edge = pick_edges(corr, args.edge_rank + 1)[args.edge_rank]
    base = BaseEdge(subject, corr, edge, args.lam, args.marginal_var, args.kernel_var)
    base_edge = pack_edge(subject, corr, edge, lam=args.lam, marginal_var=args.marginal_var, kernel_var=args.kernel_var)
    baseline, baseline_bands = band_for_edge(base_edge, z=args.z, covariance_mode=args.covariance_mode)
    rows = [row_for_class(i, pair, base, strength=args.strength, mode=args.mode, z=args.z, covariance_mode=args.covariance_mode) for i, pair in enumerate(pairs[: args.classes])]
    shifts = np.array([float(r["kappa_mean"]) - float(baseline["kappa_mean"]) for r in rows])
    artifact = {
        "schema": "sounio.moonshot_a.transport_168_modulation.v1",
        "subject": subject,
        "synthetic": bool(args.synthetic),
        "edge": list(edge),
        "lambda": args.lam,
        "strength": args.strength,
        "modulation_mode": args.mode,
        "covariance_mode": args.covariance_mode,
        "classes_evaluated": args.classes,
        "zd_census": {
            "valid_primitives": 84,
            "candidate_unordered_pairs": 462,
            "runtime_slots_evaluated": 168,
            "fiber_labels": [9, 10, 11, 12, 13, 14, 15],
            "enumeration": "deterministic_prefix_of_fiber_group_unordered_pairs",
            "padding_rows_present": False,
        },
        "baseline": baseline,
        "baseline_bands": baseline_bands,
        "shift_summary": {"min": float(shifts.min()), "max": float(shifts.max()), "mean": float(shifts.mean()), "std": float(shifts.std()), "argmin_class_index": int(rows[int(shifts.argmin())]["class_index"]), "argmax_class_index": int(rows[int(shifts.argmax())]["class_index"])},
        "boundaries": [
            "ports_exact_lean_zd_census_to_runtime_enumeration",
            "modulates_K_log_upstream_of_sinkhorn_only",
            "4d_mask_is_union_of_two_primitive_supports",
            "168_runtime_slots_are_a_deterministic_prefix_of_candidate_pairs",
            "does_not_claim_final_zd_surgery_semantics",
            "jacobian_mode_carries_shared_input_sensitivities_when_requested",
            "jacobian_mode_is_first_order_not_second_order",
        ],
        "classes": rows,
    }
    text = json.dumps(artifact, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
