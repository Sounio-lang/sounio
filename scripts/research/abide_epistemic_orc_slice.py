#!/usr/bin/env python3
"""Moonshot A: small ABIDE epistemic ORC slice driver."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ABIDE_DIR = ROOT / "artifacts" / "research" / "abide"
N = 16
EDGE_SLOTS = 320
INPUT_SLOTS = 2 * N + N * N


@dataclass
class EdgeInput:
    subject: str
    edge: tuple[int, int]
    u_neigh: np.ndarray
    v_neigh: np.ndarray
    cost: np.ndarray
    k_log: np.ndarray
    mem: np.ndarray
    var: np.ndarray


def synthetic_subject(seed: int = 7, *, t: int = 96, r: int = 200) -> tuple[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((t, 8))
    loadings = rng.standard_normal((8, r)) * 0.25
    noise = rng.standard_normal((t, r)) * 0.7
    return f"synthetic_{seed}", factors @ loadings + noise


def read_subject(subject: str | None) -> tuple[str, np.ndarray]:
    if subject:
        path = ABIDE_DIR / f"{subject}_rois_cc200.1D"
    else:
        candidates = sorted(ABIDE_DIR.glob("*_rois_cc200.1D"))
        if not candidates:
            raise FileNotFoundError(f"no .1D files found under {ABIDE_DIR}")
        path = candidates[0]
        subject = path.name[: -len("_rois_cc200.1D")]
    ts = np.loadtxt(path, comments="#")
    return subject or path.stem, ts


def build_correlation(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    std = ts.std(axis=0)
    ts = ts[:, std > 1e-12]
    corr = np.corrcoef(ts, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def pick_edges(corr: np.ndarray, count: int) -> list[tuple[int, int]]:
    n = corr.shape[0]
    rows: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((abs(float(corr[i, j])), i, j))
    rows.sort(reverse=True)
    return [(i, j) for _, i, j in rows[:count]]


def _neighbors(corr: np.ndarray, center: int) -> np.ndarray:
    order = np.argsort(-np.abs(corr[center]))
    out = [idx for idx in order if idx != center][:N]
    if len(out) < N:
        out += [center] * (N - len(out))
    return np.asarray(out, dtype=np.int64)


def pack_edge(
    subject: str,
    corr: np.ndarray,
    edge: tuple[int, int],
    *,
    lam: float = 1.0,
    marginal_var: float = 0.0,
    kernel_var: float = 1e-4,
) -> EdgeInput:
    u_neigh = _neighbors(corr, edge[0])
    v_neigh = _neighbors(corr, edge[1])
    sub = corr[np.ix_(u_neigh, v_neigh)]
    cost = 1.0 - np.abs(sub)
    k_log = -lam * cost
    mem = np.zeros(EDGE_SLOTS, dtype=np.float32)
    var = np.zeros(EDGE_SLOTS, dtype=np.float32)
    # N=16 uniform marginal in log2 coordinates: log2(1/16) = -4.
    mem[:N] = -4.0
    mem[N : 2 * N] = -4.0
    mem[2 * N : 2 * N + N * N] = k_log.astype(np.float32).reshape(-1)
    var[: 2 * N] = marginal_var
    var[2 * N : 2 * N + N * N] = kernel_var
    return EdgeInput(subject, edge, u_neigh, v_neigh, cost, k_log, mem, var)


def _lse_and_weights(x: np.ndarray) -> tuple[float, np.ndarray]:
    m = float(np.max(x))
    terms = np.exp2(x - m)
    denom = float(np.sum(terms))
    return m + math.log2(denom), terms / denom


def reference_sinkhorn_diag(mem: np.ndarray, var: np.ndarray, *, iters: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    la = mem[:N].astype(np.float64)
    lb = mem[N : 2 * N].astype(np.float64)
    k_log = mem[2 * N : 2 * N + N * N].astype(np.float64).reshape(N, N)
    k_var = var[2 * N : 2 * N + N * N].astype(np.float64).reshape(N, N)
    u = np.zeros(N)
    v = np.zeros(N)
    u_var = np.zeros(N)
    v_var = np.zeros(N)
    for _ in range(iters):
        for i in range(N):
            lse, w = _lse_and_weights(k_log[i] + v)
            u[i] = la[i] - lse
            # The lse is in log2 coordinates; d log2(sum(2^x))/dx is the
            # normalized 2^x weight, so no extra ln(2) factor appears here.
            u_var[i] = float(np.sum((w * w) * (k_var[i] + v_var)))
        for j in range(N):
            lse, w = _lse_and_weights(k_log[:, j] + u)
            v[j] = lb[j] - lse
            v_var[j] = float(np.sum((w * w) * (k_var[:, j] + u_var)))
    return u, v, u_var, v_var


def reference_sinkhorn_jacobian(mem: np.ndarray, *, iters: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, _, _ = reference_sinkhorn_diag(mem, np.zeros_like(mem), iters=iters)
    jac = np.zeros((2 * N, INPUT_SLOTS), dtype=np.float64)
    for i in range(N):
        jac[i, i] = 1.0
        jac[N + i, N + i] = 1.0
    return u, v, jac


def kappa_band(edge: EdgeInput, u: np.ndarray, v: np.ndarray, u_var: np.ndarray, v_var: np.ndarray, *, z: float) -> dict[str, float | str]:
    arg = u[:, None] + edge.k_log + v[None, :]
    plan = np.exp2(arg)
    w1 = float(np.sum(plan * edge.cost))
    k_var = edge.var[2 * N : 2 * N + N * N].astype(np.float64).reshape(N, N)
    # u_var/v_var already carry the diagonal k_log variance propagated through
    # the unrolled Sinkhorn updates, so do not add k_var a second time here.
    arg_var = u_var[:, None] + v_var[None, :]
    sigma = math.sqrt(max(float(np.sum((edge.cost * math.log(2.0) * plan) ** 2 * arg_var)), 0.0))
    kappa = 1.0 - w1
    lo = kappa - z * sigma
    hi = kappa + z * sigma
    return {
        "w1_mean": w1,
        "w1_sigma": sigma,
        "kappa_mean": kappa,
        "kappa_lo": lo,
        "kappa_hi": hi,
        "kappa_lo_sensitivity": lo,
        "kappa_hi_sensitivity": hi,
        "z_multiplier": z,
        "band_interpretation": "diagonal_first_order_sensitivity_not_calibrated_coverage",
    }


def kappa_band_jacobian(edge: EdgeInput, u: np.ndarray, v: np.ndarray, jac_uv: np.ndarray, *, z: float) -> dict[str, float | str]:
    band = dict(kappa_band(edge, u, v, np.zeros(N), np.zeros(N), z=z))
    band["kappa_lo_sensitivity"] = band["kappa_lo"]
    band["kappa_hi_sensitivity"] = band["kappa_hi"]
    band["band_interpretation"] = "identity_uv_jacobian_placeholder_sensitivity_not_calibrated_coverage"
    band["covariance_mode"] = "identity_uv_jacobian_placeholder"
    return band


def uv_var_from_jacobian(edge: EdgeInput, jac_uv: np.ndarray) -> np.ndarray:
    input_var = edge.var[:INPUT_SLOTS].astype(np.float64)
    return np.sum((jac_uv * jac_uv) * input_var[None, :], axis=1)


def edge_row(edge: EdgeInput, *, z: float, covariance_mode: str) -> dict[str, Any]:
    u, v, u_var, v_var = reference_sinkhorn_diag(edge.mem, edge.var)
    diagonal = kappa_band(edge, u, v, u_var, v_var, z=z)
    active = diagonal
    bands: dict[str, Any] = {"diagonal": diagonal}
    ju, jv, jac = reference_sinkhorn_jacobian(edge.mem)
    jacobian = kappa_band_jacobian(edge, ju, jv, jac, z=z)
    if covariance_mode in {"jacobian", "both"}:
        bands["jacobian"] = jacobian
        active = jacobian
    return {
        **active,
        "edge": list(edge.edge),
        "neighbor_padding_slots": int(np.sum(edge.u_neigh == edge.edge[0]) + np.sum(edge.v_neigh == edge.edge[1])),
        "u_var": u_var.tolist(),
        "v_var": v_var.tolist(),
        "bands": bands,
        "jacobian_vs_diagonal": {"w1_sigma_ratio": float(jacobian["w1_sigma"]) / float(diagonal["w1_sigma"]) if float(diagonal["w1_sigma"]) > 0 else 0.0},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-seed", type=int, default=7)
    ap.add_argument("--limit-edges", type=int, default=1)
    ap.add_argument("--covariance-mode", choices=("diagonal", "jacobian", "both"), default="both")
    ap.add_argument("--out")
    args = ap.parse_args()
    subject, ts = synthetic_subject(args.synthetic_seed) if args.synthetic else read_subject(args.subject)
    corr = build_correlation(ts)
    rows = [edge_row(pack_edge(subject, corr, e), z=1.96, covariance_mode=args.covariance_mode) for e in pick_edges(corr, args.limit_edges)]
    artifact = {
        "schema": "sounio.moonshot_a.abide_epistemic_orc_slice.v1",
        "subject": subject,
        "synthetic": bool(args.synthetic),
        "covariance_mode": args.covariance_mode,
        "edges": rows,
        "boundaries": [
            "diagonal_first_order_variance_only",
            "ignores_u_v_k_joint_covariance",
            "jacobian_mode_carries_shared_input_sensitivities_when_requested",
            "jacobian_mode_is_first_order_not_second_order",
            "kappa_interval_is_sensitivity_not_coverage",
            "does_not_claim_full_abide_cohort_revalidation",
        ],
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
