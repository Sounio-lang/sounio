#!/usr/bin/env python3
"""
Sub-PTX B2: feed a real ABIDE-I edge K matrix through the Sounio
Sinkhorn-LSE N=16 GPU kernel and cross-check the GPU output against
a NumPy/SciPy reference implementation of the same algorithm.

Pipeline (one subject at a time):
  1. Read a CC200 ROI time series from artifacts/research/abide/*.1D
  2. Compute the 200×200 Pearson correlation matrix
  3. Pick one edge (u, v) — by default the first non-self pair with the
     highest correlation outside the trivial diagonal
  4. Build 16-element neighborhoods around u and around v
     (each node + its 15 strongest functional partners)
  5. Compute the 16×16 cost matrix:
        C[i, j] = 1 - corr(u_neigh[i], v_neigh[j])
     and the log-domain kernel:
        K[i, j] = -C[i, j] / lambda
  6. Use uniform marginals a = b = (1/16, ..., 1/16) and compute
     la = log2(a), lb = log2(b) for the kernel input layout
  7. Pack the 320-element f32 input (la + lb + K + zeros for u/v output)
  8. Launch /tmp/kaxi_runner with the Sinkhorn-16 PTX
  9. Compute the SciPy reference Sinkhorn output in log domain
  10. Print a compact comparison and exit 0 if within tolerance

Usage:
    python3 scripts/research/abide_sinkhorn_probe.py \\
        [--subject CMU_a_0050642] \\
        [--lambda 1.0] \\
        [--ptx /tmp/sinkhorn16iter.ptx] \\
        [--runner /tmp/kaxi_runner]

This is a one-shot probe — it does NOT scale to all 1035 subjects.
Its purpose is to bridge ABIDE-I → 16-iter Sinkhorn-LSE GPU kernel
and verify the GPU output matches a CPU reference on a real-data
input. Once this bridge holds, the next step is per-edge sweep
across a single subject's connectome, then across the cohort.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ABIDE_DIR = ROOT / "artifacts" / "research" / "abide"
N = 16  # kernel size — matches kaxi_emit_sinkhorn16_asm


def read_subject(subject: str) -> np.ndarray:
    """Read a CC200 .1D ROI time series. Returns (T, 200) float64."""
    path = ABIDE_DIR / f"{subject}_rois_cc200.1D"
    if not path.is_file():
        # Default: first file in the directory.
        candidates = sorted(ABIDE_DIR.glob("*_rois_cc200.1D"))
        if not candidates:
            sys.exit(f"no .1D files found under {ABIDE_DIR}")
        path = candidates[0]
        print(f"  (subject {subject!r} not found, using {path.name})", file=sys.stderr)
    # Skip the column-name header (starts with '#')
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        sys.exit(f"unexpected shape {data.shape} in {path}")
    return data


def build_correlation(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation of (T, R) → (R, R), dropping all-zero ROIs."""
    # ABIDE-I CC200 has some ROIs that are all-zero in some subjects
    # (parcels outside the brain mask). Replace with tiny noise so corrcoef
    # doesn't return NaN.
    std = ts.std(axis=0)
    zero_cols = std < 1e-12
    if zero_cols.any():
        rng = np.random.default_rng(42)
        ts = ts.copy()
        ts[:, zero_cols] = rng.standard_normal((ts.shape[0], int(zero_cols.sum()))) * 1e-3
    # rowvar=False because we have (T, R), want correlation between ROIs
    return np.corrcoef(ts, rowvar=False)


def pick_edge(corr: np.ndarray) -> tuple[int, int]:
    """Pick the off-diagonal edge with highest correlation."""
    R = corr.shape[0]
    work = corr.copy()
    # Zero the diagonal so argmax doesn't return (i, i)
    np.fill_diagonal(work, -np.inf)
    flat_idx = int(np.argmax(work))
    return divmod(flat_idx, R)


def build_neighborhood(corr: np.ndarray, node: int, k: int = N - 1) -> np.ndarray:
    """Top-k functional partners of `node`, plus the node itself.
    Returns 16 integer indices."""
    row = corr[node].copy()
    row[node] = -np.inf
    partners = np.argpartition(row, -k)[-k:]
    # Sort by correlation descending for reproducibility
    partners = partners[np.argsort(-row[partners])]
    return np.concatenate(([node], partners)).astype(np.int64)


def build_K(corr: np.ndarray, u_neigh: np.ndarray, v_neigh: np.ndarray,
            lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C, K) — 16x16 cost and 16x16 log-kernel matrices."""
    C = 1.0 - corr[np.ix_(u_neigh, v_neigh)]
    # Standard Sinkhorn convention: K = exp(-C / lambda)
    # We're in BASE-2 log domain (matches kernel emission), so K_log = -C / (lam * ln 2)
    K_log = -C / (lam * np.log(2.0))
    return C, K_log


def gpu_run(runner: str, ptx: str, input_vec: np.ndarray) -> np.ndarray:
    """Launch the kernel; return mem[288..319] (u_out, v_out — 32 f32)."""
    csv = ",".join(f"{v}" for v in input_vec.astype(np.float32))
    proc = subprocess.run(
        [runner, ptx, "--threads", "1", "--mem-words", "320",
         "--init-mem", csv, "--type", "f32"],
        capture_output=True, text=True, check=True)
    mem_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("MEM:")),
                    None)
    if mem_line is None:
        sys.exit(f"runner produced no MEM: line.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    values = np.array([float(x) for x in mem_line.split()[1:]], dtype=np.float64)
    if values.size != 320:
        sys.exit(f"runner returned {values.size} values, expected 320")
    return values[288:]


def reference_sinkhorn_log(la: np.ndarray, lb: np.ndarray, K_log: np.ndarray,
                           iters: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Reference log-domain Sinkhorn (base-2) matching the GPU kernel.

    u, v = zeros(N), zeros(N)
    for iter in 0..iters:
        u[i] = la[i] - log2(sum_j 2^(K[i,j] + v[j]))
        v[j] = lb[j] - log2(sum_i 2^(K[i,j] + u[i]))
    """
    u = np.zeros(N, dtype=np.float64)
    v = np.zeros(N, dtype=np.float64)
    for _ in range(iters):
        # row update
        for i in range(N):
            M = np.max(K_log[i] + v)
            u[i] = la[i] - (M + np.log2(np.sum(np.exp2(K_log[i] + v - M))))
        # col update
        for j in range(N):
            M = np.max(K_log[:, j] + u)
            v[j] = lb[j] - (M + np.log2(np.sum(np.exp2(K_log[:, j] + u - M))))
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="CMU_a_0050642")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--ptx", default="/tmp/sinkhorn16iter.ptx")
    ap.add_argument("--runner", default="/tmp/kaxi_runner")
    ap.add_argument("--edge", nargs=2, type=int, default=None,
                    help="optional explicit (u, v) edge indices")
    args = ap.parse_args()

    if not os.path.isfile(args.ptx):
        sys.exit(f"PTX not found: {args.ptx}\n"
                 f"  emit with: ./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o {args.ptx}")
    if not os.path.isfile(args.runner):
        sys.exit(f"runner not found: {args.runner}\n"
                 f"  build with: cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o {args.runner}")

    print(f"== ABIDE-I → Sinkhorn-LSE N=16 GPU probe ==")
    print(f"subject:   {args.subject}")
    print(f"lambda:    {args.lam}")
    print()

    ts = read_subject(args.subject)
    print(f"timeseries shape: {ts.shape}    (T timepoints × R ROIs)")

    corr = build_correlation(ts)
    print(f"correlation:      {corr.shape}    range [{corr.min():.3f}, {corr.max():.3f}]")

    if args.edge:
        u, v = args.edge
    else:
        u, v = pick_edge(corr)
    print(f"edge picked:      ({u}, {v})    corr = {corr[u, v]:.4f}")

    u_neigh = build_neighborhood(corr, u)
    v_neigh = build_neighborhood(corr, v)
    print(f"u neighborhood:   {u_neigh.tolist()}")
    print(f"v neighborhood:   {v_neigh.tolist()}")
    print()

    C, K_log = build_K(corr, u_neigh, v_neigh, args.lam)
    print(f"cost matrix C:    shape {C.shape}, range [{C.min():.3f}, {C.max():.3f}]")
    print(f"log-kernel K:     shape {K_log.shape}, range [{K_log.min():.3f}, {K_log.max():.3f}]")
    print()

    # Uniform marginals; la = lb = log2(1/N)
    la = np.full(N, np.log2(1.0 / N), dtype=np.float64)
    lb = np.full(N, np.log2(1.0 / N), dtype=np.float64)

    # Pack input: la (16) + lb (16) + K (256, row-major) + zeros (32)
    input_vec = np.concatenate([la, lb, K_log.flatten(), np.zeros(2 * N)])
    assert input_vec.size == 320

    print(f"launching GPU kernel...")
    gpu_uv = gpu_run(args.runner, args.ptx, input_vec)
    gpu_u = gpu_uv[:N]
    gpu_v = gpu_uv[N:]
    print(f"  GPU u: {gpu_u}")
    print(f"  GPU v: {gpu_v}")
    print()

    print(f"computing CPU reference Sinkhorn (16 iters, log domain base-2)...")
    ref_u, ref_v = reference_sinkhorn_log(la, lb, K_log, iters=16)
    print(f"  CPU u: {ref_u}")
    print(f"  CPU v: {ref_v}")
    print()

    # The GPU uses f32 + .approx ex2/lg2; CPU uses f64. Tolerance ~5e-3.
    diff_u = np.abs(gpu_u - ref_u)
    diff_v = np.abs(gpu_v - ref_v)
    print(f"max |Δu| = {diff_u.max():.6f}    mean |Δu| = {diff_u.mean():.6f}")
    print(f"max |Δv| = {diff_v.max():.6f}    mean |Δv| = {diff_v.mean():.6f}")

    tol = 5e-2  # generous; .approx + chained sub/add/ex2/lg2 over 16 iters accumulates
    if diff_u.max() < tol and diff_v.max() < tol:
        print(f"\nPASS: GPU matches CPU reference within tolerance {tol}")
        sys.exit(0)
    else:
        print(f"\nFAIL: GPU diverges from CPU reference by > {tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
