#!/usr/bin/env python3
"""
Sub-PTX B2: per-subject all-edge ORC sweep using the Sounio Sinkhorn-LSE
N=16 GPU kernel.

Pipeline:
  1. Read CC200 ROI time series for one ABIDE-I subject
  2. Compute 200×200 Pearson correlation
  3. Build a directed k-NN connectome: for each node, its top-k functional
     partners → 200 × k = 3000 edges (default k=15)
  4. For each edge (u, v):
       neighborhoods N_u, N_v (each 16 nodes: the endpoint + its top-15
       functional partners; built once per node, reused across edges)
       cost matrix C[i, j] = 1 - corr(N_u[i], N_v[j])
       log-kernel K = -C / (lambda * ln 2)
       marginals la = lb = log2(1/16) * 1
       pack 320-element f32 row
  5. Concatenate all 3000 rows into a single 960k-element f32 buffer
  6. Write buffer to a temp file, launch the kernel ONCE with --threads =
     N_edges and --init-file <path>
  7. Read back the N_edges × 32 output slab (u_out, v_out per edge)
  8. Derive the transport plan P[i, j] = ex2(u_out[i] + K[i, j] + v_out[j])
  9. Compute Wasserstein-1 distance W_uv = sum_{i,j} P[i, j] * C[i, j]
  10. ORC for each k-NN edge (u, v) with direct distance d(u, v) = 1:
        kappa(u, v) = 1 - W_uv
  11. Spot-check N_check sample edges by re-running them through the
      NumPy reference Sinkhorn and comparing the GPU ORC to the CPU ORC

Output:
  - artifacts/research/abide_orc/<subject>_orc.npy   (N × N, ORC matrix)
  - prints summary statistics + spot-check pass/fail

Usage:
    python3 scripts/research/abide_orc_sweep.py \\
        [--subject CMU_a_0050642] \\
        [--k 15] \\
        [--lambda 1.0] \\
        [--ptx /tmp/sinkhorn16iter.ptx] \\
        [--runner /tmp/kaxi_runner] \\
        [--check 5]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ABIDE_DIR = ROOT / "artifacts" / "research" / "abide"
OUT_DIR = ROOT / "artifacts" / "research" / "abide_orc"
N = 16   # kernel size — matches kaxi_emit_sinkhorn16_asm
EDGE_SLOTS = 320  # f32 per edge (la 16 + lb 16 + K 256 + zeros 32)


def read_subject(subject: str) -> np.ndarray:
    path = ABIDE_DIR / f"{subject}_rois_cc200.1D"
    if not path.is_file():
        candidates = sorted(ABIDE_DIR.glob("*_rois_cc200.1D"))
        if not candidates:
            sys.exit(f"no .1D files under {ABIDE_DIR}")
        path = candidates[0]
        print(f"  (subject {subject!r} not found, using {path.name})", file=sys.stderr)
    return np.loadtxt(path, comments="#")


def build_correlation(ts: np.ndarray) -> np.ndarray:
    std = ts.std(axis=0)
    zero_cols = std < 1e-12
    if zero_cols.any():
        rng = np.random.default_rng(42)
        ts = ts.copy()
        ts[:, zero_cols] = rng.standard_normal((ts.shape[0], int(zero_cols.sum()))) * 1e-3
    return np.corrcoef(ts, rowvar=False)


def build_neighborhood(corr: np.ndarray, node: int, k: int = N - 1) -> np.ndarray:
    """Top-k functional partners of `node`, plus the node itself. Returns 16 ints."""
    row = corr[node].copy()
    row[node] = -np.inf
    partners = np.argpartition(row, -k)[-k:]
    partners = partners[np.argsort(-row[partners])]
    return np.concatenate(([node], partners)).astype(np.int64)


def reference_sinkhorn_log(la: np.ndarray, lb: np.ndarray, K_log: np.ndarray,
                           iters: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """NumPy log-domain Sinkhorn (base-2), matches kernel exactly."""
    u = np.zeros(N, dtype=np.float64)
    v = np.zeros(N, dtype=np.float64)
    for _ in range(iters):
        for i in range(N):
            M = np.max(K_log[i] + v)
            u[i] = la[i] - (M + np.log2(np.sum(np.exp2(K_log[i] + v - M))))
        for j in range(N):
            M = np.max(K_log[:, j] + u)
            v[j] = lb[j] - (M + np.log2(np.sum(np.exp2(K_log[:, j] + u - M))))
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="CMU_a_0050642")
    ap.add_argument("--k", type=int, default=15, help="k for directed k-NN connectome")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--ptx", default="/tmp/sinkhorn16iter.ptx")
    ap.add_argument("--runner", default="/tmp/kaxi_runner")
    ap.add_argument("--check", type=int, default=5,
                    help="how many edges to spot-check against NumPy reference")
    args = ap.parse_args()

    if not os.path.isfile(args.ptx):
        sys.exit(f"PTX not found: {args.ptx}")
    if not os.path.isfile(args.runner):
        sys.exit(f"runner not found: {args.runner}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== ABIDE-I → per-subject ORC sweep ==")
    print(f"subject:   {args.subject}")
    print(f"k (k-NN):  {args.k}")
    print(f"lambda:    {args.lam}")
    print()

    t0 = time.time()
    ts = read_subject(args.subject)
    R = ts.shape[1]
    print(f"timeseries shape: {ts.shape}    (T × R ROIs)")
    corr = build_correlation(ts)
    print(f"correlation range: [{corr.min():.3f}, {corr.max():.3f}]")

    # Build k-NN graph: for each node, top-k partners (excluding self)
    print(f"building directed k-NN graph (k={args.k})...")
    neighbors = np.zeros((R, args.k), dtype=np.int64)
    for u in range(R):
        row = corr[u].copy()
        row[u] = -np.inf
        top = np.argpartition(row, -args.k)[-args.k:]
        top = top[np.argsort(-row[top])]
        neighbors[u] = top
    n_edges = R * args.k
    print(f"  {n_edges} directed edges total")

    # Build 16-element neighborhoods (one per node, reused across edges)
    print(f"building per-node 16-element neighborhoods...")
    neigh = np.zeros((R, N), dtype=np.int64)
    for u in range(R):
        neigh[u] = build_neighborhood(corr, u, k=N - 1)

    # Precompute la, lb (same for all edges with uniform marginals)
    la = np.full(N, np.log2(1.0 / N), dtype=np.float32)
    lb = la.copy()

    # Construct batch input: (n_edges, 320) f32
    print(f"building batch input ({n_edges} edges × {EDGE_SLOTS} f32)...")
    inv_lam_ln2 = 1.0 / (args.lam * np.log(2.0))
    batch_input = np.zeros((n_edges, EDGE_SLOTS), dtype=np.float32)
    cost_store = np.zeros((n_edges, N, N), dtype=np.float64)  # for ORC post-process
    edge_index = np.zeros((n_edges, 2), dtype=np.int64)  # (u, v) per edge

    eid = 0
    for u in range(R):
        nu = neigh[u]
        for vi in range(args.k):
            v = neighbors[u, vi]
            nv = neigh[v]
            C = 1.0 - corr[np.ix_(nu, nv)]  # 16x16
            K_log = (-C * inv_lam_ln2).astype(np.float32)
            batch_input[eid, 0:N] = la
            batch_input[eid, N:2 * N] = lb
            batch_input[eid, 2 * N:2 * N + N * N] = K_log.flatten()
            # last 32 slots stay zero (output area)
            cost_store[eid] = C
            edge_index[eid] = (u, v)
            eid += 1
    assert eid == n_edges

    print(f"  batch input size: {batch_input.nbytes / 1024 / 1024:.2f} MB")
    print(f"setup time: {time.time() - t0:.1f}s")
    print()

    # The Sinkhorn-16 kernel uses `%tid.x` (per-block) for its base offset
    # and consumes ~255 registers per thread (ptxas info: "Used 255
    # registers").  At sm_89, max regs per block = 65536, so each block
    # is capped at 65536 / 256 ≈ 256 threads.  And multi-block launches
    # don't work without a global-tid refactor of the kernel.  So we
    # split the n_edges work into chunks of ≤ CHUNK threads, one launch
    # per chunk.
    CHUNK = 256
    n_chunks = (n_edges + CHUNK - 1) // CHUNK
    print(f"launching GPU kernel in {n_chunks} chunks of ≤ {CHUNK} threads...")
    values = np.zeros(n_edges * EDGE_SLOTS, dtype=np.float64)
    t1 = time.time()
    for c in range(n_chunks):
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, n_edges)
        n = hi - lo
        # Pad to CHUNK threads so launches are uniform (extra threads
        # process zero-filled inputs and we discard their outputs).
        chunk_buf = np.zeros((CHUNK, EDGE_SLOTS), dtype=np.float32)
        chunk_buf[:n] = batch_input[lo:hi]
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(chunk_buf.tobytes())
            bin_path = f.name
        try:
            proc = subprocess.run(
                [args.runner, args.ptx,
                 "--threads", str(CHUNK),
                 "--mem-words", str(CHUNK * EDGE_SLOTS),
                 "--init-file", bin_path,
                 "--type", "f32"],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                sys.exit(f"runner failed at chunk {c} rc={proc.returncode}")
            mem_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("MEM:")), None)
            if mem_line is None:
                sys.exit(f"no MEM: line at chunk {c}.\n{proc.stdout}")
            chunk_vals = np.array([float(x) for x in mem_line.split()[1:]], dtype=np.float64)
            expected = CHUNK * EDGE_SLOTS
            if chunk_vals.size != expected:
                sys.exit(f"chunk {c}: got {chunk_vals.size} values, expected {expected}")
            # Copy only the first n*EDGE_SLOTS entries (real edges) into the
            # full-cohort accumulator.  The remaining (CHUNK-n) entries are
            # discarded padding.
            values[lo * EDGE_SLOTS:hi * EDGE_SLOTS] = chunk_vals[:n * EDGE_SLOTS]
        finally:
            os.unlink(bin_path)
    gpu_time = time.time() - t1
    print(f"  total kernel runtime: {gpu_time:.1f}s ({gpu_time / n_edges * 1000:.2f} ms/edge avg)")
    print()

    # Reshape: per-edge u_out is at offset eid*320 + 288, 16 values; v_out at +304
    values = values.reshape(n_edges, EDGE_SLOTS)
    u_out = values[:, 288:304]  # (n_edges, 16)
    v_out = values[:, 304:320]  # (n_edges, 16)

    # Compute transport plans + Wasserstein distances + ORC
    print(f"computing transport plans + ORC...")
    t2 = time.time()
    # P[e, i, j] = ex2(u_out[e, i] + K_log[e, i, j] + v_out[e, j])
    # We need K_log: recompute (cheap) or store. We stored cost_store; recompute K_log from it.
    K_log_all = (-cost_store * inv_lam_ln2).astype(np.float64)  # (n_edges, 16, 16)
    # Outer-add: u_out + K + v_out
    arg = u_out[:, :, None] + K_log_all + v_out[:, None, :]   # (n_edges, 16, 16)
    P = np.exp2(arg)
    # Wasserstein-1: W = sum P * C
    W = (P * cost_store).sum(axis=(1, 2))                     # (n_edges,)
    # ORC for direct k-NN edges (d=1): kappa = 1 - W
    kappa = 1.0 - W
    print(f"  ORC compute: {time.time() - t2:.1f}s")
    print()

    # Build per-subject ORC matrix (asymmetric; entry [u,v] is the directed
    # ORC from u → v).  Indexed by ROI ids.
    orc_mat = np.full((R, R), np.nan, dtype=np.float64)
    for e in range(n_edges):
        u, v = edge_index[e]
        orc_mat[u, v] = kappa[e]

    # Stats
    print(f"ORC statistics (n_edges = {n_edges}):")
    print(f"  range:        [{kappa.min():.4f}, {kappa.max():.4f}]")
    print(f"  mean ± std:   {kappa.mean():.4f} ± {kappa.std():.4f}")
    print(f"  median:       {np.median(kappa):.4f}")
    # Fraction positive (positively-curved edges) vs negative
    n_pos = int((kappa > 0).sum())
    n_neg = int((kappa < 0).sum())
    print(f"  positive:     {n_pos} ({100.0 * n_pos / n_edges:.1f}%)")
    print(f"  negative:     {n_neg} ({100.0 * n_neg / n_edges:.1f}%)")
    print()

    # Spot-check against NumPy reference
    if args.check > 0:
        print(f"spot-check: {args.check} random edges against NumPy reference...")
        rng = np.random.default_rng(0)
        sample = rng.choice(n_edges, size=min(args.check, n_edges), replace=False)
        check_pass = 0
        for e in sample:
            u, v = edge_index[e]
            K_log = K_log_all[e]
            ref_u, ref_v = reference_sinkhorn_log(la.astype(np.float64),
                                                  lb.astype(np.float64),
                                                  K_log, iters=16)
            ref_arg = ref_u[:, None] + K_log + ref_v[None, :]
            ref_P = np.exp2(ref_arg)
            ref_W = float((ref_P * cost_store[e]).sum())
            ref_kappa = 1.0 - ref_W
            d_kappa = abs(kappa[e] - ref_kappa)
            ok = d_kappa < 5e-3
            check_pass += int(ok)
            mark = "OK " if ok else "FAIL"
            print(f"  [{mark}] edge ({u:3d},{v:3d}) corr={corr[u, v]:.3f}  "
                  f"gpu_kappa={kappa[e]:.6f}  cpu_kappa={ref_kappa:.6f}  "
                  f"|Δ|={d_kappa:.2e}")
        print(f"spot-check: {check_pass}/{len(sample)} PASS (tol 5e-3)")
        print()
        if check_pass != len(sample):
            sys.exit(1)

    # Save artefact
    out_npy = OUT_DIR / f"{args.subject}_orc.npy"
    np.save(out_npy, orc_mat)
    print(f"saved: {out_npy}")
    print(f"total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
