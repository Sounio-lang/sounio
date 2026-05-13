#!/usr/bin/env python3
"""GPU-accelerated feature extraction for OSSM-168.

Uses PyTorch for CUDA acceleration of the O-SSM forward pass
and F3 feature computation (168 sedenion associator triples).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[gpu_features] Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"[gpu_features] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[gpu_features] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


@lru_cache(maxsize=1)
def _mul_tensor_torch() -> torch.Tensor:
    """Pre-compute 16x16x16 sedenion multiplication tensor on GPU."""
    from . import sedenion as sed_mod

    M = np.zeros((16, 16, 16), dtype=np.float64)
    basis = [sed_mod.sed_basis(i) for i in range(16)]
    for i in range(16):
        for j in range(16):
            M[i, j, :] = sed_mod.sed_mul(basis[i], basis[j])
    return torch.from_numpy(M).to(DEVICE)


@lru_cache(maxsize=1)
def _triples_arr_torch() -> torch.Tensor:
    """168 sedenion associator triples on GPU."""
    from . import sedenion as sed_mod

    triples = np.asarray(sed_mod.O_168_sedenion_triples(), dtype=np.int64)
    return torch.from_numpy(triples).to(DEVICE)


def feature_f3_gpu(traj: np.ndarray) -> float:
    """GPU-accelerated F3 computation.

    Args:
        traj: (T, 2, 8) numpy array - O-SSM trajectory

    Returns:
        F3 scalar value
    """
    T = traj.shape[0]
    if traj.shape[1:] != (2, 8):
        raise ValueError(f"expected traj shape (T,2,8); got {traj.shape}")

    # Convert to torch and move to GPU
    s = torch.zeros((T, 16), dtype=torch.float64, device=DEVICE)
    s[:, :8] = torch.from_numpy(traj[:, 0, :]).to(DEVICE)
    s[:, 8:] = torch.from_numpy(traj[:, 1, :]).to(DEVICE)

    M = _mul_tensor_torch()
    triples = _triples_arr_torch()

    # s_products[t, I, k] = s_t * e_I at component k
    # Using einsum: s[t,j] * M[j,I,k] -> s_products[t,I,k]
    s_products = torch.einsum("tj,jIk->tIk", s, M)

    # Gather for 168 triples
    # a, b, c each: (T, 168, 16)
    a = s_products[:, triples[:, 0], :]
    b = s_products[:, triples[:, 1], :]
    c = s_products[:, triples[:, 2], :]

    # Batched sedenion multiplications
    # ab[t,n,k] = a[t,n,i] * b[t,n,j] * M[i,j,k]
    ab = torch.einsum("tni,tnj,ijk->tnk", a, b, M)
    ab_c = torch.einsum("tni,tnj,ijk->tnk", ab, c, M)

    bc = torch.einsum("tni,tnj,ijk->tnk", b, c, M)
    a_bc = torch.einsum("tni,tnj,ijk->tnk", a, bc, M)

    # Associator
    assoc = ab_c - a_bc  # (T, 168, 16)

    # L1 norm, sum over all dims, divide by (T * 168)
    result = torch.abs(assoc).sum() / (T * 168.0)

    return float(result.cpu().numpy())


def batch_feature_f3_gpu(trajectories: list[np.ndarray]) -> list[float]:
    """Batch process multiple trajectories on GPU.

    Args:
        trajectories: List of (T, 2, 8) arrays

    Returns:
        List of F3 values
    """
    results = []
    for traj in trajectories:
        results.append(feature_f3_gpu(traj))
    return results


def forward_pass_gpu(
    signal_7xt: np.ndarray,
    subject_index: int,
    sigma_init: float = 0.1,
    seed_base: int = 20260421,
) -> np.ndarray:
    """GPU-accelerated O-SSM forward pass.

    Args:
        signal_7xt: (7, T) input signal
        subject_index: for deterministic parameter init

    Returns:
        trajectory: (T, 2, 8) state tensor
    """
    from . import octonion as oct_mod

    T = signal_7xt.shape[1]

    # Initialize parameters on CPU (small), then move to GPU
    rng = np.random.default_rng(seed_base + subject_index)

    # A_diag: 2 octonions
    a_diag = torch.stack([
        torch.from_numpy(rng.normal(0.0, sigma_init, 8).astype(np.float64))
        for _ in range(2)
    ]).to(DEVICE)

    # B_matrix: 2x7 octonions
    b_mat = torch.stack([
        torch.stack([
            torch.from_numpy(rng.normal(0.0, sigma_init, 8).astype(np.float64))
            for _ in range(7)
        ])
        for _ in range(2)
    ]).to(DEVICE)

    # Input signal to GPU
    x = torch.from_numpy(signal_7xt.T).to(DEVICE)  # (T, 7)

    # Initial state
    h = [torch.zeros(8, dtype=torch.float64, device=DEVICE) for _ in range(2)]

    # Trajectory storage
    traj = torch.zeros((T, 2, 8), dtype=torch.float64, device=DEVICE)

    for t in range(T):
        # Embed input: (0, x_t)
        x_oct = torch.zeros(8, dtype=torch.float64, device=DEVICE)
        x_oct[1:8] = x[t]

        for s in range(2):
            # A_s * h_{t-1, s}
            ah = oct_mod.oct_mul(a_diag[s].cpu().numpy(), h[s].cpu().numpy())
            ah = torch.from_numpy(ah).to(DEVICE)

            # B_s @ x_t
            bx = torch.zeros(8, dtype=torch.float64, device=DEVICE)
            for k in range(7):
                b_k = b_mat[s, k]
                x_k = x[t, k] if k < len(x[t]) else 0.0
                # Contribution: B[s,k] * x_k (scalar multiply octonion)
                contribution = b_k * x_k
                bx += contribution

            # Sum and sigmoid
            new_h_val = ah + bx
            h[s] = torch.sigmoid(new_h_val)
            traj[t, s, :] = h[s]

    return traj.cpu().numpy()


def process_epoch_gpu(epoch: np.ndarray, subject_index: int) -> tuple[float, float, float]:
    """Process one epoch on GPU, return (F1, F2, F3).

    Args:
        epoch: (7, 1000) array
        subject_index: for deterministic init

    Returns:
        (F1, F2, F3) tuple
    """
    from . import features as feat_mod

    # Forward pass (can be GPU-accelerated)
    traj = forward_pass_gpu(epoch, subject_index)

    # F1 and F2 (CPU-based, fast enough)
    f1 = feat_mod.feature_f1_from_trajectory(traj)
    f2 = feat_mod.feature_f2_from_trajectory(traj)

    # F3 (GPU-accelerated)
    f3 = feature_f3_gpu(traj)

    return f1, f2, f3


def process_subject_gpu(
    epochs: np.ndarray,
    subject_index: int,
    max_epochs: int | None = 50,
    subsample_seed: int = 20260421,
) -> tuple[float, float, float]:
    """Process all epochs of one subject on GPU.

    Args:
        epochs: (n_epochs, 7, 1000) array
        subject_index: for deterministic init
        max_epochs: subsample if more than this

    Returns:
        (F1_median, F2_median, F3_median)
    """
    import statistics

    n = epochs.shape[0]
    if max_epochs is not None and n > max_epochs:
        rng = np.random.default_rng(subsample_seed + subject_index)
        idx = np.sort(rng.choice(n, size=max_epochs, replace=False))
        epochs = epochs[idx]

    f1s, f2s, f3s = [], [], []

    for epoch in epochs:
        f1, f2, f3 = process_epoch_gpu(epoch, subject_index)
        f1s.append(f1)
        f2s.append(f2)
        f3s.append(f3)

    return statistics.median(f1s), statistics.median(f2s), statistics.median(f3s)


if __name__ == "__main__":
    # Test GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
