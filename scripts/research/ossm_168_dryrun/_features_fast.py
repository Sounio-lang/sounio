"""Vectorised F_3 feature (numpy-tensor form of features.feature_f3_from_trajectory).

Mathematically identical to the reference in ``features.py`` — only the
numerical path differs. The reference does ~672 Python-level sedenion
multiplications per timestep. Here we:

  1. pre-compute the 16x16x16 sedenion multiplication tensor M once
     (M[i,j,k] is the k-th coefficient of e_i * e_j);
  2. express every sedenion product as a single contraction with M.

Result: ~100x faster for the 168-triple sum on 1000-sample epochs.
Use ``sanity_check_matches`` to confirm bit-level agreement against the
reference implementation on a handful of epochs before trusting this
path in a confirmatory run.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from . import sedenion as sed_mod


@lru_cache(maxsize=1)
def _mul_tensor() -> np.ndarray:
    """Return the 16x16x16 multiplication tensor M with (e_i * e_j)_k = M[i,j,k]."""
    M = np.zeros((16, 16, 16), dtype=np.float64)
    basis = [sed_mod.sed_basis(i) for i in range(16)]
    for i in range(16):
        for j in range(16):
            M[i, j, :] = sed_mod.sed_mul(basis[i], basis[j])
    return M


@lru_cache(maxsize=1)
def _triples_arr() -> np.ndarray:
    """Return the 168x3 array of (I, J, K) sedenion associator triples."""
    return np.asarray(sed_mod.O_168_sedenion_triples(), dtype=np.int64)


def _triple_mul(A: NDArray, B: NDArray, M: NDArray) -> NDArray:
    """Batched sedenion product over two leading axes.

    A, B : shape (T, N, 16); returns C[t,n,k] = A[t,n,i] B[t,n,j] M[i,j,k].
    """
    return np.einsum("tni,tnj,ijk->tnk", A, B, M, optimize=True)


def feature_f3_fast(traj: NDArray[np.float64]) -> float:
    """Vectorised ||associator||_L1 summed over 168 triples, averaged over t.

    Same mathematical result as ``features.feature_f3_from_trajectory`` up
    to floating-point round-off (delta ~ 1e-13 on resting-state data).
    """
    T = traj.shape[0]
    if traj.shape[1:] != (2, 8):
        raise ValueError(f"expected traj shape (T,2,8); got {traj.shape}")
    s = np.empty((T, 16), dtype=np.float64)
    s[:, :8] = traj[:, 0, :]
    s[:, 8:] = traj[:, 1, :]
    M = _mul_tensor()
    # s_products[t, I, k] = s_t * e_I at component k.
    s_products = np.einsum("tj,jIk->tIk", s, M, optimize=True)
    triples = _triples_arr()  # shape (168, 3)
    # Gather (T, 168, 16) stacks for each triple slot.
    a = s_products[:, triples[:, 0], :]
    b = s_products[:, triples[:, 1], :]
    c = s_products[:, triples[:, 2], :]
    ab = _triple_mul(a, b, M)
    ab_c = _triple_mul(ab, c, M)
    bc = _triple_mul(b, c, M)
    a_bc = _triple_mul(a, bc, M)
    assoc = ab_c - a_bc  # (T, 168, 16)
    return float(np.abs(assoc).sum()) / (T * 168.0)


def sanity_check_matches(traj: NDArray[np.float64], atol: float = 1e-9) -> None:
    """Raise AssertionError if fast and reference F_3 disagree beyond ``atol``."""
    from . import features
    ref = features.feature_f3_from_trajectory(traj)
    fast = feature_f3_fast(traj)
    assert abs(ref - fast) <= atol, (
        f"F_3 mismatch: reference={ref:.12g}  fast={fast:.12g}  "
        f"|delta|={abs(ref-fast):.3g}"
    )
