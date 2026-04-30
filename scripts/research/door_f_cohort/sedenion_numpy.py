#!/usr/bin/env python3
# Python/numpy port of the Sounio sedenion pipeline, for offline control
# analyses. The forward SSM, the associator statistic, and make_a(alpha) are
# reproduced bit-for-bit where possible; bare Cayley-Dickson multiplication
# is implemented via a precomputed 16x16x16 structure-constant tensor so that
# large trajectories can be vectorised with numpy.einsum.
#
# Parity goal: values produced here should agree with the Sounio native
# trajectory TSVs to within a few percent (the Sounio tanh uses a truncated
# Taylor-series exp; here we use math.tanh. The SSM normalisation rule is
# identical.)
from __future__ import annotations

import numpy as np


# ---- Cayley-Dickson sedenion multiplication ---------------------------------

def _oct_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Octonion product; a, b shape (8,) -> (8,). Matches stdlib oct_mul."""
    a0, a1, a2, a3, a4, a5, a6, a7 = a
    b0, b1, b2, b3, b4, b5, b6, b7 = b
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7,
        a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6,
        a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5,
        a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4,
        a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3,
        a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2,
        a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1,
        a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0,
    ])


def sed_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sedenion product via Cayley-Dickson: (p,q)(r,s) = (pr - conj(s)q, sp + q conj(r))."""
    a_lo, a_hi = a[:8], a[8:]
    b_lo, b_hi = b[:8], b[8:]
    conj = np.array([1, -1, -1, -1, -1, -1, -1, -1], dtype=float)
    lo = _oct_mul(a_lo, b_lo) - _oct_mul(b_hi * conj, a_hi)
    hi = _oct_mul(b_hi, a_lo) + _oct_mul(a_hi, b_lo * conj)
    return np.concatenate([lo, hi])


def build_structure_constants() -> np.ndarray:
    """Return C[i,j,k] such that e_i * e_j = sum_k C[i,j,k] e_k, via sed_mul."""
    C = np.zeros((16, 16, 16), dtype=float)
    for i in range(16):
        for j in range(16):
            ei = np.zeros(16); ei[i] = 1.0
            ej = np.zeros(16); ej[j] = 1.0
            C[i, j, :] = sed_mul(ei, ej)
    return C


C_SED = build_structure_constants()


def sed_mul_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised sedenion product using the structure-constant tensor.

    Accepts shape-(16,) and shape-(..., 16) arrays. Return has the broadcast
    shape. For the time-series case, pass (T,16) and the result is (T,16).
    """
    return np.einsum("ijk,...i,...j->...k", C_SED, a, b)


def sed_norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(a * a, axis=-1))


# ---- Sounio-parity tanh (truncated-Taylor exp, 20 terms) -------------------

def _exp_s_scalar(x: float) -> float:
    """Matches header.sio.part::exp_s: 20-term Taylor series, with splitting
    for |x|>20 to avoid overflow."""
    if x > 20.0:
        return _exp_s_scalar(10.0) * _exp_s_scalar(x - 10.0)
    if x < -20.0:
        return 0.0
    s = 1.0
    term = 1.0
    for i in range(1, 21):
        term = term * x / i
        s += term
    return s


def _tanh_s_scalar(x: float) -> float:
    if x > 10.0:
        return 1.0
    if x < -10.0:
        return -1.0
    e2 = _exp_s_scalar(2.0 * x)
    return (e2 - 1.0) / (e2 + 1.0)


_TANH_S_VEC = np.vectorize(_tanh_s_scalar, otypes=[float])


def sed_tanh(v: np.ndarray) -> np.ndarray:
    """Component-wise tanh using the Sounio 20-term Taylor approximation."""
    return _TANH_S_VEC(v)


# ---- make_a(alpha) matching header.sio.part --------------------------------

def make_a(alpha: float) -> np.ndarray:
    """Lerp-to-zero-divisor gate, as in header.sio.part::make_a."""
    inv2 = 1.0 / np.sqrt(2.0)
    zd0, zd1, zd4, zd5 = 0.5, 0.5, 0.5, -0.5
    g0 = (1 - alpha) * inv2 + alpha * zd0
    g1 = (1 - alpha) * inv2 + alpha * zd1
    g4 = alpha * zd4
    g5 = alpha * zd5
    n = np.sqrt(g0 * g0 + g1 * g1 + g4 * g4 + g5 * g5)
    s = 1.0 / n if n > 1e-7 else 1.0
    a = np.zeros(16, dtype=float)
    a[0] = g0 * s
    a[1] = g1 * s
    a[4] = g4 * s
    a[5] = g5 * s
    return a


# ---- SSM forward, training on II, and associator statistic -----------------

def ssm_forward(a: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Evolve h_t = normalize(tanh(sed_mul(a, h_{t-1}) + x_t)) for t=0..T-1.

    X shape (T, 16). Returns H shape (T, 16) (post-update states).
    """
    T = X.shape[0]
    H = np.empty_like(X)
    h = np.zeros(16); h[0] = 1.0  # sed_one
    for t in range(T):
        ah = sed_mul_vec(a, h)
        raw = ah + X[t]
        act = sed_tanh(raw)
        n = sed_norm(act)
        h = act / n if n > 1e-7 else act
        H[t] = h
    return H


def train_c(a: np.ndarray, X_ii: np.ndarray, Y_ii: np.ndarray,
            n_epochs: int = 200, lr: float = 0.2) -> np.ndarray:
    """200 steps of gradient descent on c to minimise mean((h_t·c - y_t)^2).

    X_ii shape (64, 16), Y_ii shape (64,).  Matches ssm_update_c semantics.
    """
    c = np.zeros(16)
    for _ in range(n_epochs):
        H = ssm_forward(a, X_ii)             # (64, 16)
        err = H @ c - Y_ii                   # (64,)
        grad = (H.T @ err) * (2.0 / X_ii.shape[0])
        c = c - lr * grad
    return c


def assoc_trajectory(a: np.ndarray, X: np.ndarray) -> float:
    """mean_t ||(a*h_t)*x_t - a*(h_t*x_t)||, with h evolving under the SSM.

    X shape (T, 16). The hidden state h is updated AFTER the associator is
    accumulated for sample t (matching template.sio.part::assoc_*).
    """
    T = X.shape[0]
    h = np.zeros(16); h[0] = 1.0
    total = 0.0
    for t in range(T):
        xt = X[t]
        ah = sed_mul_vec(a, h)
        ahx = sed_mul_vec(ah, xt)
        hx = sed_mul_vec(h, xt)
        a_hx = sed_mul_vec(a, hx)
        diff = ahx - a_hx
        total += float(sed_norm(diff))
        raw = ah + xt
        act = sed_tanh(raw)
        n = float(sed_norm(act))
        h = act / n if n > 1e-7 else act
    return total / T


# ---- Permuted-basis associator: reorder channels before lifting ------------

def assoc_trajectory_permuted(a: np.ndarray, X: np.ndarray,
                              perm: np.ndarray) -> float:
    """Apply channel permutation `perm` (length 16) before lifting into the
    sedenion basis. All other dynamics remain identical.
    """
    Xp = X[:, perm]
    return assoc_trajectory(a, Xp)


# ---- Control biomarkers: non-sedenion trilinear forms ----------------------

def random_trilinear_trajectory(a: np.ndarray, X: np.ndarray,
                                T: np.ndarray) -> float:
    """Replace sed_mul with a fixed random trilinear tensor T of shape
    (16,16,16). The SSM still evolves via sed_tanh + normalisation but the
    associator statistic uses T instead of the sedenion product. Analogous to
    mean_t ||T(T(a,h_t),x_t) - T(a, T(h_t,x_t))||.
    """
    h = np.zeros(16); h[0] = 1.0
    total = 0.0
    Tsteps = X.shape[0]
    for t in range(Tsteps):
        xt = X[t]
        ah   = np.einsum("ijk,i,j->k", T, a, h)
        ahx  = np.einsum("ijk,i,j->k", T, ah, xt)
        hx   = np.einsum("ijk,i,j->k", T, h, xt)
        a_hx = np.einsum("ijk,i,j->k", T, a, hx)
        total += float(np.linalg.norm(ahx - a_hx))
        # advance state under sedenion dynamics to keep h comparable
        raw = sed_mul_vec(a, h) + xt
        act = sed_tanh(raw)
        n = float(sed_norm(act))
        h = act / n if n > 1e-7 else act
    return total / Tsteps


def bispectrum_norm(X: np.ndarray) -> float:
    """Third-order channel-coupling proxy: ||<x_t \\otimes x_t \\otimes x_t>_t||_F
    where x_t is the 16-dim sample at time t. SSM-free, algebra-free control.
    """
    T = np.einsum("ti,tj,tk->ijk", X, X, X) / X.shape[0]
    return float(np.sqrt(np.sum(T * T)))

