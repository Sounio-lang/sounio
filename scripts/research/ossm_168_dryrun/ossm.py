"""Untrained O-SSM forward pass matching prereg v3 §6.

Architecture (pinned, no training):
- d_input  = 7 (seven regional-averaged EEG channels)
- d_state  = 2 (two octonion units = 16 real values)
- d_output = 1 (not used for classification; features read out from h_t)
- State update: h_t = sigma(A (x) h_{t-1} + B (x) x_t), component-wise sigmoid
- Per-subject deterministic seed: 20260421 + subject_index
- Parameter init: N(0, sigma_0^2) with sigma_0 = 0.1

This module is a Python mirror of the behaviour specified in
`.claude/prompts/door3_octonion_ssm.md` and realised in
`examples/octonion_ssm.sio`. A Sounio-native port is a follow-up task.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import octonion as oct_mod


STATE_DIM_OCTONIONS = 2  # d_state
INPUT_DIM = 7            # d_input (7 regional channels)
SIGMA_INIT = 0.1
SEED_BASE = 20260421


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def _sigmoid_oct(o: oct_mod.Oct) -> oct_mod.Oct:
    out = np.empty(8, dtype=np.float64)
    for i in range(8):
        out[i] = _sigmoid(float(o[i]))
    return out


def initialise_parameters(subject_index: int) -> tuple[list[oct_mod.Oct], list[list[oct_mod.Oct]]]:
    """Return (A_diag, B_matrix) for a subject.

    A_diag is a list of STATE_DIM_OCTONIONS octonions (diagonal entries).
    B_matrix is STATE_DIM_OCTONIONS x INPUT_DIM octonions.
    All entries drawn i.i.d. N(0, SIGMA_INIT^2) with seed SEED_BASE + subject_index.
    """
    rng = np.random.default_rng(SEED_BASE + subject_index)
    a_diag = [rng.normal(0.0, SIGMA_INIT, size=8).astype(np.float64)
              for _ in range(STATE_DIM_OCTONIONS)]
    b_mat = [
        [rng.normal(0.0, SIGMA_INIT, size=8).astype(np.float64)
         for _ in range(INPUT_DIM)]
        for _ in range(STATE_DIM_OCTONIONS)
    ]
    return a_diag, b_mat


def _input_to_octonion(x7: NDArray[np.float64]) -> oct_mod.Oct:
    """Embed a length-7 real vector into an octonion as (0, x_1, ..., x_7)."""
    out = oct_mod.oct_zero()
    out[1:8] = x7
    return out


def forward_pass(
    signal_7xt: NDArray[np.float64],
    subject_index: int,
    h0: list[oct_mod.Oct] | None = None,
) -> NDArray[np.float64]:
    """Run the untrained O-SSM on a (7, T) signal for one epoch.

    Returns a (T, STATE_DIM_OCTONIONS, 8) array of state trajectories.
    The i-th slot holds h_t[i] as an octonion (length-8 float64).
    """
    if signal_7xt.ndim != 2 or signal_7xt.shape[0] != INPUT_DIM:
        raise ValueError(
            f"signal must have shape (7, T); got {signal_7xt.shape}"
        )
    T = signal_7xt.shape[1]
    a_diag, b_mat = initialise_parameters(subject_index)

    if h0 is None:
        h = [oct_mod.oct_zero() for _ in range(STATE_DIM_OCTONIONS)]
    else:
        h = [hi.copy() for hi in h0]

    traj = np.empty((T, STATE_DIM_OCTONIONS, 8), dtype=np.float64)

    for t in range(T):
        x_oct = _input_to_octonion(signal_7xt[:, t])
        new_h: list[oct_mod.Oct] = []
        for s in range(STATE_DIM_OCTONIONS):
            # A_s * h_{t-1, s} (diagonal state transition)
            ah = oct_mod.oct_mul(a_diag[s], h[s])
            # B_s @ x_t  =  sum over input dim k of  B[s, k] * e_{k+1}-component of x
            bx = oct_mod.oct_zero()
            for k in range(INPUT_DIM):
                contribution = oct_mod.oct_mul(
                    b_mat[s][k],
                    oct_mod.oct_basis(k + 1) * float(signal_7xt[k, t]),
                )
                bx = bx + contribution
            pre = ah + bx
            new_h.append(_sigmoid_oct(pre))
        h = new_h
        for s in range(STATE_DIM_OCTONIONS):
            traj[t, s] = h[s]

    return traj


def permuted_forward_pass(
    signal_7xt: NDArray[np.float64],
    subject_index: int,
    sigma_perm: tuple[int, ...],
) -> NDArray[np.float64]:
    """Run forward_pass after applying the 1-indexed S_7 permutation sigma_perm
    to the input channel ordering.

    sigma_perm is a length-7 tuple (a permutation of 1..7), interpreted as:
    the input previously at region r (1-indexed) is now placed at channel
    sigma_perm(r). Equivalently the input vector y satisfies
    y[i] = signal[sigma_perm[i] - 1].

    This is the primitive used by the §7.1 labelling-invariance control.
    """
    T = signal_7xt.shape[1]
    permuted = np.empty_like(signal_7xt)
    for i in range(INPUT_DIM):
        permuted[i, :] = signal_7xt[sigma_perm[i] - 1, :]
    return forward_pass(permuted, subject_index)
