"""Features F_1, F_2, F_3 per prereg v3.

F_1 (rumination substrate, H1):
    F_1 = (1 / (T * 168)) * sum_t sum_{(i,j,k) in O_168}
            || [ h_t[0][i] * e_i, h_t[0][j] * e_j, h_t[0][k] * e_k ] ||_L2,
    where h_t[0] is the first octonion of the state and the outer norm
    is the L2 octonion norm.

F_2 (anhedonia substrate, H2):
    F_2 = (1/T) * sum_t min_{(z, w) in Z_canonical}
            ( || tilde_h_t * z - 0 ||^2 + || tilde_h_t * w - 0 ||^2 )
            / ( || tilde_h_t ||^2 + eps ).
    Smaller = closer to the zero-divisor variety.

F_3 (valence substrate, H3) per DEVIATIONS 0004 -> v3:
    F_3 = (1 / (T * 168)) * sum_t sum_{(I,J,K) in tilde_O_168}
            || [ tilde_h_t * e_I, tilde_h_t * e_J, tilde_h_t * e_K ] ||_L1.
    L1 is the componentwise sum of absolute values over the 16 sedenion
    basis components.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import fano_orbits
from . import octonion as oct_mod
from . import sedenion as sed_mod


_F2_EPS = 1e-12


def feature_f1_from_trajectory(traj: NDArray[np.float64]) -> float:
    """traj shape: (T, STATE_DIM_OCTONIONS, 8). Uses traj[:, 0, :] (the first octonion)."""
    T = traj.shape[0]
    h0 = traj[:, 0, :]  # (T, 8)
    total = 0.0
    triples = list(fano_orbits.non_fano_ordered_triples())
    assert len(triples) == 168
    # Precompute the basis associator vectors once (constant across t).
    basis_associators: list[tuple[oct_mod.Oct, tuple[int, int, int]]] = []
    for i, j, k in triples:
        assoc = oct_mod.oct_associator(
            oct_mod.oct_basis(i),
            oct_mod.oct_basis(j),
            oct_mod.oct_basis(k),
        )
        basis_associators.append((assoc, (i, j, k)))
    for t in range(T):
        a = h0[t]
        for assoc_vec, (i, j, k) in basis_associators:
            # By multilinearity: [a_i e_i, a_j e_j, a_k e_k] = a_i a_j a_k * [e_i, e_j, e_k].
            coeff = float(a[i] * a[j] * a[k])
            scaled = coeff * assoc_vec
            total += float(np.sqrt(np.dot(scaled, scaled)))
    return total / (T * 168.0)


def _sedenion_from_state(h_t: NDArray[np.float64]) -> sed_mod.Sed:
    """Lift (STATE_DIM_OCTONIONS=2, 8) octonion state to a sedenion."""
    assert h_t.shape == (2, 8)
    return sed_mod.sed_from_pair(h_t[0], h_t[1])


def feature_f2_from_trajectory(traj: NDArray[np.float64]) -> float:
    """Proximity to canonical primitive zero-divisor pair in S_16."""
    T = traj.shape[0]
    z, w = sed_mod.CANONICAL_ZERO_DIVISOR_PAIR
    total = 0.0
    for t in range(T):
        s = _sedenion_from_state(traj[t])
        prod_z = sed_mod.sed_mul(s, z)
        prod_w = sed_mod.sed_mul(s, w)
        denom = float(np.dot(s, s)) + _F2_EPS
        total += (float(np.dot(prod_z, prod_z)) + float(np.dot(prod_w, prod_w))) / denom
    return total / T


def feature_f3_from_trajectory(traj: NDArray[np.float64]) -> float:
    """L1 sedenion associator mass on the 168 triples (v3 F_3)."""
    T = traj.shape[0]
    triples = sed_mod.O_168_sedenion_triples()
    total = 0.0
    for t in range(T):
        s = _sedenion_from_state(traj[t])
        # Precompute s * e_I once per basis index to avoid O(168 * 3) re-multiplication.
        s_products: dict[int, sed_mod.Sed] = {}
        for I, J, K in triples:
            for idx in (I, J, K):
                if idx not in s_products:
                    s_products[idx] = sed_mod.sed_mul(s, sed_mod.sed_basis(idx))
        for I, J, K in triples:
            a = s_products[I]
            b = s_products[J]
            c = s_products[K]
            assoc = sed_mod.sed_associator(a, b, c)
            total += float(np.sum(np.abs(assoc)))
    return total / (T * 168.0)


def compute_all_features(
    traj: NDArray[np.float64],
) -> dict[str, float]:
    """Return {'F1': ..., 'F2': ..., 'F3': ...} for one epoch's trajectory."""
    return {
        "F1": feature_f1_from_trajectory(traj),
        "F2": feature_f2_from_trajectory(traj),
        "F3": feature_f3_from_trajectory(traj),
    }
