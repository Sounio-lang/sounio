"""Sedenion algebra S_16, primitive zero-divisor pairs, and v3 F_3.

A sedenion is stored as a 1D float64 ndarray of shape (16,) with basis
[e0, e1, ..., e15]. e0..e7 span the "octonion half"; e8..e15 span the
doubled half where e_{8+i} = e_i * e_8.

Multiplication follows the Cayley-Dickson doubling:
    (a, b) * (c, d) = (a*c - d_bar * b,  d*a + b*c_bar)
where a, b, c, d in O (length-8 octonions) and x_bar is octonion
conjugation. This is implemented in terms of oct_mul from
scripts/research/ossm_168_dryrun/octonion.py.

Structural note (v2 → v3 correction): every Cayley-Dickson algebra is
FLEXIBLE ([a,b,a] = 0), which forces [c,b,a] = -[a,b,c], hence
||[c,b,a]|| = ||[a,b,c]|| identically. Any feature defined as a
norm-asymmetry under (a ↔ c) reversal collapses to zero throughout the
CD tower (C, H, O, S_16, T_32, ...). v2's F_3 had this flaw; v3 F_3
replaces it with a componentwise-L¹ associator mass (sed_l1_norm of
the associator vector), which is generically non-zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import fano_orbits
from . import octonion as oct_mod

Sed = NDArray[np.float64]  # shape (16,)

__all__ = [
    "Sed",
    "sed_zero",
    "sed_basis",
    "sed_conjugate_oct",
    "sed_from_pair",
    "sed_mul",
    "sed_associator",
    "sed_norm",
    "sed_l1_norm",
    "f3_v3_l1_associator_mass",
    "O_168_sedenion_triples",
    "CANONICAL_ZERO_DIVISOR_PAIR",
]


def sed_zero() -> Sed:
    return np.zeros(16, dtype=np.float64)


def sed_basis(i: int) -> Sed:
    if not (0 <= i <= 15):
        raise ValueError(f"sedenion basis index out of range: {i}")
    s = sed_zero()
    s[i] = 1.0
    return s


def sed_conjugate_oct(a: oct_mod.Oct) -> oct_mod.Oct:
    """Octonion conjugation: (a0, a1, ..., a7) -> (a0, -a1, ..., -a7)."""
    out = a.copy()
    out[1:] *= -1.0
    return out


def sed_from_pair(a: oct_mod.Oct, b: oct_mod.Oct) -> Sed:
    """Build a sedenion from a pair of octonions in Cayley-Dickson order."""
    s = sed_zero()
    s[0:8] = a
    s[8:16] = b
    return s


def _split(s: Sed) -> tuple[oct_mod.Oct, oct_mod.Oct]:
    return s[0:8].copy(), s[8:16].copy()


def sed_mul(s: Sed, t: Sed) -> Sed:
    """Sedenion product (a, b) * (c, d) = (a*c - d_bar * b, d*a + b*c_bar)."""
    a, b = _split(s)
    c, d = _split(t)
    d_bar = sed_conjugate_oct(d)
    c_bar = sed_conjugate_oct(c)
    real_half = oct_mod.oct_mul(a, c) - oct_mod.oct_mul(d_bar, b)
    imag_half = oct_mod.oct_mul(d, a) + oct_mod.oct_mul(b, c_bar)
    return sed_from_pair(real_half, imag_half)


def sed_associator(a: Sed, b: Sed, c: Sed) -> Sed:
    """[a, b, c] = (a*b)*c - a*(b*c) in S_16."""
    return sed_mul(sed_mul(a, b), c) - sed_mul(a, sed_mul(b, c))


def sed_norm(s: Sed) -> float:
    return float(np.sqrt(np.dot(s, s)))


def sed_l1_norm(s: Sed) -> float:
    """Componentwise L¹ norm: sum of absolute values of all 16 components."""
    return float(np.sum(np.abs(s)))


def f3_v3_l1_associator_mass(
    h: Sed, triples: tuple[tuple[int, int, int], ...] | None = None
) -> float:
    """v3 F_3 per-state: (1/168) * sum_{(I,J,K) in O_168} ||[h*e_I, h*e_J, h*e_K]||_{L1}.

    Uses the 168-orbit sedenion triples from O_168_sedenion_triples() by
    default. For a time-series use-case, average this over T states externally.
    Non-zero on generic h; vanishes only when all three arguments lie in an
    associative subalgebra.
    """
    if triples is None:
        triples = O_168_sedenion_triples()
    total = 0.0
    for I, J, K in triples:
        a = sed_mul(h, sed_basis(I))
        b = sed_mul(h, sed_basis(J))
        c = sed_mul(h, sed_basis(K))
        total += sed_l1_norm(sed_associator(a, b, c))
    return total / len(triples)


def O_168_sedenion_triples() -> tuple[tuple[int, int, int], ...]:
    """Return 168 ordered sedenion-basis triples (I, J, K) used by v2 F_3.

    Operational resolution of v2's indexing ambiguity: for each non-Fano
    ordered triple (i, j, k) in O_168 on {1..7}, map to the sedenion
    triple (I, J, K) = (i, 8+j, k). See DEVIATIONS.md entry 0003.

    The returned tuple is deterministically ordered by the lex order
    of (i, j, k).
    """
    triples: list[tuple[int, int, int]] = []
    for i, j, k in sorted(fano_orbits.non_fano_ordered_triples()):
        triples.append((i, 8 + j, k))
    assert len(triples) == 168, f"expected 168 sedenion triples, got {len(triples)}"
    return tuple(triples)


# Canonical primitive zero-divisor pair catalogued in stdlib/math/sedenion.sio
# and de Marrais 2000. Used by H2 for F_2 proximity.
def _canonical_zd_pair() -> tuple[Sed, Sed]:
    z = sed_basis(3) + sed_basis(10)
    w = sed_basis(6) - sed_basis(15)
    return z, w


CANONICAL_ZERO_DIVISOR_PAIR: tuple[Sed, Sed] = _canonical_zd_pair()
"""Primitive zero-divisor pair (e_3 + e_10, e_6 - e_15). Verify sed_mul(z, w) = 0."""
