"""Python mirror of stdlib/algebra/octonion.sio's oct_mul, bit-for-bit.

The canonical Sounio implementation is in stdlib/algebra/octonion.sio
(function oct_mul at lines 80-144). This module replicates its fully
expanded 8x8 Cayley-Dickson sign convention (Albuquerque-Majid 1999)
in NumPy, so that features computed in this dry-run agree with the
Sounio-native port to be produced in a follow-up slice.

An octonion is a 1D ndarray of shape (8,) with float64 dtype, indexed
as [e0, e1, e2, e3, e4, e5, e6, e7], where e0 is the real part and
e1..e7 are the seven imaginary units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Oct = NDArray[np.float64]  # shape (8,)

__all__ = [
    "Oct",
    "oct_zero",
    "oct_basis",
    "oct_mul",
    "oct_sub",
    "oct_associator",
    "oct_norm",
    "oct_norm_sq",
    "permute_input",
]


def oct_zero() -> Oct:
    return np.zeros(8, dtype=np.float64)


def oct_basis(i: int) -> Oct:
    """Unit octonion e_i, i in {0..7}."""
    if not (0 <= i <= 7):
        raise ValueError(f"basis index out of range: {i}")
    x = oct_zero()
    x[i] = 1.0
    return x


def oct_mul(a: Oct, b: Oct) -> Oct:
    """Cayley-Dickson octonion multiplication, non-commutative and
    non-associative, verbatim copy of stdlib/algebra/octonion.sio:oct_mul
    (lines 86-143 there)."""
    out = np.empty(8, dtype=np.float64)
    # e0 (real part):
    out[0] = (
        a[0] * b[0]
        - a[1] * b[1]
        - a[2] * b[2]
        - a[3] * b[3]
        - a[4] * b[4]
        - a[5] * b[5]
        - a[6] * b[6]
        - a[7] * b[7]
    )
    # e1:
    out[1] = (
        a[0] * b[1] + a[1] * b[0]
        + a[2] * b[3] - a[3] * b[2]
        + a[4] * b[5] - a[5] * b[4]
        - a[6] * b[7] + a[7] * b[6]
    )
    # e2:
    out[2] = (
        a[0] * b[2] + a[2] * b[0]
        - a[1] * b[3] + a[3] * b[1]
        + a[4] * b[6] - a[6] * b[4]
        + a[5] * b[7] - a[7] * b[5]
    )
    # e3:
    out[3] = (
        a[0] * b[3] + a[3] * b[0]
        + a[1] * b[2] - a[2] * b[1]
        + a[4] * b[7] - a[7] * b[4]
        - a[5] * b[6] + a[6] * b[5]
    )
    # e4:
    out[4] = (
        a[0] * b[4] + a[4] * b[0]
        - a[1] * b[5] + a[5] * b[1]
        - a[2] * b[6] + a[6] * b[2]
        - a[3] * b[7] + a[7] * b[3]
    )
    # e5:
    out[5] = (
        a[0] * b[5] + a[5] * b[0]
        + a[1] * b[4] - a[4] * b[1]
        - a[2] * b[7] + a[7] * b[2]
        + a[3] * b[6] - a[6] * b[3]
    )
    # e6:
    out[6] = (
        a[0] * b[6] + a[6] * b[0]
        + a[1] * b[7] - a[7] * b[1]
        + a[2] * b[4] - a[4] * b[2]
        - a[3] * b[5] + a[5] * b[3]
    )
    # e7:
    out[7] = (
        a[0] * b[7] + a[7] * b[0]
        - a[1] * b[6] + a[6] * b[1]
        + a[2] * b[5] - a[5] * b[2]
        + a[3] * b[4] - a[4] * b[3]
    )
    return out


def oct_sub(a: Oct, b: Oct) -> Oct:
    return a - b


def oct_associator(a: Oct, b: Oct, c: Oct) -> Oct:
    """[a, b, c] = (a*b)*c - a*(b*c)."""
    ab = oct_mul(a, b)
    bc = oct_mul(b, c)
    return oct_mul(ab, c) - oct_mul(a, bc)


def oct_norm_sq(a: Oct) -> float:
    return float(np.dot(a, a))


def oct_norm(a: Oct) -> float:
    return float(np.sqrt(oct_norm_sq(a)))


def permute_input(x7: NDArray[np.float64], sigma: tuple[int, ...]) -> NDArray[np.float64]:
    """Apply 1-indexed permutation sigma in S_7 to a length-7 input vector.

    y[i] = x[sigma(i) - 1] for i in {1..7}, 0-indexed in the output.
    Equivalently: the region previously at position i is relabeled so
    that its signal is fed to octonion basis slot sigma(i).
    """
    if len(sigma) != 7:
        raise ValueError(f"sigma must be length 7, got {len(sigma)}")
    y = np.empty(7, dtype=np.float64)
    for i in range(7):
        y[i] = x7[sigma[i] - 1]
    return y
