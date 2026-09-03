#!/usr/bin/env python3
"""NEGATIVE fixture for evidential-independence checking (rung R6).

This file pretends to be an independent corroborator of the E6 claim while
COPY-PASTING the harness's own derivation verbatim from
scripts/research/functor_f_e6_albert_shadow_contract.py. A corroborator built
this way inherits whatever that derivation encodes, so it corroborates nothing.

The independence check MUST reject it. If it does not, the guard is vacuous.

Not executed by anything: it exists to be measured.
"""

import numpy as np


def cds(a, b, bits=3):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah, bh = a >= h, b >= h
        al, bl = a & (h - 1), b & (h - 1)
        if not ah and not bh:
            a, b = al, bl
        elif not ah and bh:
            a, b = bl, al
        elif ah and not bh:
            a, b, s = ((al, 0, s) if bl == 0 else (al, bl, -s))
        else:
            a, b, s = ((0, al, -s) if bl == 0 else (bl, al, s))
        bits -= 1
    return s


def o(A, B):
    C = np.zeros(8)
    for i in range(8):
        if A[i] == 0.0:
            continue
        for j in range(8):
            if B[j] == 0.0:
                continue
            C[i ^ j] += cds(i, j) * A[i] * B[j]
    return C


def e(i):
    v = np.zeros(8); v[i] = 1.0; return v


def corroborate():
    """Would 'check' the E6 identity using the harness's own multiplication."""
    x, y, z = e(1), e(2), e(3)
    return float(o(o(x, y), z)[0])
