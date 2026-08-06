#!/usr/bin/env python3
"""`tri3(P3~)` at the fibre references — a CLOSED FORM, found and then confirmed out of sample.

§57.8 reduced the deviation law to one moment of one Seidel matrix: `tri3(P3~)`, where `P3~` is
`P3` masked to the off-diagonal nonzero indices.  §57.9/§57.10 pinned the maximal seam and the
`g = 0` reference.  What was left open was the value at a reference with `g != 0` -- at `m = 5` the
four numbers `-39654, 15642, -7398, 11034` had no formula.

THE ANSWER.  Fix a level `m`, write `b = m - 3`, `N = 2^(m+1) - 1`.  The fibre references are
`W = 8g + 1`; `P3_top_switch` (Tier 64) makes `g` and `g + 2^b` switching-equivalent, so `g` runs
over `[0, 2^b)`.  Expand the map `g -> tri3(P3~)(8g+1)` in WALSH characters of `(Z/2)^b`:

    tri3(P3~)(8g+1) = sum_k  w_k * (-1)^popcount(g & k)

Then:

  (1) `w_k = 0` UNLESS the set bits of `k` form a CONTIGUOUS BLOCK of bit positions.
      So only `b(b+1)/2` of the `2^b` characters are present.

  (2) for the block `[i, i+L-1]`,

          w_k = - 2304 * (2^(i+1) - 1) * 8^(m-4-i) / 2^(L-1)

      -- i.e. everything is fixed by the SINGLE-BIT coefficients `s_i = -2304 (2^(i+1)-1) 8^(m-4-i)`,
      and lengthening a block by one halves its coefficient.

  (3) the mean is pinned by §57.10's closed form at `g = 0`:

          w_0 = N(N-1)(N-2) - 1728*[m,3]_2 - 288*[m-1,2]_2 - sum_{k != 0} w_k

EVIDENCE.  Discovered on `m = 4,5,6`.  At `m = 7` the seven coefficients that already existed at
`m = 6` were PREDICTED (`w_{m+1}[k] = 8 * w_m[k]`) and confirmed exactly, and `w_5 = 0` was
predicted by (1) and confirmed.  At `m = 8` the WHOLE table -- all 32 reference values, none of them
used to build the formula -- was predicted and confirmed: 0 mismatches.

WHAT THIS IS NOT.  It is a closed form for the REFERENCES, not a proof of the deviation law.  The
law `D[tri3(P3~)] = 27*8^(n-j)*[j,3]_2` is about `W = 8g + 2^j` against `8g + 1`; this file computes
only the second of those.  Nothing here is formalised in Lean.
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_yrow_probe import _P3  # noqa: E402


def gauss(j, k):
    """[j choose k]_2."""
    if k > j:
        return 0
    num = den = 1
    for i in range(k):
        num *= 2**j - 2**i
        den *= 2**k - 2**i
    return num // den


def block(k):
    """(start, length) if k's set bits are a contiguous block, else None."""
    if k == 0:
        return None
    lo = (k & -k).bit_length() - 1
    hi = k.bit_length() - 1
    return (lo, hi - lo + 1) if k == (1 << (hi + 1)) - (1 << lo) else None


def coefficients(m):
    """The Walsh coefficients w_k of g -> tri3(P3~)(8g+1), by the closed form."""
    b = m - 3
    N = (1 << (m + 1)) - 1
    w = [0] * (1 << b)
    for k in range(1, 1 << b):
        bl = block(k)
        if bl:
            i, L = bl
            w[k] = -2304 * (2**(i + 1) - 1) * 8**(m - 4 - i) // 2**(L - 1)
    w[0] = N * (N - 1) * (N - 2) - 1728 * gauss(m, 3) - 288 * gauss(m - 1, 2) - sum(w[1:])
    return w


def predict(m):
    w = coefficients(m)
    b = m - 3
    return [sum(w[k] * (-1)**(bin(g & k).count("1")) for k in range(1 << b))
            for g in range(1 << b)]


def measure(m, W):
    V = 1 << (m + 1)
    Q = np.array([[_P3(l, y, W, m) for y in range(1, V)] for l in range(1, V)], dtype=np.int64)
    np.fill_diagonal(Q, 0)
    return int(np.sum(Q * (Q @ Q)))


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["4", "5", "6", "7"])]
    for m in levels:
        b = m - 3
        pred = predict(m)
        bad = 0
        for g in range(1 << b):
            if measure(m, 8 * g + 1) != pred[g]:
                bad += 1
        w = coefficients(m)
        nz = [k for k in range(1, 1 << b) if w[k]]
        print(f"m={m} (N={(1 << (m+1)) - 1}, {1 << b} references): {bad} mismatches / {1 << b}")
        print(f"      nonzero characters {len(nz)} = b(b+1)/2 = {b*(b+1)//2}: {nz}")
        print(f"      single-bit coefficients s_i = {[w[1 << i] for i in range(b)]}")


if __name__ == "__main__":
    main()
