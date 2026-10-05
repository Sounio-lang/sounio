#!/usr/bin/env python3
"""`t3(box)`'s variation inside a fibre, attacked WITH the edge-disjointness (Tier 60).

Setting, in the Lean indexing.  Fix a level `m` and a low label `W < 2^(m+1)`.  Two matrices on the
SAME vertex set `[0, 2^(m+1))` (index 0 is isolated at both levels):

    Alo(l,y) = Asig l y  W              m        -- the level-m matrix of the LOW label
    Ahi(l,y) = Asig l y (W + 2^(m+1)) (m+1)      -- THE BOX: the level-(m+1) matrix of the HIGH
                                                    label, restricted to the low indices

Tier 60 proves `supp(Ahi) ∩ supp(Alo) = ∅`.  What this probe found is that they also COVER, which
makes their difference mask-free, and that the deviation law splits between them by parity.

C1  disjointness (Tier 60) -- re-measured as a control on the transcription
C2  is `U = Alo + Ahi` label-independent?  NO -- that route is dead, and the live one is C3
C3  ** THE PARTITION **  the off-diagonal nonzero pairs where BOTH masks are false: there are NONE.
    §57.4 asserted such pairs existed; they do not.  Hence
        Alo − Ahi = −P1        (level m, off the diagonal, no mask)
    a SEIDEL matrix (symmetric, ±1 off the diagonal), of which the two `Asig` are the two edge
    classes.  Both facts are now Lean theorems (Tier 61 `resB_hi_or_lo`, `Asig_hi_lo_diff`).
C4  the exact four-term expansion that follows, `P := Alo − Ahi`:
        t3(Ahi) = t3(Alo) − 3 tr(Alo² P) + 3 tr(Alo P²) − t3(P)
    This is a HIGH-BRANCH DESCENT.  It does NOT contradict §54.3, which refuted a two-term affine
    recursion in `(t3', t2')`: the extra terms here are mixed traces, not multiples of those two.
C5  the box's own deviation obeys the law at level `n = m+2`:
        t3(box_W) − t3(box_{8g+1}) = −27·8^(n−j)·[j,3]₂   (with the parity rule on g(W+2^(m+1)))
    and the FULL level-(m+1) matrix's deviation is exactly 8× the box's -- so the box is the level-n
    object and the `8` is `tri3_kron`'s, now checked on general labels and not only the max seam.
C6  the parity corollary, and why it is NOT new content.  `g(W + 2^(m+1)) = g(W) + 2^(m−2)`, so
    popcount's PARITY flips: the low label and its box sit in fibres of opposite parity.  Given (A)
    -- the parity half of the law, already a theorem -- exactly one of the two deviations is zero
    and the other is the full prediction, so `Dlow + Dbox = −27·8^(n−j)·[j,3]₂` with no parity case
    split.  That is a restatement of (A) plus the g-shift, not an independent finding.
"""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_yrow_probe import Asig, _P1  # noqa: E402


def g_of(W):
    return (W & (W - 1)) >> 3


def lsb(W):
    return (W & -W).bit_length() - 1


def gauss3(j):
    """[j choose 3]_2 -- the number of 3-dim subspaces of F_2^j."""
    return 0 if j < 3 else (2**j - 1) * (2**j - 2) * (2**j - 4) // 168


def t3(X):
    return int(np.einsum('ab,bc,ca->', X, X, X))


def mix(X, Y, Z):
    return int(np.einsum('ab,bc,ca->', X, Y, Z))


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["2", "3", "4"])]
    for m in levels:
        V = 1 << (m + 1)
        N = 2 * V
        n = m + 2
        print(f"=== Lean level m={m} (box on [0,{V}), contract level n={n}) ===")
        c1 = c3 = cdiff = 0
        us = set()
        box = {}
        low = {}
        full = {}
        four = {}
        for W in range(1, V):
            F = np.array([[Asig(l, y, W + V, m + 1) for y in range(N)] for l in range(N)],
                         dtype=np.int64)
            Ahi = F[:V, :V]
            Alo = np.array([[Asig(l, y, W, m) for y in range(V)] for l in range(V)], dtype=np.int64)
            P = np.zeros_like(Alo)
            for l in range(1, V):
                for y in range(1, V):
                    if l != y:
                        P[l, y] = -_P1(l, y, W, m)
                        if Alo[l, y] == 0 and Ahi[l, y] == 0:
                            c3 += 1
            c1 += int(np.any((Alo != 0) & (Ahi != 0)))
            cdiff += int(np.any(Alo - Ahi != P))
            us.add((Alo + Ahi).tobytes())
            box[W], low[W], full[W] = t3(Ahi), t3(Alo), t3(F)
            four[W] = (low[W] - 3 * mix(Alo, Alo, P) + 3 * mix(Alo, P, P) - t3(P) == box[W])
        print(f"  C1 supports disjoint                        : {c1} labels with an overlap")
        print(f"  C2 distinct U = Alo + Ahi                   : {len(us)} / {V-1}"
              f"  ({'label-independent' if len(us) == 1 else 'NOT label-independent -- route dead'})")
        print(f"  C3 pairs with BOTH masks false (must be 0)  : {c3}")
        print(f"     Alo - Ahi = -P1 off-diagonal, mask-free  : {cdiff} viol / {V-1}")
        print(f"  C4 t3(Ahi) = t3(Alo) -3tr(A^2 P) +3tr(A P^2) -t3(P): "
              f"{sum(1 for W in four if not four[W])} viol / {V-1}")
        c5 = c5f = tot = c6 = 0
        for W in range(1, V):
            g = g_of(W)
            ref = 8 * g + 1
            if not (0 < ref < V):
                continue
            tot += 1
            j = lsb(W)
            law = -27 * 8**(n - j) * gauss3(j)
            ghi = g_of(W + V)
            pred = 0 if bin(ghi).count("1") % 2 == 1 else law
            c5 += int(box[W] - box[ref] != pred)
            c5f += int(full[W] - full[ref] != 8 * (box[W] - box[ref]))
            c6 += int((low[W] - low[ref]) + (box[W] - box[ref]) != law)
        print(f"  C5 box deviation = law at level n           : {c5} viol / {tot}")
        print(f"     D(full level m+1) = 8 * D(box)           : {c5f} viol / {tot}")
        print(f"  C6 Dlow + Dbox = law, no parity split       : {c6} viol / {tot}"
              f"   [corollary of (A) + the g-shift, not new]")
        gsh = sum(int(g_of(W + V) != g_of(W) + (1 << (m - 2))) for W in range(1, V)) if m >= 2 else 0
        print(f"     g(W + 2^(m+1)) = g(W) + 2^(m-2)          : {gsh} viol / {V-1}")


if __name__ == "__main__":
    main()
