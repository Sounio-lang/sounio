#!/usr/bin/env python3
"""tr(B E^2) = 0 -- the mechanism.  §47's diagnosis of the blocker was about the wrong index.

§47 recorded: "the hub rows are dense, so tr(B E^2) has O(h^2) nonzero terms and its vanishing is a
CANCELLATION, not a sparsity argument".  The premise is true and the conclusion does not follow.
Write

    tr(B E^2) = sum_{u,v,w} B[u,v] E[v,w] E[w,u].

B = J2 (x) A' is zero on the rows AND columns of the three special vertices  W, W+h, h  (row W of A'
is the isolated vertex, and h is outside B's index range), so u and v are both non-special; the dense
rows can only ever appear as the MIDDLE index w.  Summing over u and v first,

    ***  tr(B E^2) = sum_w  y_w^T A' y_w ,     y_w(a) = E[w,a] + E[w,a+h],  a in [1,h)  ***

-- one quadratic form per vertex w, over the level-(n-1) index, because B collapses its two
arguments through the 2x2 block.  Each form vanishes, and for a reason already in the Lean file:

  (L2) w in {W, W+h, h}:  y_w = 0 OUTRIGHT.
         w = W    : row W of A is zero          -- `Asig_isolated_row`
         w = h    : A[h, y+h] = -A[h, y]        -- `Asig_hub0`
         w = W+h  : A[W+h, y+h] = -A[W+h, y]    -- `Asig_hubL`
       The block flip is an antisymmetry of the hub rows, so collapsing the block ADDS a number to
       its own negative.  Density is irrelevant.

  (L1) w = (b,delta) non-special:  y_w = e_b - e_{b^W}, off the index W where A' is null
         +1 at b     -- the matching edge, `Asig_matching`
         -1 at b^W   -- the coset edge, `Asig_coset` (cross block only; the within-block coset
                        entry is zero by `resB_coset`)
       so  y_w^T A' y_w = A'(b,b) - 2 A'(b, b^W) + A'(b^W, b^W) = 0  by `Asig_diag` (zero diagonal)
       and `resB_coset` (a vertex is never adjacent to its coset partner).

So it IS a support argument after all -- just at the collapsed index, not the ambient one.  Every
pointwise ingredient is already a kernel-clean theorem; what is left is the sum bookkeeping.

C1 collapse | C2 y_w = 0 on the three special w, with the three reasons | C3 (L1) shape | C4 A'
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402
from zd_v1_18_1_decomposition_probe import blowup  # noqa: E402


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["6", "7", "8"])]
    for n in levels:
        h = 1 << (n - 2)
        Sn, Sm = sign_table_fast(n), sign_table_fast(n - 1)
        N = (1 << (n - 1)) - 1
        lo = np.arange(1, h)

        c1 = c2 = c3 = c4 = 0
        why = {"row W = 0": 0, "A[h,y+h]=-A[h,y]": 0, "A[W+h,y+h]=-A[W+h,y]": 0}
        n3 = 0
        for W in range(1, h):
            A = A_sig_fast(n, W, Sn).astype(np.int64)
            Ap = A_sig_fast(n - 1, W, Sm).astype(np.int64)
            E = A - blowup(Ap, n)
            Y = E[:, lo - 1] + E[:, lo + h - 1]                    # y_w for every w
            quad = np.einsum('wa,ab,wb->w', Y, Ap, Y)

            # C1: the collapse is an identity, and the trace really is zero
            c1 += int(quad.sum() != np.einsum('uv,vw,wu->', blowup(Ap, n), E, E))
            c1 += int(quad.sum() != 0)

            # C2: y_w = 0 on the three special vertices, and each for its own stated reason
            for v in (W, W + h, h):
                c2 += int(np.count_nonzero(Y[v - 1]))
            why["row W = 0"] += int(np.count_nonzero(A[W - 1]))
            why["A[h,y+h]=-A[h,y]"] += int(np.count_nonzero(A[h - 1, lo + h - 1] + A[h - 1, lo - 1]))
            why["A[W+h,y+h]=-A[W+h,y]"] += int(
                np.count_nonzero(A[W + h - 1, lo + h - 1] + A[W + h - 1, lo - 1]))

            # C3: y_w = e_b - e_{b^W} for every non-special w
            for w in range(1, N + 1):
                if w in (W, W + h, h):
                    continue
                b = w - h if w > h else w
                exp = np.zeros(h - 1, dtype=np.int64)
                exp[b - 1] = 1
                if (b ^ W) >= 1:
                    exp[(b ^ W) - 1] -= 1
                d = Y[w - 1] - exp
                d[W - 1] = 0                                        # A' kills this coordinate
                c3 += int(np.count_nonzero(d))
                n3 += 1

            # C4: A' has zero diagonal and no vertex is adjacent to its coset partner
            c4 += int(np.count_nonzero(np.diag(Ap)))
            for b in lo:
                if b != W and (b ^ W) >= 1:
                    c4 += int(Ap[b - 1, (b ^ W) - 1] != 0)

        print(f"n={n}  ({h-1} low labels)")
        print(f"  C1 tr(B E^2) == sum_w y_w^T A' y_w == 0            violations {c1}")
        print(f"  C2 y_w = 0 for w in W, W+h, h                      violations {c2}")
        print(f"     reasons: {why}")
        print(f"  C3 y_w = e_b - e_(b^W) off index W                 violations {c3} / {n3}")
        print(f"  C4 A' zero diagonal + coset non-adjacency          violations {c4}")


if __name__ == "__main__":
    main()
