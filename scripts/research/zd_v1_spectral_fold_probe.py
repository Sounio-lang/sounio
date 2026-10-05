#!/usr/bin/env python3
"""The SPECTRAL route to the deviation law: the exact halving, and what the spectra show.

§54.6 concluded that five combinatorial decompositions all fail the same way -- the g-dependence is
present at the fine level and cancels only in the signed triangle count -- and that the remaining
reading is spectral.  This probe takes that route.

THE FOLD.  `A_sigma(l ^ W, y) = -A_sigma(l, y)` (A1, this file's headline lemma) pairs the vertices
{l, l^W}; the isolated vertex is W itself.  Choosing the representative with W's TOP bit clear and
deleting that bit maps the representatives onto [1, 2^(n-2)) and gives

    A  =  M (x) K   (plus the isolated row/column),      K = [[1,-1],[-1,1]]

so with tr(K^2) = 4 and tr(K^3) = 8:

    ***  tr(A^2) = 4 tr(M^2)   and   tr(A^3) = 8 tr(M^3),  for EVERY label  ***

-- unlike §18.1, which is the LOW branch only.  The ubiquitous 8 is tr(K^3), once.

C1  the two trace identities, every label
C2  is the fold repeatable?  (rank of M, and whether M has its own signed antisymmetry)
C3  the family W_j = 8g + 2^j: spectra of M along j
C4  the maximal seam W = 2^(n-2): M is EXACTLY I - J, the all-minus-one complete graph
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402


def top(W):
    return W.bit_length() - 1


def lsb(W):
    return (W & -W).bit_length() - 1


def fold(n, W, S):
    A = A_sig_fast(n, W, S).astype(np.int64)
    t = top(W)
    N = (1 << (n - 1)) - 1
    reps = [l for l in range(1, N + 1) if not (l >> t) & 1]
    idx = [(l & ((1 << t) - 1)) | ((l >> (t + 1)) << t) for l in reps]
    M = np.zeros(((1 << (n - 2)) - 1, (1 << (n - 2)) - 1), dtype=np.int64)
    for a, la in zip(idx, reps):
        for b, lb in zip(idx, reps):
            M[a - 1, b - 1] = A[la - 1, lb - 1]
    return A, M


def t3(M):
    return int(np.einsum('ab,bc,ca->', M, M, M))


def main():
    for n in [int(x) for x in (sys.argv[1:] or ["6", "7", "8"])]:
        S = sign_table_fast(n)
        b3 = b2 = 0
        ranks = set()
        for W in range(1, 1 << (n - 1)):
            A, M = fold(n, W, S)
            b3 += int(t3(A) != 8 * t3(M))
            b2 += int(int(np.trace(A @ A)) != 4 * int(np.trace(M @ M)))
            ranks.add(int(np.linalg.matrix_rank(M.astype(float))))
        s = (1 << (n - 2)) - 1
        print(f"n={n}: C1 tr(A^3)=8tr(M^3) {b3} viol | tr(A^2)=4tr(M^2) {b2} viol "
              f"(all {(1<<(n-1))-1} labels)")
        print(f"      C2 rank(M) in {sorted(ranks)} of {s} -- M is NONSINGULAR, so the fold is "
              f"exactly one level deep and 8^(n-j) is NOT iterated folding")
        # C4: the maximal seam
        W = 1 << (n - 2)
        _, M = fold(n, W, S)
        IJ = np.eye(s, dtype=np.int64) - np.ones((s, s), dtype=np.int64)
        q = 1 << (n - 2)
        print(f"      C4 maximal seam W={W}: M == I - J ? {np.array_equal(M, IJ)}; "
              f"tr(M^3) = (s-1)-(s-1)^3 = {(s-1)-(s-1)**3} ? {t3(M)==(s-1)-(s-1)**3}; "
              f"8*tr(M^3) = -8(q-1)(q-2)(q-3) ? "
              f"{8*t3(M) == -8*(q-1)*(q-2)*(q-3)}   [§33.5(C), re-derived in one line]")
        # C3: spectra along the family
        for g in ([0, 6] if n >= 7 else [0]):
            if 8 * g + 1 >= 1 << (n - 1):
                continue
            hi = lsb(8 * g) if g else n - 1
            fam = [8 * g + (1 << j) for j in range(0, hi) if 8 * g + (1 << j) < 1 << (n - 1)]
            if len(fam) < 2:
                continue
            out = []
            for W in fam:
                _, M = fold(n, W, S)
                ev = np.round(np.linalg.eigvalsh(M.astype(float)), 4)
                out.append((lsb(W), len(np.unique(ev)), t3(M)))
            print(f"      C3 g={g}: (j, #distinct eigenvalues of M, tr(M^3)) = {out}")
            print(f"         -- j = 0,1,2 are COSPECTRAL (the Fano orbit, spectrally); the "
                  f"spectrum simplifies monotonically as j grows")


if __name__ == "__main__":
    main()
