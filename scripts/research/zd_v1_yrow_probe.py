#!/usr/bin/env python3
"""The COLLAPSED ROW closes BOTH of §34's remaining sums -- and the index 0 carries no weight.

Two independent things, both prerequisites for turning §49 into Lean.

C0  THE INDEX 0.  Lean's sum ranges are [0, 2^(n+1)), one element wider than the contract's vertex
    set [1, 2^(n+1)), and Tier 43's `blow m A x y = A (x % m) (y % m)` sends the hub 2^(k+1) to 0.
    So every denotation claim ("this sumLtI IS t3'") needs row and column 0 of A_sig to be zero.
    `Asig_isolated` does NOT cover it -- it requires l != 0.  Checked here against a Lean-FAITHFUL
    re-implementation of cdSigma / P1 / P3 / resB / Asig (not the fast builder, which never
    constructs the index at all), over every label at n = 2..5.

C1  THE COLLAPSED ROW.  y_w(a) = E[w,a] + E[w,a+h] for a generic w = b + delta*h is
    e_b - e_{b^W}: two support points, and INDEPENDENT OF delta.

C2  Hence the 2x2 block sum of E is  Z(b,a) = 2[(a=b) - (a=b^W)]  off the isolated row/column, and
    since B^2 = 2 J2 (x) A'^2,

        tr(B^2 E) = 2 sum_{a,b} (A'^2)(a,b) Z(b,a) = 4[t2' - S(W)] = 8 t2'

    by the already-proven S(W) = -t2' (`cosetSum_eq`, Tier 42).  So `3 tr(B^2 E) = 24 t2'` and
    `tr(B E^2) = 0` are ONE lemma apart, not two, and §34.3's `24 = 3 x (4+4)` is exactly the two
    support points of the collapsed row.
"""

import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402
from zd_v1_18_1_decomposition_probe import blowup  # noqa: E402


# --- Lean-faithful transcription of the definitions in SounioZDFiberAntisym.lean -------------
@lru_cache(maxsize=None)
def cdS(a, b, n):
    if n == 0:
        return -1
    if n == 1:
        return 1 if (a == 0 or b == 0) else -1
    if a == 0 or b == 0:
        return 1
    half = 1 << (n - 1)
    au, bu = a >= half, b >= half
    if not au and not bu:
        return cdS(a % half, b % half, n - 1)
    if not au and bu:
        return cdS(b % half, a % half, n - 1)
    if au and not bu:
        return cdS(a % half, 0, n - 1) if b % half == 0 else -cdS(a % half, b % half, n - 1)
    return -cdS(0, a % half, n - 1) if b % half == 0 else cdS(b % half, a % half, n - 1)


def _hi(x, L, n):
    return (x ^ L) + (1 << (n + 1))


def _P1(l, y, L, n):
    return cdS(l, y, n + 2) * cdS(_hi(l, L, n), _hi(y, L, n), n + 2)


def _P3(l, y, L, n):
    return cdS(l, _hi(y, L, n), n + 2) * cdS(_hi(l, L, n), y, n + 2)


def Asig(l, y, L, n):
    ok = (_P1(l, y, L, n) == _P1(y, l, L, n) and _P3(l, y, L, n) == _P3(y, l, L, n)
          and _P1(l, y, L, n) == _P3(l, y, L, n))
    return -_P1(l, y, L, n) if ok else 0


def bridge():
    """C-1  The Lean `Asig` IS the contract's builder matrix.

    Everything else in this file, and every Lean theorem about `Asig`, is only about §34's ledger
    if the transcription and the builder agree.  `Asig x y W m` has x,y in [0, 2^(m+1)); the
    builder's level-n matrix has vertices [1, 2^(n-1)) at index vertex-1.  So n = m + 2.
    """
    print("C-1 Lean `Asig` vs the contract's builder A_sig_fast")
    for m in (2, 3, 4):
        n = m + 2
        S = sign_table_fast(n)
        V = 1 << (m + 1)
        bad = tot = 0
        for W in range(1, V):
            A = A_sig_fast(n, W, S).astype(np.int64)
            for x in range(1, V):
                for y in range(1, V):
                    tot += 1
                    if int(A[x - 1, y - 1]) != Asig(x, y, W, m):
                        bad += 1
        print(f"    m={m} <-> n={n}: {tot} entries over {V-1} labels, {bad} mismatches")


def denotation():
    """C-2  The theorem's own sum over [0,N) equals the contract's over [1,N), and both are 0."""
    print("C-2 index-0 padding is inert in tr(B E^2)")
    for k in (1, 2, 3):
        m = 1 << (k + 1)
        N = m + m
        for W in range(1, m):
            A1 = [[Asig(x, y, W, k + 1) for y in range(N)] for x in range(N)]
            A0 = [[Asig(x, y, W, k) for y in range(m)] for x in range(m)]
            B = [[A0[x % m][y % m] for y in range(N)] for x in range(N)]
            E = [[A1[x][y] - B[x][y] for y in range(N)] for x in range(N)]
            s0 = sum(B[a][b] * E[b][c] * E[c][a]
                     for a in range(N) for b in range(N) for c in range(N))
            s1 = sum(B[a][b] * E[b][c] * E[c][a]
                     for a in range(1, N) for b in range(1, N) for c in range(1, N))
            assert s0 == s1 == 0, (k, W, s0, s1)
        print(f"    k={k} (N={N}, {m-1} labels): sum[0,N) == sum[1,N) == 0")


def main():
    bridge()
    denotation()
    print("C0  row and column 0 of A_sig, from the Lean definitions")
    for n in (2, 3, 4, 5):
        H = 1 << (n + 1)
        bad = 0
        for L in range(1, H):
            bad += sum(1 for y in range(H) if Asig(0, y, L, n) != 0)
            bad += sum(1 for x in range(H) if Asig(x, 0, L, n) != 0)
        print(f"    n={n}: {H-1} labels, nonzero entries in row/col 0: {bad}")

    print("C1/C2  the collapsed row and tr(B^2 E)")
    for n in [int(x) for x in (sys.argv[1:] or ["6", "7", "8"])]:
        h = 1 << (n - 2)
        Sn, Sm = sign_table_fast(n), sign_table_fast(n - 1)
        lo = np.arange(1, h)
        badz = badt = 0
        for W in range(1, h):
            A = A_sig_fast(n, W, Sn).astype(np.int64)
            Ap = A_sig_fast(n - 1, W, Sm).astype(np.int64)
            B = blowup(Ap, n)
            E = A - B
            t2 = int(np.trace(Ap @ Ap))
            Z = (E[np.ix_(lo - 1, lo - 1)] + E[np.ix_(lo - 1, lo + h - 1)]
                 + E[np.ix_(lo + h - 1, lo - 1)] + E[np.ix_(lo + h - 1, lo + h - 1)])
            P = np.zeros_like(Z)
            for b in lo:
                if b == W:
                    continue
                P[b - 1, b - 1] += 2
                c = b ^ W
                if c >= 1:
                    P[b - 1, c - 1] -= 2
            D = Z - P
            D[W - 1, :] = 0                      # the isolated row/column, killed by A'^2
            D[:, W - 1] = 0
            badz += int(np.count_nonzero(D))
            pred = 2 * int(np.einsum('ab,ba->', Ap @ Ap, P))
            act = int(np.einsum('uv,vw,wu->', B, B, E))
            badt += int(pred != act or act != 8 * t2)
        print(f"    n={n} ({h-1} labels): Z = 2[(a=b)-(a=b^W)] violations {badz};  "
              f"tr(B^2 E) = 8 t2' from the closed form, violations {badt}")


if __name__ == "__main__":
    main()
