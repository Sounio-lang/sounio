#!/usr/bin/env python3
"""The deviation law as a SEPARATION OF VARIABLES -- and the death of §33.3's candidate mechanism.

§54.3 reframed what is open: on the high labels `t3(n, .)` is exactly a function of `(g, lsb)`, so
the law is the statement that this function is additively separable,

    t3(g, j) = f0(g) + delta(n, j)       on the even-popcount(g) fibres.

Parametrisation, fixed once: a seam is  W = 8*(g + 2^i)  with 2^i strictly below every bit of g;
then lsb(W) = i+3 and g(W) = g.  Its fibre reference is 8g+1, the j=0 member of the same fibre.

C1  the law, per seam
C2  THE FIRST DIFFERENCE IN j, inside a fibre: t3 of consecutive seams differs by
    delta(n,j) - delta(n,j-1).  This is the local form of separability -- it compares two labels
    that differ in one bit position, instead of comparing to the Fano reference.
C3  §33.3's CANDIDATE MECHANISM, tested for the first time: "the triangle deficit is a count over
    independent triples strictly below the seam".  Split the signed triangle sum by whether the
    three vertices' LOW j BITS are linearly independent over F_2.
C4  §33.3's proposed bijection `a |-> a XOR 8y`, at TRIANGLE level (the edge level was already
    refuted there).

C5  THE LOCAL FORM, attacked (§54.5).  The fibre-g family is ONE parameter:
        W_j = 8g + 2^j   for j = 0 .. lsb(8g)-1     (j = 0,1,2 Fano, j >= 3 seams)
    -- so the reference 8g+1 is the j=0 member and consecutive members differ in TWO BITS.
    ⚠ The range is j < lsb(8g), NOT j <= n-2: a first version of this probe used the latter and
    silently compared labels from DIFFERENT fibres (e.g. 8*10+2^4 = 96 = 8*12 has g = 8, not 10).
    Tested there, all NEGATIVE:
      - is A_{W_j} - A_{W_{j-1}} a function of the low m bits of the two vertices?  No, for every
        m < n-1 (i.e. for every non-vacuous m).  There is no locality.
      - the perturbative split t3(W_j) - t3(W_{j-1}) = 3tr(A'^2 D) + 3tr(A' D^2) + tr(D^3):
        all three pieces are g-DEPENDENT and each is as large as the total.  Only the sum is
        g-independent.
    One regularity did survive: nnz(A_{W_j} - A_{W_{j-1}}) depends only on (n, g) -- it is the
    SAME for every j in the family -- while the triangle-count difference depends only on (n, j).

⚠ THE LAW IS COMPUTED BY ONE FUNCTION, `dev_pred`, ON PURPOSE.  The popcount(g) parity was dropped
by hand THREE times in one session while re-deriving the prediction inline, and each time the
resulting check FAILED and looked like a refutation.  Do not inline it again.
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402


def lsb(W):
    return (W & -W).bit_length() - 1


def gauss3(j):
    return 0 if j < 3 else (2**j - 1) * (2**j - 2) * (2**j - 4) // 168


def dev_pred(n, W):
    """THE deviation law.  Parity included.  Never re-derive this inline."""
    g = (W & (W - 1)) >> 3
    if bin(g).count("1") % 2 == 1:
        return 0
    return -27 * 8**(n - lsb(W)) * gauss3(lsb(W))


def t3(A):
    return int(np.einsum('ab,bc,ca->', A, A, A))


def seams(n):
    """W = 8(g + 2^i) -> (g, i, W), grouped by fibre."""
    fam = {}
    for W in range(1, 1 << (n - 1)):
        if W % 8 or (W >> 3) == 0:
            continue
        y = W >> 3
        i = lsb(y)
        fam.setdefault(y - (1 << i), {})[i] = W
    return fam


def split_by_independence(A, j):
    """Signed triangle sum split by independence of the vertices' low j bits."""
    N = A.shape[0]
    low = np.arange(1, N + 1) & ((1 << j) - 1)
    a = low[:, None, None]
    b = low[None, :, None]
    c = low[None, None, :]
    ind = ((a != 0) & (b != 0) & (c != 0) & (a != b) & (b != c) & (a != c) & ((a ^ b ^ c) != 0))
    T = A[:, :, None] * A[None, :, :] * A.T[:, None, :]
    return int(T[ind].sum()), int(T[~ind].sum())


def main():
    for n in [int(x) for x in (sys.argv[1:] or ["6", "7", "8"])]:
        S = sign_table_fast(n)
        N = (1 << (n - 1)) - 1
        c1 = n1 = c2 = n2 = 0
        print(f"n={n}")
        for g, byi in seams(n).items():
            ref = 8 * g + 1
            if ref >= 1 << (n - 1):
                continue
            Aref = A_sig_fast(n, ref, S).astype(np.int64)
            for i, W in sorted(byi.items()):
                Aw = A_sig_fast(n, W, S).astype(np.int64)
                c1 += int(t3(Aw) - t3(Aref) != dev_pred(n, W))
                n1 += 1
                if i - 1 in byi:
                    Wp = byi[i - 1]
                    Ap = A_sig_fast(n, Wp, S).astype(np.int64)
                    c2 += int(t3(Aw) - t3(Ap) != dev_pred(n, W) - dev_pred(n, Wp))
                    n2 += 1
                # C3, only where there is something to explain (even parity) and g != 0
                if g and bin(g).count("1") % 2 == 0:
                    j = lsb(W)
                    iw, dw = split_by_independence(Aw, j)
                    ir, dr = split_by_independence(Aref, j)
                    print(f"  C3 W={W:4d} j={j} g={g:3d}: D={t3(Aw)-t3(Aref):9d} "
                          f"(law {dev_pred(n,W):9d}) | independent-class {iw-ir:9d} "
                          f"| dependent-class {dw-dr:9d}")
                    # C4: the proposed bijection a |-> a ^ 8g, at TRIANGLE level
                    s = 8 * g
                    base = 1 << j
                    Ab = A_sig_fast(n, base, S).astype(np.int64)
                    v = np.arange(1, N + 1)
                    vs = v ^ s
                    ok = (vs >= 1) & (vs <= N)
                    idx = np.where(ok)[0]
                    tgt = vs[ok] - 1
                    Mw = Aw[np.ix_(idx, idx)]
                    Mb = Ab[np.ix_(tgt, tgt)]
                    Pw = Mw[:, :, None] * Mw[None, :, :] * Mw.T[:, None, :]
                    Pb = Mb[:, :, None] * Mb[None, :, :] * Mb.T[:, None, :]
                    print(f"     C4 a|->a^{s}: edge mismatches "
                          f"{int(np.count_nonzero(Mw - Mb))}, TRIANGLE-product mismatches "
                          f"{int(np.count_nonzero(Pw - Pb))} of {len(idx)**3}")
        print(f"  C1 per-seam law: {c1} viol / {n1}")
        print(f"  C2 consecutive-j difference inside the fibre: {c2} viol / {n2} pairs")


if __name__ == "__main__":
    main()
