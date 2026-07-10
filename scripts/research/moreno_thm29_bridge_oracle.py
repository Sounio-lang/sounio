#!/usr/bin/env python3
"""Oracle: the Moreno seam-index bridge, verified at dim 16 / 32 / 64.

Settles the "unchecked Moreno bridge" caveat of §4 of the Paper-3 manuscript
(docs/papers/cd-tower-seam-obstruction.md).  For every lower x upper pair (l, u) of a
Cayley-Dickson algebra A_n (1 <= l < H = 2^(n-1) <= u < 2^n) it certifies, with EXACT integer
arithmetic (Bareiss determinants), that our zero-divisor predicate coincides with Moreno's
spectral criterion -- in BOTH its literal lower-level form and an equivalent direct form.

Moreno (q-alg/9710013, Chapter II) studies a zero divisor (a,b) in A_{n} written via the doubling
A_n = A_{n-1} x A_{n-1}, with a, b norm-1 imaginary in A_{n-1}, and derives (his eq. after Thm 2.7):

      L^2_{a+b}(y) = (a+b)[(a+b)y] = -2 y        i.e.   -2 in spec(L^2_{a+b}),

where a+b lives in the LOWER algebra A_{n-1}.  Under the standard doubling our element
e_l + e_u  in  A_n  (loHi:  l < H <= u)  is exactly Moreno's (a, b) = (e_l, e_{u'}) with
u' = u - H, so his a+b is the lower sum  e_l + e_{u'}  in  A_{n-1}  (a 2^(n-1)-dim operator), and
his special-couple hypotheses (both imaginary, linearly independent) require u' != 0 and u' != l --
which are precisely the on-seam pairs (u = H, u = H+l), correctly the non-zero-divisors.

  LOWER (Moreno's literal form, A_{n-1}):   -2 in spec(L^2_{e_l + e_{u'}}),  u' = u - H.

There is ALSO an equivalent DIRECT form at A_n, obtained via L_i^2 = -I (proved forall-n as
SounioCDCocycle.cocycle_bundle / cdSigma_cocycle):

      L^2_{e_l+e_u} = L_l^2 + L_u^2 + {L_l, L_u} = -2 I + {L_l, L_u},
  UPPER (direct form, A_n):                 -4 in spec(L^2_{e_l+e_u})  <=>  det(L^2_{e_l+e_u}+4I)=0.

The two spectral forms sit one level apart (the -2 vs -4 is the DOUBLING scaling of the operator,
NOT a normalization of the vector -- both e_l+e_{u'} and e_l+e_u have norm^2 = 2).  Both agree with
each other and with the textbook ZD definition and with our combinatorial predicate.  Certified over
EVERY lower x upper pair at dims 16/32/64 (0 mismatches):

  our 2-term isZD  ==  ker L_{e_l+e_u} != 0 (textbook ZD def)
                   ==  Moreno LOWER (-2 in spec L^2_{e_l+e_{u'}} at A_{n-1}, seam pairs excluded)
                   ==  Moreno UPPER (-4 in spec L^2_{e_l+e_u} at A_n).

So the earlier caveat's LEVEL placement (A_{n-1}) was correct; what was unchecked is now checked, and
the equivalent A_n form additionally exhibits Moreno's operator as -2 I + our own anticommutator.

HONEST STATUS: a *verified correspondence* at dims 16, 32, 64 -- it upgrades the caveat from
"unchecked" to "checked".  It is NOT a forall-n Lean theorem: the spectral/determinant half lives
outside the Mathlib-free combinatorial encoding.  What IS proved forall-n (in SounioCDConverse.lean)
is the combinatorial leg (isZD = hasXorAnnih, anti0 = not isZD on the full box); this oracle certifies
that that leg is Moreno's Thm 2.9 criterion, so the Moreno/BDI forall-n result and ours are the same
statement, not merely parallel.
"""


def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def bareiss_det(M):
    """Exact integer determinant (fraction-free Bareiss elimination)."""
    A = [row[:] for row in M]
    n = len(A)
    prev = 1
    sign = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            aik = A[i][k]
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - aik * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def _Lsum(idxs, bits):
    """Left-multiplication matrix of sum_i e_{idxs[i]} in A_bits."""
    N = 1 << bits
    M = [[0] * N for _ in range(N)]
    for i in idxs:
        for k in range(N):
            M[i ^ k][k] += cd_sigma(i, k, bits)
    return M


def _mm(A, B):
    n = len(A)
    return [[sum(A[r][t] * B[t][c] for t in range(n)) for c in range(n)] for r in range(n)]


def analyze(bits, verbose=True):
    N = 1 << bits
    H = N // 2
    lb = bits - 1  # lower level A_{n-1}

    Lfull = [_Lsum([i], bits) for i in range(N)]

    def madd(A, B, s=1):
        n = len(A)
        return [[A[r][c] + s * B[r][c] for c in range(n)] for r in range(n)]

    def is_zd_comb(l, u):
        """Our 2-term combinatorial isZD (proved forall-n = not anti0 = hasXorAnnih)."""
        for a in range(1, N):
            for b in range(a + 1, N):
                for s in (1, -1):
                    if all(((cd_sigma(l, a, bits) if (l ^ a) == k else 0)
                            + (s * cd_sigma(l, b, bits) if (l ^ b) == k else 0)
                            + (cd_sigma(u, a, bits) if (u ^ a) == k else 0)
                            + (s * cd_sigma(u, b, bits) if (u ^ b) == k else 0)) == 0
                           for k in range(N)):
                        return True
        return False

    pairs = [(l, u) for l in range(1, H) for u in range(H, N)]
    identity_ok = True
    n_mismatch = 0
    for (l, u) in pairs:
        up = u - H
        comb = is_zd_comb(l, u)                          # our combinatorial 2-term isZD
        Lx = madd(Lfull[l], Lfull[u], 1)                 # L_{e_l+e_u}  at A_n
        ker = bareiss_det(Lx) == 0                       # ker L != 0  (textbook ZD def)
        # identity  L^2_{e_l+e_u} == -2 I + {L_l, L_u}
        Lsq = _mm(Lx, Lx)
        anti = madd(_mm(Lfull[l], Lfull[u]), _mm(Lfull[u], Lfull[l]), 1)
        diff = madd(Lsq, anti, -1)
        for r in range(N):
            diff[r][r] += 2
        if any(diff[r][c] != 0 for r in range(N) for c in range(N)):
            identity_ok = False
        # UPPER (direct A_n form): -4 in spec(L^2_{e_l+e_u})
        Lsq4 = [row[:] for row in Lsq]
        for r in range(N):
            Lsq4[r][r] += 4
        upper = bareiss_det(Lsq4) == 0
        # LOWER (Moreno literal, A_{n-1}): -2 in spec(L^2_{e_l+e_{u'}}) on special couples
        if up == 0 or up == l:                           # hypotheses fail => not a ZD couple
            lower = False
        else:
            Q = _Lsum([l, up], lb)
            Qsq = _mm(Q, Q)
            for r in range(1 << lb):
                Qsq[r][r] += 2
            lower = bareiss_det(Qsq) == 0
        if not (comb == ker == upper == lower):
            n_mismatch += 1
    if verbose:
        print(f"bits={bits} (dim {N}, {len(pairs)} lower x upper pairs):")
        print(f"  identity  L^2_(e_l+e_u) == -2I + {{L_l,L_u}}                     : {'OK' if identity_ok else 'FAIL'}")
        print(f"  isZD == ker L != 0 == Moreno-lower(-2@A_(n-1)) == upper(-4@A_n)  mismatches : {n_mismatch}")
    return identity_ok and n_mismatch == 0


if __name__ == "__main__":
    ok = True
    for bits in (4, 5, 6):
        ok = analyze(bits) and ok
    print()
    print("MORENO seam-index BRIDGE:", "CERTIFIED (dims 16/32/64)" if ok else "FAILED")
