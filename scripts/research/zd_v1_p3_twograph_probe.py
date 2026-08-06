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


# --- §57.12: attacking the contiguous-block law itself -----------------------------------------
#
# Three mechanisms were tried and the first two are REFUTED.  Recorded so the next attempt does not
# re-spend them.
#
# (a) "each TRIPLE's coherence is +-a character of g, so the support is the set of realised k".
#     REFUTED: 10464 / 39711 triples at m = 5 and 139032 / 333375 at m = 6 have a coherence vector
#     that is not +-a character.
#
# (b) "the ENTRIES are characters and the block structure is inherited".  REFUTED twice over: only
#     13164 / 16002 entries at m = 6 are single characters at all, and those that are REALISE THE
#     NON-BLOCK CHARACTER — k = 5 = 0b101 occurs at 1624 entries.  Block-ness is invisible at the
#     entry level.
#
# (c) what IS true, measured at m = 6 by splitting the triple sum into the two classes of (a):
#     the non-block coefficient vanishes IN EACH CLASS SEPARATELY (k = 5 gets exactly 0 from the
#     character triples and exactly 0 from the rest).  So it is not a conspiracy between the two;
#     any proof has to explain a cancellation that already holds inside each class.  Note also that
#     k = 3, 6, 7 get NOTHING from the character triples — their whole coefficient comes from the
#     non-character ones.
#
# An exact reformulation of the law (algebra, not measurement; checked at m = 5..8, 0 violations):
# writing x_t = (-1)^(bit t of g) and
#
#     R_i = sum_{L>=1} 2^-(L-1) * x_i x_(i+1) ... x_(i+L-1),   equivalently   R_i = x_i (1 + R_(i+1)/2)
#
# the two halves of the law -- interval support AND the halving -- are together equivalent to
#
#     tri3(P3~)(8g+1) = w_0 + sum_i s_i * R_i(g)
#
# i.e. the whole g-dependence enters through b nested dyadic quantities, one per bit position, each
# an affine function of the binary fraction whose digits are the PREFIX PARITIES of g from that
# position on.  That is what a proof would have to produce.

def check_reformulation(m):
    """`tri3 = w_0 + sum_i s_i R_i` and the recursion for `R_i`.  Returns (violations, cases)."""
    b = m - 3
    w = coefficients(m)
    vals = predict(m)
    s = [w[1 << i] for i in range(b)]
    bad = 0
    for g in range(1 << b):
        x = [(-1)**((g >> t) & 1) for t in range(b)]
        R = [0.0] * (b + 1)
        for i in range(b - 1, -1, -1):
            R[i] = x[i] * (1 + R[i + 1] / 2)
        if abs(w[0] + sum(s[i] * R[i] for i in range(b)) - vals[g]) > 1e-6:
            bad += 1
    return bad, 1 << b


# --- §57.13: the IDENTIFICATION check (both standard families EXCLUDED) --------------------------
#
# (1) NOT a bilinear-form two-graph.  If P3~ were (-1)^(B(l,y)+f(l)+f(y)) with B F2-bilinear, then
#     switching away f and taking the F2 log would leave a matrix of rank <= m+1.  Measured
#     rank_F2 = 2^m - 4 exactly -- 4, 12, 28, 60, 124 at m = 3..7 against the bound 4, 5, 6, 7, 8 --
#     at EVERY label of m = 3,4,5 and six labels of m = 6,7.  The maximal seam has rank 0, which is
#     consistent: its two-graph is empty (Tier 65).  m = 3 is the coincidence level where
#     2^m - 4 = m + 1, i.e. the level at which this test cannot distinguish anything.
#
# (2) NOT a regular two-graph.  The descendant graph is regular but not STRONGLY regular: at m = 5
#     lambda takes 12,14,16,18,20,... and mu takes 0,2,4,6,8,...
#
# Neither result establishes novelty -- the two-graph catalogues have not been searched -- but they
# exclude the two identifications that would have made tri3(P3~) a known computation, and they hand
# over a clean invariant (rank_F2 = 2^m - 4, zero at the seam) to search WITH.

def f2rank(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i, c]), None)
        if p is None:
            continue
        if p != r:
            M[[r, p]] = M[[p, r]]
        sel = (M[:, c] == 1)
        sel[r] = False
        M[sel] = (M[sel] + M[r]) % 2
        r += 1
        if r == rows:
            break
    return r


def switched_log(m, W, root=1):
    """`P3~` switched so the root row is all `+1`, as an F2 matrix (1 where the entry is -1)."""
    V = 1 << (m + 1)
    S = np.array([[_P3(l, y, W, m) for y in range(1, V)] for l in range(1, V)], dtype=np.int64)
    np.fill_diagonal(S, 0)
    r = root - 1
    eps = S[r].copy()
    eps[r] = 1
    Sp = S * np.outer(eps, eps)
    np.fill_diagonal(Sp, 1)
    return ((1 - Sp) // 2).astype(np.int64)


# --- §57.16: the level-transfer orthants ---------------------------------------------------------
#
# Tier 66/67 give the four blocks of the level transfer.  Summing triples by which half each vertex
# lands in gives eight orthants; the sum depends only on the WEIGHT of (lambda_a, lambda_b, lambda_c)
# (§57.14), so there are four numbers O_0..O_3 and tri3_(m+1) = O_0 + 3 O_1 + 3 O_2 + O_3.
#
#   O_0 = tri3(P3~)_m                                       (proved-shaped: Tier 66's block)
#   O_1 - O_2 = 26*2^m - 64      | every label EXCEPT the maximal seam, where both
#   O_3 - O_0 = 54*2^m - 90      | shift by +288*[m-1,2]_2 -- the SAME constant as §57.10's
#                                  maximal-seam excess in the deviation law
#
# found on m = 3,4,5 and confirmed OUT OF SAMPLE at m = 6 (c1 = 1600, c3 = 3366 at three labels; the
# seam shift 44640 = 288*[5,2]_2 for both).  Substituting:
#
#   tri3(P3~)_(m+1) = 2 * tri3(P3~)_m + 6 * O_1 - 24*2^m + 102        (off the seam, 0 viol m=4,5,6)
#
# -- one unknown left, and 2 + 6 = 8 recovers the heuristic factor with the weights the refutation
# in §57.14 forced.  At the fibre references O_1 follows from the closed form at both levels:
#
#   O_1(8g+1) = [ T_(m+1)(g) - 2 T_m(g) + 24*2^m - 102 ] / 6
#
# which is bookkeeping on top of measured closed forms, not independent evidence.

def transfer_constants(m):
    """The two level constants of §57.16, off the maximal seam."""
    return 26 * 2**m - 64, 54 * 2**m - 90


def O1_at_reference(m, g):
    """`O_1` at the fibre reference `W = 8g+1`, from the closed form at both levels."""
    c1, c3 = transfer_constants(m)
    t_m = predict(m)[g % (1 << (m - 3))]
    t_m1 = predict(m + 1)[g % (1 << (m - 2))]
    return (t_m1 - 2 * t_m + 24 * 2**m - 102) // 6


# --- §57.17: O_1 DERIVED from the block identities -----------------------------------------------
#
# The honest route §57.16 owed.  Take the weight-1 orthant (lambda_a, lambda_b, lambda_c) = (0,0,1):
# `a`, `b` low, `c = c0 + H` high.  Tiers 66/67 give each of the three factors:
#
#   M(a,b)  = block (0,0) = S(a,b)            the level-m Seidel matrix (same masking)
#   M(b,c)  = block (0,1) = eps01 * P(b,c0)   eps01 = -1 iff b = W or b^c0 = W
#   M(c,a)  = block (1,0) = eps10 * P(c0,a)   eps10 = -1 iff a = W or c0^a = W
#
# where P is the level-m P3 with index 0 zeroed but the DIAGONAL KEPT (P(x,x) = -1, `P3_diag`) --
# because `b = c0` and `c0 = a` are legal at level m+1, the two indices being in different halves.
# `c0 = 0` is legal too (the index H), and Tier 67's lemmas exclude it, so it is a separate slice.
# Expanding eps01*eps10 = 1 - 2u - 2v + 4uv:
#
#   O_1 = S0 - 2 Su - 2 Sv + 4 Suv + slice0        (verified for every label, m = 3,4)
#
# and each piece is then identified:
#
#   S0  = sum S(a,b) P(b,c) P(c,a) = tri3_m - 2 tr(S^2) = tri3_m - 2 N(N-1)     [P = S - I_1]
#   Su  = Sv, and Su splits as {b = W} + {b^c = W} with EMPTY overlap (the overlap forces c = 0);
#         its ISOLATED-ROW half is itself a level constant 10*2^m - 22 (58, 138, 298 at m = 3,4,5;
#         154, 810, 3658 at the seam), so ALL the label dependence sits in the COSET-LINE term
#             Sigma_coset(W) = sum_{a,b} S(a,b) P(b, b^W) P(b^W, a)
#   Suv = 2(N-1)             -- a level constant
#   slice0 = 18*2^m - 30 off the maximal seam, and N(N-1) AT it
#
# giving, off the seam,
#
#   O_1 = tri3_m - 4*Su + 2(N-1)(4-N) + 18*2^m - 30          0 violations, every label, m = 3,4
#
# so the single remaining unknown is Su -- a SECOND-order sum, over the isolated row and the coset
# line, the two loci this lane already has theorems about.  Suv, slice0 and the seam value were all
# confirmed OUT OF SAMPLE at m = 5.
#
# N = 2^(m+1) - 1 throughout.

def O1_pieces(m):
    """The level constants of the O_1 derivation: (Suv, slice0 off the seam, slice0 at the seam)."""
    N = (1 << (m + 1)) - 1
    return 2 * (N - 1), 18 * 2**m - 30, N * (N - 1)


def O1_from_Su(m, tri3_m, Su):
    """`O_1` from the level-m triangle count and the second-order sum `Su`, off the maximal seam."""
    N = (1 << (m + 1)) - 1
    return tri3_m - 4 * Su + 2 * (N - 1) * (4 - N) + 18 * 2**m - 30


# --- §57.18: Sigma_coset in closed form, and the transfer recursion COMPLETES --------------------
#
# Sigma_coset(W) = sum_{a,b} S(a,b) P(b, b^W) P(b^W, a) -- the last label-dependent piece of O_1.
#
# It is CONSTANT ON EACH g-FIBRE (every label, m = 3,4,5), and its Walsh expansion in g -- now over
# b = m-2 bits, with NO top-bit identification -- is far sparser than tri3's:
#
#   nonzero exactly at the b blocks ANCHORED AT THE TOP BIT, k_L = 2^b - 2^(b-L), L = 1..b
#   w[k_L] = 24*2^(m+L-1) - 96*4^(L-1)
#   w[0]   = 12*2^m - 24
#
# so b+1 coefficients instead of tri3's b(b+1)/2, and lengthening a top-anchored block MULTIPLIES by
# 4 rather than dividing by 2.  Found on m = 3,4,5; confirmed OUT OF SAMPLE at m = 6 (0 mismatches /
# 16 references) and m = 7 (0 / 32).
#
# Substituting into §57.17's O_1 and §57.16's assembly, the whole transfer collapses to
#
#   tri3(P3~)_(m+1) = 8 * tri3(P3~)_m - 24 * Sigma_coset(W) + c(m)
#   c(m) = -156*2^m + 450 + 12 (N-1)(4-N),   N = 2^(m+1)-1
#
# verified for EVERY non-seam label at m = 3,4,5 (14/14, 30/30, 62/62).  ** The factor 8 is back **:
# the heuristic "each level-m triangle lifts 2^3 ways" survives after all, but only once the defect
# is identified -- it is exactly -24 times the coset-line sum, plus a level constant.  That also
# retro-explains §57.11's s_i(m+1) = 8 s_i(m): it IS this 8.
#
# NOTE none of §57.18 is formalised.  The block identities it rests on (Tiers 66/67) are theorems;
# the constants and the Sigma_coset closed form are measured.

def sigma_coset_coefficients(m):
    """Walsh coefficients of `g -> Sigma_coset(8g+1)`, over `b = m-2` bits."""
    b = m - 2
    w = [0] * (1 << b)
    w[0] = 12 * 2**m - 24
    for L in range(1, b + 1):
        w[(1 << b) - (1 << (b - L))] = 24 * 2**(m + L - 1) - 96 * 4**(L - 1)
    return w


def sigma_coset(m, g):
    w = sigma_coset_coefficients(m)
    return sum(w[k] * (-1)**(bin(g & k).count("1")) for k in range(1 << (m - 2)))


def transfer_constant(m):
    """`c(m)` in `tri3_(m+1) = 8 tri3_m - 24 Sigma_coset + c(m)`, off the maximal seam."""
    N = (1 << (m + 1)) - 1
    return -156 * 2**m + 450 + 12 * (N - 1) * (4 - N)
