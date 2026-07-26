#!/usr/bin/env python3
"""
Cayley-Dickson tower nullity histogram law — contract.

Companion to:
  docs/research/cd_tower_nullity_histogram_law_spec_2026-07-26.md

Builds on the level-7 routon scan (scripts/research/routon_zd_contract.py,
docs/research/routon_zd_spec_2026-07-26.md), which produced the exact nullity
spectrum of canonical zero divisors at levels 4..7 and left the histogram
multiplicities {4:684, 8:504, 12:504, 16..44:336, 48:504, 52:504, 56:684,
60:870} as an open combinatorial question.

This contract states and verifies the law that EXPLAINS those multiplicities:

  * Fiber type invariant.  For a ZD label ell = 2^(m-1) + r (m = birth level,
    1 <= r <= 2^(m-1) - 1), define the FIBER TYPE

        tau(r) = (r - lowbit(r)) >> 3,   lowbit(r) = r & (-r).

    The nullity histogram of the fiber of ell at every level b >= m depends
    only on (m, tau(r), b) — verified exhaustively at b = 4..8 for all
    7+22+53+116+243 = 441 fibers.

  * Label count law.  The number of labels born at level m of type tau is
    A(m, tau) = m + 3 (tau = 0) and A(m, tau) = 7 + v2(tau) (tau >= 1),
    where v2 is the 2-adic valuation.  Note: independent of m for tau >= 1.

  * Fiber histogram recursion (exact).  At birth level m a fiber has
    2^(m-1) - 2 index pairs; writing nullity = 8u + 4 (u = (t-1)/2 with
    t the odd part of the nullity law), the histogram c_m(tau, u) obeys
        base (m = 4):            c_4(0, 0) = 6
        old type tau <= M':      c_m(tau, 2u+1) = 2 c_{m-1}(tau, u),
                                 top class c_m(tau, M) += 2
        new type tau = M'+1+tau': c_m(tau, (M-1)-2u) = 2 c_{m-1}(tau', u),
                                 c_m(tau, M) = 2
    with M = 2^(m-4) - 1, M' = 2^(m-5) - 1.  Lifting from birth level m to
    level b scales nullities and counts by exactly 2^(b-m).

  * Aggregate generation law (the headline).  At level b the nullity
    histogram consists of exactly 2^(b-k) distinct nullity values, each
    occurring with multiplicity

        6 * 2^(b-k) * f(k),   f(k) = (2k - 5) * 2^(k-3) + 1,

    for each generation k = 4..b.  The L7 multiplicities are therefore
    336 = 6*2^3*f(4), 504 = 6*2^2*f(5), 684 = 6*2*f(6), 870 = 6*f(7).

  * Census identity.  The histogram law sums to the growth law of the
    routon contract:  12 * sum_{k=4}^b 4^(b-k) f(k) = Z(b) identically,
    so the multiplicity law yields an independent derivation of the census
    law Z(b) = 4^b - (3b-1)*2^b + 2^(b-1) - 4.

  * Level-8 falsification test (out of sample).  The law predicts the
    full L8 histogram, including the census Z(8) = 59772 triples (29886
    index pairs), 243 fibers, 31 distinct nullities, max nullity 124, and
    multiplicities {672:16, 1008:8, 1368:4, 1740:2, 2118:1}.  The exact
    2-cycle scan at b = 8 confirms every clause.

Method: exact integer arithmetic only (the 2-cycle criterion of the routon
contract); no floating point, no SVD.
"""

from collections import Counter, defaultdict

import numpy as np

np.seterr(all='ignore')


# ------------------------------------------------------------------
# Exact 2-cycle scan (integer arithmetic; cf. routon_zd_contract.py)
# ------------------------------------------------------------------

def cds(a, b, bits):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah = a >= h
        bh = b >= h
        al = a & (h - 1)
        bl = b & (h - 1)
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


_SIGN_TABLES = {}


def get_sign_matrix(bits):
    if bits not in _SIGN_TABLES:
        n = 1 << bits
        S = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                S[i, j] = cds(i, j, bits)
        _SIGN_TABLES[bits] = S
    return _SIGN_TABLES[bits]


def exact_nullity_index_pairs(bits):
    """All (i, j), 1 <= i < j < 2^bits, with a = e_i +/- e_j a zero divisor,
    mapped to the exact nullity of L_a (identical for both signs).

    Vectorized over j for each i.  p(k) = S[i,k]*S[j,k]*S[i,k^l]*S[j,k^l];
    nullity = #{k : p(k) = 1} / 2.
    """
    n = 1 << bits
    S = get_sign_matrix(bits).astype(np.int16)
    rows = np.arange(n)
    out = {}
    for i in range(1, n):
        Si = S[i]
        J = np.arange(i + 1, n)
        if len(J) == 0:
            continue
        L = J ^ i
        K = rows[None, :] ^ L[:, None]          # (len(J), n): k ^ l per pair
        SJ = S[J]                               # (len(J), n): S[j, k]
        # p(k) = S[i,k] * S[j,k] * S[i,k^l] * S[j,k^l]
        p = Si[None, :] * SJ * Si[K] * np.take_along_axis(SJ, K, axis=1)
        bad = np.count_nonzero(p == 1, axis=1) // 2
        for idx, b in enumerate(bad):
            if b > 0:
                out[(i, int(J[idx]))] = int(b)
    return out


# ------------------------------------------------------------------
# The histogram law
# ------------------------------------------------------------------

def lowbit(r):
    return r & (-r)


def fiber_type(r):
    """Fiber type invariant: tau(r) = (r - lowbit(r)) >> 3."""
    return (r - lowbit(r)) >> 3


def v2(x):
    v = 0
    while x % 2 == 0:
        x //= 2
        v += 1
    return v


def label_count_law(m, tau):
    """A(m, tau): number of ZD labels born at level m of fiber type tau."""
    return m + 3 if tau == 0 else 7 + v2(tau)


def generation_f(k):
    """f(k) = (2k - 5) * 2^(k-3) + 1: 7, 21, 57, 145, 353, ..."""
    return (2 * k - 5) * (1 << (k - 3)) + 1


def build_fiber_law(mmax):
    """U[m][tau] = {u: count} at birth level m; nullity at birth = 8u + 4.

    Recursion (see module docstring): old types lift u -> 2u+1 with the top
    class gaining 2 extra pairs; new types tau = 2^(m-5) + tau' reflect
    u -> (M-1) - 2u with a fresh top class of 2.
    """
    U = {4: {0: {0: 6}}}
    for m in range(5, mmax + 1):
        M = (1 << (m - 4)) - 1
        Mp = (1 << (m - 5)) - 1
        Um = {}
        for tau in range(0, M + 1):
            if tau <= Mp:
                d = {}
                for u, c in U[m - 1][tau].items():
                    d[2 * u + 1] = 2 * c
                d[M] += 2
                Um[tau] = d
            else:
                tp = tau - (1 << (m - 5))
                d = {(M - 1) - 2 * u: 2 * c for u, c in U[m - 1][tp].items()}
                d[M] = 2
                Um[tau] = d
        U[m] = Um
    return U


def predicted_fiber_hist(U, m, tau, b):
    """Predicted nullity -> count for a fiber of type tau born at m, at
    level b >= m: nullities and counts scale by exactly 2^(b-m)."""
    scale = 1 << (b - m)
    return {(8 * u + 4) * scale: c * scale for u, c in U[m][tau].items()}


def predicted_aggregate_hist(U, b):
    """Predicted full nullity histogram at level b (index pairs)."""
    hist = Counter()
    for m in range(4, b + 1):
        for tau in range(0, 1 << (m - 4)):
            for _ in range(label_count_law(m, tau)):
                for nullity, c in predicted_fiber_hist(U, m, tau, b).items():
                    hist[nullity] += c
    return dict(hist)


def census_law(b):
    return 4 ** b - (3 * b - 1) * (2 ** b) + 2 ** (b - 1) - 4


# ------------------------------------------------------------------
# Contract clauses
# ------------------------------------------------------------------

def check_C1_type_invariant(scans, U):
    """Fiber histograms depend only on (m, tau, b); lifting scales by
    exactly 2^(b-m).  Verified for every fiber at b = 4..8."""
    ok = True
    n_fibers = 0
    for b, pairs in scans.items():
        per_label = defaultdict(Counter)
        for (i, j), v in pairs.items():
            per_label[i ^ j][v] += 1
        seen_types = {}
        for ell, hist in per_label.items():
            m = (ell).bit_length()
            r = ell - (1 << (m - 1))
            tau = fiber_type(r)
            n_fibers += 1
            key = (m, tau)
            h = dict(hist)
            if key in seen_types and seen_types[key] != h:
                ok = False
                print(f"  C1 invariant fail b={b} ell={ell} (m={m},tau={tau})")
            seen_types[key] = h
            # lifting: hist at b equals birth hist scaled by 2^(b-m)
            birth = predicted_fiber_hist(U, m, tau, m)
            scaled = {v * (1 << (b - m)): c * (1 << (b - m))
                      for v, c in birth.items()}
            if h != scaled:
                ok = False
                print(f"  C1 lifting fail b={b} ell={ell} (m={m},tau={tau})")
    print(f"C1_TYPE_INVARIANT fibers={n_fibers} (m,tau,b)-reduction + "
          f"2^(b-m) lifting -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C2_label_count_law(scans):
    """A(m, tau) = m+3 (tau=0) resp. 7 + v2(tau) (tau>=1), m = 4..8."""
    ok = True
    for b, pairs in scans.items():
        born = set()
        for (i, j) in pairs:
            ell = i ^ j
            if ell.bit_length() == b:
                born.add(ell)
        acnt = Counter(fiber_type(ell - (1 << (b - 1))) for ell in born)
        for tau in range(0, 1 << (b - 4)):
            if acnt.get(tau, 0) != label_count_law(b, tau):
                ok = False
                print(f"  C2 fail b={b} tau={tau}: exact={acnt.get(tau, 0)} "
                      f"law={label_count_law(b, tau)}")
    detail = " ".join(f"A({m},0)={m+3}" for m in (4, 5, 6, 7, 8))
    print(f"C2_LABEL_COUNT_LAW A(m,tau)=7+v2(tau) (tau>=1), {detail} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C3_fiber_recursion(scans, U):
    """The recursion-predicted per-fiber histogram equals the exact scan
    for every fiber at b = 4..8."""
    ok = True
    n_fibers = 0
    for b, pairs in scans.items():
        per_label = defaultdict(Counter)
        for (i, j), v in pairs.items():
            per_label[i ^ j][v] += 1
        for ell, hist in per_label.items():
            m = (ell).bit_length()
            r = ell - (1 << (m - 1))
            pred = predicted_fiber_hist(U, m, fiber_type(r), b)
            n_fibers += 1
            if dict(hist) != pred:
                ok = False
                print(f"  C3 fail b={b} ell={ell}: exact={dict(sorted(hist.items()))} "
                      f"pred={pred}")
    print(f"C3_FIBER_HISTOGRAM_RECURSION fibers={n_fibers} at b=4..8 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C4_aggregate_generation_law(scans, U):
    """Headline: at level b the histogram has exactly 2^(b-k) distinct
    nullity values of multiplicity 6*2^(b-k)*f(k) each, k = 4..b.
    Checked value-by-value (full histogram equality) and as a multiset of
    multiplicities, at b = 4..8."""
    ok = True
    for b, pairs in scans.items():
        exact = Counter(pairs.values())
        pred = predicted_aggregate_hist(U, b)
        if dict(exact) != pred:
            ok = False
            print(f"  C4 full-histogram fail at b={b}")
        mult_exact = Counter(exact.values())
        mult_law = Counter()
        for k in range(4, b + 1):
            mult_law[6 * (1 << (b - k)) * generation_f(k)] += (1 << (b - k))
        if mult_exact != mult_law:
            ok = False
            print(f"  C4 multiset fail at b={b}: exact={dict(sorted(mult_exact.items()))} "
                  f"law={dict(sorted(mult_law.items()))}")
        if b == 7:
            print(f"  C4 L7 multiplicities: {dict(sorted(mult_exact.items()))} "
                  f"(336=6*8*f(4), 504=6*4*f(5), 684=6*2*f(6), 870=6*f(7))")
    print(f"C4_AGGREGATE_GENERATION_LAW mult=6*2^(b-k)*f(k), "
          f"f(k)=(2k-5)*2^(k-3)+1, b=4..8 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C5_l8_falsification(scans):
    """Out-of-sample test at level 8 (predicted, not fitted):
    census Z(8) = 59772 triples, F(8) = 243 fibers, 31 distinct nullities,
    max nullity 124 = 2^7 - 4."""
    pairs = scans[8]
    triples = 2 * len(pairs)
    hist = Counter(pairs.values())
    labels = {i ^ j for (i, j) in pairs}
    c1 = triples == 59772 == census_law(8)
    c2 = len(labels) == 243
    c3 = len(hist) == 31
    c4 = max(hist) == 124
    ok = c1 and c2 and c3 and c4
    print(f"C5_L8_FALSIFICATION Z(8)={triples}/59772 fibers={len(labels)}/243 "
          f"distinct={len(hist)}/31 max={max(hist)}/124 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C6_extremal_corollaries(scans):
    """Max nullity 2^(b-1)-4 has multiplicity 6*f(b); min nullity 4 has
    multiplicity 12*f(b-1) (b >= 5; at b=4 the min IS the max, generation
    k=4, multiplicity 6*f(4)=42); #distinct nullities = 2^(b-3) - 1."""
    ok = True
    for b, pairs in scans.items():
        hist = Counter(pairs.values())
        if hist[(1 << (b - 1)) - 4] != 6 * generation_f(b):
            ok = False
            print(f"  C6 max fail b={b}")
        if b >= 5 and hist[4] != 12 * generation_f(b - 1):
            ok = False
            print(f"  C6 min fail b={b}: exact={hist[4]} "
                  f"law={12 * generation_f(b - 1)}")
        if len(hist) != (1 << (b - 3)) - 1:
            ok = False
            print(f"  C6 distinct-count fail b={b}")
    print(f"C6_EXTREMAL_COROLLARIES max_mult=6*f(b), min_mult=12*f(b-1), "
          f"distinct=2^(b-3)-1, b=4..8 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C7_census_identity():
    """The histogram law sums to the census law:
    12 * sum_{k=4}^b 4^(b-k) f(k) = Z(b) = 4^b - (3b-1)*2^b + 2^(b-1) - 4.
    Verified as an integer identity for b = 4..16 (both sides closed form;
    induction step Z(b) - 4*Z(b-1) = 12*f(b) is exact)."""
    ok = True
    for b in range(4, 17):
        hist_sum = 12 * sum(4 ** (b - k) * generation_f(k)
                            for k in range(4, b + 1))
        if hist_sum != census_law(b):
            ok = False
            print(f"  C7 identity fail at b={b}: {hist_sum} != {census_law(b)}")
    print(f"C7_CENSUS_IDENTITY 12*sum 4^(b-k)f(k) = Z(b) for b=4..16 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 70)
    print("CD-TOWER NULLITY HISTOGRAM LAW — contract")
    print("=" * 70)
    U = build_fiber_law(8)
    # internal consistency: every predicted fiber has 2^(m-1)-2 pairs
    for m in range(4, 9):
        for tau in range(0, 1 << (m - 4)):
            assert sum(U[m][tau].values()) == (1 << (m - 1)) - 2
    scans = {b: exact_nullity_index_pairs(b) for b in (4, 5, 6, 7, 8)}
    results = []
    results.append(("C1", check_C1_type_invariant(scans, U)))
    results.append(("C2", check_C2_label_count_law(scans)))
    results.append(("C3", check_C3_fiber_recursion(scans, U)))
    results.append(("C4", check_C4_aggregate_generation_law(scans, U)))
    results.append(("C5", check_C5_l8_falsification(scans)))
    results.append(("C6", check_C6_extremal_corollaries(scans)))
    results.append(("C7", check_C7_census_identity()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"CD_HISTOGRAM_LAW_VERDICT C_GREEN ({passed}/{total} clauses PASS)")
        print("CD_HISTOGRAM_LAW_NOTE fiber_type_tau; label_count_7+v2; "
              "recursion_exact_b4..8; generation_law_mult=6*2^(b-k)*f(k); "
              "L8_falsification_survived; census_identity; novel_theorem")
        return 0
    print(f"CD_HISTOGRAM_LAW_VERDICT C_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
