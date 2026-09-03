#!/usr/bin/env python3
"""
Nullity-histogram counting law — exact multiplicities of the canonical
zero-divisor nullity spectrum at levels 4..7 of the Cayley-Dickson tower.

Companion to:
  docs/research/nullity_histogram_law_spec_2026-07-26.md

Solves the open question left in docs/research/routon_zd_spec_2026-07-26.md
("Not an explanation of the odd-part distribution"): WHY the level-7 nullity
histogram is {4:684, 8:504, 12:504, 16..44:336 each, 48:504, 52:504, 56:684,
60:870}.

The law
-------
Let N(m, b, t) be the number of canonical ZD index pairs at level b, born at
level m (xor-label in (2^(m-1), 2^m)), with nullity 2^(b-m+2)*t, t odd,
1 <= t <= 2^(m-3)-1.  Then:

  (i)   N(m, b, t) = 2^(b-m) * N(m, m, t)         (doubling doubles kernel);
  (ii)  N(m, m, t) = 2^(V+1) * c0(m_s - 1),
        where c0(b) = 3*(2b-3)*2^(b-2) + 3 is the invertible-pair census at
        level b, and (V, m_s) come from the 2-adic descent of t: iterate
            (m, t) -> (m - v, (2^(m-3)-1-t) / 2^v),  v = v2(2^(m-3)-1-t),
        until t = 2^(m-3)-1; V is the accumulated valuation, m_s = m - V the
        terminal level.

Equivalently (terminal-level form): the level-b histogram has exactly b-3
distinct multiplicities.  For each terminal level s in {4..b} the
multiplicity
        mu_s = 3 * 2^(b-s+1) * ((2s-5)*2^(s-3) + 1)
is attained by exactly 2^(b-s) distinct nullity values.  At level 7:
336 x 8 values, 504 x 4, 684 x 2, 870 x 1 — the observed histogram.

Derivation chain (all steps verified exactly below, b = 3..7):

  L1 eps-lemma:   S(i,l)*S(j,l) = -1 for every candidate pair {i,j},
                  l = i^j.  Provable by induction on bits from the cds()
                  block recursion (spec section 2); here verified
                  exhaustively.
  L2 L=R lemma:   nullity(L_a) = nullity(R_a) for a = e_i + e_j, via the
                  exact 2-cycle formulas
                  pL(k) = S[i,k]S[j,k]S[i,k^l]S[j,k^l],
                  pR(k) = S[k,i]S[k,j]S[k^l,i]S[k^l,j].
  L3 master native recursion: a native pair {i0, h+j0} at level m
                  (h = 2^(m-1), label h+r, j0 = i0^r) has
                      nullity = h - 2*nu - 4,
                  nu the nullity of {i0,j0} at level m-1 (0 if invertible).
                  From the 2-cycle criterion: p((k0,c)) = -ptilde(k0)
                  generically, and the 8 exceptional k (k0 in {0,r,i0,j0},
                  c in {0,1}) contribute a total correction of 4*eps-4 = -8.
  L4 doubling:    the embedded pair {i,j} and the high pair {h+i,h+j} at
                  level b both have nullity 2*nu (block decomposition of
                  L_a into L/R blocks plus L2).

Counting: every candidate pair P at level b-1 (ZD or not, nullity nu)
yields exactly 2 native ZD pairs at level b with nullity 2^(b-1)-4-2*nu
(L3; positivity is automatic since nu <= 2^(b-2)-4), and every ZD pair
yields exactly 2 lifted pairs (embedded + high) with doubled nullity (L4).
This gives Z_b(nu) = 2*Z_{b-1}(nu/2) + 2*c_{b-1}((2^(b-1)-4-nu)/2), which
unrolls to the descent law above; the census law Z(b) and the odd-part law
of the routon contract follow as corollaries (spec section 3).
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routon_zd_contract import (  # noqa: E402
    exact_nullity_index_pairs,
    fiber_birth_level,
    get_sign_matrix,
    census_law,
)

LEVELS = (3, 4, 5, 6, 7)


def candidate_pairs(b):
    """All index pairs 1 <= i < j < 2^b (canonical 2-unit-sum candidates)."""
    n = 1 << b
    return [(i, j) for i in range(1, n) for j in range(i + 1, n)]


def c0_law(b):
    """Invertible canonical candidate pairs at level b: 3*(2b-3)*2^(b-2)+3."""
    return 3 * (2 * b - 3) * (1 << (b - 2)) + 3


def descent(m, t):
    """2-adic descent of the odd part t at birth level m.

    Iterate (m, t) -> (m - v, (max-t)/2^v) with max = 2^(m-3)-1 and
    v = v2(max-t), until t = max.  Returns (V, m_s): accumulated 2-adic
    valuation and terminal level (m_s = m - V).  Terminates because m
    strictly decreases and the only odd t at m = 4 is t = 1 = max.
    """
    V = 0
    while True:
        mx = (1 << (m - 3)) - 1
        if not (1 <= t <= mx and t % 2 == 1):
            raise ValueError(f"invalid (m={m}, t={t})")
        if t == mx:
            return V, m
        u = mx - t
        v = (u & -u).bit_length() - 1
        V += v
        m -= v
        t = u >> v


def law_N(m, b, t):
    """N(m, b, t): # index pairs at level b, born at m, nullity 2^(b-m+2)*t."""
    V, m_s = descent(m, t)
    return (1 << (b - m + V + 1)) * c0_law(m_s - 1)


def law_histogram(b):
    """Full level-b nullity histogram {nullity: multiplicity} from the law."""
    hist = Counter()
    for m in range(4, b + 1):
        for t in range(1, 1 << (m - 3), 2):
            hist[(1 << (b - m + 2)) * t] += law_N(m, b, t)
    return dict(sorted(hist.items()))


def terminal_multiplicities(b):
    """{multiplicity mu_s: #distinct nullity values} = {mu_s: 2^(b-s)}."""
    out = {}
    for s in range(4, b + 1):
        mu = 3 * (1 << (b - s + 1)) * ((2 * s - 5) * (1 << (s - 3)) + 1)
        out[mu] = 1 << (b - s)
    return dict(sorted(out.items()))


# ------------------------------------------------------------------
# Lemma clauses
# ------------------------------------------------------------------

def check_L1_eps_identity(nullities):
    """S(i,l)S(j,l) = -1 for all 1 <= i < j < 2^b, l = i^j, b = 3..7.

    Equivalently (e_i e_l) e_l = -e_i for distinct imaginary units, the
    basis-element instance of the right alternative law; provable by
    induction on bits from the cds() block recursion (spec section 2).
    """
    ok = True
    for b in LEVELS:
        S = get_sign_matrix(b)
        n_bad = 0
        for (i, j) in candidate_pairs(b):
            l = i ^ j
            if int(S[i, l]) * int(S[j, l]) != -1:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  L1 violation b={b} (i={i}, j={j}, l={l})")
        ok = ok and n_bad == 0
        print(f"  L1 b={b}: {n_bad} violations over "
              f"{len(candidate_pairs(b))} candidate pairs")
    print(f"L1_EPS_IDENTITY -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L2_left_right_nullity():
    """nullity(L_a) = nullity(R_a) for a = e_i + e_j, all candidate pairs.

    Exact 2-cycle counts on both sides (integer arithmetic, no SVD):
    nullity = #{k : p(k) = 1} / 2 with
    pL(k) = S[i,k]S[j,k]S[i,k^l]S[j,k^l] (left) and
    pR(k) = S[k,i]S[k,j]S[k^l,i]S[k^l,j] (right).
    """
    ok = True
    for b in (4, 5, 6, 7):
        n = 1 << b
        S = get_sign_matrix(b).astype(np.int16)
        rows = np.arange(n)
        n_bad = 0
        for (i, j) in candidate_pairs(b):
            l = i ^ j
            pL = S[i] * S[j] * S[i][rows ^ l] * S[j][rows ^ l]
            pR = S[:, i] * S[:, j] * S[rows ^ l][:, i] * S[rows ^ l][:, j]
            if int(np.count_nonzero(pL == 1)) != int(np.count_nonzero(pR == 1)):
                n_bad += 1
                if n_bad <= 3:
                    print(f"  L2 mismatch b={b} (i={i}, j={j})")
        ok = ok and n_bad == 0
        print(f"  L2 b={b}: {n_bad} mismatches over "
              f"{len(candidate_pairs(b))} candidate pairs")
    print(f"L2_LEFT_RIGHT_NULLITY -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L3_native_recursion(nullities):
    """Native pair {i0, h+j0} at level m, label h+r: nullity = h - 2*nu - 4,
    nu = nullity of {i0, j0=i0^r} at level m-1 (0 if invertible)."""
    ok = True
    for m in (4, 5, 6, 7):
        h = 1 << (m - 1)
        n_bad = 0
        for (i, j), nul in nullities[m].items():
            l = i ^ j
            if l < h:
                continue
            r = l - h
            i0, j0 = i, j - h
            assert j0 == (i0 ^ r) and 1 <= i0 < h and i0 != r
            nu = nullities[m - 1].get((min(i0, j0), max(i0, j0)), 0)
            if nul != h - 2 * nu - 4:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  L3 mismatch m={m} ({i},{j}) nu={nu} "
                          f"pred={h - 2 * nu - 4} actual={nul}")
        ok = ok and n_bad == 0
        print(f"  L3 m={m}: {n_bad} mismatches over native pairs")
    print(f"L3_NATIVE_RECURSION nullity=h-2nu-4 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L4_doubling(nullities):
    """Embedded pair {i,j} and high pair {h+i,h+j} at level b both have
    nullity 2*nu, nu the level-(b-1) nullity (0 if invertible)."""
    ok = True
    for b in (4, 5, 6, 7):
        h = 1 << (b - 1)
        bad_e = bad_h = 0
        for (i, j) in candidate_pairs(b - 1):
            nu = nullities[b - 1].get((i, j), 0)
            if nullities[b].get((i, j), 0) != 2 * nu:
                bad_e += 1
            if nullities[b].get((h + i, h + j), 0) != 2 * nu:
                bad_h += 1
        ok = ok and bad_e == 0 and bad_h == 0
        print(f"  L4 b={b}: embedded_mismatch={bad_e} high_mismatch={bad_h}")
    print(f"L4_DOUBLING nullity_b=2*nullity_(b-1) -> {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------
# Counting-law clauses
# ------------------------------------------------------------------

def check_C1_invertible_census(nullities):
    """c0(b) = 3*(2b-3)*2^(b-2) + 3 reproduces the invertible census at
    b = 3..7: 21, 63, 171, 435, 1059."""
    ok = True
    for b in LEVELS:
        inv = len(candidate_pairs(b)) - len(nullities[b])
        if inv != c0_law(b):
            ok = False
            print(f"  C1 mismatch b={b}: invertible={inv} law={c0_law(b)}")
    detail = ",".join(str(c0_law(b)) for b in LEVELS)
    print(f"C1_INVERTIBLE_CENSUS c0(3..7)={detail} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C2_descent_law(nullities):
    """The descent law reproduces the full nullity histogram at b = 4..7,
    both per birth class (N(m,b,t) for every m and odd t) and in total."""
    ok = True
    for b in (4, 5, 6, 7):
        pred = law_histogram(b)
        actual = dict(sorted(Counter(nullities[b].values()).items()))
        if pred != actual:
            ok = False
            print(f"  C2 mismatch b={b}: law={pred} scan={actual}")
        # per birth class as well
        by_class = {}
        for (i, j), nul in nullities[b].items():
            m = fiber_birth_level(i ^ j)
            by_class.setdefault(m, Counter())[nul] += 1
        for m, cnt in by_class.items():
            pred_class = Counter()
            for t in range(1, 1 << (m - 3), 2):
                pred_class[(1 << (b - m + 2)) * t] += law_N(m, b, t)
            if dict(pred_class) != dict(cnt):
                ok = False
                print(f"  C2 class mismatch b={b} m={m}: "
                      f"law={dict(sorted(pred_class.items()))} "
                      f"scan={dict(sorted(cnt.items()))}")
    print(f"C2_DESCENT_LAW N(m,b,t)=2^(b-m+V+1)*c0(m_s-1) exhaustive b=4..7 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C3_terminal_structure(nullities):
    """Terminal-level form: at level b the histogram has exactly b-3
    distinct multiplicities; mu_s = 3*2^(b-s+1)*((2s-5)*2^(s-3)+1) is
    attained by exactly 2^(b-s) distinct nullity values (s = 4..b)."""
    ok = True
    for b in (4, 5, 6, 7):
        mult_hist = dict(sorted(Counter(Counter(nullities[b].values()).values()).items()))
        pred = terminal_multiplicities(b)
        if pred != mult_hist:
            ok = False
            print(f"  C3 mismatch b={b}: law={pred} scan={mult_hist}")
        else:
            print(f"  C3 b={b}: {pred}")
    print(f"C3_TERMINAL_STRUCTURE mu_s x 2^(b-s) values -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C4_l7_headline(nullities):
    """Headline: the level-7 histogram is exactly
    {4:684, 8:504, 12:504, 16..44:336 each, 48:504, 52:504, 56:684, 60:870},
    totaling 6942 index pairs = Z(7)/2 with Z(7) = 13884."""
    expected = {4: 684, 8: 504, 12: 504, 16: 336, 20: 336, 24: 336, 28: 336,
                32: 336, 36: 336, 40: 336, 44: 336, 48: 504, 52: 504,
                56: 684, 60: 870}
    actual = dict(sorted(Counter(nullities[7].values()).items()))
    law = law_histogram(7)
    ok = (actual == expected and law == expected
          and sum(expected.values()) == 6942
          and 2 * sum(expected.values()) == census_law(7))
    print(f"C4_L7_HEADLINE histogram={actual} law_match={law == expected} "
          f"total={sum(actual.values())} -> {'PASS' if ok else 'FAIL'}")
    return ok


# Level-8 reference histogram, tabulated in
# docs/research/l8_zd_census_benchmark_spec_2026-07-26.md section 4
# (exact 2-cycle census, independently audited by full GF(65521) rank
# computations on all 64770 pair-signs, 0 mismatches).
def l8_reference_histogram():
    ref = {4: 1740, 8: 1368, 12: 1368, 120: 1740, 124: 2118}
    for k in range(16, 29, 4):
        ref[k] = 1008
    for k in range(96, 109, 4):
        ref[k] = 1008
    for k in range(32, 93, 4):
        ref[k] = 672
    for k in (112, 116):
        ref[k] = 1368
    return dict(sorted(ref.items()))


def check_C5_l8_out_of_sample():
    """Out-of-sample: the law's level-8 prediction — multiplicities
    mu_s(8) = 2^(9-s)*c0(s-1) = 672, 1008, 1368, 1740, 2118 attained by
    16, 8, 4, 2, 1 distinct nullity values — matches the tabulated L8
    histogram (see l8_reference_histogram provenance).  Set
    NULLITY_LAW_L8_EXACT=1 to additionally re-run the exact 32385-pair
    L8 scan (slow) instead of trusting the tabulation."""
    pred = law_histogram(8)
    ok = pred == l8_reference_histogram() and sum(pred.values()) == 29886 \
        and 2 * sum(pred.values()) == census_law(8)
    term = terminal_multiplicities(8)
    ok = ok and term == {672: 16, 1008: 8, 1368: 4, 1740: 2, 2118: 1}
    exact_note = "tabulated_reference"
    if os.environ.get("NULLITY_LAW_L8_EXACT") == "1":
        scan = dict(sorted(Counter(exact_nullity_index_pairs(8).values()).items()))
        ok = ok and pred == scan
        exact_note = "exact_L8_scan"
    print(f"C5_L8_OUT_OF_SAMPLE law==L8({exact_note}) total={sum(pred.values())} "
          f"terminal={term} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("NULLITY-HISTOGRAM COUNTING LAW (LEVELS 4..7) — contract")
    print("=" * 70)
    nullities = {b: exact_nullity_index_pairs(b) for b in LEVELS}
    results.append(("L1", check_L1_eps_identity(nullities)))
    results.append(("L2", check_L2_left_right_nullity()))
    results.append(("L3", check_L3_native_recursion(nullities)))
    results.append(("L4", check_L4_doubling(nullities)))
    results.append(("C1", check_C1_invertible_census(nullities)))
    results.append(("C2", check_C2_descent_law(nullities)))
    results.append(("C3", check_C3_terminal_structure(nullities)))
    results.append(("C4", check_C4_l7_headline(nullities)))
    results.append(("C5", check_C5_l8_out_of_sample()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"NULLITY_HISTOGRAM_LAW_VERDICT C_GREEN ({passed}/{total} clauses PASS)")
        print("NULLITY_HISTOGRAM_LAW_NOTE multiplicities=3*2^(b-s+1)*((2s-5)*2^(s-3)+1) "
              "x 2^(b-s) values; descent_law; eps_identity; native_recursion; "
              "L8_out_of_sample_confirmed; routon_open_question_resolved")
        return 0
    print(f"NULLITY_HISTOGRAM_LAW_VERDICT C_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
