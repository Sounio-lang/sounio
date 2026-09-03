#!/usr/bin/env python3
"""Counting analysis reducing the CD-tower converse (off-seam ⟹ zero-divisor) to a clean
autocorrelation bound on the sign cocycle, with the inductive levers identified.

Context. `cd_tower_converse_probe.py` verifies off-seam ⟹ hasXorAnnih empirically to n≤10.
`SounioCDConverse.lean` proves the reduction `hasXorAnnih ⟹ isZD` and the (now unconditional)
doubling recursion `P_(l,u)(a) = -P_(l,u_lo)(a)`. This script pins the REMAINING existence
argument down to a counting statement `Q` and exhibits the structure a proof of `Q` would use.

Definitions (k = level, N = 2^k, indices in [0,N)):
  f_(l,m)(a) = cdSigma(l,a,k) * cdSigma(m,a,k)          ∈ {±1}
  P_(l,m)(a) = f(a) * f(a⊕d),   d = l⊕m                  (constant on the orbit {a, a⊕d})
  An orbit "agrees" if P=+1, "disagrees" if P=-1.
  k_d(l,m) = #disagreeing orbits;  S = Σ_a P(a) = N - 4·k_d.

The converse reduces (via the recursion, P_up = -P_down on non-exceptional low a) to:

  Q(k):  every distinct-nonzero pair (l,m) in A_k (k≥3) has a NON-EXCEPTIONAL disagreeing orbit,
         where the exceptional orbits are exactly {0,d} and {l,m}  (these coincide: l⊕d = m).

FINDINGS (all verified below over k = 3..7/8):
  (1) BOUND:      k_d ≥ 4  (equivalently S ≤ N-16) for EVERY distinct-nonzero pair — uniform in k.
                  Always ≥ 2 non-exceptional disagreeing orbits. So Q holds with margin.
  (2) DOUBLING:   for a both-low pair (l,m < 2^(k-1)),  k_d(l,m,k) = 2·k_d(l,m,k-1)
                  (consequence of cdSigma_stable: signs on the low block are level-independent).
  (3) BASE:       octonions (k=3, a division algebra) — EVERY orbit disagrees (k_d = N/2 = 4).
  (4) WITNESS:    a MIXED pair (l, 2^(k-1)+m_lo) with m_lo ∉ {0,l} has an EXPLICIT non-exceptional
                  disagreeing orbit at a = m_lo (and at a = l⊕m_lo).
  RESIDUAL: (1) for ALL pairs (needed for Q ∀k) does not follow from (2)+(3)+(4) alone, because a
            general pair reduces by (2) to a MINIMAL-level pair that may be a both-low pair the
            recursion cannot peel. That both-low minimal case is the open seam.
"""
import sys
sys.path.insert(0, '/workspace/sounio-exact-algebra/scripts/research')
from cd_tower_converse_probe import cd_sigma


def f(l, m, a, k):
    return cd_sigma(l, a, k) * cd_sigma(m, a, k)


def P(l, m, a, k):
    d = l ^ m
    return f(l, m, a, k) * f(l, m, a ^ d, k)


def stats(l, m, k):
    N = 1 << k
    d = l ^ m
    S = sum(P(l, m, a, k) for a in range(N))
    kd = (N - S) // 4
    excep = {0, d, l, m}
    seen, nonexcep_dis = set(), 0
    for a in range(N):
        b = a ^ d
        if a in seen or b in seen:
            continue
        seen.add(a); seen.add(b)
        if P(l, m, a, k) == -1 and a not in excep and b not in excep:
            nonexcep_dis += 1
    return S, kd, nonexcep_dis


def main():
    print("(1) BOUND  k_d ≥ 4  (S ≤ N-16), ≥2 non-exceptional disagreeing orbits:")
    for k in range(3, 8):
        N = 1 << k
        pairs = [(l, m) for l in range(1, N) for m in range(l + 1, N)]
        min_kd = min(stats(l, m, k)[1] for l, m in pairs)
        max_S = max(stats(l, m, k)[0] for l, m in pairs)
        min_ne = min(stats(l, m, k)[2] for l, m in pairs)
        print(f"    k={k} N={N}: min k_d={min_kd}, max S={max_S} (N-16={N-16}), "
              f"min #nonexcep-disagree={min_ne}")

    print("(2) DOUBLING  k_d(k)=2·k_d(k-1) for both-low pairs:")
    bad = tot = 0
    for k in range(3, 7):
        for l in range(1, 1 << k):
            for m in range(l + 1, 1 << k):  # both < 2^k ⇒ both-low at level k+1
                tot += 1
                if stats(l, m, k + 1)[1] != 2 * stats(l, m, k)[1]:
                    bad += 1
    print(f"    {tot - bad}/{tot} both-low pairs satisfy the doubling identity")

    print("(3) BASE  octonions (k=3): every orbit disagrees:")
    alldis = all(P(l, m, a, 3) == -1
                 for l in range(1, 8) for m in range(l + 1, 8) for a in range(8))
    print(f"    all octonion orbits disagree: {alldis}")

    print("(4) WITNESS  a=m_lo disagrees for mixed off-seam pairs (l, 2^(k-1)+m_lo), m_lo∉{0,l}:")
    ok = tot = 0
    for k in range(4, 8):
        H = 1 << (k - 1)
        for l in range(1, H):
            for m_lo in range(1, H):
                if m_lo == l:
                    continue
                tot += 1
                if P(l, H + m_lo, m_lo, k) == -1:
                    ok += 1
    print(f"    {ok}/{tot} mixed off-seam pairs have P(m_lo) = -1  (explicit witness)")


if __name__ == "__main__":
    main()
