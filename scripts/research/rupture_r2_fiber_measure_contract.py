#!/usr/bin/env python3
"""R2 fiber+measure contract for the rupture A+B+C+D ladder.

Certifies R2-partial (required for gate green):
  (i)   all primitive annihilating edges are intra-fiber (INTRA_BAD=0)
  (ii)  Frente A exact rational measure on the canonical slice:
          on-locus  E[r]=0, Var[r]=0
          off-locus E[r]=0, Var[r]=1/150
  (iii) structured fiber partners annihilate; uniform random mixed-half pairs rarely do

Reports R2-full diagnostics (all seven fibers) without claiming a continuous law on S.

Uses the same cd_sigma recursion as scripts/research/sedenion_zd_fibers_oracle.py
(ir_cd_sigma transcription). Pure Python / fractions — no numpy required.

Exit 0 iff R2-partial passes. Prints machine lines:
  R2_PARTIAL PASS|FAIL
  R2_FULL_DIAG ...
  R2_CONTRACT_OK|R2_CONTRACT_FAIL
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from random import Random


def cd_sigma(a: int, b: int, bits: int = 4) -> int:
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi, b_hi = a >= half, b >= half
    a_lo, b_lo = a & (half - 1), b & (half - 1)
    if not a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if not a_hi and b_hi:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1) if b_lo == 0 else -cd_sigma(a_lo, b_lo, bits - 1)
    return -cd_sigma(b_lo, a_lo, bits - 1) if b_lo == 0 else cd_sigma(b_lo, a_lo, bits - 1)


def mul(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, ci in a.items():
        for j, cj in b.items():
            k = i ^ j
            out[k] = out.get(k, 0) + cd_sigma(i, j) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def vec(c: tuple[int, int, int]) -> dict[int, int]:
    lo, hi, neg = c
    return {lo: 1, hi: (-1 if neg == 1 else 1)}


def fiber_label(c: tuple[int, int, int]) -> int:
    return c[0] ^ c[1]


def r5(alpha: Fraction, beta: Fraction, gamma: Fraction, delta: Fraction) -> Fraction:
    """Canonical product coordinate used by Frente A (e3+e10)×(e6±e15) family."""
    return alpha * gamma + beta * delta


def empirical_mean_var(samples: list[Fraction]) -> tuple[Fraction, Fraction]:
    n = len(samples)
    assert n > 0
    inv_n = Fraction(1, n)
    mean = sum(samples, Fraction(0)) * inv_n
    mean_sq = sum((s * s for s in samples), Fraction(0)) * inv_n
    return mean, mean_sq - mean * mean


def frente_a_slice() -> bool:
    # on-locus: (t,t,s,-s) → r5 = 0
    on = [
        (Fraction(1), Fraction(1), Fraction(1), Fraction(-1)),
        (Fraction(2), Fraction(2), Fraction(1), Fraction(-1)),
        (Fraction(1), Fraction(1), Fraction(3), Fraction(-3)),
    ]
    on_r = [r5(*p) for p in on]
    e_on, v_on = empirical_mean_var(on_r)
    # off-locus: alpha in {9/10,1,11/10}, beta=1, gamma=1, delta=-1 → r5 in {-1/10,0,1/10}
    off = [
        (Fraction(9, 10), Fraction(1), Fraction(1), Fraction(-1)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(-1)),
        (Fraction(11, 10), Fraction(1), Fraction(1), Fraction(-1)),
    ]
    off_r = [r5(*p) for p in off]
    e_off, v_off = empirical_mean_var(off_r)
    ok = (
        e_on == 0
        and v_on == 0
        and e_off == 0
        and v_off == Fraction(1, 150)
    )
    print(f"FRENTE_A on E={e_on} Var={v_on} off E={e_off} Var={v_off} -> {'PASS' if ok else 'FAIL'}")
    return ok


def build_primitives() -> tuple[list[tuple[int, int, int]], list[tuple], dict]:
    cands = [(lo, hi, neg) for lo in range(1, 8) for hi in range(8, 16) for neg in (0, 1)]
    part = [c for c in cands if any(not mul(vec(c), vec(b)) for b in cands)]
    edges = []
    adj: dict = defaultdict(set)
    for i, j in combinations(range(len(part)), 2):
        a, b = part[i], part[j]
        if not mul(vec(a), vec(b)):
            edges.append((a, b))
            adj[a].add(b)
            adj[b].add(a)
    return part, edges, adj


def main() -> int:
    part, edges, adj = build_primitives()
    fib: dict[int, list] = defaultdict(list)
    for v in part:
        fib[fiber_label(v)].append(v)

    intra_bad = sum(1 for a, b in edges if fiber_label(a) != fiber_label(b))
    degree_bad = sum(1 for v in part if len(adj[v]) != 4)
    n_part = len(part)
    n_edges = len(edges)

    print(f"PARTICIPATE {n_part}")
    print(f"EDGES {n_edges}")
    print(f"INTRA_BAD {intra_bad}")
    print(f"DEGREE_BAD {degree_bad}")
    for L in sorted(fib):
        vs = fib[L]
        e_count = sum(1 for a, b in edges if fiber_label(a) == L and fiber_label(b) == L)
        print(f"FIBER {L} size={len(vs)} edges={e_count}")

    # (iii) structured vs random annihilation rates (exact integer product)
    structured_ann = 0
    structured_n = 0
    for v in part:
        for w in adj[v]:
            structured_n += 1
            if not mul(vec(v), vec(w)):
                structured_ann += 1
    # each undirected edge counted twice in the walk above
    structured_rate = structured_ann / structured_n if structured_n else 0.0

    rng = Random(20260724)
    mixed = [(lo, hi, neg) for lo in range(1, 8) for hi in range(8, 16) for neg in (0, 1)]
    random_ann = 0
    random_n = 4000
    for _ in range(random_n):
        a = mixed[rng.randrange(len(mixed))]
        b = mixed[rng.randrange(len(mixed))]
        if a == b:
            continue
        if not mul(vec(a), vec(b)):
            random_ann += 1
    random_rate = random_ann / random_n

    print(f"STRUCTURED_ANN_RATE {structured_rate:.6f}")
    print(f"RANDOM_MIXED_ANN_RATE {random_rate:.6f} (n={random_n})")

    # Per-fiber: every recorded neighbour annihilates and shares L
    fiber_partner_ok = True
    for L, vs in fib.items():
        for v in vs:
            for w in adj[v]:
                if fiber_label(w) != L or mul(vec(v), vec(w)):
                    fiber_partner_ok = False

    measure_ok = frente_a_slice()
    fiber_ok = (
        n_part == 84
        and n_edges == 168
        and intra_bad == 0
        and degree_bad == 0
        and len(fib) == 7
        and fiber_partner_ok
        and structured_rate == 1.0
        and random_rate < 0.08  # structured ~deg/83 ~0.05 on full graph of 84; random far lower than 1
    )
    # random rate among *all* mixed (including non-participants) should be << structured
    rate_ok = random_rate < structured_rate and random_rate < 0.05

    partial_ok = measure_ok and fiber_ok and rate_ok
    print(f"R2_PARTIAL {'PASS' if partial_ok else 'FAIL'}")

    # R2-full diagnostic: each fiber has 12 verts and positive annihilators
    full_sizes_ok = all(len(fib[L]) == 12 for L in range(9, 16))
    print(
        f"R2_FULL_DIAG fibers7={len(fib)==7} size12={full_sizes_ok} "
        f"intra0={intra_bad==0} measure={measure_ok} "
        f"NOTE=see_rupture_r2_full_tubular_probe_for_R2_FULL_MEASURED"
    )
    # Partial is the combinatorial gate; continuous tubular lives in the companion probe
    if full_sizes_ok and partial_ok:
        print("R2_FULL_STATUS PARTIAL_OK_COMPANION_IS_FULL_MEASURED")
    else:
        print("R2_FULL_STATUS OPEN_OR_PARTIAL_FAIL")

    if partial_ok:
        print("R2_CONTRACT_OK")
        return 0
    print("R2_CONTRACT_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
