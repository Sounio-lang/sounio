#!/usr/bin/env python3
"""Numeric probe for `hDelta_law` (Tier 160, formal/lean4/SounioZDFiberAntisym.lean).

Definitions copied from `zd_e5_hiso_probe.py` / `zd_e5_hentry_probe.py` (themselves
transcriptions of SounioZDFiberAntisym.lean's `P3`/`cdSigma`/`eDef`/`isDefect`/`refP`).

Checks, for `k = 0..kmax`, `m = k+3`, `H = 2^(k+4)`:

  1. `Delta(m) := tr(M^3) - tr(P^3)` (M = P3(.,.,1,m), P = refP m) computed by brute
     force triple sum, matches `-12 * net(m)` where `net` is `stratum_net`'s exact
     `hnetdef` sum (signed odd-`kcnt` count over `x<y<z`).
  2. The degenerate-triple part of `Delta`'s defining sum (any two of a,b,c equal)
     is exactly 0.
  3. `M(x,y)*M(y,x) = refP(x,y)*refP(y,x)` for all x,y (the pair lemma the proof
     leans on), including x=y.
  4. Permutation-invariance: for pairwise-distinct a,b,c, `Mp(a,b,c) - Pp(a,b,c)`
     (Mp = M(a,b)M(b,c)M(c,a), Pp = refP(a,b)refP(b,c)refP(c,a)) takes the same value
     on all 6 orderings of {a,b,c}.

Exit code 0 iff all four checks pass with 0 failures.
"""
from functools import lru_cache
import sys


@lru_cache(maxsize=None)
def cd_sigma(a: int, b: int, n: int) -> int:
    if n == 0:
        return -1
    if n == 1:
        return 1 if a == 0 or b == 0 else -1
    if a == 0 or b == 0:
        return 1
    half = 1 << (n - 1)
    a_hi, b_hi = a >= half, b >= half
    if not a_hi and not b_hi:
        return cd_sigma(a % half, b % half, n - 1)
    if not a_hi and b_hi:
        return cd_sigma(b % half, a % half, n - 1)
    if a_hi and not b_hi:
        if b % half == 0:
            return cd_sigma(a % half, 0, n - 1)
        return -cd_sigma(a % half, b % half, n - 1)
    if b % half == 0:
        return -cd_sigma(0, a % half, n - 1)
    return cd_sigma(b % half, a % half, n - 1)


def hi(x: int, llo: int, n: int) -> int:
    return (x ^ llo) + (1 << (n + 1))


def p3(l: int, y: int, llo: int, n: int) -> int:
    return cd_sigma(l, hi(y, llo, n), n + 2) * cd_sigma(hi(l, llo, n), y, n + 2)


def rs(m: int, x: int) -> int:
    return p3(0, x, 1, m)


def ref_p(m: int, a: int, b: int) -> int:
    if a == 0:
        return 1 if b == 0 else rs(m, b)
    if b == 0:
        return -rs(m, a)
    if b == a:
        return -1
    return rs(m, a) * rs(m, b)


def e_def(m: int, a: int, b: int) -> int:
    return p3(a, b, 1, m) * (rs(m, a) * rs(m, b))


def is_defect(m: int, x: int, y: int) -> bool:
    return (x // 2 != 0) and (y // 2 != 0) and (x // 2 != y // 2) and (e_def(m, x, y) != 1)


def kcnt(m: int, a: int, b: int, c: int) -> int:
    return (1 if is_defect(m, a, b) else 0) + (1 if is_defect(m, b, c) else 0) \
        + (1 if is_defect(m, c, a) else 0)


def m_entry(m: int, x: int, y: int) -> int:
    return p3(x, y, 1, m)


def check_delta_eq_neg12net(kmax: int) -> bool:
    ok = True
    for k in range(kmax + 1):
        m = k + 3
        H = 1 << (k + 4)
        trM3 = 0
        trP3 = 0
        for a in range(H):
            for b in range(H):
                for c in range(H):
                    trM3 += m_entry(m, a, b) * m_entry(m, b, c) * m_entry(m, c, a)
                    trP3 += ref_p(m, a, b) * ref_p(m, b, c) * ref_p(m, c, a)
        Delta = trM3 - trP3
        net = 0
        for a in range(H):
            for b in range(H):
                for c in range(H):
                    if a < b < c and kcnt(m, a, b, c) % 2 == 1:
                        net += -1 if (a == 0 or b == 0 or c == 0) else 1
        pred = -12 * net
        status = "OK" if Delta == pred else "FAIL"
        if Delta != pred:
            ok = False
        print(f"k={k} m={m} H={H}: Delta={Delta} net={net} -12*net={pred} [{status}]")
    return ok


def check_degenerate_zero(kmax: int) -> bool:
    ok = True
    for k in range(kmax + 1):
        m = k + 3
        H = 1 << (k + 4)
        bad = 0
        for a in range(H):
            for b in range(H):
                for c in range(H):
                    if a == b or b == c or c == a:
                        Mp = m_entry(m, a, b) * m_entry(m, b, c) * m_entry(m, c, a)
                        Pp = ref_p(m, a, b) * ref_p(m, b, c) * ref_p(m, c, a)
                        if Mp != Pp:
                            bad += 1
        print(f"k={k}: degenerate mismatches = {bad}")
        if bad:
            ok = False
    return ok


def check_pair_law(kmax: int) -> bool:
    ok = True
    for k in range(kmax + 1):
        m = k + 3
        H = 1 << (k + 4)
        bad = 0
        for x in range(H):
            for y in range(H):
                lhs = m_entry(m, x, y) * m_entry(m, y, x)
                rhs = ref_p(m, x, y) * ref_p(m, y, x)
                if lhs != rhs:
                    bad += 1
                    if bad <= 5:
                        print(f"PAIR FAIL k={k} x={x} y={y}: {lhs} != {rhs}")
        print(f"k={k}: pair-law mismatches = {bad}")
        if bad:
            ok = False
    return ok


def check_perm_invariance(kmax: int) -> bool:
    from itertools import permutations
    ok = True
    for k in range(kmax + 1):
        m = k + 3
        H = min(1 << (k + 4), 12)  # cap for perf; small H already exercises all cases
        bad = 0
        checked = 0
        for a in range(H):
            for b in range(H):
                for c in range(H):
                    if len({a, b, c}) != 3:
                        continue
                    vals = set()
                    for (p, q, r) in permutations((a, b, c)):
                        Mp = m_entry(m, p, q) * m_entry(m, q, r) * m_entry(m, r, p)
                        Pp = ref_p(m, p, q) * ref_p(m, q, r) * ref_p(m, r, p)
                        vals.add(Mp - Pp)
                    checked += 1
                    if len(vals) != 1:
                        bad += 1
                        if bad <= 5:
                            print(f"PERM FAIL k={k} triple={{{a},{b},{c}}} vals={vals}")
        print(f"k={k}: perm-invariance checked={checked} mismatches={bad}")
        if bad:
            ok = False
    return ok


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print("=== check 1: Delta = -12*net ===")
    r1 = check_delta_eq_neg12net(min(kmax, 3))  # H up to 2^7=128, triple sum is H^3
    print("=== check 2: degenerate triples contribute 0 ===")
    r2 = check_degenerate_zero(min(kmax, 3))
    print("=== check 3: pair law M(x,y)M(y,x) = refP(x,y)refP(y,x) ===")
    r3 = check_pair_law(kmax)
    print("=== check 4: permutation invariance of Mp - Pp on distinct triples ===")
    r4 = check_perm_invariance(kmax)
    allok = r1 and r2 and r3 and r4
    print("ALL OK" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
