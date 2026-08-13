#!/usr/bin/env python3
"""Numeric probe for `hiso_law` (Tier 158, formal/lean4/SounioZDFiberAntisym.lean).

Definitions copied from scripts/research/zd_e5_hentry_probe.py (itself a transcription
of SounioZDFiberAntisym.lean's `P3`/`cdSigma`/`eDef`/`isDefect`).  Checks, in order:

  1. The bounded claim itself: for `k = 0..5`, `m = k+3`, every spine vertex
     `x in {0, 1, 2^(k+3), 2^(k+3)+1}` has `isDefect(m, x, y) = False` for EVERY
     `y < 2^(k+4)` (the window `hiso_law`'s `hy` hypothesis restricts to).
  2. An UNBOUNDED sweep (`y` allowed past `2^(k+4)`, up to a few multiples of the
     window) — this is expected to FAIL, the same situation as `hentry_law`'s bound
     hypotheses.  It is why `hiso_law` carries an explicit `hy : y < 2^(k+4)` bound,
     unlike the literal unbounded `∀ y` that `cherry_total`/`stratum_handshake`/
     `odd_has0_eq_k1`/`k1_has0_eq_edges`/`k3_has0_vanishes` request of their `hiso`
     hypothesis parameter.  See the Tier 158 docstring in SounioZDFiberAntisym.lean.

Exit code 0 iff check 1 passes (0 failures) AND check 2 fails as expected (nonzero
failures) -- i.e. iff the measured shape matches what got proved.
"""
from functools import lru_cache


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


def e_def(m: int, a: int, b: int) -> int:
    return p3(a, b, 1, m) * (rs(m, a) * rs(m, b))


def is_defect(m: int, x: int, y: int) -> bool:
    return (x // 2 != 0) and (y // 2 != 0) and (x // 2 != y // 2) and (e_def(m, x, y) != 1)


def spine(k: int):
    h3 = 1 << (k + 3)
    return [0, 1, h3, h3 + 1]


def bounded_check(kmax: int = 5) -> bool:
    bad = 0
    total = 0
    for k in range(kmax + 1):
        h4 = 1 << (k + 4)
        m = k + 3
        for x in spine(k):
            for y in range(h4):
                total += 1
                if is_defect(m, x, y):
                    bad += 1
                    if bad <= 10:
                        print(f"BOUNDED FAIL k={k} x={x} y={y}")
    print(f"bounded: total={total} bad={bad}")
    return bad == 0


def unbounded_check(kmax: int = 4, extra: int = 3) -> bool:
    bad = 0
    total = 0
    for k in range(kmax + 1):
        h4 = 1 << (k + 4)
        m = k + 3
        ymax = h4 * (1 + extra)
        for x in spine(k):
            for y in range(h4, ymax):
                total += 1
                if is_defect(m, x, y):
                    bad += 1
                    if bad <= 20:
                        print(f"UNBOUNDED FAIL k={k} x={x} y={y} (h4={h4})")
    print(f"unbounded: total={total} bad={bad}")
    return bad == 0


if __name__ == "__main__":
    ok_bounded = bounded_check(5)
    ok_unbounded = unbounded_check(4, 3)
    print("bounded OK" if ok_bounded else "bounded FAILS (unexpected!)")
    print("unbounded OK (true for all y)" if ok_unbounded
          else "unbounded FAILS (expected, matches hentry_law's situation)")
    # PASS iff bounded holds and unbounded is genuinely false (matches what got proved).
    success = ok_bounded and not ok_unbounded
    print("PASS" if success else "FAIL")
    raise SystemExit(0 if success else 1)
