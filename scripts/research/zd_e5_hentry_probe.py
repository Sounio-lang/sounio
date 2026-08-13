#!/usr/bin/env python3
"""Numeric probe for `hentry_law` (Tier 156, formal/lean4/SounioZDFiberAntisym.lean).

Definitions copied from scripts/research/zd_e5_reference_anatomy_probe.py (itself a
transcription of SounioZDFiberAntisym.lean's `P3`/`cdSigma`).  Checks, in order:

  1. The bounded claim itself: for all `m`, all `0 <= x,y < 2^(m+1)`, `x != y`:
       P3(x,y,1,m) = refP(m,x,y) * (-1 if isDefect(m,x,y) else 1)
  2. The three "escape loci" sub-lemmas used to prove it (`eDef == 1` at `x=1`,
     `y=1`, and `x/2=y/2` i.e. `y = x^1`), plus the bridging fact
     `P3(1,y,1,m) == P3(0,y,1,m)`.
  3. An UNBOUNDED sweep (`x,y` allowed past `2^(m+1)`) — this is expected to FAIL.
     It is why `hentry_law`'s Lean statement carries explicit bound hypotheses
     `hx : x < 2^(m+1)`, `hy : y < 2^(m+1)`, unlike the literal `∀ x y` that
     `hlaw`/`hlaw_pow`/`hlaw_class_dev` request of their `hentry` hypothesis
     parameter.  See the Tier 156 docstring in SounioZDFiberAntisym.lean.

Exit code 0 iff checks 1-2 pass (0 failures) AND check 3 fails as expected
(nonzero failures) — i.e. iff the measured shape matches what got proved.
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


def ref_p(m: int, a: int, b: int) -> int:
    if a == 0:
        return 1 if b == 0 else rs(m, b)
    if b == 0:
        return -rs(m, a)
    if b == a:
        return -1
    return rs(m, a) * rs(m, b)


def bounded_checks() -> bool:
    bad_full = bad_x1 = bad_y1 = bad_xor1 = bad_p3eq = 0
    total = 0
    for m in range(0, 7):
        H = 1 << (m + 1)
        for x in range(H):
            for y in range(H):
                if x == y:
                    continue
                total += 1
                want = ref_p(m, x, y) * (-1 if is_defect(m, x, y) else 1)
                got = p3(x, y, 1, m)
                if got != want:
                    bad_full += 1
                if x != 0 and y != 0:
                    if x == 1 and e_def(m, x, y) != 1:
                        bad_x1 += 1
                    if y == 1 and e_def(m, x, y) != 1:
                        bad_y1 += 1
                    if x >= 2 and y >= 2 and (x // 2 == y // 2) and e_def(m, x, y) != 1:
                        bad_xor1 += 1
        for y in range(2, H):
            if p3(1, y, 1, m) != p3(0, y, 1, m):
                bad_p3eq += 1

    print(f"bounded total pairs: {total}")
    print(f"  bad_full (whole hentry claim):                       {bad_full}")
    print(f"  bad_x1   (eDef m 1 y == 1, y!=0,1):                  {bad_x1}")
    print(f"  bad_y1   (eDef m x 1 == 1, x!=0,1):                  {bad_y1}")
    print(f"  bad_xor1 (eDef m x y == 1, x/2=y/2, x,y>=2):         {bad_xor1}")
    print(f"  bad_p3eq (P3(1,y,1,m) == P3(0,y,1,m), y>=2):         {bad_p3eq}")
    return bad_full == bad_x1 == bad_y1 == bad_xor1 == bad_p3eq == 0


def unbounded_check_fails_as_expected() -> bool:
    bad = 0
    total = 0
    for m in range(0, 5):
        H = 1 << (m + 1)
        for x in range(0, 3 * H):
            for y in range(0, 3 * H):
                if x == y:
                    continue
                total += 1
                want = ref_p(m, x, y) * (-1 if is_defect(m, x, y) else 1)
                got = p3(x, y, 1, m)
                if got != want:
                    bad += 1
    print(f"unbounded sweep (m=0..4, x,y up to 3*2^(m+1)): total={total} bad={bad}")
    return bad > 0


def main() -> int:
    ok = bounded_checks()
    unbounded_fails = unbounded_check_fails_as_expected()
    if not ok:
        print("FAIL: bounded hentry claim or a sub-lemma has counterexamples")
        return 1
    if not unbounded_fails:
        print("FAIL: unbounded sweep found no counterexample — hx/hy bounds may be droppable "
              "(re-examine before trusting this)")
        return 1
    print("PASS: bounded claim exact; unbounded sweep confirms the bound hypotheses are load-bearing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
