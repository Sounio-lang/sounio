#!/usr/bin/env python3
"""Independent oracle for the sedenion ASSOCIATOR side (Frente B): 1848 = 11*168 (confirms the
SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT conjecture that the factor 11 lives on the associator side).

Emits, with pure-integer arithmetic:
  TOTAL <n>     ordered distinct triples (i,j,k) of {1..15} with nonzero associator  (1848)
  GRADE8 <n>    of those, with output grade i^j^k == 8                                (168)
  OTHER <n>     with grade != 8                                                        (1680 = 10*168)
  OCT <n>       within the octonion sub-tower {1..7}                                   (168)
  GRADEC <s> <n>  per output-grade count (s = 1..15)                                  (120 each, 168 at s=8)
  ASSOC <OK|FAIL>
The cd_sigma recursion transcribes ir_cd_sigma. Non-souc leg of scripts/ci/sedenion_associator_1848_gate.sh.
"""
from __future__ import annotations
from itertools import permutations


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


def assoc(i: int, j: int, k: int) -> int:
    return cd_sigma(i, j) * cd_sigma(i ^ j, k) - cd_sigma(j, k) * cd_sigma(i, j ^ k)


def main() -> None:
    total = oct = 0
    grade = [0] * 16
    for i, j, k in permutations(range(1, 16), 3):
        if assoc(i, j, k) != 0:
            total += 1
            grade[i ^ j ^ k] += 1
            if i <= 7 and j <= 7 and k <= 7:
                oct += 1
    from itertools import combinations
    class0 = class2 = class6 = g8_notfull = 0
    for i, j, k in combinations(range(1, 16), 3):
        cnt = sum(1 for p in permutations((i, j, k)) if assoc(*p) != 0)
        if cnt == 0: class0 += 1
        elif cnt == 2: class2 += 1
        elif cnt == 6: class6 += 1
        if (i ^ j ^ k) == 8 and cnt != 6: g8_notfull += 1
    grade8 = grade[8]
    other = sum(grade[s] for s in range(1, 16) if s != 8)
    uniform = all(grade[s] == 120 for s in range(1, 16) if s != 8)
    print(f"TOTAL {total}")
    print(f"GRADE8 {grade8}")
    print(f"OTHER {other}")
    print(f"OCT {oct}")
    print(f"CLASS0 {class0}")
    print(f"CLASS2 {class2}")
    print(f"CLASS6 {class6}")
    print(f"G8_NOTFULL {g8_notfull}")
    for s in range(1, 16):
        print(f"GRADEC {s} {grade[s]}")
    ok = (total == 1848 and grade8 == 168 and other == 1680 and oct == 168 and uniform
          and class0 == 35 and class2 == 168 and class6 == 252 and g8_notfull == 0)
    print(f"ASSOC {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
