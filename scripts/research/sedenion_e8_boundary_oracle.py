#!/usr/bin/env python3
"""Independent oracle for the sedenion zero-divisor e8-boundary (Frente B).

Emits, deterministically and with pure-integer arithmetic, the participation split of the 112
mixed-half signed two-support primitives v = e_lo (+/-) e_hi (lo in 1..7, hi in 8..15, sign in {+,-})
under sedenion (16D Cayley-Dickson) multiplication, and the exact characterization of the excluded
set. This is the NON-souc leg of the cross-toolchain check run by
scripts/ci/sedenion_e8_boundary_gate.sh; the souc leg is tests/run-pass/sedenion_e8_boundary.sio.

The cd_sigma recursion is the exact transcription of the compiler's own sign law
(self-hosted/ir/algebra.sio::ir_cd_sigma), matching scripts/research/generate_sedenion_zero_divisor_geometry.py.

Output lines (sorted, so /usr/bin/diff can compare against the souc output verbatim):
  EXCL <code>        one per excluded primitive, code = lo*10000 + hi*100 + neg   (neg: 0=+, 1=-)
  PARTICIPATE <n>    count that participate in some zero-divisor pair
  EXCLUDED <n>       count that participate in none
  TOUCH_E8 <n>       excluded because hi == 8
  DIAGONAL <n>       excluded because lo XOR hi == 8
  INVARIANT <HOLDS|FAILS>   excluded  <=>  (hi == 8) or (lo XOR hi == 8)
"""
from __future__ import annotations


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


def vec(lo: int, hi: int, neg: int) -> dict[int, int]:
    return {lo: 1, hi: (-1 if neg == 1 else 1)}


def main() -> None:
    cands = [(lo, hi, neg) for lo in range(1, 8) for hi in range(8, 16) for neg in (0, 1)]

    def participates(a: tuple[int, int, int]) -> bool:
        A = vec(*a)
        return any(not mul(A, vec(*b)) for b in cands)

    part = [c for c in cands if participates(c)]
    excl = [c for c in cands if not participates(c)]

    def in_e8(c: tuple[int, int, int]) -> bool:
        lo, hi, _ = c
        return hi == 8 or (lo ^ hi) == 8

    invariant = all(in_e8(c) for c in excl) and all(not in_e8(c) for c in part)
    touch8 = sum(1 for c in excl if c[1] == 8)
    diag = sum(1 for c in excl if (c[0] ^ c[1]) == 8)

    for lo, hi, neg in sorted(excl):
        print(f"EXCL {lo * 10000 + hi * 100 + neg}")
    print(f"PARTICIPATE {len(part)}")
    print(f"EXCLUDED {len(excl)}")
    print(f"TOUCH_E8 {touch8}")
    print(f"DIAGONAL {diag}")
    print(f"INVARIANT {'HOLDS' if invariant and len(part) == 84 and len(excl) == 28 else 'FAILS'}")


if __name__ == "__main__":
    main()
