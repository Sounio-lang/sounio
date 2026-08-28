#!/usr/bin/env python3
"""Full-box oracle for the recursive Cayley-Dickson seam predicate.

Checks every distinct nonzero pair at levels 4..8 against the independent
O(N) XOR-annihilator test.  The Lean theorem is scale-independent; this script
is a reproducible executable cross-check, not a proof dependency.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def cd_sigma(a: int, b: int, bits: int) -> int:
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


def has_xor_annih(bits: int, l: int, u: int) -> bool:
    dimension = 1 << bits
    delta = l ^ u
    return any(
        a != delta
        and cd_sigma(l, a, bits)
        * cd_sigma(u, a, bits)
        * cd_sigma(l, a ^ delta, bits)
        * cd_sigma(u, a ^ delta, bits)
        == 1
        for a in range(1, dimension)
    )


def recursive_off_seam(bits: int, l: int, u: int) -> bool:
    while bits > 3 and l != 0 and u != 0 and l != u:
        top = 1 << (bits - 1)
        l_hi, u_hi = l >= top, u >= top
        if l_hi == u_hi:
            l %= top
            u %= top
            bits -= 1
            continue
        low, high = (u, l) if l_hi else (l, u)
        return high != top and (low ^ high) != top
    return False


def main() -> None:
    expected = {4: 42, 5: 294, 6: 1518, 7: 6942, 8: 29886}
    for bits in range(4, 9):
        dimension = 1 << bits
        mismatches = []
        geometric_count = 0
        for l in range(1, dimension):
            for u in range(l + 1, dimension):
                geometric = recursive_off_seam(bits, l, u)
                intrinsic = has_xor_annih(bits, l, u)
                geometric_count += geometric
                if geometric != intrinsic:
                    mismatches.append((l, u, geometric, intrinsic))
        pair_count = (dimension - 1) * (dimension - 2) // 2
        print(
            f"RECURSIVE_SEAM bits={bits} pairs={pair_count} "
            f"zd={geometric_count} mismatches={len(mismatches)}"
        )
        if mismatches or geometric_count != expected[bits]:
            raise SystemExit(f"recursive seam mismatch at bits={bits}: {mismatches[:8]}")
    print("CD_TOWER_RECURSIVE_SEAM_OK")


if __name__ == "__main__":
    main()
