#!/usr/bin/env python3
"""Independent oracle for the 42 support-quartets of the sedenion ZD geometry (Frente B).

The 168 unordered zero-divisor pairs group by support-union into exactly 42 quartets, each a 4-set
with 2 lower {1..7} + 2 upper {8..15} indices, each hosting exactly 4 pairs (42*4 = 168). Emits the
42 quartet bitmasks (sorted) + the summary. Non-souc leg of scripts/ci/sedenion_zd_quartets_gate.sh;
souc leg tests/run-pass/sedenion_zd_quartets.sio; Lean leg formal/lean4/SounioSedenionQuartets.lean.

Output (sorted):
  QMASK <mask>   one per distinct quartet, mask = sum of (1<<idx) over its 4 support indices
  PAIRS <n>      unordered zero-divisor pairs (168)
  QUARTETS <n>   distinct support-quartets (42)
  BAD_SIZE <n>   quartets not of shape (2 lower, 2 upper) (0)
  BAD_COUNT <n>  quartets not hosting exactly 4 pairs (0)
  QUARTETS_V <OK|FAIL>
"""
from __future__ import annotations
from collections import defaultdict


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


def mul(a, b):
    out = {}
    for i, ci in a.items():
        for j, cj in b.items():
            k = i ^ j
            out[k] = out.get(k, 0) + cd_sigma(i, j) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def vec(c):
    lo, hi, neg = c
    return {lo: 1, hi: (-1 if neg == 1 else 1)}


def main() -> None:
    cands = [(lo, hi, neg) for lo in range(1, 8) for hi in range(8, 16) for neg in (0, 1)]
    part = [c for c in cands if any(not mul(vec(c), vec(b)) for b in cands)]
    q = defaultdict(int)
    npair = 0
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if not mul(vec(part[i]), vec(part[j])):
                a, b = part[i], part[j]
                mask = (1 << a[0]) | (1 << a[1]) | (1 << b[0]) | (1 << b[1])
                q[mask] += 1
                npair += 1

    def popc(m):
        return bin(m).count("1")

    def lower(m):
        return sum(1 for x in range(1, 8) if m & (1 << x))

    bad_size = sum(1 for m in q if popc(m) != 4 or lower(m) != 2)
    bad_count = sum(1 for m in q if q[m] != 4)
    for m in sorted(q):
        print(f"QMASK {m}")
    print(f"PAIRS {npair}")
    print(f"QUARTETS {len(q)}")
    print(f"BAD_SIZE {bad_size}")
    print(f"BAD_COUNT {bad_count}")
    ok = npair == 168 and len(q) == 42 and bad_size == 0 and bad_count == 0
    print(f"QUARTETS_V {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
