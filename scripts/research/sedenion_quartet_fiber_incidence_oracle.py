#!/usr/bin/env python3
"""Independent oracle for the quartet<->fiber incidence of the sedenion ZD geometry (Frente B):
the 42 support-quartets, as edges on the 7 fibers L=lo^hi in {9..15}, form exactly 2*K_7 — every one
of the C(7,2)=21 fiber-pairs joined by exactly 2 quartets, every fiber of incidence degree 12.

Non-souc leg of scripts/ci/sedenion_quartet_fiber_incidence_gate.sh.
Output:
  FP <l1> <l2> <n>   per used fiber-pair, count of quartets on it (all 21 pairs, n=2)
  PAIRS <n>          zero-divisor pairs (168)
  FIBERPAIRS <n>     distinct fiber-pairs used (21)
  BAD_FIBERS <n>     quartets not spanning exactly 2 fibers (0)
  BAD_PAIRCT <n>     fiber-pairs not carrying exactly 2 quartets (0)
  BAD_DEG <n>        fibers not of incidence-degree 12 (0)
  INCIDENCE <OK|FAIL>
"""
from __future__ import annotations
from collections import defaultdict


def cd_sigma(a, b, bits=4):
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
    qfibers = defaultdict(set)
    npair = 0
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if not mul(vec(part[i]), vec(part[j])):
                a, b = part[i], part[j]
                q = (1 << a[0]) | (1 << a[1]) | (1 << b[0]) | (1 << b[1])
                qfibers[q].add(a[0] ^ a[1])
                qfibers[q].add(b[0] ^ b[1])
                npair += 1
    bad_fibers = sum(1 for s in qfibers.values() if len(s) != 2)
    fp = defaultdict(int)
    for s in qfibers.values():
        l1, l2 = sorted(s)[:2] if len(s) >= 2 else (min(s), min(s))
        fp[(l1, l2)] += 1
    bad_pairct = sum(1 for c in fp.values() if c != 2)
    deg = defaultdict(int)
    for (l1, l2), c in fp.items():
        deg[l1] += c
        deg[l2] += c
    bad_deg = sum(1 for f in range(9, 16) if deg[f] != 12)
    for (l1, l2) in sorted(fp):
        print(f"FP {l1} {l2} {fp[(l1, l2)]}")
    print(f"PAIRS {npair}")
    print(f"FIBERPAIRS {len(fp)}")
    print(f"BAD_FIBERS {bad_fibers}")
    print(f"BAD_PAIRCT {bad_pairct}")
    print(f"BAD_DEG {bad_deg}")
    ok = npair == 168 and len(fp) == 21 and bad_fibers == 0 and bad_pairct == 0 and bad_deg == 0
    print(f"INCIDENCE {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
