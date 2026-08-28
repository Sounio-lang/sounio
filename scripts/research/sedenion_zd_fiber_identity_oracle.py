#!/usr/bin/env python3
"""Independent oracle for the isomorphism type of the sedenion ZD fibers (Frente B, brick 3).

Each of the 7 fibers is isomorphic to K_{6,6} minus a 2-factor of three disjoint 4-cycles
(= K_{6,6} - 3*K_{2,2}), the bipartite 3-block color-mismatch graph. Certified two ways:
  * common-neighbor PROFILE over vertex-pairs = (4:6, 2:24, 0:36)   [BFS-free; the souc leg]
  * the complement 2-factor decomposes into three 4-cycles           [BFS; oracle-only]

NON-souc leg of scripts/ci/sedenion_zd_fiber_identity_gate.sh; the souc leg is
tests/run-pass/sedenion_zd_fiber_identity.sio; the Lean leg is
formal/lean4/SounioSedenionFiberIdentity.lean. cd_sigma transcribes ir_cd_sigma.

Output (sorted for diff):
  FIBERID <code>     per fiber, code = L*1000000 + n4*10000 + n2*100 + n0  (n4/n2/n0 = pairs with 4/2/0 common)
  VERTICES <n>
  COMPLEMENT_C4 <n>  fibers whose K_{6,6}-complement is exactly three 4-cycles (7 expected; oracle-only)
  FIBER_ID <OK|FAIL>
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


def main() -> None:
    cands = [(lo, hi, neg) for lo in range(1, 8) for hi in range(8, 16) for neg in (0, 1)]
    part = [c for c in cands if any(not mul(vec(c), vec(b)) for b in cands)]
    adj: dict[tuple, set] = defaultdict(set)
    for i in range(len(part)):
        for j in range(len(part)):
            if i != j and not mul(vec(part[i]), vec(part[j])):
                adj[part[i]].add(part[j])
    fib: dict[int, list] = defaultdict(list)
    for v in part:
        fib[v[0] ^ v[1]].append(v)

    def profile(vs) -> tuple[int, int, int, int]:
        S = set(vs)
        n4 = n2 = n0 = nx = 0
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                common = len((adj[vs[i]] & adj[vs[j]]) & S)
                if common == 4:
                    n4 += 1
                elif common == 2:
                    n2 += 1
                elif common == 0:
                    n0 += 1
                else:
                    nx += 1
        return n4, n2, n0, nx

    def bipartition(vs):
        color = {vs[0]: 0}
        stack = [vs[0]]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in vs and w not in color:
                    color[w] = 1 - color[u]
                    stack.append(w)
        A = [v for v in vs if color.get(v, 0) == 0]
        B = [v for v in vs if color.get(v, 0) == 1]
        return A, B

    def complement_three_c4(vs) -> bool:
        A, B = bipartition(vs)
        comp: dict[tuple, set] = defaultdict(set)
        for a in A:
            for b in B:
                if b not in adj[a]:
                    comp[a].add(b)
                    comp[b].add(a)
        if any(len(comp[v]) != 2 for v in vs):  # 2-regular complement
            return False
        seen, cycles = set(), []
        for s in vs:
            if s in seen:
                continue
            length, cur, prev = 0, s, None
            while cur not in seen:
                seen.add(cur)
                length += 1
                nxts = [w for w in comp[cur] if w != prev]
                prev, cur = cur, (nxts[0] if nxts else None)
                if cur is None:
                    break
            cycles.append(length)
        return sorted(cycles) == [4, 4, 4]

    lines = []
    ok = True
    for L in range(9, 16):
        n4, n2, n0, nx = profile(fib[L])
        lines.append(f"FIBERID {L * 1000000 + n4 * 10000 + n2 * 100 + n0}")
        if (n4, n2, n0, nx) != (6, 24, 36, 0):
            ok = False
    c4 = sum(1 for L in range(9, 16) if complement_three_c4(fib[L]))
    for ln in sorted(lines):
        print(ln)
    print(f"VERTICES {len(part)}")
    print(f"COMPLEMENT_C4 {c4}")
    print(f"FIBER_ID {'OK' if ok and len(part) == 84 and c4 == 7 else 'FAIL'}")


if __name__ == "__main__":
    main()
