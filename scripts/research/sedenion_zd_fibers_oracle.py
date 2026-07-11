#!/usr/bin/env python3
"""Independent oracle for the 7-fiber structure of the sedenion zero-divisor graph (Frente B, brick 2).

Emits, deterministically with pure-integer arithmetic, the fiber decomposition of the 84 participating
mixed-half primitives: the 7 fibers indexed by L = lo XOR hi in {9..15}, each of 12 vertices / 24 edges,
degree 4, with all annihilation edges intra-fiber. It ALSO verifies (Python-side, via BFS) the two
companion facts the souc test does not execute: each fiber is connected and bipartite (6,6).

This is the NON-souc leg of scripts/ci/sedenion_zd_fibers_gate.sh; the souc leg is
tests/run-pass/sedenion_zd_fibers.sio. The cd_sigma recursion transcribes ir_cd_sigma (same as the
other lane oracles), so the cross-check certifies implementation-agreement against souc miscompiles,
not spec-independence.

Output (sorted for diff):
  FIBER <code>       one per fiber, code = L*10000 + size*100 + edges   (L=lo^hi in 9..15)
  PARTICIPATE <n>    participating vertices
  DEGREE_BAD <n>     vertices whose annihilator degree != 4   (0 = uniform degree 4)
  INTRA_BAD <n>      annihilation edges crossing fibers        (0 = all intra-fiber)
  BIPARTITE_OK <n>   fibers that are bipartite (6,6)           (7 expected; oracle-only)
  CONNECTED_OK <n>   fibers that are a single connected component (7 expected; oracle-only)
  FIBERS <OK|FAIL>
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
    edges = []
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if not mul(vec(part[i]), vec(part[j])):
                edges.append((part[i], part[j]))
                adj[part[i]].add(part[j])
                adj[part[j]].add(part[i])

    fib: dict[int, list] = defaultdict(list)
    for v in part:
        fib[v[0] ^ v[1]].append(v)

    degree_bad = sum(1 for v in part if len(adj[v]) != 4)
    intra_bad = sum(1 for a, b in edges if (a[0] ^ a[1]) != (b[0] ^ b[1]))
    fiber_edges = defaultdict(int)
    for a, b in edges:
        if (a[0] ^ a[1]) == (b[0] ^ b[1]):
            fiber_edges[a[0] ^ a[1]] += 1

    def connected(vs) -> bool:
        seen = {vs[0]}
        stack = [vs[0]]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen == set(vs)

    def bipartite66(vs) -> bool:
        color: dict[tuple, int] = {}
        for s in vs:
            if s in color:
                continue
            color[s] = 0
            stack = [s]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w in color:
                        if color[w] == color[u]:
                            return False
                    else:
                        color[w] = 1 - color[u]
                        stack.append(w)
        return sum(color[v] for v in vs) == len(vs) // 2

    connected_ok = sum(1 for L in range(9, 16) if connected(fib[L]))
    bipartite_ok = sum(1 for L in range(9, 16) if len(fib[L]) == 12 and bipartite66(fib[L]))

    lines = []
    for L in range(9, 16):
        lines.append(f"FIBER {L * 10000 + len(fib[L]) * 100 + fiber_edges[L]}")
    ok = (all(len(fib[L]) == 12 and fiber_edges[L] == 24 for L in range(9, 16))
          and len(part) == 84 and degree_bad == 0 and intra_bad == 0)
    for ln in sorted(lines):
        print(ln)
    print(f"PARTICIPATE {len(part)}")
    print(f"DEGREE_BAD {degree_bad}")
    print(f"INTRA_BAD {intra_bad}")
    print(f"BIPARTITE_OK {bipartite_ok}")
    print(f"CONNECTED_OK {connected_ok}")
    print(f"FIBERS {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
