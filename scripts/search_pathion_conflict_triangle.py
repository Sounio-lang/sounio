#!/usr/bin/env python3
"""Search for a pathion associator-conflict triangle.

Mirrors the Lean definitions in SounioPathionBridge + SounioErdosUnitDistance.
Pathion level = 5 (32 dimensions). Edge (x,y) iff ||[x,y,c]||^2 == 4,
where [x,y,c] = (x*y)*c - x*(y*c).
"""

from itertools import combinations


def cd_sigma(a: int, b: int, bits: int) -> int:
    """Recursive Cayley-Dickson sign function (matches Lean cdSigma)."""
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi = 1 if a >= half else 0
    b_hi = 1 if b >= half else 0
    a_lo = a & (half - 1)
    b_lo = b & (half - 1)
    if a_hi == 0 and b_hi == 0:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if a_hi == 0 and b_hi == 1:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi == 1 and b_hi == 0:
        if b_lo == 0:
            return cd_sigma(a_lo, b_lo, bits - 1)
        else:
            return -cd_sigma(a_lo, b_lo, bits - 1)
    # both hi
    if b_lo == 0:
        return -cd_sigma(b_lo, a_lo, bits - 1)
    return cd_sigma(b_lo, a_lo, bits - 1)


def path_sigma(a: int, b: int) -> int:
    return cd_sigma(a, b, 5)


def smul(a: list, b: list) -> list:
    """Pathion product of two sparse vectors (list of (idx, coeff)).
    Returns combined sparse vector."""
    out = {}
    for ia, ca in a:
        for ib, cb in b:
            k = ia ^ ib
            c = ca * cb * path_sigma(ia, ib)
            out[k] = out.get(k, 0) + c
    return [(k, v) for k, v in out.items() if v != 0]


def nsq(v: list) -> int:
    return sum(c * c for _, c in v)


def prim_vec(lo: int, hi: int, neg: bool) -> list:
    return [(lo, 1), (hi, -1 if neg else 1)]


def basis_vec(i: int) -> list:
    return [(i, 1)]


def neg_vec(v: list) -> list:
    return [(k, -c) for k, c in v]


def sub_vec(a: list, b: list) -> list:
    out = {}
    for k, c in a:
        out[k] = out.get(k, 0) + c
    for k, c in b:
        out[k] = out.get(k, 0) - c
    return [(k, v) for k, v in out.items() if v != 0]


def add_vec(a: list, b: list) -> list:
    out = {}
    for k, c in a:
        out[k] = out.get(k, 0) + c
    for k, c in b:
        out[k] = out.get(k, 0) + c
    return [(k, v) for k, v in out.items() if v != 0]


def assoc_normsq(x: list, y: list, c: list) -> int:
    xy = smul(x, y)
    xy_c = smul(xy, c)
    yc = smul(y, c)
    x_yc = smul(x, yc)
    diff = sub_vec(xy_c, x_yc)
    return nsq(diff)


def valid_path_prims():
    prims = []
    for lo in range(1, 16):
        for hi in range(17, 32):
            if (lo ^ hi) != 16:
                for neg in [False, True]:
                    prims.append((lo, hi, neg))
    return prims


def main():
    prims = valid_path_prims()
    print(f"valid pathion primitives: {len(prims)}")

    # Search for triangle on single basis-vector triples in the lower half first.
    found = False
    for r in [range(1, 16), range(17, 32)]:
        for i, j, k in combinations(r, 3):
            for lo, hi, neg in prims:
                c = prim_vec(lo, hi, neg)
                a = assoc_normsq(basis_vec(i), basis_vec(j), c)
                b = assoc_normsq(basis_vec(i), basis_vec(k), c)
                c2 = assoc_normsq(basis_vec(j), basis_vec(k), c)
                if a == 4 and b == 4 and c2 == 4:
                    print(f"TRIANGLE on e_{i}, e_{j}, e_{k} with c = e_{lo} {'-' if neg else '+'} e_{hi}")
                    print(f"  associator norms: {a}, {b}, {c2}")
                    found = True
                    break
            if found:
                break
        if found:
            break

    if not found:
        print("No triangle on single-basis triples within one half; searching mixed triples...")
        candidates = list(range(1, 8)) + list(range(17, 24))
        for i, j, k in combinations(candidates, 3):
            for lo, hi, neg in prims:
                c = prim_vec(lo, hi, neg)
                a = assoc_normsq(basis_vec(i), basis_vec(j), c)
                b = assoc_normsq(basis_vec(i), basis_vec(k), c)
                c2 = assoc_normsq(basis_vec(j), basis_vec(k), c)
                if a == 4 and b == 4 and c2 == 4:
                    print(f"TRIANGLE on e_{i}, e_{j}, e_{k} with c = e_{lo} {'-' if neg else '+'} e_{hi}")
                    print(f"  associator norms: {a}, {b}, {c2}")
                    found = True
                    break
            if found:
                break

    if not found:
        print("Still no triangle. Searching weight-2 probes...")
        # Probe family: {0} ∪ {e_i} ∪ {e_a+e_b} over small index ranges.
        lower = list(range(1, 8))
        upper = list(range(17, 24))
        points = [(0, basis_vec(0))]
        for i in lower + upper:
            points.append((i, basis_vec(i)))
        for a, b in combinations(lower, 2):
            points.append((f"{a}+{b}", add_vec(basis_vec(a), basis_vec(b))))
        for a, b in combinations(upper, 2):
            points.append((f"{a}+{b}", add_vec(basis_vec(a), basis_vec(b))))

        for lo, hi, neg in prims:
            c = prim_vec(lo, hi, neg)
            edges = []
            for (n1, p1), (n2, p2) in combinations(points, 2):
                if assoc_normsq(p1, p2, c) == 4 or assoc_normsq(p2, p1, c) == 4:
                    edges.append((n1, n2))
            # Look for a triangle.
            adj = {}
            for n1, n2 in edges:
                adj.setdefault(n1, set()).add(n2)
                adj.setdefault(n2, set()).add(n1)
            for n1 in adj:
                for n2 in adj[n1]:
                    common = adj[n1].intersection(adj[n2])
                    if common:
                        n3 = next(iter(common))
                        print(f"TRIANGLE in weight-2 probe: {n1} -- {n2} -- {n3} -- {n1}")
                        print(f"  c = e_{lo} {'-' if neg else '+'} e_{hi}")
                        found = True
                        break
                if found:
                    break
            if found:
                break

    if not found:
        print("No triangle found in searched configurations.")


if __name__ == "__main__":
    main()
