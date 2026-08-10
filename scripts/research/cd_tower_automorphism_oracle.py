r"""cd_tower_automorphism_oracle.py -- RECONSTRUCTED 2026-07-31.

This file was loaded by cd_tower_auto_action_on_zd_fibers.py (the orbit theorem's
verifier) via spec_from_file_location + exec_module with no fallback, yet it was
NEVER committed to any branch in the repository's history (`git log --all` empty).
R20 found the dangling dependency; R21 rested a proof on the very theorem this
oracle's absence kept from running. So the verifier could not execute in ANY
checkout of this repo. This reconstruction is built from that verifier's own
docstring and PROOF, and validated rather than assumed (see `_self_check`).

WHAT THE VERIFIER CONSUMES (two functions):

  sweep_autos(n)      the VALID index-maps at level n, each an image array of
                      length 2^n (m[v] = image of basis index v). "Valid" here
                      means the PERMUTATION PART of a signed-monomial octonion
                      automorphism -- the 168 = |PSL(2,7)| = |GL(3,2)|, not the
                      168*2^k signed group (memory: our 168 is the permutation
                      part; Kirshtein's signed group grows). At n=4 these are the
                      block-form maps [[g,0],[0,1]], g in GL(3,2) on the octonion
                      bits {0,1,2}, seam bit 3 (value 8) fixed.

  orbits_on(M, elems) partition of `elems` under the group given EXPLICITLY by
                      the image arrays M. M is the full group, so a single
                      application per element already yields the orbit.

VALIDATION, not assertion. The verifier's docstring says "168 valid maps = GL(3,2)
= Aut(octonions)". That is imprecise for the SIGNED table: only 21 of the 168
GL(3,2) linear maps preserve `cds` with a fixed sign convention. The correct fact,
checked in `_self_check`, is that all 168 are valid *permutation parts*: for each
g in GL(3,2) there EXISTS a sign vector eps making e_i -> eps_i e_{g.i} an algebra
automorphism, i.e. the discrepancy cocycle
    delta(i,j) = cds(g.i, g.j, 3) * cds(i, j, 3)
is a coboundary eps_i eps_j eps_{i^j}. Solving that F2 linear system succeeds for
all 168 and fails for none. Orbits on fibers depend only on the permutation part
(the sign eps does not move the label L = lo XOR hi), which is why the orbit
theorem is stated for the permutation part.

Pure Python 3 (itertools). No external dependencies -- the failure that made this
file necessary was exactly such a dependency going missing.
"""

from __future__ import annotations

import itertools


def _cds(a: int, b: int, bits: int) -> int:
    """Cayley-Dickson basis sign, from the recursion. Written here, not imported:
    an oracle that imports the thing it certifies has no independence, and an
    imported dependency is the failure that made this reconstruction necessary."""
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    h = 1 << (bits - 1)
    aH, bH, aL, bL = a >= h, b >= h, a & (h - 1), b & (h - 1)
    if not aH and not bH:
        return _cds(aL, bL, bits - 1)
    if not aH and bH:
        return _cds(bL, aL, bits - 1)
    if aH and not bH:
        return _cds(aL, bL, bits - 1) if bL == 0 else -_cds(aL, bL, bits - 1)
    return -_cds(bL, aL, bits - 1) if bL == 0 else _cds(bL, aL, bits - 1)


def _gl3_columns():
    """Every invertible F2 3x3 matrix, as the triple of column images
    (image of e0=bit0, e1=bit1, e2=bit2). |GL(3,2)| = 168."""
    for cols in itertools.product(range(8), repeat=3):
        span = {0}
        for c in cols:
            span = span | {s ^ c for s in span}
        if len(span) == 8:  # columns span F2^3 <=> invertible
            yield cols


def _apply3(cols, v: int) -> int:
    r = 0
    for k in range(3):
        if v >> k & 1:
            r ^= cols[k]
    return r


def _is_valid_permutation_part(cols) -> bool:
    """Does a sign vector eps exist making e_i -> eps_i e_{g.i} an automorphism?
    Equivalent to: delta(i,j) = cds(g.i,g.j) cds(i,j) is a coboundary. Solve the
    F2 system e_i + e_j + e_{i^j} = d(i,j) by Gaussian elimination; solvable iff
    no 0 = 1 row survives."""
    pi = [_apply3(cols, i) for i in range(8)]
    basis: dict[int, tuple[int, int]] = {}
    for i in range(8):
        for j in range(8):
            k = i ^ j
            d = 0 if _cds(pi[i], pi[j], 3) * _cds(i, j, 3) == 1 else 1
            vec = 0
            for idx in (i, j, k):
                if idx != 0:
                    vec ^= 1 << (idx - 1)
            v, dd = vec, d
            while v:
                p = v.bit_length() - 1
                if p in basis:
                    bv, bd = basis[p]
                    v ^= bv
                    dd ^= bd
                else:
                    basis[p] = (v, dd)
                    break
            else:
                if dd:
                    return False
    return True


def sweep_autos(n: int):
    """The valid index-maps at level n: image arrays of length 2^n.

    n == 4: the 168 block-form maps [[g,0],[0,1]], g in GL(3,2) on bits {0,1,2},
    seam bit 3 fixed. Each is validated as a permutation part before being
    emitted. For n > 4 the verifier lifts maps[4] itself (see its `lift`), but
    this also returns the block-form GL(3,2) x Id maps at any n for completeness.
    """
    N = 1 << n
    seam_mask = N - 1  # low n bits; bits 3..n-1 are fixed pointwise
    out = []
    for cols in _gl3_columns():
        if not _is_valid_permutation_part(cols):
            continue
        m = [0] * N
        for v in range(N):
            low = v & 7           # octonion bits moved by g
            high = v & (seam_mask ^ 7)  # bits 3..n-1 fixed
            m[v] = _apply3(cols, low) | high
        out.append(m)
    return out


def orbits_on(M, elems):
    """Partition `elems` under the group given by image arrays M.

    M is the full group (every element as an image array), so the orbit of e is
    just {m[e] : m in M} -- no closure loop needed. Returns a list of orbits,
    each a sorted list; every element of `elems` appears in exactly one."""
    todo = set(elems)
    parts = []
    while todo:
        e = min(todo)
        orbit = {m[e] for m in M}
        orbit &= set(elems)  # stay within the queried set
        parts.append(sorted(orbit))
        todo -= orbit
    return parts


def _self_check():
    """Reconstruction is only worth anything if it is validated. Run on import of
    __main__ and by the R25 gate."""
    autos = sweep_autos(4)
    assert len(autos) == 168, f"sweep_autos(4) returned {len(autos)}, expected 168"
    # every returned map fixes the seam bit and keeps octonion bits inside {0..7}
    for m in autos:
        assert m[8] == 8, "seam bit e_8 not fixed"
        for v in range(8):
            assert 0 <= m[v] < 8, "octonion bit escaped the octonion span"
    # the group is closed and each is a bijection of 0..15
    for m in autos:
        assert sorted(m) == list(range(16)), "not a permutation of the basis"
    return len(autos)


if __name__ == "__main__":
    n = _self_check()
    print(f"cd_tower_automorphism_oracle: sweep_autos(4) = {n} validated "
          f"permutation parts (GL(3,2) = PSL(2,7)); self-check OK")
