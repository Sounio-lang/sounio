#!/usr/bin/env python3
"""Resolve L8_64_192: benign, or a cospectral counterexample?

R14 found exactly one level-8 sign flip that leaves the ZD-fiber spectral
contract's verdict unchanged: sigma(64, 192). Three explanations were refuted
(shared blind spot, doubling diagonal, arithmetic form). The remaining fork is
the one that matters, and it is decidable:

  (a) the flip leaves every fiber's ADJACENCY MATRIX untouched
      -> the product never participates in an annihilation; fully benign, and
         the contract is right not to react;
  (b) some adjacency matrix CHANGES while its spectrum does not
      -> a cospectral pair of distinct annihilation graphs, i.e. the spectrum
         fails to separate them. The contract's claim
         (ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8) is about graphs from the
         real CD tower, and a perturbed sign table is outside that family -- so
         (b) would NOT refute the claim. It would show the CHECK cannot detect
         the difference, which is a different and weaker statement, and it is
         the one this rung is entitled to make.

Independent re-implementation of the committed construction (fresh code, cross
checked against the contract's own reported spectrum counts 6/12/24 for
n = 6, 7, 8 before anything is concluded).
"""
from __future__ import annotations

import numpy as np

FLIP = (64, 192)          # the surviving perturbation, at level 8


def sigma(a, b, bits, flip=None):
    """Cayley-Dickson basis sign, written from the recursion, not copied."""
    s = _sigma(a, b, bits)
    if flip and bits == 8 and (a, b) in (flip, (flip[1], flip[0])):
        return -s
    return s


def _sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    h = 1 << (bits - 1)
    aH, bH, aL, bL = a >= h, b >= h, a & (h - 1), b & (h - 1)
    if not aH and not bH:
        return _sigma(aL, bL, bits - 1)
    if not aH and bH:
        return _sigma(bL, aL, bits - 1)
    if aH and not bH:
        return _sigma(aL, bL, bits - 1) if bL == 0 else -_sigma(aL, bL, bits - 1)
    return -_sigma(bL, aL, bits - 1) if bL == 0 else _sigma(bL, aL, bits - 1)


def mul(x, y, bits, flip):
    out = {}
    for i, ci in x.items():
        for j, cj in y.items():
            k = i ^ j
            out[k] = out.get(k, 0) + sigma(i, j, bits, flip) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def fiber_adj(n, Llo, flip=None):
    H, N = 1 << (n - 1), 1 << n
    L = Llo | H
    V = [{lo: 1, hi: (-1 if neg else 1)}
         for lo in range(1, H) for hi in range(H, N) for neg in (0, 1)
         if (lo ^ hi) == L]
    m = len(V)
    A = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        for j in range(i + 1, m):
            if not mul(V[i], V[j], n, flip) and not mul(V[j], V[i], n, flip):
                A[i, j] = A[j, i] = 1
    return A, V


def spec(A):
    return tuple(np.round(np.linalg.eigvalsh(A.astype(float)), 3).tolist())


def main():
    # cross-check the fresh implementation against the contract's own figures
    print("cross-check: distinct spectra per level (contract reports 6 / 12 / 24)")
    for n in (6, 7):
        s = {spec(fiber_adj(n, Llo)[0]) for Llo in range(1, 1 << (n - 1))}
        print(f"  n={n}: {len(s)}  (expected {3 * 2 ** (n - 5)})")

    n = 8
    print(f"\nn=8: comparing every fiber with and without sigma{FLIP} flipped")
    changed_adj, changed_spec, same_spec_diff_adj = [], [], []
    for Llo in range(0, 1 << (n - 1)):          # include Llo=0: L=128, the
        A0, V = fiber_adj(n, Llo)               # fiber that CONTAINS (64,192)
        A1, _ = fiber_adj(n, Llo, FLIP)
        da = not np.array_equal(A0, A1)
        ds = spec(A0) != spec(A1)
        if da:
            changed_adj.append(Llo)
        if ds:
            changed_spec.append(Llo)
        if da and not ds:
            same_spec_diff_adj.append((Llo, int((A0 != A1).sum()) // 2))

    print(f"  fibers whose ADJACENCY changes: {len(changed_adj)}")
    print(f"  fibers whose SPECTRUM changes : {len(changed_spec)}")
    print(f"  adjacency changed but spectrum did NOT: {len(same_spec_diff_adj)}")
    for Llo, ne in same_spec_diff_adj[:8]:
        print(f"      Llo={Llo}  ({ne} edges differ)  -> COSPECTRAL, NON-IDENTICAL")

    print()
    if not changed_adj:
        print("VERDICT (a): the flip touches no adjacency. The product never")
        print("  participates in an annihilation; the contract is RIGHT not to react.")
    elif same_spec_diff_adj:
        print("VERDICT (b): some fiber graph CHANGES while its spectrum does not.")
        print("  The spectrum cannot see this difference. Note the scope limit in")
        print("  the docstring: the perturbed table is outside the CD family, so")
        print("  this does NOT refute the n<=8 completeness claim.")
    else:
        print("VERDICT (c): adjacency and spectrum both change on some fiber --")
        print("  so the contract's INSENSITIVITY is not explained by either. The")
        print("  verdict token must be aggregating over something coarser.")


if __name__ == "__main__":
    main()
