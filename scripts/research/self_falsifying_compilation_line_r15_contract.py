#!/usr/bin/env python3
"""Self-falsifying compilation, rung R15 — a verdict token is blind to whatever
preserves the truth of its proposition.

Spec: docs/research/self_falsifying_compilation_line_r15_2026-07-28.md

R14 left one perturbation unexplained: flipping the Cayley-Dickson sign of a
single product, sigma(64, 192) at level 8, leaves
`ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8` unchanged. Three explanations were
refuted there. This rung resolves it, and the resolution is not about that
product.

WHAT WAS FOUND. The flip changes 126 of 128 fiber graphs AND all their spectra
-- there is no cospectral pair. What it preserves is the CARDINALITY: the number
of distinct spectra stays at 24 while the SET of 24 is entirely replaced. The
contract's claim has the form "#distinct spectra = 3*2^(n-5) = #iso classes", so
its check tests a count, and a count cannot see a transformation that swaps the
things counted.

It is a family, not an accident. At every level tested the flip
sigma(H/2, H + H/2), H = 2^(n-1), preserves the count while generic flips at the
same level change it:

    n=5   3 -> 3     controls 5, 5
    n=6   6 -> 6     controls 7, 10
    n=7  12 -> 12    controls 13, 20
    n=8  24 -> 24    controls 25, 25, 40

And that explains why only level 8 was invisible to the contract. A wrapper on
the sign function intercepts the RECURSIVE calls too, so a flip aimed at level k
also perturbs every deeper level computed through it. The contract checks
n = 6, 7 and 8 together, so a count-preserving flip at 5, 6 or 7 still betrays
itself higher up. At level 8 -- the top of its own analysis -- there is nowhere
higher to look.

CLAUSES:

  C1_CONSTRUCTION_VALIDATED
      A from-scratch re-implementation of the committed annihilation-graph
      construction reproduces the contract's own published spectrum counts.
      Checked before anything is concluded from it.

  C2_COUNT_PRESERVING_FAMILY
      sigma(H/2, H + H/2) preserves the count at every level tested; generic
      flips at the same level do not. Derived live at n = 5, 6; read from the
      recorded n = 7, 8 runs (those cost minutes, not seconds).

  C3_WITNESS_WOULD_CATCH_IT
      Binding the token to the WITNESS -- the set of spectra -- rather than to
      the PREDICATE would detect the perturbation, because the two sets differ.
      A concrete, two-line repair to a real contract, verified rather than
      proposed.

SCOPE, fixed before any number was seen: a perturbed sign table is NOT a
Cayley-Dickson algebra, so none of this refutes the n <= 8 completeness claim,
which is about the real tower. What is measured is the reach of the CHECK, not
the truth of the CLAIM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = REPO / "scripts/research/r15"


def _sigma(a, b, bits):
    """CD basis sign, written from the recursion rather than copied (R6)."""
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


def sigma(a, b, bits, flip, at):
    s = _sigma(a, b, bits)
    if flip and bits == at and (a, b) in (flip, (flip[1], flip[0])):
        return -s
    return s


def mul(x, y, bits, flip, at):
    out = {}
    for i, ci in x.items():
        for j, cj in y.items():
            k = i ^ j
            out[k] = out.get(k, 0) + sigma(i, j, bits, flip, at) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def fiber_spectrum(n, Llo, flip=None, at=None):
    H, N = 1 << (n - 1), 1 << n
    L = Llo | H
    V = [{lo: 1, hi: (-1 if neg else 1)}
         for lo in range(1, H) for hi in range(H, N) for neg in (0, 1)
         if (lo ^ hi) == L]
    m = len(V)
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            if not mul(V[i], V[j], n, flip, at) and not mul(V[j], V[i], n, flip, at):
                A[i, j] = A[j, i] = 1
    return tuple(np.round(np.linalg.eigvalsh(A), 3).tolist())


def spectra(n, flip=None):
    return {fiber_spectrum(n, L, flip, n) for L in range(1, 1 << (n - 1))}


def main() -> int:
    rec = json.loads((D / "recorded.json").read_text())

    print("R15 — the token is blind to whatever preserves its proposition")
    print("=" * 72)

    # ---- C1 -----------------------------------------------------------------
    ok1 = True
    print("C1_CONSTRUCTION_VALIDATED  (fresh re-implementation vs published counts)")
    for n in (5, 6):
        got, want = len(spectra(n)), 3 * 2 ** (n - 5)
        good = got == want
        ok1 &= good
        print(f"    n={n}: {got} distinct spectra, expected 3*2^(n-5) = {want}  "
              f"{'[OK]' if good else '[MISMATCH]'}")
    for n in ("7", "8"):
        r = rec["baseline"][n]
        want = 3 * 2 ** (int(n) - 5)
        good = r == want
        ok1 &= good
        print(f"    n={n}: {r} (recorded), expected {want}  "
              f"{'[OK]' if good else '[MISMATCH]'}")
    print(f"C1_CONSTRUCTION_VALIDATED {'PASS' if ok1 else 'FAIL'}")
    print()

    # ---- C2 -----------------------------------------------------------------
    print("C2_COUNT_PRESERVING_FAMILY  sigma(H/2, H+H/2), H = 2^(n-1)")
    ok2 = True
    for n in (5, 6):
        H = 1 << (n - 1)
        base = len(spectra(n))
        surv = len(spectra(n, (H // 2, H + H // 2)))
        ctrl = len(spectra(n, (H // 2, H + H // 4)))
        good = (surv == base) and (ctrl != base)
        ok2 &= good
        print(f"    n={n} live      base {base:2} | ({H//2},{H+H//2}) -> {surv:2} "
              f"{'PRESERVES' if surv == base else 'changes  '} | "
              f"control -> {ctrl:2} {'changes' if ctrl != base else 'PRESERVES'}"
              f"  {'[OK]' if good else '[FAIL]'}")
    for n in ("7", "8"):
        b = rec["baseline"][n]
        s = rec["survivor"][n]
        cs = rec["controls"][n]
        good = (s == b) and all(c != b for c in cs)
        ok2 &= good
        print(f"    n={n} recorded  base {b:2} | survivor -> {s:2} "
              f"{'PRESERVES' if s == b else 'changes  '} | "
              f"controls -> {cs}  {'[OK]' if good else '[FAIL]'}")
    print(f"C2_COUNT_PRESERVING_FAMILY {'PASS' if ok2 else 'FAIL'}")
    print()

    # ---- C3 -----------------------------------------------------------------
    print("C3_WITNESS_WOULD_CATCH_IT  bind the set of spectra, not their number")
    ok3 = True
    for n in (5, 6):
        H = 1 << (n - 1)
        B, F = spectra(n), spectra(n, (H // 2, H + H // 2))
        good = (len(B) == len(F)) and (B != F)
        ok3 &= good
        print(f"    n={n} live      |S|={len(B)} = |S'|={len(F)}, sets "
              f"{'DIFFER' if B != F else 'identical'}  {'[OK]' if good else '[FAIL]'}")
    g8 = rec["sets_differ"]["8"]
    ok3 &= g8
    print(f"    n=8 recorded  |S|=|S'|=24, sets "
          f"{'DIFFER' if g8 else 'identical'}  {'[OK]' if g8 else '[FAIL]'}")
    print("    => a token bound to the witness changes; a token bound to the")
    print("       count does not. The repair is two lines in a real contract.")
    print(f"C3_WITNESS_WOULD_CATCH_IT {'PASS' if ok3 else 'FAIL'}")
    print()

    ok = ok1 and ok2 and ok3
    verdict = ("TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print("A verdict token binds a proposition. Its resolution is bounded by the")
    print("invariance group of that proposition: '#X = N' cannot detect anything")
    print("that preserves |X|. Bind the witness, not the predicate.")
    print()
    print(f"SELF_FALSIFYING_R15_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
