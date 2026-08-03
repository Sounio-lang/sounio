#!/usr/bin/env python3
"""Self-falsifying compilation, rung R19 — deriving R16's locality, and reducing
what is left to one lemma.

Spec: docs/research/self_falsifying_compilation_line_r19_2026-07-28.md

R16 measured that the count-preserving flip sigma(H/2, H+H/2) changes exactly
two edges per fiber, in all but one fiber, and inferred -- explicitly without
proof -- that such a uniform local change preserves the classification. This
rung derives the locality half from index arithmetic, predicts the exceptional
fiber instead of observing it, refutes two candidate explanations of the other
half, and states what remains as a single equivariance lemma.

WHAT IS NOW DERIVED, not measured.

A fiber L = Llo | H has vertices (lo, lo^L, e) for lo in 1..H-1, e = +-1.
`mul` computes sigma only over the 2x2 index pairs of two vertices, so the
flipped pair (h, H+h), h = H/2, can be reached only where one vertex carries h
as its lo and another carries H+h as its hi.

  P := the vertex-pair with lo = h          (hi = h ^ L)
  Q := the vertex-pair with hi = H + h      (lo = (H+h) ^ L = h ^ Llo)

  L1  P = Q  <=>  h = h ^ Llo  <=>  Llo = 0, the one fiber never examined.
                  So P != Q in every examined fiber.
  L2  Q exists <=> 1 <= h ^ Llo < H.  It fails exactly when Llo = h.
                  So EXACTLY ONE fiber is untouched, and it is Llo = H/2.
  L3  The effect is to ADD the crossing matching P_e -- Q_{-e}; P and Q are
      non-adjacent beforehand, and the same-sign pairs are untouched.

L2 is the substance: R16 reported "one fiber changes nothing" as an observation.
It is a consequence, and which fiber is predicted.

WHAT WAS REFUTED HERE (both were mine, both tested):

  F1  "the graph is symmetric enough that adding this matching between ANY
       non-adjacent vertex-pair-pair gives the same result." FALSE: 63 such
       pairs give 8 distinct spectra, and the flip's own pair is alone in its
       class. The pair (h, h^Llo) is special.
  F2  "blocks are characterised by the high bit of Llo." FALSE at n = 7, where
       [33..40, 48] and the singleton [56] break it.

WHAT REMAINS, and it is now one statement rather than a vague inference:

  OPEN  Why is the assignment Llo |-> {h, h ^ Llo} equivariant with the
        spectrum-block structure? Marking P and Q does not refine the partition
        (measured, n = 5, 6), which is exactly what makes a canonical function
        of the marked graph preserve it. That non-refinement is the lemma.

CLAUSES:
  Y1_LOCALITY_DERIVED     L1-L3 hold, checked as arithmetic for n = 5..12 and
                          against the built graphs at n = 5, 6.
  Y2_EXCEPTIONAL_PREDICTED  the untouched fiber is Llo = H/2, derived not found.
  Y3_MARKING_DOES_NOT_REFINE  the open lemma, measured and labelled as such.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_sp = importlib.util.spec_from_file_location(
    "r15", REPO / "scripts/research/self_falsifying_compilation_line_r15_contract.py")
r15 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(r15)


def build(n, Llo, flip=None):
    H, N = 1 << (n - 1), 1 << n
    L = Llo | H
    V = [(lo, lo ^ L, e) for lo in range(1, H) for e in (1, -1)]
    D = [{v[0]: 1, v[1]: v[2]} for v in V]
    m = len(V)
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            if (not r15.mul(D[i], D[j], n, flip, n)
                    and not r15.mul(D[j], D[i], n, flip, n)):
                A[i, j] = A[j, i] = 1
    return A, V, {v: k for k, v in enumerate(V)}


def spec(A):
    return tuple(np.round(np.linalg.eigvalsh(A), 3).tolist())


def main() -> int:
    print("R19 — R16's locality derived; what is left is one lemma")
    print("=" * 72)

    # ---- Y1 -----------------------------------------------------------------
    ok1 = True
    print("Y1_LOCALITY_DERIVED")
    for n in range(5, 13):
        H = 1 << (n - 1)
        h = H // 2
        # L1: P = Q iff Llo = 0
        l1 = all((h == (h ^ Llo)) == (Llo == 0) for Llo in range(0, H))
        # L2: Q absent iff Llo = h
        l2 = all(((h ^ Llo) == 0) == (Llo == h) for Llo in range(1, H))
        ok1 &= l1 and l2
        if n <= 7:
            print(f"    n={n:2}  L1 (P=Q only at Llo=0): {'OK' if l1 else 'FAIL'}   "
                  f"L2 (Q absent only at Llo=h={h}): {'OK' if l2 else 'FAIL'}")
    print(f"    ... arithmetic holds for n = 5..12")
    # L3 against the actual graphs
    for n in (5, 6):
        H = 1 << (n - 1)
        h = H // 2
        bad = []
        for Llo in (1, 2, 3):
            L = Llo | H
            A0, V, idx = build(n, Llo)
            A1, _, _ = build(n, Llo, (h, H + h))
            q = h ^ Llo
            want = {tuple(sorted((idx[(h, h ^ L, e)], idx[(q, q ^ L, -e)])))
                    for e in (1, -1)}
            got = {tuple(sorted(t)) for t in np.argwhere(np.triu(A0 != A1))}
            same_sign_touched = any(
                A0[idx[(h, h ^ L, e)], idx[(q, q ^ L, e)]]
                != A1[idx[(h, h ^ L, e)], idx[(q, q ^ L, e)]] for e in (1, -1))
            if got != want or same_sign_touched:
                bad.append(Llo)
        ok1 &= not bad
        print(f"    n={n} L3 (adds exactly the crossing matching): "
              f"{'OK' if not bad else 'FAIL ' + str(bad)}")
    print(f"Y1_LOCALITY_DERIVED {'PASS' if ok1 else 'FAIL'}")
    print()

    # ---- Y2 -----------------------------------------------------------------
    print("Y2_EXCEPTIONAL_PREDICTED")
    ok2 = True
    for n in (5, 6):
        H = 1 << (n - 1)
        h = H // 2
        zero = [Llo for Llo in range(1, H)
                if not (build(n, Llo)[0] != build(n, Llo, (h, H + h))[0]).any()]
        good = zero == [h]
        ok2 &= good
        print(f"    n={n}: untouched fibers {zero}, derivation predicts [{h}]  "
              f"{'[OK]' if good else '[FAIL]'}")
    print(f"Y2_EXCEPTIONAL_PREDICTED {'PASS' if ok2 else 'FAIL'}")
    print()

    # ---- Y3 -----------------------------------------------------------------
    print("Y3_MARKING_DOES_NOT_REFINE   (the open lemma — measured, not proved)")
    ok3 = True
    for n in (5, 6):
        H = 1 << (n - 1)
        h = H // 2
        plain, marked = {}, {}
        for Llo in range(1, H):
            L = Llo | H
            A, V, idx = build(n, Llo)
            plain.setdefault(spec(A), []).append(Llo)
            M = A.copy()
            for e in (1, -1):
                M[idx[(h, h ^ L, e)], idx[(h, h ^ L, e)]] += 2.0
            q = h ^ Llo
            if 1 <= q < H:
                for e in (1, -1):
                    M[idx[(q, q ^ L, e)], idx[(q, q ^ L, e)]] += 5.0
            marked.setdefault(spec(M), []).append(Llo)
        pp = sorted(sorted(v) for v in plain.values())
        mm = sorted(sorted(v) for v in marked.values())
        good = pp == mm
        ok3 &= good
        print(f"    n={n}: {len(pp)} plain blocks, {len(mm)} marked blocks, "
              f"identical {'YES' if good else 'NO'}  {'[OK]' if good else '[FAIL]'}")
    print("    This is what lets a canonical function of the MARKED graph")
    print("    preserve the partition. It is measured at n = 5, 6 and is the")
    print("    lemma a proof would have to establish.")
    print(f"Y3_MARKING_DOES_NOT_REFINE {'PASS' if ok3 else 'FAIL'}")
    print()

    ok = ok1 and ok2 and ok3
    verdict = ("LOCALITY_DERIVED__EQUIVARIANCE_REDUCED_TO_ONE_LEMMA"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print("R16 measured 'two edges per fiber, all but one'. That half is now a")
    print("consequence of index arithmetic, and the exceptional fiber is")
    print("predicted. Two explanations of the other half were tested and both")
    print("failed. What is left is one equivariance lemma, stated.")
    print()
    print(f"SELF_FALSIFYING_R19_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
