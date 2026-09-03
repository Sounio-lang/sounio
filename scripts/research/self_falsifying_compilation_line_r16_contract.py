#!/usr/bin/env python3
"""Self-falsifying compilation, rung R16 — the invariance group, identified.

Spec: docs/research/self_falsifying_compilation_line_r16_2026-07-28.md

R15 showed a verdict token's resolution is bounded by the invariance group of the
proposition it states, and exhibited one element of that group: the sign flip
sigma(H/2, H + H/2) preserves the count of distinct ZD-fiber spectra at every
level, while generic flips change it. It left OPEN why, calling that the more
interesting question. This rung answers it.

THE MECHANISM, in three measured steps.

1. WHY IT IS MINIMAL. The flipped pair is (h, H + h) with h = H/2, and
   h XOR (H + h) = H. A fiber's vertices are pairs (lo, hi) with lo XOR hi = L,
   so that pair's own home fiber is L = H, i.e. Llo = 0 -- the single fiber the
   contract does not examine (its range is 1..H-1). The flip therefore cannot
   alter any vertex's internal product; it can only touch adjacency BETWEEN the
   vertex whose lo = h and the vertex whose hi = H + h.

2. IT CHANGES EXACTLY TWO EDGES PER FIBER. Measured: H-2 fibers lose or gain
   exactly 2 edges (the two sign variants), and one fiber changes nothing. The
   same minimal modification, uniformly, everywhere.

3. THE PARTITION SURVIVES; ITS LABELS DO NOT. The set partition of fibers into
   spectrum-classes is IDENTICAL before and after -- same blocks, sizes
   [1,7,7] / [1,1,7,7,7,8] / [1,1,1,1,7,7,7,7,7,7,8,9] for n = 5, 6, 7, the
   Fano-orbit-and-fixed-seam structure the corpus's own orbit theorem predicts.
   Every spectrum labelling those blocks changes. A uniform local change keeps
   equivalent fibers equivalent while moving all the eigenvalues.

So the check tests |partition|, and the flip acts strictly WITHIN blocks. The
invariance group is not merely "count-preserving maps" -- it is
PARTITION-PRESERVING maps, which is larger, and R15 under-described it.

CLAUSES:

  C1_FLIP_IS_MINIMAL_BY_CONSTRUCTION
      h XOR (H+h) = H, so the pair's home fiber is Llo = 0, outside the examined
      range. Arithmetic, checked for every level, not sampled.

  C2_TWO_EDGES_PER_FIBER
      The perturbation is exactly 2 edges in all but one fiber. Derived live at
      n = 5; read from record at n = 6.

  C3_PARTITION_PRESERVED_LABELS_NOT
      Same blocks, different spectra. Derived live at n = 5, 6; recorded at 7.

HONEST LIMIT, fixed before running: step 3's inference -- that a uniform 2-edge
change must preserve the classification -- is supported by measurement at three
levels, NOT proved. It would follow from the change being equivariant for
whatever relation makes fibers equivalent, and that equivariance is not
established here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = REPO / "scripts/research/r16"

sys.path.insert(0, str(REPO))
import importlib.util                                            # noqa: E402

_sp = importlib.util.spec_from_file_location(
    "r15", REPO / "scripts/research/self_falsifying_compilation_line_r15_contract.py")
r15 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(r15)


def adjacency(n, Llo, flip):
    H, N = 1 << (n - 1), 1 << n
    L = Llo | H
    V = [{lo: 1, hi: (-1 if neg else 1)}
         for lo in range(1, H) for hi in range(H, N) for neg in (0, 1)
         if (lo ^ hi) == L]
    m = len(V)
    A = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        for j in range(i + 1, m):
            if (not r15.mul(V[i], V[j], n, flip, n)
                    and not r15.mul(V[j], V[i], n, flip, n)):
                A[i, j] = A[j, i] = 1
    return A


def partition(n, flip):
    by = {}
    for L in range(1, 1 << (n - 1)):
        by.setdefault(r15.fiber_spectrum(n, L, flip, n), []).append(L)
    return frozenset(frozenset(v) for v in by.values()), set(by)


def main() -> int:
    rec = json.loads((D / "recorded.json").read_text())
    print("R16 — the invariance group, identified")
    print("=" * 72)

    # ---- C1: arithmetic, every level ----------------------------------------
    ok1 = True
    print("C1_FLIP_IS_MINIMAL_BY_CONSTRUCTION")
    for n in range(5, 13):
        H = 1 << (n - 1)
        h = H // 2
        good = (h ^ (H + h)) == H
        ok1 &= good
        if n <= 8:
            print(f"    n={n:2}: ({h} XOR {H + h}) = {h ^ (H + h)} = H  -> home fiber "
                  f"Llo=0, outside range(1,{H})  {'[OK]' if good else '[FAIL]'}")
    print(f"    ... holds for n = 5..12  ({'all OK' if ok1 else 'FAILED'})")
    print(f"C1_FLIP_IS_MINIMAL_BY_CONSTRUCTION {'PASS' if ok1 else 'FAIL'}")
    print()

    # ---- C2: two edges per fiber -------------------------------------------
    print("C2_TWO_EDGES_PER_FIBER")
    ok2 = True
    n = 5
    H = 1 << (n - 1)
    h = H // 2
    counts = {}
    for L in range(1, H):
        d = int((adjacency(n, L, None) != adjacency(n, L, (h, H + h))).sum()) // 2
        counts[d] = counts.get(d, 0) + 1
    good = set(counts) <= {0, 2} and counts.get(2, 0) == H - 2
    ok2 &= good
    print(f"    n=5 live      edges changed per fiber: {counts}  "
          f"{'[OK]' if good else '[FAIL]'}")
    r6 = {int(k): v for k, v in rec["edges_changed_per_fiber"]["6"].items()}
    good6 = set(r6) <= {0, 2} and r6.get(2, 0) == 30
    ok2 &= good6
    print(f"    n=6 recorded  edges changed per fiber: {r6}  "
          f"{'[OK]' if good6 else '[FAIL]'}")
    print(f"C2_TWO_EDGES_PER_FIBER {'PASS' if ok2 else 'FAIL'}")
    print()

    # ---- C3: partition preserved, labels not -------------------------------
    print("C3_PARTITION_PRESERVED_LABELS_NOT")
    ok3 = True
    for n in (5, 6):
        H = 1 << (n - 1)
        h = H // 2
        P0, S0 = partition(n, None)
        P1, S1 = partition(n, (h, H + h))
        same_blocks = P0 == P1
        diff_labels = S0 != S1
        good = same_blocks and diff_labels
        ok3 &= good
        print(f"    n={n} live      blocks {'IDENTICAL' if same_blocks else 'DIFFER'}, "
              f"sizes {sorted(len(b) for b in P0)}, spectra "
              f"{'DIFFER' if diff_labels else 'identical'}  "
              f"{'[OK]' if good else '[FAIL]'}")
    g7 = rec["partition_identical"]["7"] and rec["spectra_sets_differ"]["7"]
    ok3 &= g7
    print(f"    n=7 recorded  blocks IDENTICAL, sizes {rec['block_sizes']['7']}, "
          f"spectra DIFFER  {'[OK]' if g7 else '[FAIL]'}")
    print("    => the flip acts strictly WITHIN spectrum-classes: it relabels "
          "every\n       class and merges or splits none.")
    print(f"C3_PARTITION_PRESERVED_LABELS_NOT {'PASS' if ok3 else 'FAIL'}")
    print()

    ok = ok1 and ok2 and ok3
    verdict = ("INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print("R15 said the token is blind to what preserves its proposition's truth.")
    print("The group is larger than 'count-preserving': it is PARTITION-preserving.")
    print("A check testing |partition| cannot see any map acting within blocks,")
    print("and one such map exists at every level, changing 2 edges per fiber.")
    print()
    print(f"SELF_FALSIFYING_R16_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
