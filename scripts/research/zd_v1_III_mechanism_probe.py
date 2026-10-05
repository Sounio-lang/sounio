#!/usr/bin/env python3
"""(III) mechanism: which proof strategies for the deviation law are still alive.

The law (§33) is  D(W) = tr(A_W^3) - tr(A_(8g+1)^3) = [popcount(g) even] * delta(n, lsb W),
whose open content is that D does not depend on g. This probe tests the natural mechanisms.

  C1  the odd regime, for contrast: popcount(g) odd -> D_k = 0 for EVERY k
      (that is (c), and it is proven by an ISOMORPHISM of the signed graph, so it must hold
      at every power -- the probe confirms the shape of the proven case)

  C2  the triple-level transport a |-> a ^ 8y  (the repaired form of the edge-level map
      already refuted in §33.3): does the per-triple deficit tensor transport?

  C3  disjoint-union trace equality: is  A(8(y+2^i)) (+) A(1)  trace-equal to
      A(8*2^i) (+) A(8y+1)  at every power? Any isomorphism / switching / spectral argument
      would force YES at every k.

  C4  the sharp version: for which k is D_k determined by lsb alone?
"""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, g_of, lsb, sign_table_fast  # noqa: E402


def trk(A, K=7):
    P = np.eye(A.shape[0])
    out = []
    for _ in range(K):
        P = P @ A
        out.append(int(round(float(np.trace(P)))))
    return out


def tri_tensor(A):
    return A[:, :, None] * A[None, :, :] * A.T[:, None, :]


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["7", "8", "9"])]
    for n in levels:
        H = 1 << (n - 1)
        S = sign_table_fast(n)
        A = {W: A_sig_fast(n, W, S).astype(np.float64) for W in range(1, H)}
        T = {W: trk(A[W]) for W in range(1, H)}
        print(f"n={n}")

        # C1 / C4
        byk = [defaultdict(set) for _ in range(7)]
        odd_bad = 0
        for W in range(1, H):
            if W & 7:
                continue
            y = g_of(W)
            if bin(y).count("1") % 2:
                odd_bad += sum(1 for k in range(7) if T[W][k] != T[8 * y + 1][k])
                continue
            for k in range(7):
                byk[k][lsb(W)].add(T[W][k] - T[8 * y + 1][k])
        print(f"  C1  popcount(g) ODD (= (c), proven by isomorphism): "
              f"{odd_bad} deviations over k=1..7 -- expected 0")
        verdict = []
        for k in range(1, 7):
            worst = max((len(v) for v in byk[k].values()), default=0)
            verdict.append(f"k={k+1}:{'YES' if worst <= 1 else f'NO({worst})'}")
        print(f"  C4  D_k determined by lsb alone?  {'  '.join(verdict)}")

        # C2 / C3
        for y in range(1, 1 << (n - 4)):
            if bin(y).count("1") % 2:
                continue
            for i in range((y & -y).bit_length() - 1):
                Ws, Wf, Ws0 = 8 * (y + (1 << i)), 8 * y + 1, 8 * (1 << i)
                if max(Ws, Wf, Ws0) >= H:
                    continue
                lhs = [a + b for a, b in zip(T[Ws], T[1])]
                rhs = [a + b for a, b in zip(T[Ws0], T[Wf])]
                eq = [a == b for a, b in zip(lhs, rhs)]
                Fy = tri_tensor(A[Ws]) - tri_tensor(A[Wf])
                F0 = tri_tensor(A[Ws0]) - tri_tensor(A[1])
                sh = 8 * y
                good = [v for v in range(1, H) if (v ^ sh) != 0]
                gi = np.array([v - 1 for v in good])
                gj = np.array([(v ^ sh) - 1 for v in good])
                tr_ok = np.array_equal(Fy[np.ix_(gi, gi, gi)], F0[np.ix_(gj, gj, gj)])
                print(f"  y={y:3d} i={i} j={i+3}:  C2 triple transport={tr_ok} "
                      f"(|supp F_y|={int(np.count_nonzero(Fy))}, "
                      f"|supp F_0|={int(np.count_nonzero(F0))}) | "
                      f"C3 disjoint-union equal at k=1..7: {''.join('T' if e else 'F' for e in eq)}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
