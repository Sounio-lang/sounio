#!/usr/bin/env python3
"""tr(E^3) = -24(h-2): the constant is 6 x 2 hubs x 2(h-2) edges, and every triangle has sign -1.

The last of §34's four terms, and the only one with no `B` in it -- so none of the blow-up
machinery that carried the other three applies.  The structure, measured here:

C1  On the GENERIC vertices (a, delta), a in [1,h), a != W, E has exactly TWO edges: the matching
    partner (a, 1-delta) with sign +1 and the coset partner (a^W, 1-delta) with sign -1.  That is
    2-regular, and the component of a is the 4-cycle

        (a,0) -- (a,1) -- (a^W,0) -- (a^W,1) -- (a,0)

    so the generic subgraph has NO triangles.

C2  Row W of E is identically zero, and the three special vertices W, W+h, h are PAIRWISE
    NON-ADJACENT.  So there is no 2-hub and no 3-hub triangle either: EVERY triangle of E uses
    exactly one of the two real hubs, h and W+h.

C3  Each hub is adjacent to EVERY generic vertex -- degree exactly 2(h-2).

C4  Every hub-edge triangle has sign product -1, and for two DIFFERENT reasons:
      matching edge:  s(a) * (+1) * (-s(a))     = -1   -- the hub row flips under the BLOCK flip
      coset edge:     s(a) * (-1) * (-s(a^W))   = -1   -- ... and under a |-> a^W

    where s = E[H, .].  The first flip is `Asig_hub0`/`Asig_hubL` (Tier 38); the second is `A1`,
    THIS FILE'S OWN HEADLINE LEMMA (`Asig (l ^ L_lo) y = - Asig l y`), which none of §34's other
    three terms had needed.

Hence: 2 hubs x 2(h-2) generic edges = 4(h-2) triangles, each contributing 6 ordered terms of -1,

        tr(E^3) = 6 * 4(h-2) * (-1) = -24(h-2).
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402
from zd_v1_18_1_decomposition_probe import blowup  # noqa: E402


def main():
    for n in [int(x) for x in (sys.argv[1:] or ["5", "6", "7"])]:
        h = 1 << (n - 2)
        N = (1 << (n - 1)) - 1
        Sn, Sm = sign_table_fast(n), sign_table_fast(n - 1)
        tot = np.zeros(4, dtype=np.int64)
        bad = c1 = c2 = c3 = c4 = c5 = c6 = 0
        ntri = 0
        degs = set()
        for W in range(1, h):
            A = A_sig_fast(n, W, Sn).astype(np.int64)
            Ap = A_sig_fast(n - 1, W, Sm).astype(np.int64)
            E = A - blowup(Ap, n)
            hubs = (h, W + h)
            gen = [(a, d) for a in range(1, h) if a != W for d in (0, 1)]

            def ix(t):
                return t[0] + t[1] * h

            # --- C1: the generic subgraph is matching(+1) + coset(-1), 2-regular
            edges = []
            for a in range(1, h):
                if a == W:
                    continue
                edges.append(((a, 0), (a, 1)))
                b = a ^ W
                if 1 <= b < h and b != W:
                    edges.append(((a, 0), (b, 1)))
            for (u, v) in edges:
                e = int(E[ix(u) - 1, ix(v) - 1])
                c1 += int(e != (1 if u[0] == v[0] else -1))
            for u in gen:
                c1 += int(np.count_nonzero([E[ix(u) - 1, ix(v) - 1] for v in gen]) != 2)

            # --- C2: row W null; the three special vertices pairwise non-adjacent
            c2 += int(np.count_nonzero(E[W - 1]))
            for x, y in ((W, W + h), (W, h), (W + h, h)):
                c2 += int(E[x - 1, y - 1] != 0)

            # --- C3: each hub sees every generic, and nothing else
            for H in hubs:
                for u in gen:
                    c3 += int(E[H - 1, ix(u) - 1] == 0)
                degs.add(int(np.count_nonzero(E[H - 1])))
                c3 += int(np.count_nonzero(E[H - 1]) != 2 * (h - 2))
                # the DEGREE COUNT as Lean states it: the square-sum of the hub row.
                # (Equal to the nonzero count only because the entries are +-1.)
                c3 += int(int((E[H - 1] * E[H - 1]).sum()) != 2 * (h - 2))

            # --- C4: every hub-edge triangle has sign product -1; and the two flips
            for H in hubs:
                for (u, v) in edges:
                    p = (int(E[H - 1, ix(u) - 1]) * int(E[ix(u) - 1, ix(v) - 1])
                         * int(E[ix(v) - 1, H - 1]))
                    c4 += int(p != -1)
                    ntri += 1
                for a in range(1, h):
                    if a == W:
                        continue
                    c5 += int(E[H - 1, a - 1] != -E[H - 1, a + h - 1])          # block flip
                    b = a ^ W
                    if 1 <= b < h and b != W:
                        c6 += int(E[H - 1, a - 1] != -E[H - 1, b - 1])          # A1

            # --- the trace itself, split by how many of the three vertices are hubs
            ishub = np.zeros(N + 1, dtype=bool)
            for x in hubs:
                ishub[x] = True
            adj = {}
            for u in range(1, N + 1):
                nzv = np.nonzero(E[u - 1])[0] + 1
                if len(nzv):
                    adj[u] = list(map(int, nzv))
            split = np.zeros(4, dtype=np.int64)
            for a in range(1, N + 1):
                for b in adj.get(a, ()):
                    for c in adj.get(b, ()):
                        e = int(E[c - 1, a - 1])
                        if e:
                            t = int(ishub[a]) + int(ishub[b]) + int(ishub[c])
                            split[t] += int(E[a - 1, b - 1]) * int(E[b - 1, c - 1]) * e
            tot += split
            bad += int(split.sum() != -24 * (h - 2))

        print(f"n={n} (h={h}, {h-1} labels): tr(E^3) = -24(h-2) = {-24*(h-2)}, violations {bad}")
        print(f"   split by #hubs in the triple:  0:{tot[0]}  1:{tot[1]}  2:{tot[2]}  3:{tot[3]}")
        print(f"   C1 generic graph = matching(+1)+coset(-1), 2-regular : {c1} viol")
        print(f"   C2 row W null; W, W+h, h pairwise non-adjacent       : {c2} viol")
        print(f"   C3 each hub sees every generic, degree {sorted(degs)} = 2(h-2) : {c3} viol")
        print(f"   C4 every hub-edge triangle has sign -1               : {c4} viol / {ntri}")
        print(f"   C5 hub row flips under the BLOCK flip (Asig_hub0/L)  : {c5} viol")
        print(f"   C6 hub row flips under a |-> a^W  (A1)               : {c6} viol")


if __name__ == "__main__":
    main()
