#!/usr/bin/env python3
"""The SIGN DEFECT of Phi = tau_j, and where its curvature lives.

§36 showed the deviation is a pure sign phenomenon on a cospectral pair. This probe names the
map and the defect.

  Phi = tau_j on vertices, j = lsb(W),  tau j x = x if bit0(x)==bitj(x) else x ^ (1|2^j)
  (the same tau as Qgen'_tau, :1714; note tau_j(W) = 8*g(W)+1 -- the Fano representative)

  S1  Phi is an UNSIGNED isomorphism of the annihilation graph for EVERY seam, both parities:
      |A_W|(a,b) = |A_(tau W)|(tau a, tau b).

  S2  so the whole difference is the sign defect  eps(a,b) = A_W(a,b) * A_(tau W)(tau a, tau b),
      a +-1 function on the common support, and:

          eps is BALANCED (every triangle +1)  <=>  popcount(g(W)) is ODD

      Balanced means eps is a coboundary, i.e. A_W = D (Phi* A_f) D -- a switching, which
      preserves every trace. That IS (c), and it is why (c) holds at every power k while the
      even regime differs only at k=3 (§35).

  S3  the curvature eps_T = eps(a,b)eps(b,c)eps(c,a) is DETERMINED BY THE LOW j+1 BITS of
      (a,b,c) alone -- the coordinates at and below the seam, exactly the ones g forgets.
      That is the mechanism behind "delta does not depend on g", and behind the [j choose 3]_2
      in delta's closed form.
"""
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, g_of, lsb, sign_table_fast  # noqa: E402


def tau(j, x):
    return x if (x & 1) == ((x >> j) & 1) else x ^ (1 | (1 << j))


def t3(M):
    F = M.astype(np.float64)
    return int(round(float(np.sum(F * (F @ F).T))))


def qb3(j):
    return (2**j - 1) * (2**(j-1) - 1) * (2**(j-2) - 1) // 21


def main():
    for n in [int(x) for x in (sys.argv[1:] or ["7", "8"])]:
        H = 1 << (n - 1)
        S = sign_table_fast(n)
        bad_iso = bad_bal = bad_delta = 0
        seams = 0
        for W in range(1, H):
            if W & 7:
                continue
            seams += 1
            j = lsb(W)
            tW = tau(j, W)
            assert tW == 8 * g_of(W) + 1
            A = A_sig_fast(n, W, S).astype(int)
            B = A_sig_fast(n, tW, S).astype(int)
            p = np.array([tau(j, a) for a in range(1, H)])
            Bp = B[np.ix_(p - 1, p - 1)]
            bad_iso += not np.array_equal(np.abs(A), np.abs(Bp))
            eps = A * Bp
            neg = (t3(np.abs(A)) - t3(eps)) // 2       # ordered triangles with curvature -1
            odd = bin(g_of(W)).count("1") % 2
            bad_bal += (neg == 0) != bool(odd)
            want = 0 if odd else -27 * 8**(n - j) * qb3(j)
            bad_delta += (t3(A) - t3(Bp)) != want
        print(f"n={n}: {seams} seams | S1 unsigned-iso violations {bad_iso} | "
              f"S2 'balanced <=> popcount(g) odd' violations {bad_bal} | "
              f"delta violations {bad_delta}")

        # S3: is the curvature low-determined?
        for W in (8, 16):
            if W >= H:
                continue
            j = lsb(W)
            A = A_sig_fast(n, W, S).astype(int)
            B = A_sig_fast(n, tau(j, W), S).astype(int)
            p = np.array([tau(j, a) for a in range(1, H)])
            eps = A * B[np.ix_(p - 1, p - 1)]
            M = (1 << (j + 1)) - 1
            d = defaultdict(set)
            cnt = 0
            for a in range(1, H):
                for b in range(a + 1, H):
                    if eps[a-1, b-1] == 0:
                        continue
                    for c in range(b + 1, H):
                        if eps[b-1, c-1] == 0 or eps[c-1, a-1] == 0:
                            continue
                        d[(a & M, b & M, c & M)].add(
                            eps[a-1, b-1] * eps[b-1, c-1] * eps[c-1, a-1])
                        cnt += 1
                if cnt > 400000:
                    break
            amb = sum(1 for v in d.values() if len(v) > 1)
            print(f"      S3 W={W} j={j}: curvature vs low {j+1} bits -> {len(d)} classes, "
                  f"{amb} ambiguous ({cnt} triangles sampled)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
