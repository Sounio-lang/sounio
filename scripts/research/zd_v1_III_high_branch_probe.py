#!/usr/bin/env python3
"""§34 carried to the HIGH branch, and where the deviation actually lives.

H1  the high branch is the COMPLEMENT, not a blow-up: for a high label W = W' + h
    (h = 2^(n-2)), the level-n resonance on the doubled vertices is the complement of the
    level-(n-1) one -- exact on the (0,0) block, off by 2(h-2) entries on each other block.
    So the level-n EDGES sit exactly where level n-1 had NO edge, and their signs are not
    determined by A' at all. That is the structural reason behind §21.3.

H2  the top bit FLIPS the parity: g(W'+h) = g(W') + 2^(n-5), so popcount(g) changes by one.
    Hence D(n-1,W') != 0  <=>  D(n,W'+h) == 0, and when nonzero the value is delta(n, lsb W')
    -- never a multiple of the level-(n-1) deviation. The high step CREATES the deviation from
    a pair that is (c)-merged one level down.

H3  the deviation is a PURE SIGN DEFECT: a seam and its Fano reference have the same edge count
    AND the same unsigned triangle count -- indeed their unsigned graphs are cospectral at every
    power tested and share degree sequences -- while tr(A^3) differs by delta.
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, g_of, lsb, sign_table_fast  # noqa: E402


def trk(A, K=6):
    P = np.eye(A.shape[0]); out = []
    for _ in range(K):
        P = P @ A; out.append(int(round(float(np.trace(P)))))
    return out


def res_full(n, Llo, S):
    H = 1 << (n - 1); L = Llo | H; los = np.arange(1, H); hi = los ^ L
    P1 = S[np.ix_(los, los)].astype(np.int16) * S[np.ix_(hi, hi)].astype(np.int16)
    P3 = S[np.ix_(los, hi)].astype(np.int16) * S[np.ix_(hi, los)].astype(np.int16)
    r = (P1 == P1.T) & (P3 == P3.T) & (P1 == P3)
    np.fill_diagonal(r, False)
    return r


def qbin3(j):
    return (2**j - 1) * (2**(j-1) - 1) * (2**(j-2) - 1) // 21


def main():
    for n in [int(x) for x in (sys.argv[1:] or ["7", "8", "9"])]:
        h = 1 << (n - 2); H = 1 << (n - 1)
        Sn = sign_table_fast(n); Sm = sign_table_fast(n - 1)

        # H1
        rows = []
        for Wp in (1, 3, 5, 12):
            if Wp >= h:
                continue
            rn = res_full(n, Wp + h, Sn); rm = res_full(n - 1, Wp, Sm)
            off = ~np.eye(h - 1, dtype=bool)
            bad = {}
            for e in (0, 1):
                for f in (0, 1):
                    blk = rn[np.ix_(np.arange(1, h) + e*h - 1, np.arange(1, h) + f*h - 1)]
                    bad[(e, f)] = int((blk[off] != (~rm)[off]).sum())
            rows.append((Wp, bad))
        print(f"n={n}  H1 complement law, mismatches per block (2(h-2) = {2*(h-2)}):")
        for Wp, bad in rows:
            print(f"     W'={Wp:3d}: " + "  ".join(f"{k}:{v}" for k, v in sorted(bad.items())))

        # H2
        Tn = {W: A_sig_fast(n, W, Sn) for W in range(1, H)}
        t3n = {W: trk(Tn[W].astype(np.float64), 3)[2] for W in range(1, H)}
        Tm = {W: A_sig_fast(n - 1, W, Sm) for W in range(1, h)}
        t3m = {W: trk(Tm[W].astype(np.float64), 3)[2] for W in range(1, h)}
        ok = bad2 = 0
        for Wp in range(1, h):
            if Wp & 7:
                continue
            gW = g_of(Wp); refp = 8 * gW + 1
            if refp >= h:
                continue
            Dlow = t3m[Wp] - t3m[refp]
            Dhigh = t3n[Wp + h] - t3n[refp + h]
            j = lsb(Wp)
            want = 0 if Dlow != 0 else -27 * 8**(n - j) * qbin3(j)
            ok += (Dhigh == want); bad2 += (Dhigh != want)
        print(f"     H2 parity flip: D(n,W'+h) = 0 when D(n-1,W') != 0, else delta(n,lsb W') "
              f"-- {ok} ok, {bad2} violations")

        # H3
        agree = [0]*6; tot = 0; deg = 0; sig = 0
        for W in range(1, H):
            if W & 7:
                continue
            ref = 8 * g_of(W) + 1
            A = np.abs(Tn[W]).astype(np.float64); B = np.abs(Tn[ref]).astype(np.float64)
            ta, tb = trk(A), trk(B); tot += 1
            for k in range(6):
                agree[k] += (ta[k] == tb[k])
            deg += int(np.array_equal(np.sort(A.sum(1)), np.sort(B.sum(1))))
            sig += int(t3n[W] != t3n[ref])
        print(f"     H3 UNSIGNED |A| vs reference: tr equal at k=1..6 {agree} of {tot}; "
              f"degree sequences equal {deg}; SIGNED tr(A^3) differs on {sig}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
