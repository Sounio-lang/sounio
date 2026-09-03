#!/usr/bin/env python3
"""Oracle: the e_top-seam bridge across the Cayley-Dickson tower (dim 16, 32, 64). Emits, per dim, the
cocycle lemma flag (L_i^2=-I), the operator/seam coincidence, the ZD coincidence (INCLUDING the full
O(n^3) zero-divisor scan at dim 64 which the souc test skips for speed), and the non-anticommuting /
off-seam counts. See cd_tower_seam.md."""


def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def analyze(bits):
    N = 1 << bits
    top = N // 2

    def anti0(l, u):
        return all(cd_sigma(l, u ^ c, bits) * cd_sigma(u, c, bits)
                   + cd_sigma(u, l ^ c, bits) * cd_sigma(l, c, bits) == 0 for c in range(N))

    def llsq(l, u):
        return all(cd_sigma(l, l ^ c, bits) * cd_sigma(u, l ^ u ^ c, bits)
                   * cd_sigma(l, u ^ c, bits) * cd_sigma(u, c, bits) == -1 for c in range(N))

    def off_seam(l, u):
        return not (u == top or (l ^ u) == top)

    def is_zd(l, u):
        for a in range(1, N):
            for b in range(a + 1, N):
                for s in (1, -1):
                    if all(((cd_sigma(l, a, bits) if (l ^ a) == k else 0)
                            + (s * cd_sigma(l, b, bits) if (l ^ b) == k else 0)
                            + (cd_sigma(u, a, bits) if (u ^ a) == k else 0)
                            + (s * cd_sigma(u, b, bits) if (u ^ b) == k else 0)) == 0 for k in range(N)):
                        return True
        return False

    lsq = 1 if all(cd_sigma(i, j, bits) * cd_sigma(i, i ^ j, bits) == -1
                   for i in range(1, N) for j in range(N)) else 0
    pairs = [(l, u) for l in range(1, top) for u in range(top, N)]
    fwd = 1 if all(anti0(l, u) == llsq(l, u) for (l, u) in pairs) else 0
    seam = 1 if all(anti0(l, u) != off_seam(l, u) for (l, u) in pairs) else 0
    zdeq = 1 if all(off_seam(l, u) == is_zd(l, u) for (l, u) in pairs) else 0
    na = sum(1 for (l, u) in pairs if not anti0(l, u))
    os = sum(1 for (l, u) in pairs if off_seam(l, u))
    return lsq, fwd, seam, zdeq, na, os


def main():
    for bits, tagN in ((4, "16"), (5, "32"), (6, "64")):
        lsq, fwd, seam, zdeq, na, os = analyze(bits)
        print(f"LSQ{tagN} {lsq}")
        print(f"FWD{tagN} {fwd}")
        print(f"SEAM{tagN} {seam}")
        print(f"ZDEQ{tagN} {zdeq}")     # includes the full dim-64 scan (souc skips ZDEQ64)
        print(f"NA{tagN} {na}")
        print(f"OS{tagN} {os}")


if __name__ == "__main__":
    main()
