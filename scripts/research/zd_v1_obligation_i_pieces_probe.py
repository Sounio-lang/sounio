#!/usr/bin/env python3
"""Obligation (i) / E4b: the three orthant sums, taken APART.

Tier 90 writes the level transfer as

    tri3(m+1) = tri3(m) + 3*T1 + 3*T2 + T3

with T1, T2, T3 the weight-1/2/3 orthant sums.  Here they are computed DIRECTLY from the
level-(m+1) sign matrix (no E-weights needed): with B_ij the H-blocks of P3(.,.,W,m+1),
H = 2^(m+1),

    T1 = tr(B00 B0H BH0),  T2 = tr(B0H BHH BH0),  T3 = tr(BHH^3),  s3 = tr(B00^3).

RESULT.  Sec 57.50 recorded that T1, T2, T3 each fall OUTSIDE span{s3, cp2, cp3, 1, H, H^2, H^3}
and only their combination closes -- "the closure is a property of the orthant SUM".  That reading
is a MAXIMAL-SEAM ARTIFACT.  Off the single label W = 2^m, each piece has a closed form:

    T1 = s3 + 4*cp2 + 16*H - 64
    T2 = s3 + 4*cp2 -  8*H + 64
    T3 = s3          + 48*H - 176

exact on 486/486 off-seam label-levels at m = 3..7, and out of sample at m = 8.

At W = 2^m the three deviate from those forms by exactly

    (delta1, delta2, delta3) = (1, -2, 3) * 2*(H-4)*(H-8)

and (3, 3, 1) . (1, -2, 3) = 0 -- which is precisely why the combination never saw it.  So
obligation (i) is a CONSEQUENCE, holding on and off the seam alike:

    3*T1 + 3*T2 + T3 = 7*s3 + 24*cp2 + 72*H - 176.

And the "7" is not a cancellation: it is 3 + 3 + 1.  Every one of the eight orthants of the
level-(m+1) triple sum contributes s3 plus a correction in span{cp2, H, 1}; the transfer's 8 is
the orthant count, 1 + 3 + 3 + 1.

For E4b (within-fibre differences): dT1 = dT2 = ds3 + 4*dcp2 and dT3 = ds3 off the seam, so with
E4a's dcp2 = 0 all four within-fibre deviations COINCIDE.

This is the second time the maximal seam has produced a spurious "irreducible" reading; the first
was the 288*[m-1,2]_2 term deflated in Sec 57.49.
"""
import sys
sys.path.insert(0, "/workspace/sounio/scripts/research")
import numpy as np
from collections import defaultdict
from zd_v1_III_deviation_probe import sign_table_fast

def M_of(m, W):
    """P3(.,.,W,m) as a 2^(m+1) x 2^(m+1) integer matrix."""
    H = 1 << (m + 1)
    S = sign_table_fast(m + 2).astype(np.int64)
    idx = np.arange(H)
    hi = (idx ^ W) + H
    return S[np.ix_(idx, hi)] * S[np.ix_(hi, idx)]

def pieces(m, W):
    """T1, T2, T3 at the transition m -> m+1, plus s3 and cp2 at level m."""
    H = 1 << (m + 1)
    M1 = M_of(m + 1, W)                       # 2H x 2H
    B = lambda i, j: M1[i*H:(i+1)*H, j*H:(j+1)*H]
    B00, B0H, BH0, BHH = B(0,0), B(0,1), B(1,0), B(1,1)
    tr = lambda X: int(np.trace(X))
    T1 = tr(B00 @ B0H @ BH0)
    T2 = tr(B0H @ BHH @ BH0)
    T3 = tr(BHH @ BHH @ BHH)
    M = M_of(m, W)
    assert np.array_equal(M, B00), "low orthant must be the level-m matrix"
    idx = np.arange(H)
    Pi = np.zeros((H, H), dtype=np.int64); Pi[idx, idx ^ W] = 1
    s3 = tr(M @ M @ M)
    cp2 = tr(M @ M @ Pi)
    return T1, T2, T3, s3, cp2

def g_of(W):
    return (W & (W - 1)) >> 3

def main():
    levels = range(3, 7)
    rows = {}
    bad = 0; tot = 0
    for m in levels:
        H = 1 << (m + 1)
        for W in range(1, H):
            T1, T2, T3, s3, cp2 = pieces(m, W)
            rows[(m, W)] = (T1, T2, T3, s3, cp2)
            tot += 1
            if 3*T1 + 3*T2 + T3 != 7*s3 + 24*cp2 - 176 + 72*H:
                bad += 1
                print(f"  CONTROL FAIL m={m} W={W}")
    print(f"[control] obligation (i): {tot-bad}/{tot} exact\n")

    # T1 - T2 label independence
    print("[T1 - T2] per level, by label")
    for m in levels:
        H = 1 << (m + 1)
        vals = defaultdict(list)
        for W in range(1, H):
            T1, T2, T3, s3, cp2 = rows[(m, W)]
            vals[T1 - T2].append(W)
        print(f"  m={m}: " + "; ".join(f"{v} on {len(ws)} labels {ws[:6]}{'...' if len(ws)>6 else ''}"
                                        for v, ws in sorted(vals.items())))
    print()

    # within-fibre differences, fibre = g(W)
    print("[within-fibre differences]  fibre classes by g = (W & (W-1)) >> 3")
    for m in levels:
        H = 1 << (m + 1)
        fib = defaultdict(list)
        for W in range(1, H):
            fib[g_of(W)].append(W)
        for g, ws in sorted(fib.items()):
            if len(ws) < 2: continue
            base = ws[0]
            b = rows[(m, base)]
            deltas = set()
            for W in ws[1:]:
                r = rows[(m, W)]
                deltas.add(tuple(r[i] - b[i] for i in range(5)))
            zero = all(d == (0,0,0,0,0) for d in deltas)
            print(f"  m={m} g={g} |class|={len(ws)} "
                  + ("ALL EQUAL (T1,T2,T3,s3,cp2 fibre-constant)" if zero
                     else f"deltas(T1,T2,T3,s3,cp2)={sorted(deltas)[:4]}"))
    print()

    # is each of T1,T2,T3 fibre-constant?  and Ds3 vs DT's on reference pairs
    print("[reference pairs]  W = 2^j against W = 1, per level")
    for m in levels:
        H = 1 << (m + 1)
        r1 = rows[(m, 1)]
        for j in range(1, m + 2):
            W = 1 << j
            if W >= H: continue
            r = rows[(m, W)]
            d = [r[i] - r1[i] for i in range(5)]
            print(f"  m={m} j={j}: dT1={d[0]} dT2={d[1]} dT3={d[2]} ds3={d[3]} dcp2={d[4]}"
                  f"   check 3dT1+3dT2+dT3 = {3*d[0]+3*d[1]+d[2]} vs 7ds3 = {7*d[3]}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# The closed forms, and the seam.  Run as `python3 <this> forms`.
def closed_forms(lo=3, hi=8):
    """T_i - s3 against the closed forms, every label at levels [lo, hi)."""
    ok = bad = 0
    seam = {}
    for m in range(lo, hi):
        H = 1 << (m + 1)
        for W in range(1, H):
            T1, T2, T3, s3, cp2 = pieces(m, W)
            p = (4*cp2 + 16*H - 64, 4*cp2 - 8*H + 64, 48*H - 176)
            d = (T1 - s3 - p[0], T2 - s3 - p[1], T3 - s3 - p[2])
            if W == (1 << m):
                seam[m] = d
                assert d == tuple(k * 2 * (H-4) * (H-8) for k in (1, -2, 3)), (m, W, d)
            elif d == (0, 0, 0):
                ok += 1
            else:
                bad += 1
                print(f"  MISS m={m} W={W} d={d}")
            assert 3*T1 + 3*T2 + T3 == 7*s3 + 24*cp2 + 72*H - 176
    print(f"[closed forms] off-seam exact on {ok}/{ok+bad}; "
          f"seam deviations = (1,-2,3)*2(H-4)(H-8) at every level {sorted(seam)}")
