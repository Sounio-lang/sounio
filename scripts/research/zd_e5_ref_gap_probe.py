#!/usr/bin/env python3
"""E5 residual — s3(1) via transfer, and the induction that produces the q-binomial.

Given:
  (T)  s3(m+1, W=1) = 8·s3(m,1) + 24·cp2(m,1) − 176 + 72·H
       with H = 2^{m+1}                                    [transfer; E4b chain]
  (C)  cp2(m,1) = −(H−2)(H−6)                              [E4a / Tier 109, g=0]
  (B)  s3(3,1) = −272                                      [finite base]
  (S)  s3(m, 2^m) = P(H) := H³ − 12 H² + 28 H − 16        [seam; E5 note]

this probe verifies the pure-arithmetic package that closes E5:

  (I)  the function
           F(m) := P(2^{m+1}) − 1728 · [m,3]_2
       satisfies the same recurrence (T)+(C) as s3(m,1);
  (II) F(3) = −272 = s3(3,1);
  (III) therefore s3(m,1) = F(m) for all m ≥ 3, once (T)(C)(B) hold;
  (IV) therefore E5:  s3(2^m) − s3(1) = 1728 · [m,3]_2
       once (S) holds as well.

The induction step factors as the identity

  8 P(H) − P(2H) + 24(−(H−2)(H−6)) + 72 H − 176
      = 1728 · ( 8 [m,3]_2 − [m+1,3]_2 )
      = 1728 · ( −[m,2]_2 )
      = −72 (H−2)(H−4)

with H = 2^{m+1}, which is checked here symbolically on integers and by
Gaussian-binomial algebra:

  [m,3]_2 = (H−2)(H−4)(H−8)/1344
  [m+1,3]_2 = (2H−2)(2H−4)(2H−8)/1344 = 8(H−1)(H−2)(H−4)/1344
  8[m,3]_2 − [m+1,3]_2 = −(H−2)(H−4)/24 = −[m,2]_2

Run:  .venv/bin/python scripts/research/zd_e5_ref_gap_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zd_v1_III_deviation_probe import sign_table_fast  # noqa: E402


def gauss(j: int, k: int) -> int:
    if k < 0 or k > j:
        return 0
    num = den = 1
    for i in range(k):
        num *= (1 << j) - (1 << i)
        den *= (1 << k) - (1 << i)
    assert num % den == 0
    return num // den


def P(H: int) -> int:
    return H**3 - 12 * H * H + 28 * H - 16


def F(m: int) -> int:
    H = 1 << (m + 1)
    return P(H) - 1728 * gauss(m, 3)


def M_of(m: int, W: int) -> np.ndarray:
    H = 1 << (m + 1)
    S = sign_table_fast(m + 2).astype(np.int64)
    idx = np.arange(H)
    hi = (idx ^ W) + H
    return S[np.ix_(idx, hi)] * S[np.ix_(hi, idx)]


def s3(m: int, W: int) -> int:
    M = M_of(m, W)
    return int(np.trace(M @ M @ M))


def cp2(m: int, W: int) -> int:
    H = 1 << (m + 1)
    M = M_of(m, W)
    idx = np.arange(H)
    Pi = np.zeros((H, H), dtype=np.int64)
    Pi[idx, idx ^ W] = 1
    return int(np.trace((M @ M) @ Pi))


def transfer_rhs(s: int, m: int) -> int:
    H = 1 << (m + 1)
    c = -(H - 2) * (H - 6)
    return 8 * s + 24 * c - 176 + 72 * H


def main() -> int:
    ok_all = True

    print("=" * 72)
    print("(B) base s3(3,1) = -272")
    print("=" * 72)
    base = s3(3, 1)
    print(f"s3(3,1)={base}  F(3)={F(3)}  match={base == F(3) == -272}")
    ok_all &= base == -272 == F(3)

    print()
    print("=" * 72)
    print("(C)+(T) transfer + cp2 form for W=1, m=3..8")
    print("=" * 72)
    for m in range(3, 9):
        H = 1 << (m + 1)
        s = s3(m, 1)
        c = cp2(m, 1)
        cform = -(H - 2) * (H - 6)
        pred = transfer_rhs(s, m)
        actual = s3(m + 1, 1)
        ok = c == cform and pred == actual
        ok_all &= ok
        print(
            f"m={m}: cp2={c} form={cform}  "
            f"s3(m+1) pred={pred} actual={actual} ok={ok}"
        )

    print()
    print("=" * 72)
    print("(I) F closed under transfer (induction step), m=3..20")
    print("=" * 72)
    for m in range(3, 21):
        pred = transfer_rhs(F(m), m)
        want = F(m + 1)
        ok = pred == want
        ok_all &= ok
        if m <= 8 or not ok:
            print(f"m={m}: step(F)={pred} F(m+1)={want} ok={ok}")
    print("... m=3..20 all checked" if ok_all else "FAILURE in range")

    print()
    print("=" * 72)
    print("Arithmetic factorisation of the induction step")
    print("=" * 72)
    # poly residual = -72(H-2)(H-4)
    # g3 residual = 1728*(8[m,3]-[m+1,3]) = 1728*(-[m,2]_2)
    for m in range(3, 12):
        H = 1 << (m + 1)
        poly_res = (
            8 * P(H)
            + 24 * (-(H - 2) * (H - 6))
            + 72 * H
            - 176
            - P(2 * H)
        )
        g_res = 1728 * (8 * gauss(m, 3) - gauss(m + 1, 3))
        g2 = gauss(m, 2)
        ok = (
            poly_res == -72 * (H - 2) * (H - 4)
            and g_res == -1728 * g2
            and poly_res == g_res
            and 8 * gauss(m, 3) - gauss(m + 1, 3) == -g2
        )
        ok_all &= ok
        if m <= 7 or not ok:
            print(
                f"m={m}: poly_res={poly_res}  "
                f"-72(H-2)(H-4)={-72*(H-2)*(H-4)}  "
                f"1728*(-[m,2])={-1728*g2}  "
                f"8g3-g3'={8*gauss(m,3)-gauss(m+1,3)}  ok={ok}"
            )

    print()
    print("=" * 72)
    print("Gaussian algebra in H = 2^{m+1}")
    print("=" * 72)
    for m in range(3, 15):
        H = 1 << (m + 1)
        g3 = (H - 2) * (H - 4) * (H - 8) // 1344
        g3n = (2 * H - 2) * (2 * H - 4) * (2 * H - 8) // 1344
        g2 = (H - 2) * (H - 4) // 24
        ok = (
            g3 == gauss(m, 3)
            and g3n == gauss(m + 1, 3)
            and g2 == gauss(m, 2)
            and 8 * g3 - g3n == -g2
        )
        ok_all &= ok
        if m <= 6 or not ok:
            print(f"m={m}: [m,3]={g3} [m+1,3]={g3n} [m,2]={g2}  8g3-g3'=-g2 ok={ok}")

    print()
    print("=" * 72)
    print("(III)+(IV) s3(1)=F and E5, m=3..8 (measured)")
    print("=" * 72)
    for m in range(3, 9):
        H = 1 << (m + 1)
        sr = s3(m, 1)
        ss = s3(m, 1 << m)
        ok = sr == F(m) and ss == P(H) and ss - sr == 1728 * gauss(m, 3)
        ok_all &= ok
        print(
            f"m={m}: s3(1)={sr} F={F(m)}  s3(seam)={ss} P={P(H)}  "
            f"Delta={ss-sr} 1728g3={1728*gauss(m,3)} ok={ok}"
        )

    print()
    print("=" * 72)
    print("SUMMARY — proof obligations remaining")
    print("=" * 72)
    print("PROVED here as arithmetic (no CD input):")
    print("  • F is closed under (T)+(C)")
    print("  • 8[m,3]_2 − [m+1,3]_2 = −[m,2]_2 = −(H−2)(H−4)/24")
    print("  • poly residual = −72(H−2)(H−4) = 1728·(−[m,2]_2)")
    print("MEASURED / to be discharged by other tiers:")
    print("  • (T) transfer at W=1          — E4b / obligation (i)")
    print("  • (C) cp2 on g=0               — E4a, Tier 109 PROVED")
    print("  • (B) s3(3,1)=−272             — finite (verified)")
    print("  • (S) s3(2^m)=P(H)             — seam S0–S4 (prior E5 note)")
    print(f"\nprobe overall: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
