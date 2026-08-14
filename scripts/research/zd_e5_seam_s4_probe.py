#!/usr/bin/env python3
"""E5 / (S) — maximal-seam S4 closed form.

At W = 2^m the structural model of the E5 note (§3) says:

  S3  B := M|_{V*×V*} = s sᵀ − 2 I     for some s ∈ {±1}^{H−1}
  S4  row-0 on V* equals s

This probe finds the **explicit** s and reduces S4 to one cocycle column:

  (1) s_b = +1  if 1 ≤ b ≤ W,
           −1  if W < b ≤ H−1
      i.e.  s_b = seamS(W,b) := 1_{b ≤ W} − 1_{b > W}

  (2) P3(0,b) = s_b  for every b ∈ V*           (S4, explicit)

  (3) P3(b,c) = s_b s_c  for b ≠ c in V*       (S3 off-diag)
      P3(b,b) = −1                              (already P3_diag)

  (4) Via the tip's P3_row0_reduce
          P3(0,b) = − cdSigma(W, b)(m+1)   (b ≠ 0),
      (2) is equivalent to the pure cocycle evaluation

          cdSigma(2^m, b)(m+1)
            = +1  if b = 0 or b > 2^m
            = −1  if 1 ≤ b ≤ 2^m.

  Proof sketch of (4) (ready for Lean; uses tip R_ul / R_uu):
      Write 2^m = 0 + 2^m as a seam lift of 0 at level m+1 = (m−1)+2 (m ≥ 1).
      • b = 0:           R_ul 0 0 → 1
      • 0 < b < 2^m:     R_ul 0 b → −σ(0,b) = −1
      • b = 2^m:         R_uu 0 0 → −1
      • b = v + 2^m, v>0: R_uu 0 v → σ(v,0) = 1

S0–S2 are already tip theorems (P3_zero_zero, P3_diag, P3_col0_eq_neg_row0).
With (1)–(4) the seam matrix is completely known, so tr(M³) = P(H) is pure
arithmetic (E5 note §3).  That is (S).

Run:  .venv/bin/python scripts/research/zd_e5_seam_s4_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zd_v1_III_deviation_probe import sign_table_fast  # noqa: E402


def cd_sigma(a: int, b: int, n: int) -> int:
    """Lean `cdSigma` (Int ±1), levels as in SounioZDFiberAntisym."""
    if n == 0:
        return -1
    if n == 1:
        return 1 if a == 0 or b == 0 else -1
    half = 1 << (n - 1)
    if a == 0 or b == 0:
        return 1
    a_hi, b_hi = a >= half, b >= half
    a_lo, b_lo = a % half, b % half
    if not a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, n - 1)
    if not a_hi and b_hi:
        return cd_sigma(b_lo, a_lo, n - 1)
    if a_hi and not b_hi:
        return cd_sigma(a_lo, 0, n - 1) if b_lo == 0 else -cd_sigma(a_lo, b_lo, n - 1)
    return -cd_sigma(0, a_lo, n - 1) if b_lo == 0 else cd_sigma(b_lo, a_lo, n - 1)


def M_of(m: int, W: int) -> np.ndarray:
    H = 1 << (m + 1)
    S = sign_table_fast(m + 2).astype(np.int64)
    idx = np.arange(H)
    hi = (idx ^ W) + H
    return S[np.ix_(idx, hi)] * S[np.ix_(hi, idx)]


def seam_s(W: int, b: int) -> int:
    return 1 if b <= W else -1


def main() -> int:
    ok_all = True

    print("=" * 72)
    print("cdSigma(2^m, b) at level m+1 — closed form")
    print("=" * 72)
    for m in range(1, 10):
        W = 1 << m
        H = 1 << (m + 1)
        bad = []
        for b in range(H):
            got = cd_sigma(W, b, m + 1)
            if b == 0:
                want = 1
            elif b <= W:
                want = -1
            else:
                want = 1
            if got != want:
                bad.append((b, got, want))
        ok = not bad
        ok_all &= ok
        print(f"m={m}: H={H}  cdSigma_pow2_col ok={ok}" + (f"  first_bad={bad[0]}" if bad else ""))

    print()
    print("=" * 72)
    print("S4: P3(0,b) = seamS(W,b) at W=2^m")
    print("=" * 72)
    for m in range(3, 9):
        W = 1 << m
        H = 1 << (m + 1)
        M = M_of(m, W)
        bad = []
        for b in range(1, H):
            got = int(M[0, b])
            want = seam_s(W, b)
            # tip reduction P3_row0_reduce
            via = -cd_sigma(W, b, m + 1)
            if got != want or got != via:
                bad.append((b, got, want, via))
        ok = not bad
        ok_all &= ok
        print(f"m={m}: row0=seamS ok={ok}" + (f"  first_bad={bad[0]}" if bad else ""))

    print()
    print("=" * 72)
    print("S3: B = s sᵀ − 2 I  (pointwise P3(b,c)=s_b s_c off-diag)")
    print("=" * 72)
    for m in range(3, 8):
        W = 1 << m
        H = 1 << (m + 1)
        M = M_of(m, W)
        s = np.array([seam_s(W, b) for b in range(1, H)], dtype=np.int64)
        B = M[1:, 1:]
        pred = np.outer(s, s) - 2 * np.eye(H - 1, dtype=np.int64)
        ok = bool(np.array_equal(B, pred))
        # also diag of M is −1 on V*
        ok_diag = bool(np.all(np.diag(M)[1:] == -1))
        ok_all &= ok and ok_diag
        print(f"m={m}: B=ssT-2I ok={ok}  diag_V*=-1 ok={ok_diag}")

    print()
    print("=" * 72)
    print("Forced seam polynomial: tr(M³) via the model")
    print("=" * 72)
    for m in range(3, 9):
        W = 1 << m
        H = 1 << (m + 1)
        M = M_of(m, W)
        # after diag switch by s: constant border model
        # n = H-1, p = n ⇒ tr = H³ − 12 H² + 28 H − 16
        tr = int(np.trace(M @ M @ M))
        poly = H**3 - 12 * H * H + 28 * H - 16
        # direct block arithmetic check
        n = H - 1
        tr_model = n**3 - 9 * n * n + 7 * n + 1
        ok = tr == poly == tr_model
        ok_all &= ok
        print(f"m={m}: tr={tr} poly={poly} model={tr_model} ok={ok}")

    print()
    print("=" * 72)
    print("Multiplicative S4: P3(0,b)·P3(0,c) = P3(b,c)  (b≠c in V*)")
    print("=" * 72)
    for m in range(3, 7):
        W = 1 << m
        H = 1 << (m + 1)
        M = M_of(m, W)
        bad = 0
        for b in range(1, H):
            for c in range(1, H):
                if b == c:
                    continue
                if int(M[0, b] * M[0, c]) != int(M[b, c]):
                    bad += 1
        ok = bad == 0
        ok_all &= ok
        print(f"m={m}: multiplicative S4 ok={ok} bad={bad}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("S4 closed form:     s_b = +1 on {1..W}, −1 on {W+1..H−1}")
    print("S4 ⇔ cdSigma col:   σ(2^m, b)_{m+1} = −s_b  (b≠0)")
    print("S3 pointwise:       P3(b,c) = s_b s_c off-diag at W=2^m")
    print("S0–S2:              tip theorems (P3_zero_zero / P3_diag / col0↔row0)")
    print("Lean foothold:      P3_row0_reduce + R_ul/R_uu case split (tip)")
    print(f"probe overall: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
