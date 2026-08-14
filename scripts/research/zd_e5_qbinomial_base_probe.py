#!/usr/bin/env python3
"""E5 base case — absolute closed forms and the q-binomial's true seat.

Obligation (iii) of §57.50 / edge E5 of the completeness pincer:

    s3(2^m) − s3(1) = 1728 · [m,3]_2          (measured m = 3..7 in the source note)

This probe does three things that the earlier transfer-matrix work left open:

(1) Reconfirm the difference through m = 8 (one level past the source range).
(2) Fit ABSOLUTE closed forms for each end separately.  Finding:
       s3(2^m) = H³ − 12 H² + 28 H − 16          (H = 2^{m+1}; NO q-binomial)
       s3(1)   = s3(2^m) − 1728 · [m,3]_2
               = H³ − 12 H² + 28 H − 16 − (9/7)(H−2)(H−4)(H−8)
   So the entire Gaussian content of E5 sits in the g = 0 reference W = 1,
   not in the maximal seam.  (The seam still carries [m−1,2]_2 in the orthant
   pieces T1/T2/T3 and in Q — different object, same location.)
(3) Exhibit the matrix model that forces (2) at the seam:

       M_00 = +1,   M_ii = −1 (i ≠ 0),
       M_0b = − M_b0  (b ≠ 0),
       B := M|_{V*×V*} = s sᵀ − 2 I     for some s ∈ {±1}^{H−1},
       and row-0 on V* equals s.

   After the diagonal switch diag(1, s) one gets the constant matrix

       M' = [  1 | 1ᵀ  ]
            [ −1 | J−2I ]

   whose cubed trace is H³ − 12 H² + 28 H − 16 by pure arithmetic
   (n = H−1, p = n ⇒ tr = n³ − 9 n² + 7 n + 1).

Status: MEASURED + algebraically reduced.  Not a Lean theorem.  The proof
targets this isolates are: (a) the four structural laws at W = 2^m, of which
P3_zero_zero / P3_diag / P3_col0_eq_neg_row0 / P3_pow2_coherent already cover
most; (b) the alignment row0 = s; (c) the arithmetic of M'; (d) either the
absolute form of s3(1) or a direct difference argument for E5.

Run:  .venv/bin/python scripts/research/zd_e5_qbinomial_base_probe.py
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


def M_of(m: int, W: int) -> np.ndarray:
    H = 1 << (m + 1)
    S = sign_table_fast(m + 2).astype(np.int64)
    idx = np.arange(H)
    hi = (idx ^ W) + H
    return S[np.ix_(idx, hi)] * S[np.ix_(hi, idx)]


def s3(m: int, W: int) -> int:
    M = M_of(m, W)
    return int(np.trace(M @ M @ M))


def s_from_B(B: np.ndarray) -> np.ndarray:
    """Recover s with B = s sᵀ − 2 I (diag of ssᵀ is 1, force diag of B to −1)."""
    n = B.shape[0]
    s = np.empty(n, dtype=np.int64)
    s[0] = 1
    for j in range(1, n):
        s[j] = int(B[0, j]) * s[0]  # B_0j = s0 s_j for j > 0
    pred = np.outer(s, s) - 2 * np.eye(n, dtype=np.int64)
    if not np.array_equal(B, pred):
        raise AssertionError("nonzero block is not ss^T - 2I")
    return s


def seam_structural_laws(m: int) -> dict:
    W = 1 << m
    M = M_of(m, W)
    H = M.shape[0]
    n = H - 1
    out = {"m": m, "H": H}
    out["M00"] = int(M[0, 0])
    out["diag_nonzero_all_m1"] = bool(np.all(np.diag(M)[1:] == -1))
    out["row0_col0_antisym"] = all(
        int(M[0, b] * M[b, 0]) == -1 for b in range(1, H)
    )
    B = M[1:, 1:]
    out["B_symmetric"] = bool(np.array_equal(B, B.T))
    try:
        s = s_from_B(B)
        out["B_is_ssT_minus_2I"] = True
    except AssertionError:
        out["B_is_ssT_minus_2I"] = False
        return out
    u = M[0, 1:].astype(np.int64)
    out["row0_equals_s"] = bool(np.array_equal(u, s))
    v = u * s
    out["v_all_plus"] = bool(np.all(v == 1))
    out["p"] = int(v.sum())
    # nonzero triple products all +1 (empty two-graph on V*)
    if m <= 5:
        bad = 0
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if int(B[i, j] * B[j, k] * B[k, i]) != 1:
                        bad += 1
        out["nonzero_triple_neg"] = bad
    return out


def main() -> int:
    print("=" * 72)
    print("E5 reconfirm: s3(2^m) - s3(1) = 1728 * [m,3]_2")
    print("=" * 72)
    e5_ok = True
    rows = []
    for m in range(3, 9):
        H = 1 << (m + 1)
        ss = s3(m, 1 << m)
        sr = s3(m, 1)
        d = ss - sr
        g3 = gauss(m, 3)
        pred = 1728 * g3
        ok = d == pred
        e5_ok &= ok
        form_s = H**3 - 12 * H**2 + 28 * H - 16
        form_r = form_s - pred
        # also (9/7)(H-2)(H-4)(H-8)
        g3_from_H = (H - 2) * (H - 4) * (H - 8) // 1344
        assert g3 == g3_from_H
        rows.append((m, H, ss, sr, d, pred, form_s, form_r))
        print(
            f"m={m}: Delta={d} pred={pred} ok={ok}  "
            f"s3_seam={ss} form={form_s} match={ss == form_s}  "
            f"s3_ref={sr} form={form_r} match={sr == form_r}"
        )

    print()
    print("=" * 72)
    print("Maximal-seam structural laws (force the polynomial form)")
    print("=" * 72)
    struct_ok = True
    for m in range(3, 8):
        st = seam_structural_laws(m)
        flags = [
            st["M00"] == 1,
            st["diag_nonzero_all_m1"],
            st["row0_col0_antisym"],
            st["B_symmetric"],
            st["B_is_ssT_minus_2I"],
            st["row0_equals_s"],
            st["v_all_plus"],
            st.get("nonzero_triple_neg", 0) == 0,
        ]
        ok = all(flags)
        struct_ok &= ok
        print(
            f"m={st['m']}: M00={st['M00']} diag*={st['diag_nonzero_all_m1']} "
            f"anti0={st['row0_col0_antisym']} B=ssT-2I={st['B_is_ssT_minus_2I']} "
            f"u=s={st['row0_equals_s']} v=1={st['v_all_plus']} "
            f"p={st['p']} ok={ok}"
        )

    print()
    print("=" * 72)
    print("Arithmetic identity: n=H-1, p=n => tr = H^3 - 12 H^2 + 28 H - 16")
    print("=" * 72)
    for m, H, *_ in rows:
        n = H - 1
        tr_n = n**3 - 9 * n**2 + 7 * n + 1  # p = n in n^3-6n^2+7n+1-3p^2
        tr_H = H**3 - 12 * H**2 + 28 * H - 16
        print(f"m={m}: tr_n={tr_n} tr_H={tr_H} equal={tr_n == tr_H}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"E5 difference exact on m=3..8:           {e5_ok}")
    print(f"Seam structural laws on m=3..7:          {struct_ok}")
    print("Absolute seam form (poly in H only):     MEASURED + model-forced")
    print("Absolute ref form (poly - 1728 g3):      MEASURED")
    print("q-binomial seat of E5:                   W=1 (reference), NOT seam")
    print("Sibling [m-1,2]_2 in T1/T2/T3/Q:         still at the seam")
    print()
    print("Lean targets isolated:")
    print("  (a) structural laws at W=2^m  (partially present as Tiers 65, 111–112)")
    print("  (b) alignment P3(0,b)=s_b with B=ssT-2I")
    print("  (c) tr(M') arithmetic")
    print("  (d) s3(1) or direct Delta — the residual combinatorial content of E5")
    return 0 if (e5_ok and struct_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
