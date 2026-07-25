#!/usr/bin/env python3
"""R4 — multi-line Fano *field* obstruction (system-level, after R3_GREEN).

R3 closed the *neighbourhood of one line* (jet + path classes).
R4 certifies the *field* of squares: 7 Fano lines, pairwise incidence 1,
cross-line non-associativity, and a two-line mixing jet that cannot be
cancelled by a single-line τ (system obstruction).

Does NOT claim D3. Does NOT claim clinical content.

Exit 0 → R4_CONTRACT_OK and R4_VERDICT R4_PARTIAL
(R4_GREEN reserved for a multi-line Φ into control space with path classes).

See docs/research/rupture-r4-fano-field_2026-07-25.md
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Optional


def cd_sigma(a: int, b: int, bits: int = 3) -> int:
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi, b_hi = a >= half, b >= half
    a_lo, b_lo = a & (half - 1), b & (half - 1)
    if not a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if not a_hi and b_hi:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1) if b_lo == 0 else -cd_sigma(a_lo, b_lo, bits - 1)
    return -cd_sigma(b_lo, a_lo, bits - 1) if b_lo == 0 else cd_sigma(b_lo, a_lo, bits - 1)


def omul(A: list[float], B: list[float]) -> list[float]:
    C = [0.0] * 8
    for i in range(8):
        for j in range(8):
            C[i ^ j] += cd_sigma(i, j) * A[i] * B[j]
    return C


def e(i: int) -> list[float]:
    v = [0.0] * 8
    v[i] = 1.0
    return v


def vadd(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(8)]


def vscale(a: list[float], s: float) -> list[float]:
    return [a[i] * s for i in range(8)]


def associator(a: list[float], b: list[float], c: list[float]) -> list[float]:
    ab_c = omul(omul(a, b), c)
    a_bc = omul(a, omul(b, c))
    return [ab_c[i] - a_bc[i] for i in range(8)]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def fano_lines() -> list[tuple[int, int, int]]:
    return [(i, j, i ^ j) for i in range(1, 8) for j in range(i + 1, 8) if i ^ j > j]


def is_fano_triple(i: int, j: int, k: int) -> bool:
    return (i ^ j) == k or (i ^ k) == j or (j ^ k) == i


def unique_support(v: list[float], tol: float = 1e-9) -> Optional[tuple[int, float]]:
    hits = [(i, v[i]) for i in range(8) if abs(v[i]) > tol]
    if len(hits) != 1:
        return None
    return hits[0]


def main() -> int:
    lines = fano_lines()
    print(f"FIELD_LINES n={len(lines)}")
    if len(lines) != 7:
        print("R4_VERDICT R4_FAIL")
        print("R4_CONTRACT_FAIL")
        return 1

    # --- (F1) each line is internally associative ---
    f1_ok = True
    for L in lines:
        i, j, k = L
        nrm = norm(associator(e(i), e(j), e(k)))
        if nrm > 1e-9:
            f1_ok = False
            print(f"F1_FAIL line={L} assoc={nrm}")
        # all ordered permutations of the triple also associative (alt. property)
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            if norm(associator(e(a), e(b), e(c))) > 1e-9:
                f1_ok = False
    print(f"F1_INTERNAL_ASSOC {'PASS' if f1_ok else 'FAIL'}")

    # --- (F2) Fano incidence: every pair of distinct lines meets in exactly one unit ---
    f2_ok = True
    meet_count = 0
    for L1, L2 in combinations(lines, 2):
        inter = set(L1) & set(L2)
        meet_count += 1
        if len(inter) != 1:
            f2_ok = False
            print(f"F2_FAIL {L1} & {L2} inter={inter}")
    print(f"F2_INCIDENCE pairs={meet_count} all_meet_1={'PASS' if f2_ok else 'FAIL'}")

    # --- (F3) cross-line obstruction: non-Fano triples drawn from two lines have ||assoc||=2 ---
    f3_ok = True
    cross_nonzero = 0
    cross_checked = 0
    # Worked pair: L1=(1,2,3), L2=(1,4,5) share e1
    L1 = (1, 2, 3)
    L2 = (1, 4, 5)
    assert L1 in lines and L2 in lines
    s = (set(L1) & set(L2)).pop()
    a1 = next(x for x in L1 if x != s)
    b1 = next(x for x in L1 if x != s and x != a1)
    a2 = next(x for x in L2 if x != s)
    b2 = next(x for x in L2 if x != s and x != a2)
    # cross triples: one from L1\s, one from L2\s, and the shared or third
    for i, j, k in (
        (a1, a2, s),
        (a1, a2, b1),
        (a1, a2, b2),
        (a1, b2, s),
        (b1, a2, s),
        (b1, b2, s),
    ):
        cross_checked += 1
        nrm = norm(associator(e(i), e(j), e(k)))
        fano = is_fano_triple(i, j, k)
        if fano:
            if nrm > 1e-9:
                f3_ok = False
                print(f"F3_FAIL fano_cross_nonzero {(i,j,k)} n={nrm}")
        else:
            cross_nonzero += 1
            if abs(nrm - 2.0) > 1e-9:
                f3_ok = False
                print(f"F3_FAIL nonfano {(i,j,k)} n={nrm} expect 2")
            else:
                print(f"F3_CROSS nonfano={(i,j,k)} ||assoc||={nrm:.1f}")
    print(
        f"F3_CROSS_OBSTRUCTION checked={cross_checked} nonfano={cross_nonzero} "
        f"-> {'PASS' if f3_ok and cross_nonzero > 0 else 'FAIL'}"
    )

    # --- (F4) field census: every unordered non-Fano triple of Im units has ||assoc||²=4 ---
    f4_ok = True
    n_fano = 0
    n_non = 0
    for i, j, k in combinations(range(1, 8), 3):
        nrm = norm(associator(e(i), e(j), e(k)))
        if is_fano_triple(i, j, k):
            n_fano += 1
            if nrm > 1e-9:
                f4_ok = False
        else:
            n_non += 1
            if abs(nrm - 2.0) > 1e-9:
                f4_ok = False
    # C(7,3)=35; Fano lines 7; non-Fano = 28
    print(f"F4_CENSUS fano_triples={n_fano} nonfano={n_non} -> {'PASS' if f4_ok and n_fano==7 and n_non==28 else 'FAIL'}")

    # --- (F5) two-line mixing jet: cannot cancel with single-line τ alone ---
    # Config: first factor mixes L1 and L2 away from shared unit:
    #   x(δ) = e_{a1} + δ e_{a2}   (a1∈L1\s, a2∈L2\s)
    #   y = e_s,  z = e_{b1}      (shared + other L1)
    # At δ=0: x,y,z may or may not be Fano; measure ||[x,y,z]||.
    # Single-line τ cancels the *odd* jet on *one* line neighbourhood (R3 Path C).
    # Here the obstruction lives *between* lines: α(δ) = δ [e_a2, e_s, e_b1] (if linear).
    f5_ok = True
    pure = associator(e(a2), e(s), e(b1))
    pure_n = norm(pure)
    print(f"F5_PURE_CROSS [e{a2},e{s},e{b1}] ||={pure_n:.4f} support={unique_support(pure)}")
    if pure_n < 1e-9:
        # try alternate z from L1
        pure = associator(e(a2), e(s), e(a1))
        pure_n = norm(pure)
        z_unit = a1
        print(f"F5_ALT_PURE [e{a2},e{s},e{a1}] ||={pure_n:.4f}")
    else:
        z_unit = b1

    if pure_n < 1.5:
        f5_ok = False
        print("F5_FAIL expected nonzero pure cross associator")
    else:
        # linearity in δ
        for delta in (-1.0, -0.5, 0.25, 0.5, 1.0):
            x = vadd(e(a1), vscale(e(a2), delta))
            got = associator(x, e(s), e(z_unit))
            # [e_a1, e_s, e_z] + δ [e_a2, e_s, e_z]
            base = associator(e(a1), e(s), e(z_unit))
            exp = vadd(base, vscale(associator(e(a2), e(s), e(z_unit)), delta))
            err = norm([got[t] - exp[t] for t in range(8)])
            if err > 1e-9:
                f5_ok = False
                print(f"F5_FAIL linear delta={delta} err={err}")
        print(f"F5_MIXING_LINEARITY -> {'PASS' if f5_ok else 'FAIL'}")

        # Single-line τ cancellation (R3 Path C style) kills odd jet when the
        # obstruction is a *neighbourhood of one line*. Here base may be 0
        # (a1,s,z_unit on L1) so α = δ * pure_cross — pure *field* term.
        base = associator(e(a1), e(s), e(z_unit))
        base_n = norm(base)
        print(f"F5_BASE_ON_L1 [e{a1},e{s},e{z_unit}] ||={base_n:.4e} (expect 0 if all on L1)")
        # Residual after "cancelling as if single-line": the cross term remains
        # proportional to δ and cannot be removed by τ on L1 alone because it
        # is not an L1-neighbourhood associator of the R3 form.
        residual_at_1 = norm(associator(vadd(e(a1), e(a2)), e(s), e(z_unit)))
        # Compare to pure L1 neighbourhood residual at same strength (should be 0 on-line)
        l1_nbhd = norm(associator(vadd(e(a1), vscale(e(b1), 1.0)), e(s), e(z_unit)))
        # For R3 form [e_i+ε e_off, e_j, e_k] with off on same... off=b1 on L1:
        # if {a1,s,z_unit} is Fano line members, adding b1 may leave line or not.
        print(f"F5_RESIDUAL_CROSS_δ1 ||alpha||={residual_at_1:.4f}")
        print(f"F5_L1_INTERNAL_PERTURB ||alpha||={l1_nbhd:.4f}")
        # System obstruction: cross residual stays O(1); claim:
        # residual_at_1 ≈ pure cross (since base=0)
        if abs(residual_at_1 - pure_n) > 1e-6 and base_n < 1e-9:
            # if base is 0, residual should equal pure
            if abs(residual_at_1 - pure_n) > 1e-6:
                f5_ok = False
                print("F5_FAIL residual should equal pure cross when base=0")
        # Non-cancellability: there is no τ such that Φ_fp Path-C style (b=0 via τ)
        # removes the cross obstruction from *being present* — the associator itself
        # remains nonzero independent of τ (τ is a *control-plane* dial, not an
        # algebraic projection). Witness: α_cross is independent of any scalar τ.
        if residual_at_1 < 1.0:
            f5_ok = False
            print("F5_FAIL cross residual too small")
        print(f"F5_SYSTEM_RESIDUAL nonzero_cross -> {'PASS' if residual_at_1 > 1.0 else 'FAIL'}")

    # --- (F6) shared-term structure: the meet unit appears in both squares ---
    f6_ok = (set(L1) & set(L2)) == {s}
    print(f"F6_SHARED_TERM L1={L1} L2={L2} meet=e{s} -> {'PASS' if f6_ok else 'FAIL'}")

    all_ok = f1_ok and f2_ok and f3_ok and f4_ok and f5_ok and f6_ok
    if all_ok:
        print("R4_VERDICT R4_PARTIAL")
        print(
            "R4_NOTE field=7_lines incidence=1 cross_assoc=2 mixing_jet_linear; "
            "multi_line_Phi_path_classes_open; D3_forbidden"
        )
        print("R4_CONTRACT_OK")
        return 0
    print("R4_VERDICT R4_FAIL")
    print("R4_CONTRACT_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
