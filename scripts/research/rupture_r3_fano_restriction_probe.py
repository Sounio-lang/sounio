#!/usr/bin/env python3
"""R3 Fano-restriction → Petitot strata probe (rupture A+B+C+D).

Does NOT claim D3 (Petitot square = ZD/associator). It:

  1. Reconfirms the documented divergence:
       Fano lines  → associator = 0  (isolated square Booleanisable in 𝕆)
       non-Fano    → associator > 0  (field of squares not globally Booleanisable)
  2. Builds a *candidate* map Φ_ε from a neighbourhood of one Fano line into the
     cusp control plane (a,b) of V = x⁴/4 + a·x²/2 + b·x.
  3. Checks whether non-Fano ambient strength ε drives fold crossings (Δ=4a³+27b²).
  4. Emits an honest verdict:
       R3_OPEN   — hypothesis not yet demonstrated under the B-contract
       R3_HINT   — candidate Φ shows ε-driven well-count change (necessary but not sufficient)
       R3_GREEN  — reserved; requires (i)–(iii) of docs/research/rupture-abcd-claims_2026-07-24.md
                   including associator *direction* ↔ crossing type (not implemented here)

Exit 0 if the probe infrastructure is sound (divergence reconfirmed + Φ evaluated).
A green process exit does **not** mean R3_GREEN.

Uses the same Cayley–Dickson sign law as other research oracles (bits=3 for 𝕆).
"""
from __future__ import annotations

import math
from typing import Iterable


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


def cusp_n_minima(a: float, b: float) -> int:
    """Number of local minima of V=x^4/4 + a x^2/2 + b x  (via V'=x^3+a x+b)."""
    # Solve x^3 + a x + b = 0
    # depressed cubic: use Cardano / numpy-free discriminant path
    # For real roots of x^3 + p x + q = 0 with p=a, q=b:
    disc = -4 * a * a * a - 27 * b * b  # related to 4a^3+27b^2 with sign
    # Count minima by scanning critical points via companion derivative test
    # V'' = 3x^2 + a; critical pts from Cardano
    roots = _real_roots_cubic(1.0, 0.0, a, b)
    mins = 0
    for x in roots:
        if 3.0 * x * x + a > 1e-9:
            mins += 1
    return mins


def _real_roots_cubic(A: float, B: float, C: float, D: float) -> list[float]:
    """Real roots of A x^3 + B x^2 + C x + D = 0 (here B=0)."""
    # Depress: x = z - B/(3A)
    assert abs(A - 1.0) < 1e-15 and abs(B) < 1e-15
    p = C
    q = D
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    roots: list[float] = []
    if disc > 1e-14:
        s = math.sqrt(disc)
        u = _cbrt(-q / 2.0 + s)
        v = _cbrt(-q / 2.0 - s)
        roots.append(u + v)
    elif abs(disc) <= 1e-14:
        u = _cbrt(-q / 2.0)
        roots.append(2.0 * u)
        roots.append(-u)
    else:
        r = math.sqrt(-p / 3.0)
        phi = math.acos(max(-1.0, min(1.0, (-q / 2.0) / (r ** 3))))
        for k in range(3):
            roots.append(2.0 * r * math.cos((phi + 2.0 * math.pi * k) / 3.0))
    # dedupe
    out: list[float] = []
    for x in sorted(roots):
        if not out or abs(x - out[-1]) > 1e-8:
            out.append(x)
    return out


def _cbrt(x: float) -> float:
    if x >= 0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


def fold_delta(a: float, b: float) -> float:
    """Δ = 4a³ + 27b²; bistable (2 wells for cusp) when Δ < 0 and a < 0."""
    return 4.0 * a * a * a + 27.0 * b * b


def candidate_phi(eps: float, tilt: float, assoc_scale: float) -> tuple[float, float]:
    """Candidate Φ — *not* derived from first principles.

    a = -1 + c·(ambient associator scale)²
    b = tilt

    At eps=0 / assoc_scale=0: (a,b)=(-1, tilt) sits in the classical bistable cusp band for small tilt.
    Growing ambient non-associativity pushes a toward the monostable side (fold crossing).
    This is a *probe* of whether an ε-driven crossing is even arrangeable — not a proof of R3.
    """
    c = 0.85
    a = -1.0 + c * (assoc_scale ** 2)
    b = tilt
    return a, b


def ambient_assoc_scale(line: tuple[int, int, int], eps: float, off: int) -> float:
    """Associator norm of (e_i + eps e_off, e_j, e_k) for Fano line (i,j,k)."""
    i, j, k = line
    a = vadd(e(i), vscale(e(off), eps))
    return norm(associator(a, e(j), e(k)))


def main() -> int:
    lines = fano_lines()
    assert len(lines) == 7
    fano_max = max(norm(associator(e(i), e(j), e(k))) for (i, j, k) in lines)
    noncol = [
        (i, j, k)
        for i in range(1, 8)
        for j in range(i + 1, 8)
        for k in range(j + 1, 8)
        if k != (i ^ j)
    ]
    non_fano_min = min(norm(associator(e(i), e(j), e(k))) for (i, j, k) in noncol)
    # non-Fano basis triples have ||assoc|| = 2
    div_ok = fano_max < 1e-9 and non_fano_min > 1.5
    print(f"DIVERGENCE fano_max_assoc={fano_max:.3e} non_fano_min_assoc={non_fano_min:.3f} -> {'PASS' if div_ok else 'FAIL'}")

    line0 = lines[0]
    i0, j0, k0 = line0
    # pick off-line unit
    off = next(u for u in range(1, 8) if u not in (i0, j0, k0))
    print(f"WORKED_LINE Fano={line0} off={off}")

    # Scan ε and tilt under candidate Φ
    eps_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    tilt_grid = [-0.4, -0.2, 0.0, 0.2, 0.4]
    wells_by_eps: dict[float, list[int]] = {}
    crossings = 0
    for eps in eps_grid:
        scale = ambient_assoc_scale(line0, eps, off)
        row = []
        for tilt in tilt_grid:
            a, b = candidate_phi(eps, tilt, scale)
            n = cusp_n_minima(a, b)
            row.append(n)
        wells_by_eps[eps] = row
        print(f"PHI_SCAN eps={eps:.2f} assoc_scale={scale:.4f} wells_by_tilt={row}")

    # Did well-counts change as eps increased for some fixed tilt?
    for t_idx, tilt in enumerate(tilt_grid):
        series = [wells_by_eps[eps][t_idx] for eps in eps_grid]
        if len(set(series)) > 1:
            crossings += 1
            print(f"FOLD_CROSSING_HINT tilt={tilt} well_series={series}")

    # Isolated square (eps=0): associator of pure line triple is 0
    pure = norm(associator(e(i0), e(j0), e(k0)))
    print(f"ISOLATED_SQUARE assoc={pure:.3e} (expect 0)")

    # Direction stub: sign of a component of associator under ±eps (for future (iii))
    assoc_pos = associator(vadd(e(i0), vscale(e(off), 0.5)), e(j0), e(k0))
    assoc_neg = associator(vadd(e(i0), vscale(e(off), -0.5)), e(j0), e(k0))
    # components may flip with eps sign — record first nonzero Im component signs
    def first_sign(v: list[float]) -> int:
        for x in v[1:]:
            if abs(x) > 1e-9:
                return 1 if x > 0 else -1
        return 0

    s_pos, s_neg = first_sign(assoc_pos), first_sign(assoc_neg)
    print(f"ASSOC_DIRECTION_STUB sign(+eps)={s_pos} sign(-eps)={s_neg} (iii not yet contracted)")

    if not div_ok:
        print("R3_VERDICT R3_PROBE_BROKEN")
        print("R3_CONTRACT_FAIL")
        return 1

    if crossings > 0:
        # Necessary arrangement exists under candidate Φ; not B-contract green
        print(f"R3_VERDICT R3_HINT crossings_on_tilt_slices={crossings}")
        print("R3_NOTE candidate_Phi_is_not_first_principles; direction_clause_open; D3_forbidden")
    else:
        print("R3_VERDICT R3_OPEN no_eps_driven_well_change_under_candidate_Phi")

    print("R3_CONTRACT_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
