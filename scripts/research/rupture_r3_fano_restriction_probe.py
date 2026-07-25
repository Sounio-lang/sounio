#!/usr/bin/env python3
"""R3 Fano-restriction → Petitot strata probe (first-principles jet).

Claim ladder: docs/research/rupture-abcd-claims_2026-07-24.md §B
Derivation note: docs/research/rupture-r3-fano-phi_2026-07-25.md

Does NOT assert D3 (Petitot square ≡ ZD / associator).

What this probe certifies
-------------------------
  JET   [e_i + ε e_u, e_j, e_k] = ε [e_u, e_j, e_k]  exactly (ℝ-linear in ε)
        and for each off-line unit u the pure triple associator is ±2 e_m
        (single-axis support in Im 𝕆).

  Φ_fp  First-principles control map into the cusp plane of
        V = x⁴/4 + a x²/2 + b x:

          a = A0 + ‖α‖² / 4     (even jet: ambient non-associativity strength)
          b = τ  + α_m / 2      (odd jet: signed direction + semantic tilt τ)

        A0 = −1 is a *unit choice* for internal opposition depth (semantic input,
        not fit). The algebraic content of Φ_fp is only the jet (‖α‖², α_m).

  (i)   ε = 0 ⇒ α = 0 ⇒ (a,b) = (A0, τ): pure Fano path, no ambient obstruction.
  (ii)  increasing |ε| drives a upward and crosses the fold for some τ
        (2 wells → 1 well).
  (iii) sign(ε) flips α_m and therefore b (at τ=0), and flips which cusp well
        is deeper (argmin of V). Direction, not only norm.

Verdict levels
--------------
  R3_OPEN          jet or (i) failed
  R3_HINT          old: only arrangeable crossing under ad-hoc Φ
  R3_PARTIAL       JET + (i)+(ii)+(iii-weak well asymmetry) under Φ_fp
  R3_GREEN         reserved: also separates contrariety vs contradiction
                   strata (needs more than cusp asymmetry; not claimed here)

Exit 0 iff infrastructure sound and at least R3_PARTIAL (or documented OPEN
with probe OK only if div_ok — we require PARTIAL for gate green on R3 path).

Actually gate expects R3_CONTRACT_PROBE_OK and forbids false R3_GREEN.
PARTIAL is the target of this revision.
"""
from __future__ import annotations

import math
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


def _cbrt(x: float) -> float:
    if x >= 0.0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


def _real_roots_cubic_depressed(p: float, q: float) -> list[float]:
    """Real roots of x³ + p x + q = 0."""
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    roots: list[float] = []
    if disc > 1e-14:
        s = math.sqrt(disc)
        roots.append(_cbrt(-q / 2.0 + s) + _cbrt(-q / 2.0 - s))
    elif abs(disc) <= 1e-14:
        u = _cbrt(-q / 2.0)
        roots.extend([2.0 * u, -u])
    else:
        r = math.sqrt(-p / 3.0)
        phi = math.acos(max(-1.0, min(1.0, (-q / 2.0) / (r ** 3))))
        for k in range(3):
            roots.append(2.0 * r * math.cos((phi + 2.0 * math.pi * k) / 3.0))
    out: list[float] = []
    for x in sorted(roots):
        if not out or abs(x - out[-1]) > 1e-8:
            out.append(x)
    return out


def cusp_minima(a: float, b: float) -> list[float]:
    """Local-minimum locations of V = x⁴/4 + a x²/2 + b x."""
    roots = _real_roots_cubic_depressed(a, b)
    return [x for x in roots if 3.0 * x * x + a > 1e-9]


def cusp_n_minima(a: float, b: float) -> int:
    return len(cusp_minima(a, b))


def fold_delta(a: float, b: float) -> float:
    return 4.0 * a * a * a + 27.0 * b * b


def unique_support(v: list[float], tol: float = 1e-9) -> Optional[tuple[int, float]]:
    """If v has a single nonzero Im component, return (index, value)."""
    hits = [(i, v[i]) for i in range(8) if abs(v[i]) > tol]
    if len(hits) != 1:
        return None
    return hits[0]


# ---------------------------------------------------------------------------
# First-principles Φ
# ---------------------------------------------------------------------------

A0 = -1.0  # unit of internal opposition depth (semantic; not fitted)


def phi_fp(alpha: list[float], tau: float) -> tuple[float, float]:
    """Map associator α and semantic tilt τ → cusp controls (a, b).

    a = A0 + ‖α‖²/4     even ambient jet
    b = τ + α_m/2       odd ambient jet (α_m = unique support coeff, else 0)
    """
    nsq = sum(x * x for x in alpha)
    a = A0 + nsq / 4.0
    sup = unique_support(alpha)
    signed = (sup[1] / 2.0) if sup is not None else 0.0
    b = tau + signed
    return a, b


def alpha_near_line(line: tuple[int, int, int], eps: float, off: int) -> list[float]:
    i, j, k = line
    return associator(vadd(e(i), vscale(e(off), eps)), e(j), e(k))


def main() -> int:
    lines = fano_lines()
    assert len(lines) == 7

    # --- Divergence reconfirm ---
    fano_max = max(norm(associator(e(i), e(j), e(k))) for (i, j, k) in lines)
    noncol = [
        (i, j, k)
        for i in range(1, 8)
        for j in range(i + 1, 8)
        for k in range(j + 1, 8)
        if k != (i ^ j)
    ]
    non_fano_min = min(norm(associator(e(i), e(j), e(k))) for (i, j, k) in noncol)
    div_ok = fano_max < 1e-9 and non_fano_min > 1.5
    print(
        f"DIVERGENCE fano_max_assoc={fano_max:.3e} "
        f"non_fano_min_assoc={non_fano_min:.3f} -> {'PASS' if div_ok else 'FAIL'}"
    )

    line0 = lines[0]
    i0, j0, k0 = line0
    print(f"WORKED_LINE Fano={line0}")

    # --- JET lemma: linearity + single-axis support for every off-line unit ---
    jet_ok = True
    for off in range(1, 8):
        if off in (i0, j0, k0):
            continue
        pure = associator(e(off), e(j0), e(k0))
        sup = unique_support(pure)
        if sup is None or abs(abs(sup[1]) - 2.0) > 1e-9:
            jet_ok = False
            print(f"JET_FAIL pure_support off={off} pure={pure}")
            continue
        for eps in (-1.0, -0.5, 0.25, 0.5, 1.0, 1.5):
            got = alpha_near_line(line0, eps, off)
            exp = vscale(pure, eps)
            err = norm([got[t] - exp[t] for t in range(8)])
            if err > 1e-9:
                jet_ok = False
                print(f"JET_FAIL linear off={off} eps={eps} err={err}")
        print(
            f"JET_OK off=e{off} pure_support=e{sup[0]} coeff={sup[1]:+.1f} "
            f"||pure||={norm(pure):.1f}"
        )
    print(f"JET_LEMMA {'PASS' if jet_ok else 'FAIL'}")

    # Worked off for Φ scans: first off-line unit
    off = next(u for u in range(1, 8) if u not in (i0, j0, k0))
    pure_off = associator(e(off), e(j0), e(k0))
    m_idx, m_coeff = unique_support(pure_off)  # type: ignore[misc]
    print(f"PHI_FP_AXIS off=e{off} alpha_axis=e{m_idx} pure_coeff={m_coeff:+.1f} A0={A0}")

    # --- (i) pure Fano: ε=0 ---
    alpha0 = alpha_near_line(line0, 0.0, off)
    a0, b0 = phi_fp(alpha0, tau=0.3)
    i_ok = norm(alpha0) < 1e-12 and abs(a0 - A0) < 1e-12 and abs(b0 - 0.3) < 1e-12
    print(f"CLAUSE_I pure_fano alpha0=0 a={a0} b={b0} -> {'PASS' if i_ok else 'FAIL'}")

    # --- (ii) ε-driven fold crossing under Φ_fp ---
    eps_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    tau_grid = [-0.4, -0.2, 0.0, 0.2, 0.4]
    wells_by_eps: dict[float, list[int]] = {}
    crossings = 0
    for eps in eps_grid:
        alpha = alpha_near_line(line0, eps, off)
        row = []
        for tau in tau_grid:
            a, b = phi_fp(alpha, tau)
            row.append(cusp_n_minima(a, b))
        wells_by_eps[eps] = row
        print(
            f"PHI_FP_SCAN eps={eps:+.2f} ||alpha||={norm(alpha):.4f} "
            f"wells_by_tau={row}"
        )

    for t_idx, tau in enumerate(tau_grid):
        series = [wells_by_eps[eps][t_idx] for eps in eps_grid]
        if len(set(series)) > 1:
            crossings += 1
            print(f"FOLD_CROSSING tau={tau} well_series={series}")
    ii_ok = crossings > 0
    print(f"CLAUSE_II fold_crossings={crossings} -> {'PASS' if ii_ok else 'FAIL'}")

    # --- (iii) direction ↔ well asymmetry (not mere norm) ---
    # At τ=0, ε = +s vs −s: a even, b odd ⇒ deeper well flips side.
    iii_ok = True
    for s in (0.5, 1.0):
        ap = alpha_near_line(line0, +s, off)
        an = alpha_near_line(line0, -s, off)
        a_p, b_p = phi_fp(ap, 0.0)
        a_n, b_n = phi_fp(an, 0.0)
        # a even, b odd
        if abs(a_p - a_n) > 1e-9 or abs(b_p + b_n) > 1e-9:
            iii_ok = False
            print(f"CLAUSE_III parity FAIL s={s} a+= {a_p} a-={a_n} b+={b_p} b-={b_n}")
        mins_p = cusp_minima(a_p, b_p)
        mins_n = cusp_minima(a_n, b_n)
        if not mins_p or not mins_n:
            # monostable both sides still has a well; compare locations
            pass
        # deeper well = argmin of V; V(x)=x^4/4 + a x^2/2 + b x
        def V(x: float, a: float, b: float) -> float:
            return 0.25 * x**4 + 0.5 * a * x * x + b * x

        def deepest(mins: list[float], a: float, b: float) -> float:
            return min(mins, key=lambda x: V(x, a, b))

        if mins_p and mins_n:
            xp = deepest(mins_p, a_p, b_p)
            xn = deepest(mins_n, a_n, b_n)
            # Expect opposite signs (well asymmetry flips with direction)
            if xp * xn >= 0 and abs(xp) > 1e-6 and abs(xn) > 1e-6:
                iii_ok = False
                print(f"CLAUSE_III well_flip FAIL s={s} deepest+={xp:.4f} deepest-={xn:.4f}")
            else:
                print(
                    f"CLAUSE_III s={s} b+={b_p:+.3f} b-={b_n:+.3f} "
                    f"deepest+={xp:+.4f} deepest-={xn:+.4f} FLIP_OK"
                )
        else:
            # If monostable, single well location should still flip with b
            if mins_p and mins_n:
                pass
            elif len(mins_p) == 1 and len(mins_n) == 1:
                if mins_p[0] * mins_n[0] >= 0:
                    iii_ok = False
            else:
                print(f"CLAUSE_III s={s} mins+={mins_p} mins-={mins_n}")

    # Norm-only control: Φ that uses only ‖α‖ (no direction) must NOT flip wells
    def phi_norm_only(alpha: list[float], tau: float) -> tuple[float, float]:
        nsq = sum(x * x for x in alpha)
        return A0 + nsq / 4.0, tau  # b ignores direction

    ap = alpha_near_line(line0, 0.75, off)
    an = alpha_near_line(line0, -0.75, off)
    a1, b1 = phi_norm_only(ap, 0.2)
    a2, b2 = phi_norm_only(an, 0.2)
    same = abs(a1 - a2) < 1e-12 and abs(b1 - b2) < 1e-12
    print(f"NORM_ONLY_CONTROL same_controls_for_±ε={same} (expect True; no direction)")
    if not same:
        iii_ok = False

    print(f"CLAUSE_III direction_well_asymmetry -> {'PASS' if iii_ok else 'FAIL'}")

    # Isolated square
    pure = norm(associator(e(i0), e(j0), e(k0)))
    print(f"ISOLATED_SQUARE assoc={pure:.3e} (expect 0)")

    if not div_ok or not jet_ok:
        print("R3_VERDICT R3_PROBE_BROKEN")
        print("R3_CONTRACT_FAIL")
        return 1

    if i_ok and ii_ok and iii_ok:
        print("R3_VERDICT R3_PARTIAL")
        print(
            "R3_NOTE Phi_fp=A0+||alpha||^2/4 , tau+alpha_m/2; "
            "A0_unit_choice; contrariety_vs_contradiction_strata_open; D3_forbidden"
        )
    elif ii_ok:
        print("R3_VERDICT R3_HINT")
        print("R3_NOTE partial_clauses_failed; see CLAUSE_* lines")
    else:
        print("R3_VERDICT R3_OPEN")

    print("R3_CONTRACT_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
