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
  (iii+) Two path classes from the same jet α, operationalising Petitot's
        two opposition types (not D3 identity):

        Path C — CONTRARIETY (even jet only): cancel odd part by
          τ = −α_m/2 so b≡0; sweep ε. Both poles approach the origin and
          merge into a *neutral* monostable well (x≈0). "Both false" /
          neutralisation is possible.

        Path D — CONTRADICTION (odd jet): τ=0 so b=α_m/2; sweep ε.
          One polar well deepens as the other vanishes; monostable well
          stays *polar* (|x| ≳ 0.5). Sign(ε) selects which pole. "Neither"
          (no well / only neutral) does not occur on this path.

        Boolean lattice 2² has a single complement type and cannot host
        both path classes — operational non-Booleanisability under Φ_fp.

Verdict levels
--------------
  R3_OPEN          jet or (i) failed
  R3_HINT          arrangeable crossing only
  R3_PARTIAL       JET + (i)+(ii)+(iii-weak)
  R3_GREEN         PARTIAL + (iii+) path-class separation (this revision's target)

Gate expects R3_CONTRACT_PROBE_OK and R3_VERDICT R3_GREEN (or PARTIAL during
transition). Still forbids claiming D3.
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
    """Local-minimum locations of V = x⁴/4 + a x²/2 + b x.

    Includes the neutral monostable well at x=0 when a>0, b=0, and the flat
    inflection-minimum at the cusp tip (a,b)=(0,0).
    """
    roots = _real_roots_cubic_depressed(a, b)
    mins = [x for x in roots if 3.0 * x * x + a > 1e-9]
    if mins:
        return mins
    # Degenerate / monostable-neutral cases the cubic root finder may miss cleanly
    if abs(b) < 1e-12 and a >= -1e-12:
        return [0.0]
    return []


def cusp_n_minima(a: float, b: float) -> int:
    return len(cusp_minima(a, b))


def V_cusp(x: float, a: float, b: float) -> float:
    return 0.25 * x**4 + 0.5 * a * x * x + b * x


def deepest_well(mins: list[float], a: float, b: float) -> float:
    return min(mins, key=lambda x: V_cusp(x, a, b))


def alpha_signed_coeff(alpha: list[float]) -> float:
    """α_m for single-support associator; 0 if not unique."""
    sup = unique_support(alpha)
    return sup[1] if sup is not None else 0.0


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
            xp = deepest_well(mins_p, a_p, b_p)
            xn = deepest_well(mins_n, a_n, b_n)
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
            print(f"CLAUSE_III s={s} mins+={mins_p} mins-={mins_n}")
            iii_ok = False

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

    # --- (iii+) Contrariety vs contradiction path classes from the same jet ---
    # Path C: cancel odd jet (τ = -α_m/2) ⇒ b≡0; even jet only → neutral merge
    # Path D: τ=0 ⇒ b=α_m/2; odd jet → polar selection
    eps_path = [0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.25]
    path_c_records = []  # (eps, n, deepest, |deepest|)
    path_d_records = []
    for eps in eps_path:
        alpha = alpha_near_line(line0, eps, off)
        am = alpha_signed_coeff(alpha)
        # Path C — contrariety / neutralisation
        tau_c = -am / 2.0
        a_c, b_c = phi_fp(alpha, tau_c)
        mins_c = cusp_minima(a_c, b_c)
        n_c = len(mins_c)
        d_c = deepest_well(mins_c, a_c, b_c) if mins_c else float("nan")
        path_c_records.append((eps, n_c, d_c, abs(d_c) if mins_c else float("nan"), a_c, b_c))
        print(
            f"PATH_C contrariety eps={eps:.2f} tau={tau_c:+.3f} "
            f"a={a_c:.3f} b={b_c:.3e} n={n_c} deepest={d_c:+.4f}"
            if mins_c
            else f"PATH_C contrariety eps={eps:.2f} n=0"
        )
        # Path D — contradiction / polar
        a_d, b_d = phi_fp(alpha, 0.0)
        mins_d = cusp_minima(a_d, b_d)
        n_d = len(mins_d)
        d_d = deepest_well(mins_d, a_d, b_d) if mins_d else float("nan")
        path_d_records.append((eps, n_d, d_d, abs(d_d) if mins_d else float("nan"), a_d, b_d))
        print(
            f"PATH_D contradiction eps={eps:.2f} "
            f"a={a_d:.3f} b={b_d:+.3f} n={n_d} deepest={d_d:+.4f}"
            if mins_d
            else f"PATH_D contradiction eps={eps:.2f} n=0"
        )

    # Path C: starts bistable, ends monostable NEUTRAL (|deepest| small), b≈0 always
    c_start_bi = path_c_records[0][1] == 2
    c_b_always_0 = all(abs(r[5]) < 1e-9 for r in path_c_records)
    c_end = path_c_records[-1]
    c_end_neutral = c_end[1] == 1 and c_end[3] < 0.25
    # Along C, when monostable, well stays near 0 (not polar)
    c_mono_neutral = all(
        (r[1] != 1) or (r[3] < 0.35) for r in path_c_records
    )
    path_c_ok = c_start_bi and c_b_always_0 and c_end_neutral and c_mono_neutral
    print(
        f"PATH_C_CHECK start_bi={c_start_bi} b0={c_b_always_0} "
        f"end_neutral={c_end_neutral} mono_neutral={c_mono_neutral} "
        f"-> {'PASS' if path_c_ok else 'FAIL'}"
    )

    # Path D: for |eps| large enough, monostable POLAR; ±eps flip pole; never neutral mono
    d_pos = [r for r in path_d_records if r[0] >= 0.75]
    d_neg_eps = alpha_near_line(line0, -1.0, off)
    a_dn, b_dn = phi_fp(d_neg_eps, 0.0)
    mins_dn = cusp_minima(a_dn, b_dn)
    d_pos_ok = all(r[1] == 1 and r[3] > 0.5 for r in d_pos)
    d_neg_ok = len(mins_dn) == 1 and abs(mins_dn[0]) > 0.5
    d_flip = bool(d_pos and mins_dn and d_pos[-1][2] * mins_dn[0] < 0)
    # No neutral monostable on D: when n==1, |x| large
    d_no_neutral = all((r[1] != 1) or (r[3] > 0.5) for r in path_d_records if r[0] != 0.0)
    path_d_ok = d_pos_ok and d_neg_ok and d_flip and d_no_neutral
    print(
        f"PATH_D_CHECK polar_mono+={d_pos_ok} polar_mono-={d_neg_ok} "
        f"flip={d_flip} no_neutral={d_no_neutral} -> {'PASS' if path_d_ok else 'FAIL'}"
    )

    # Distinctness: C ends neutral, D ends polar — two non-equivalent monostable outcomes
    distinct = path_c_ok and path_d_ok and c_end[3] < 0.25 and d_pos[-1][3] > 0.5
    print(f"PATH_DISTINCTNESS neutral_vs_polar -> {'PASS' if distinct else 'FAIL'}")

    # Boolean impossibility operational: single-complement Boolean lattice cannot
    # host both a neutralisation path and a polar-selection path as distinct types.
    # We only claim the *operational* witness (two path classes), not a topos theorem.
    iii_plus_ok = distinct
    print(
        f"CLAUSE_III_PLUS contrariety_vs_contradiction_paths -> "
        f"{'PASS' if iii_plus_ok else 'FAIL'}"
    )

    # Isolated square
    pure = norm(associator(e(i0), e(j0), e(k0)))
    print(f"ISOLATED_SQUARE assoc={pure:.3e} (expect 0)")

    if not div_ok or not jet_ok:
        print("R3_VERDICT R3_PROBE_BROKEN")
        print("R3_CONTRACT_FAIL")
        return 1

    if i_ok and ii_ok and iii_ok and iii_plus_ok:
        print("R3_VERDICT R3_GREEN")
        print(
            "R3_NOTE Phi_fp jet even/odd split = Path_C_contrariety + Path_D_contradiction; "
            "A0_unit_choice; operational_non_Booleanisability; D3_identity_still_forbidden"
        )
    elif i_ok and ii_ok and iii_ok:
        print("R3_VERDICT R3_PARTIAL")
        print("R3_NOTE iii_plus_failed; strata_paths_open; D3_forbidden")
    elif ii_ok:
        print("R3_VERDICT R3_HINT")
        print("R3_NOTE partial_clauses_failed; see CLAUSE_* lines")
    else:
        print("R3_VERDICT R3_OPEN")

    print("R3_CONTRACT_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
