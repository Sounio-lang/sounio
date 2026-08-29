#!/usr/bin/env python3
"""R2-full tubular probe — continuous d_sing law on the sedenion ZD hypersurface.

Companion to rupture_r2_fiber_measure_contract.py (R2-partial, unchanged).
Claims doc: docs/research/rupture-abcd-claims_2026-07-24.md §A/R2 ("R2-full").

EXACT certificates (fraction arithmetic; Bareiss elimination; exact polynomial
interpolation of det L_{x(α)}):
  A0  Control: det L_{e1} = ±1 (basis elements are non-singular).
  A1  All 84 census primitives x: det L_x = 0 and rank L_x = 12 (corank 4) —
      the primitives are NON-GENERIC (high-contact) points of the discriminant
      hypersurface {det L_x = 0} ⊂ 𝕊¹⁵.
  A2  Transversal rational slice x(α) = α·e_lo + s·e_hi through every one of
      the 168 census edges: det vanishes to order ≈ 4 at α = 1 (two-point
      ratio over t = 1/100, 1/1000), with IDENTICAL statistics on all seven
      fibers — the exact local form of p_fiber = 1/7.
  A2+ For one canonical edge per fiber, det L_{x(α)} is interpolated EXACTLY
      as a degree-16 polynomial (17 rational points) and factored at α = 1:
      exact vanishing order and exact leading coefficient, per fiber.

MEASURED statements (float Monte Carlo, declared seed — measurements, not
proofs):
  M1  Uniform-MC upper bound: on 20 000 uniform samples of 𝕊¹⁵ essentially
      none have d_sing < 0.5 — the tube is a tiny fraction of μ_G = 1
      (quantitative μ_loc / μ_G separation).
  M2  Local transversal slopes: d_sing(p + t·u)/t^{1/4} at t = 1e-3 over K
      random directions per primitive — per-fiber means agree across all
      seven fibers (continuous analogue of the 7×12 census symmetry).
  M3  Model-based tube estimate (DECLARED APPROXIMATION): using the measured
      local law d_sing ≈ s_u·t^{1/4}, the tube around each primitive (and its
      antipode) extends to t*(u) = (ε/s_u)^4, capped at t*_max = 0.05 where
      the local law is not validated; cap overlaps ignored. Reported as an
      estimator with its assumptions, not as a theorem.

Verdict vocabulary (honest):
  R2_FULL_MEASURED       exact anchors pass AND measured sections ran
  R2_FULL_PROBE_BROKEN   exact anchors fail (infrastructure bug → exit 1)

This probe does NOT prove the continuous law, does NOT claim D3, and does NOT
alter the R2-partial gate contract.

Exit 0 iff exact anchors pass and the measured machinery ran.
Prints machine lines: R2_FULL_PROBE_OK, R2_FULL_VERDICT ...
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path
from random import Random

# Allow `python3 scripts/research/rupture_r2_full_tubular_probe.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rupture_r2_fiber_measure_contract import (  # noqa: E402
    build_primitives,
    cd_sigma,
    fiber_label,
)

FIBERS = list(range(9, 16))
T_STAR_MAX = 0.05  # declared truncation for the M3 model estimate


# --------------------------------------------------------------------------
# Exact rational linear algebra
# --------------------------------------------------------------------------

def lmat_exact(x: dict[int, Fraction]) -> list[list[Fraction]]:
    M = [[Fraction(0)] * 16 for _ in range(16)]
    for i, ci in x.items():
        for j in range(16):
            k = i ^ j
            M[k][j] += cd_sigma(i, j) * ci
    return M


def det_bareiss(M: list[list[Fraction]]) -> Fraction:
    """Determinant via fraction-free Bareiss with row pivoting.

    Returns 0 as soon as an elimination column has no pivot — for the
    matrices in this probe that coincides with singularity (cross-checked
    against the rank computation in A1 for all 84 primitives).
    """
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = Fraction(1)
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = next((r for r in range(k + 1, n) if A[r][k] != 0), None)
            if piv is None:
                return Fraction(0)
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        for i in range(k + 1, n):
            Ai, Ak = A[i], A[k]
            aik = Ai[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * Ak[k] - aik * Ak[j]) / prev
        prev = Ak[k]
    return sign * A[n - 1][n - 1]


def rank_bareiss(M: list[list[Fraction]]) -> int:
    n = len(M)
    A = [row[:] for row in M]
    prev = Fraction(1)
    rank = 0
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = next((r for r in range(k + 1, n) if A[r][k] != 0), None)
            if piv is None:
                continue
            A[k], A[piv] = A[piv], A[k]
        rank += 1
        for i in range(k + 1, n):
            Ai, Ak = A[i], A[k]
            aik = Ai[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * Ak[k] - aik * Ak[j]) / prev
        prev = Ak[k]
    if A[n - 1][n - 1] != 0:
        rank += 1
    return rank


def prim_vec_exact(c) -> dict[int, Fraction]:
    lo, hi, neg = c
    return {lo: Fraction(1), hi: Fraction(-1 if neg else 1)}


def det_poly_at(lo: int, hi: int, s: int, alpha: Fraction) -> Fraction:
    return det_bareiss(lmat_exact({lo: alpha, hi: Fraction(s)}))


def interp_newton(xs: list[Fraction], ys: list[Fraction]) -> list[Fraction]:
    """Newton divided-difference coefficients (exact)."""
    n = len(xs)
    coef = ys[:]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    return coef


# --------------------------------------------------------------------------
# A0/A1/A2/A2+: exact anchors
# --------------------------------------------------------------------------

def exact_anchors(part, edges) -> bool:
    det_e1 = det_bareiss(lmat_exact({1: Fraction(1)}))
    control_ok = abs(det_e1) == 1
    print(f"EXACT_CONTROL det_L_e1={det_e1} -> {'PASS' if control_ok else 'FAIL'}")

    det0 = 0
    rank12 = 0
    for c in part:
        M = lmat_exact(prim_vec_exact(c))
        if det_bareiss(M) == 0:
            det0 += 1
        if rank_bareiss(M) == 12:
            rank12 += 1
    a1_ok = det0 == 84 and rank12 == 84 and len(part) == 84
    print(f"EXACT_ANCHORS det0={det0}/84 rank12={rank12}/84 "
          f"-> {'PASS' if a1_ok else 'FAIL'}")

    # A2: two-point order estimate on all 168 edges
    t1, t2 = Fraction(1, 100), Fraction(1, 1000)
    stats: dict[int, list[tuple[float, float]]] = {L: [] for L in FIBERS}
    for a, _b in edges:
        lo, hi, neg = a
        s = -1 if neg else 1
        d1 = det_poly_at(lo, hi, s, 1 + t1)
        d2 = det_poly_at(lo, hi, s, 1 + t2)
        r1, r2 = abs(float(d1)), abs(float(d2))
        order = math.log(r1 / r2) / math.log(10.0)
        coeff = r2 / (float(t2) ** order)
        stats[fiber_label(a)].append((order, coeff))
    all_orders = [o for L in FIBERS for o, _ in stats[L]]
    order4 = sum(1 for o in all_orders if abs(o - 4.0) < 0.1)
    a2_ok = order4 == 168
    print(f"EXACT_ORDER4 edges_order≈4={order4}/168 (two-point estimate) "
          f"-> {'PASS' if a2_ok else 'FAIL'}")
    for L in FIBERS:
        os_ = [o for o, _ in stats[L]]
        cs = [c for _, c in stats[L]]
        om = sum(os_) / len(os_)
        cm = sum(cs) / len(cs)
        cvar = sum((c - cm) ** 2 for c in cs) / len(cs)
        print(f"FIBER_ORDER L={L} edges={len(os_)} order_mean={om:.4f} "
              f"order_spread={max(os_) - min(os_):.4f} coeff_mean={cm:.4e} "
              f"coeff_var={cvar:.4e}")

    # A2+: exact polynomial factorisation for one canonical edge per fiber
    exact_ok = True
    seen: set[int] = set()
    alphas = [Fraction(1) + Fraction(k, 100) for k in range(-8, 9)]
    for a, _b in edges:
        L = fiber_label(a)
        if L in seen:
            continue
        seen.add(L)
        lo, hi, neg = a
        s = -1 if neg else 1
        ys = [det_poly_at(lo, hi, s, al) for al in alphas]
        coef = interp_newton(alphas, ys)
        # exact deflation in the monomial basis: order = multiplicity of α=1
        mono = _newton_to_monomial(coef, alphas)
        order = 0
        while len(mono) > 1 and sum(mono) == 0:  # p(1) = sum of coeffs
            mono = _deflate_at_one(mono)
            order += 1
        lead = sum(mono)  # value of the deflated polynomial at α=1
        edge_ok = order == 4
        exact_ok = exact_ok and edge_ok
        print(f"EXACT_POLY L={L} edge={a} vanishing_order={order} "
              f"leading_coeff_at_1={lead} (float={float(lead):.6e}) "
              f"-> {'PASS' if edge_ok else 'FAIL'}")
    a2p_ok = exact_ok and len(seen) == 7

    return control_ok and a1_ok and a2_ok and a2p_ok


def _newton_to_monomial(coef: list[Fraction], xs: list[Fraction]) -> list[Fraction]:
    """Convert Newton form to monomial coefficients (ascending powers)."""
    n = len(coef)
    mono = [Fraction(0)] * n
    basis = [Fraction(1)]  # ∏_{j<i} (x - xs[j]), ascending powers
    for i in range(n):
        for j in range(len(basis)):
            mono[j] += coef[i] * basis[j]
        if i < n - 1:
            basis = _poly_mul_linear(basis, xs[i])
    return mono


def _poly_mul_linear(poly: list[Fraction], x0: Fraction) -> list[Fraction]:
    """Multiply ascending-power polynomial by (x - x0)."""
    out = [Fraction(0)] * (len(poly) + 1)
    for j, c in enumerate(poly):
        out[j] -= c * x0
        out[j + 1] += c
    return out


def _deflate_at_one(mono: list[Fraction]) -> list[Fraction]:
    """Divide ascending-power polynomial by (x - 1); requires p(1) = 0."""
    desc = mono[::-1]
    q = [desc[0]]
    for c in desc[1:-1]:
        q.append(c + q[-1])
    return q[::-1]


# --------------------------------------------------------------------------
# Float linear algebra for Monte Carlo
# --------------------------------------------------------------------------

def lmat_float(x: list[float]) -> list[list[float]]:
    M = [[0.0] * 16 for _ in range(16)]
    for i in range(16):
        ci = x[i]
        if ci == 0.0:
            continue
        for j in range(16):
            M[i ^ j][j] += cd_sigma(i, j) * ci
    return M


def det_float(M: list[list[float]]) -> float:
    n = 16
    A = [row[:] for row in M]
    sign = 1.0
    det = 1.0
    for k in range(n):
        piv = k
        big = abs(A[k][k])
        for r in range(k + 1, n):
            if abs(A[r][k]) > big:
                big = abs(A[r][k])
                piv = r
        if big < 1e-300:
            return 0.0
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        akk = A[k][k]
        det *= akk
        for i in range(k + 1, n):
            f = A[i][k] / akk
            Ai = A[i]
            for j in range(k + 1, n):
                Ai[j] -= f * A[k][j]
    return sign * det


def d_sing(x: list[float]) -> float:
    return abs(det_float(lmat_float(x))) ** (1.0 / 16.0)


# --------------------------------------------------------------------------
# M1–M3: measured continuous law
# --------------------------------------------------------------------------

def measured_law(part, rng: Random) -> bool:
    prims: list[tuple[list[float], int]] = []
    for c in part:
        v = [0.0] * 16
        lo, hi, neg = c
        v[lo] = 1.0 / math.sqrt(2.0)
        v[hi] = (-1.0 if neg else 1.0) / math.sqrt(2.0)
        prims.append((v, fiber_label(c)))

    # M1: uniform-MC upper bound on the tube mass
    n_mc = 20000
    below = {0.5: 0, 0.6: 0, 0.7: 0}
    dmin = 1e9
    for _ in range(n_mc):
        x = [rng.gauss(0.0, 1.0) for _ in range(16)]
        nx = math.sqrt(sum(t * t for t in x))
        dv = d_sing([t / nx for t in x])
        dmin = min(dmin, dv)
        for eps in below:
            if dv < eps:
                below[eps] += 1
    print(f"TUBE_UPPER_BOUND n={n_mc} " +
          " ".join(f"mu(<{e})<={(below[e] + 1) / n_mc:.2e}" for e in sorted(below)) +
          f" d_min={dmin:.4f} (uniform MC; tube ≪ μ_G=1)")
    m1_ok = below[0.5] <= n_mc * 0.01  # tube under 1% at eps=0.5
    print(f"SEPARATION mu_loc_tube_vs_mu_G -> {'MEASURED' if m1_ok else 'UNEXPECTEDLY_LARGE'}")

    # M2: local transversal slopes per fiber
    k_dirs = 50
    t_loc = 1e-3
    slope_samples: dict[int, list[float]] = {L: [] for L in FIBERS}
    for pv, L in prims:
        for _ in range(k_dirs):
            u = [rng.gauss(0.0, 1.0) for _ in range(16)]
            nu = math.sqrt(sum(t * t for t in u))
            x = [pv[i] + t_loc * u[i] / nu for i in range(16)]
            nx = math.sqrt(sum(t * t for t in x))
            s = d_sing([t / nx for t in x]) / (t_loc ** 0.25)
            slope_samples[L].append(s)
    for L in FIBERS:
        ss = slope_samples[L]
        sm = sum(ss) / len(ss)
        sv = sum((v - sm) ** 2 for v in ss) / len(ss)
        print(f"LOCAL_SLOPE L={L} mean={sm:.4f} var={sv:.4f} n={len(ss)}")
    means = [sum(slope_samples[L]) / len(slope_samples[L]) for L in FIBERS]
    spread = (max(means) - min(means)) / (sum(means) / len(means))
    print(f"LOCAL_SLOPE_UNIFORMITY rel_spread={spread:.4f} across 7 fibers "
          f"({'PASS' if spread < 0.05 else 'FAIL'} expect <5%)")

    # M3: model-based tube estimate (declared approximation, see docstring)
    vol_s15 = 2.0 * math.pi ** 8 / 5040.0        # Vol(𝕊¹⁵) = 2π^8/Γ(8)
    vol_b15 = math.pi ** 7.5 / math.gamma(8.5)   # 15-dim unit ball volume
    all_slopes = [s for L in FIBERS for s in slope_samples[L]]
    n_s = len(all_slopes)
    for eps in (0.025, 0.05, 0.1, 0.2):
        acc = 0.0
        for s in all_slopes:
            t_star = (eps / s) ** 4
            if t_star > T_STAR_MAX:
                t_star = T_STAR_MAX
            acc += t_star ** 15
        # 84 primitives × 2 (antipodes) × E_u[ball15(t*)] / Vol(𝕊¹⁵)
        mu_est = 168.0 * vol_b15 * (acc / n_s) / 15.0 / vol_s15
        print(f"MU_MODEL eps={eps:.3f} mu_est={mu_est:.3e} "
              f"(local t^1/4 law; t*≤{T_STAR_MAX}; overlaps ignored)")

    return True


def main() -> int:
    part, edges, _adj = build_primitives()
    anchors_ok = exact_anchors(part, edges)
    rng = Random(20260725)
    measured_law(part, rng)

    if not anchors_ok:
        print("R2_FULL_VERDICT R2_FULL_PROBE_BROKEN")
        print("R2_FULL_CONTRACT_FAIL")
        return 1
    print("R2_FULL_VERDICT R2_FULL_MEASURED (exact anchors + MC measurements; "
          "measurement not proof; D3 forbidden)")
    print("R2_FULL_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
