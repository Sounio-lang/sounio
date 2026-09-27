#!/usr/bin/env python3
"""
Functor F — Phi_fp restated G2-EQUIVARIANTLY, exhibited over a CONTINUOUS G2 orbit.

Delivers two edges from functor_f_g2_equivariance_spec §7:
  (2) propagate b_cov := <alpha, e_m> into the Phi_fp ladder as the canonical polar
      coordinate, and show the C/D path classes + Betti drop become G2-invariant;
  (3) replace the finite sample of automorphisms with a one-parameter subgroup
      g(t) = exp(t·D), D a genuine octonion derivation (D in g2 = Lie(G2)), and
      exhibit the obstruction as a smooth curve b_argmax(t) vs the FLAT b_cov(t).

Companion to:
  docs/research/functor_f_phi_fp_equivariant_spec_2026-07-25.md
  docs/research/functor_f_g2_equivariance_spec_2026-07-25.md   (parent: H_CHARACTERISED)

Self-contained: CD sign law, octonion product, a hand-rolled matrix exponential
(scaling + squaring; scipy is not available in this tree), and the derivation are
all re-implemented and numerically self-verified.
"""
import numpy as np

np.seterr(all='ignore')

INV_TOL = 1e-9
DERIV_TOL = 1e-9
BREAK_TOL = 1e-2     # a curve whose range exceeds this genuinely moves


# ---------------------------------------------------------------- algebra
def cds(a, b, bits=3):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah, bh = a >= h, b >= h
        al, bl = a & (h - 1), b & (h - 1)
        if not ah and not bh:
            a, b = al, bl
        elif not ah and bh:
            a, b = bl, al
        elif ah and not bh:
            a, b, s = ((al, 0, s) if bl == 0 else (al, bl, -s))
        else:
            a, b, s = ((0, al, -s) if bl == 0 else (bl, al, s))
        bits -= 1
    return s


def omul(A, B):
    C = np.zeros(8)
    for i in range(8):
        for j in range(8):
            C[i ^ j] += cds(i, j) * A[i] * B[j]
    return C


def e(i):
    v = np.zeros(8)
    v[i] = 1.0
    return v


def comm(u, v):
    return omul(u, v) - omul(v, u)


def assoc(u, v, w):
    return omul(omul(u, v), w) - omul(u, omul(v, w))


# ---------------------------------------------------------------- g2 derivation + expm
def inner_derivation(a, b):
    """Schafer inner derivation of the octonions: D_{a,b}(x) = [[a,b],x] - 3(a,b,x).

    For imaginary a,b this is a derivation, hence an element of g2 = Lie(G2).
    Returned as an 8x8 matrix (column i = D(e_i)); it annihilates the real unit.
    """
    D = np.zeros((8, 8))
    ab = comm(a, b)
    for i in range(8):
        D[:, i] = comm(ab, e(i)) - 3.0 * assoc(a, b, e(i))
    return D


def is_derivation(D):
    worst = 0.0
    for i in range(8):
        for j in range(8):
            lhs = D @ omul(e(i), e(j))                     # D(e_i e_j)
            rhs = omul(D[:, i], e(j)) + omul(e(i), D[:, j])  # D(e_i)e_j + e_i D(e_j)
            worst = max(worst, float(np.linalg.norm(lhs - rhs)))
    return worst


def expm(A, terms=32):
    """Matrix exponential by scaling-and-squaring + Taylor (8x8, no scipy)."""
    norm = float(np.max(np.sum(np.abs(A), axis=1)))
    s = max(0, int(np.ceil(np.log2(norm + 1e-30))))
    B = A / (2.0 ** s)
    R = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, terms):
        term = term @ B / k
        R = R + term
    for _ in range(s):
        R = R @ R
    return R


def automorphism_residual(M):
    worst = 0.0
    for i in range(8):
        for j in range(8):
            worst = max(worst, float(np.linalg.norm(M @ omul(e(i), e(j)) - omul(M[:, i], M[:, j]))))
    return worst


# ---------------------------------------------------------------- F assignment
def single_line_jet(line, off_unit, eps):
    i, j, k = line
    base = e(i) + eps * e(off_unit)
    return omul(omul(base, e(j)), e(k)) - omul(base, omul(e(j), e(k)))


def argmax_b(alpha):
    a_im = alpha.copy(); a_im[0] = 0.0
    return float(a_im[int(np.argmax(np.abs(a_im)))])


def cov_b(alpha, em):
    return float(np.dot(alpha, em))


def a_coeff(alpha, A0=-1.0):
    return A0 + float(np.dot(alpha, alpha)) / 4.0


# ---------------------------------------------------------------- cusp / path / betti
def cusp_value(x, a, b):
    return 0.25 * x ** 4 + 0.5 * a * x * x + b * x


def cusp_wells(a, b):
    roots = np.roots([1.0, 0.0, a, b])
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    return sorted([x for x in real if 3.0 * x * x + a > 1e-9])


def deepest_well(a, b):
    mins = cusp_wells(a, b)
    if not mins:
        return None
    return mins[int(np.argmin([cusp_value(x, a, b) for x in mins]))]


def betti_zero(a, b):
    roots = np.roots([1.0, 0.0, a, b])
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    mins = sorted([x for x in real if 3.0 * x * x + a > 1e-9])
    saddles = sorted([x for x in real if 3.0 * x * x + a < -1e-9])
    if not mins:
        return 0
    if len(mins) == 1:
        c = cusp_value(mins[0], a, b) + 1.0
    else:
        hi = max(cusp_value(x, a, b) for x in mins)
        c = (hi - 0.5) if not saddles else 0.5 * (cusp_value(saddles[0], a, b) + hi)
    R = max(3.0, 2.0 * (abs(b) + 1.0) / max(abs(a), 0.1) ** 0.5 if a != 0 else 3.0)
    xs = np.linspace(-R, R, 20001)
    below = cusp_value(xs, a, b) <= c
    comps, prev = 0, False
    for f in below:
        if f and not prev:
            comps += 1
        prev = f
    return comps


# worked configuration
LINE, OFF, EPS = (1, 2, 3), 4, 2.0
ALPHA0 = single_line_jet(LINE, OFF, EPS)
EM0 = e(int(np.argmax(np.abs(np.concatenate([[0.0], ALPHA0[1:]])))))  # the jet axis, config-fixed


def path_d_well(alpha, b_value):
    """Path D (tau=0): deepest well of the cusp with a=A0+||alpha||^2/4, b=b_value/2."""
    a = a_coeff(alpha)
    return deepest_well(a, b_value / 2.0)


def build_orbit():
    rng = np.random.default_rng(7)
    a_vec = rng.standard_normal(8); a_vec[0] = 0.0; a_vec /= np.linalg.norm(a_vec)
    b_vec = rng.standard_normal(8); b_vec[0] = 0.0; b_vec /= np.linalg.norm(b_vec)
    D = inner_derivation(a_vec, b_vec)
    return D


def main():
    print("=" * 70)
    print("FUNCTOR F — Phi_fp G2-EQUIVARIANT (continuous orbit)")
    print("=" * 70)

    # E1 — the generator is a genuine octonion derivation (=> exp(tD) in G2)
    D = build_orbit()
    dres = is_derivation(D)
    e1 = dres < DERIV_TOL and np.linalg.norm(D @ e(0)) < DERIV_TOL
    print(f"E1_DERIVATION residual={dres:.2e} D(1)={float(np.linalg.norm(D @ e(0))):.2e} "
          f"{'PASS' if e1 else 'FAIL'}")

    ts = np.linspace(0.0, 3.0, 25)
    # E2 — continuous orbit: g(t) automorphism; b_argmax wiggles, a & b_cov flat
    auto_worst = 0.0
    b_arg, b_cov, a_curve, x_arg, x_cov = [], [], [], [], []
    for t in ts:
        g = expm(t * D)
        auto_worst = max(auto_worst, automorphism_residual(g))
        ga = g @ ALPHA0
        gem = g @ EM0
        b_arg.append(argmax_b(ga))
        b_cov.append(cov_b(ga, gem))
        a_curve.append(a_coeff(ga))
        x_arg.append(path_d_well(ga, argmax_b(ga)))          # ladder under OLD argmax-b
        x_cov.append(path_d_well(ga, cov_b(ga, gem)))        # ladder under b_cov
    b_arg, b_cov, a_curve = map(np.array, (b_arg, b_cov, a_curve))
    rng_barg = float(b_arg.max() - b_arg.min())
    rng_bcov = float(b_cov.max() - b_cov.min())
    rng_a = float(a_curve.max() - a_curve.min())
    e2 = (auto_worst < 1e-6 and rng_barg > BREAK_TOL and rng_bcov < INV_TOL and rng_a < INV_TOL)
    print(f"E2_CONTINUOUS_ORBIT g(t)_auto_worst={auto_worst:.2e} "
          f"range[b_argmax]={rng_barg:.3f} range[b_cov]={rng_bcov:.2e} range[a]={rng_a:.2e} "
          f"{'PASS' if e2 else 'FAIL'}")

    # E3 — the Phi_fp path-D end-state: BREAKS under argmax-b, INVARIANT under b_cov
    xa = np.array([v for v in x_arg if v is not None])
    xc = np.array([v for v in x_cov if v is not None])
    rng_xarg = float(xa.max() - xa.min()) if len(xa) else 0.0
    rng_xcov = float(xc.max() - xc.min()) if len(xc) else 0.0
    e3 = (rng_xarg > BREAK_TOL and rng_xcov < INV_TOL and len(xc) == len(ts))
    print(f"E3_LADDER_COVARIANT range[x_D | argmax]={rng_xarg:.3f} (breaks)  "
          f"range[x_D | b_cov]={rng_xcov:.2e} (flat)  {'PASS' if e3 else 'FAIL'}")

    # E4 — Betti-0 drop witness is orbit-invariant under b_cov
    betti_ok = True
    for t in ts:
        g = expm(t * D)
        ga, gem = g @ ALPHA0, g @ EM0
        a = a_coeff(ga)
        b0_full = betti_zero(a, cov_b(ga, gem) / 2.0)              # at eps=2 (folded)
        # unfolded reference near eps~0 along the SAME line/off (config), transported
        ga_small = g @ single_line_jet(LINE, OFF, 0.2)
        gem_small = g @ EM0
        b0_open = betti_zero(a_coeff(ga_small), cov_b(ga_small, gem_small) / 2.0)
        if not (b0_open == 2 and b0_full == 1):
            betti_ok = False
            break
    print(f"E4_BETTI_DROP_INVARIANT open=2->folded=1 across orbit "
          f"{'PASS' if betti_ok else 'FAIL'}")

    print("-" * 70)
    print("orbit sample (t, b_argmax, b_cov, a, x_D|argmax, x_D|cov):")
    for idx in (0, 6, 12, 18, 24):
        t = ts[idx]
        xa_v = x_arg[idx]; xc_v = x_cov[idx]
        print(f"  t={t:4.2f}  b_arg={b_arg[idx]:+.4f}  b_cov={b_cov[idx]:+.4f}  a={a_curve[idx]:+.4f}"
              f"  xD_arg={('%.4f'%xa_v) if xa_v is not None else 'None':>8}"
              f"  xD_cov={('%.4f'%xc_v) if xc_v is not None else 'None':>8}")
    print("=" * 70)

    if e1 and e2 and e3 and betti_ok:
        print("FUNCTOR_F_PHI_EQUIV_VERDICT E_GREEN (4/4 clauses PASS)")
        print("FUNCTOR_F_PHI_EQUIV_NOTE b_cov_canonical; ladder_G2_equivariant; "
              "argmax_b_breaks_on_continuous_orbit; a_and_betti_invariant; D3_forbidden")
        return 0
    print("FUNCTOR_F_PHI_EQUIV_VERDICT E_INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
