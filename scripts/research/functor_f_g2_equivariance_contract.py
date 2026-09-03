#!/usr/bin/env python3
"""
Functor F — G2-EQUIVARIANCE (pointwise naturality), the step the g2-covariance
contract explicitly deferred as `weak_covariance_witness`.

Companion to:
  docs/research/functor_f_g2_equivariance_spec_2026-07-25.md
  docs/research/functor_f_g2_covariance_spec_2026-07-25.md   (parent: G_GREEN)

What this settles (and does not):
  * g2_covariance verified UNIFORMITY across the 7 basis-aligned Fano lines.
    It never applied a nontrivial g in G2 = Aut(O). This contract does.
  * Result is NOT a clean "H_GREEN". It is a CHARACTERISATION:
      - the associator jet is equivariant BY DEFINITION of automorphism (no content);
      - a = A0 + ||alpha||^2/4 is g-invariant because G2 subset SO(7) (no content);
      - Phi_fp's b, extracted by argmax over COORDINATES, is provably NOT g-covariant
        (H3_ARGMAX_B_OBSTRUCTED) — a generic g spreads a single-axis jet over all 7 axes;
      - a config-determined PAIRING b := <alpha, e_m> restores exact equivariance
        (H4_PAIRING_B_COVARIANT).
  * Not a construction of the full group G2. Not D3. Not clinical.

Self-contained; re-implements the Cayley-Dickson sign law for auditability.
"""
import numpy as np

np.seterr(all='ignore')

N_RANDOM_G = 200          # how many independent generic automorphisms to sample
AUTO_TOL = 1e-9           # a hand-built g must be an automorphism to this tol
INV_TOL = 1e-9            # invariance acceptance
OBSTRUCTION_TOL = 1e-3    # |Δb| above this = the argmax-b genuinely moved


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


# ---------------------------------------------------------------- G2 element
def _imag_unit(rng):
    v = rng.standard_normal(8)
    v[0] = 0.0
    return v / np.linalg.norm(v)


def _proj_out(v, basis):
    w = v.copy()
    for b in basis:
        w = w - np.dot(w, b) * b
    return w


def generic_automorphism(rng):
    """A generic g in G2 as an 8x8 matrix, fixing e0.

    Built from a generic Cayley triple (I,J,L): I perp J, L perp <1,I,J,IJ>.
    The 7 images are then forced by the octonion product, matching the sign law
    e_{i^j} = cds(i,j) e_i e_j on the generators 1,2,4 (with 3=1^2,5=1^4,6=2^4,7=1^2^4).
    """
    I = _imag_unit(rng)
    J = _imag_unit(rng)
    J = _proj_out(J, [I]); J /= np.linalg.norm(J)
    IJ = omul(I, J)
    L = _imag_unit(rng)
    L = _proj_out(L, [I, J, IJ]); L /= np.linalg.norm(L)
    M = np.zeros((8, 8))
    M[:, 0] = e(0)
    M[:, 1] = I
    M[:, 2] = J
    M[:, 4] = L
    M[:, 3] = cds(1, 2) * omul(I, J)
    M[:, 5] = cds(1, 4) * omul(I, L)
    M[:, 6] = cds(2, 4) * omul(J, L)
    M[:, 7] = cds(3, 4) * omul(M[:, 3], L)
    return M


def automorphism_residual(M):
    worst = 0.0
    for i in range(8):
        for j in range(8):
            worst = max(worst, float(np.linalg.norm(M @ omul(e(i), e(j)) - omul(M[:, i], M[:, j]))))
    return worst


def is_signed_permutation(M):
    return np.allclose(np.sort(np.abs(M).ravel()), np.sort(np.eye(8).ravel()))


# ---------------------------------------------------------------- F assignment
def single_line_jet(line, off_unit, eps):
    i, j, k = line
    base = e(i) + eps * e(off_unit)
    return omul(omul(base, e(j)), e(k)) - omul(base, omul(e(j), e(k)))


def argmax_b(alpha):
    a_im = alpha.copy(); a_im[0] = 0.0
    idx = int(np.argmax(np.abs(a_im)))
    return float(a_im[idx]), idx


def a_coeff(alpha, A0=-1.0):
    return A0 + float(np.dot(alpha, alpha)) / 4.0


# The worked configuration (line, off-line unit) whose jet is a single axis e_m.
LINE, OFF = (1, 2, 3), 4


def clause_H1_generic_g_exists(rng_seeds):
    worst_auto, worst_orth, n_generic = 0.0, 0.0, 0
    for s in rng_seeds:
        M = generic_automorphism(np.random.default_rng(s))
        worst_auto = max(worst_auto, automorphism_residual(M))
        worst_orth = max(worst_orth, float(np.linalg.norm(M.T @ M - np.eye(8))))
        n_generic += 0 if is_signed_permutation(M) else 1
    ok = worst_auto < AUTO_TOL and worst_orth < INV_TOL and n_generic == len(rng_seeds)
    print(f"H1_GENERIC_G_EXISTS worst_auto={worst_auto:.2e} worst_orth={worst_orth:.2e} "
          f"generic={n_generic}/{len(rng_seeds)} {'PASS' if ok else 'FAIL'}")
    return ok


def clause_H2_a_invariant(rng_seeds):
    alpha = single_line_jet(LINE, OFF, 1.0)
    a0 = a_coeff(alpha)
    worst = 0.0
    for s in rng_seeds:
        M = generic_automorphism(np.random.default_rng(s))
        worst = max(worst, abs(a_coeff(M @ alpha) - a0))
    ok = worst < INV_TOL
    print(f"H2_A_INVARIANT max|da|={worst:.2e} (a={a0:+.3f}) {'PASS' if ok else 'FAIL'}")
    return ok


def clause_H3_argmax_b_obstructed(rng_seeds):
    """The obstruction: argmax-b is NOT g-covariant. PASS = obstruction confirmed."""
    alpha = single_line_jet(LINE, OFF, 1.0)
    b0, _ = argmax_b(alpha)
    deltas, supports = [], []
    for s in rng_seeds:
        M = generic_automorphism(np.random.default_rng(s))
        g_alpha = M @ alpha
        b1, _ = argmax_b(g_alpha)
        deltas.append(abs(abs(b1) - abs(b0)))
        supports.append(int(np.sum(np.abs(g_alpha[1:]) > 1e-9)))
    deltas = np.array(deltas)
    frac_moved = float(np.mean(deltas > OBSTRUCTION_TOL))
    ok = frac_moved > 0.99                      # essentially every generic g breaks b
    print(f"H3_ARGMAX_B_OBSTRUCTED |b0|={abs(b0):.3f} mean_support(g.alpha)={np.mean(supports):.2f} "
          f"median|db|={np.median(deltas):.3f} frac_moved={frac_moved:.3f} "
          f"{'PASS(obstruction confirmed)' if ok else 'FAIL'}")
    return ok


def clause_H4_pairing_b_covariant(rng_seeds):
    """The fix: b := <alpha, e_m> with a config-determined axis e_m that g transports."""
    alpha = single_line_jet(LINE, OFF, 1.0)
    _, m = argmax_b(alpha)          # e_m fixed by the (line, off) configuration, not by g
    em = e(m)
    b0 = float(np.dot(alpha, em))
    worst = 0.0
    for s in rng_seeds:
        M = generic_automorphism(np.random.default_rng(s))
        b1 = float(np.dot(M @ alpha, M @ em))   # <g.alpha, g.e_m>
        worst = max(worst, abs(b1 - b0))
    ok = worst < INV_TOL
    print(f"H4_PAIRING_B_COVARIANT max|db|={worst:.2e} (b={b0:+.3f}) {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    seeds = list(range(N_RANDOM_G))
    print("=" * 70)
    print("FUNCTOR F — G2-EQUIVARIANCE (pointwise naturality)")
    print("=" * 70)
    h1 = clause_H1_generic_g_exists(seeds)
    h2 = clause_H2_a_invariant(seeds)
    h3 = clause_H3_argmax_b_obstructed(seeds)
    h4 = clause_H4_pairing_b_covariant(seeds)
    print("=" * 70)
    if h1 and h2 and h3 and h4:
        print("FUNCTOR_F_G2_EQUIV_VERDICT H_CHARACTERISED (4/4 clauses PASS)")
        print("FUNCTOR_F_G2_EQUIV_NOTE argmax_b_NOT_covariant(obstruction); "
              "pairing_b_covariant(fix); a_invariant; jet_equivariance_definitional; D3_forbidden")
        return 0
    print("FUNCTOR_F_G2_EQUIV_VERDICT H_INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
