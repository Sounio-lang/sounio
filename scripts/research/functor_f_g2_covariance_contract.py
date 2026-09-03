#!/usr/bin/env python3
"""
Functor F — G2-covariance across the seven Fano lines.

Companion to:
  docs/research/functor_f_g2_covariance_spec_2026-07-25.md
  docs/research/functor_f_g2_covariance_falsifiers_2026-07-25.md

Self-contained; re-implements the Cayley-Dickson sign law for auditability.
"""

import numpy as np

np.seterr(all='ignore')


def cds(a, b, bits=3):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah = a >= h
        bh = b >= h
        al = a & (h - 1)
        bl = b & (h - 1)
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


def fano_lines():
    return [(i, j, i ^ j) for i in range(1, 8) for j in range(i + 1, 8) if (i ^ j) > j]


def cusp_value(x, a, b):
    return 0.25 * x ** 4 + 0.5 * a * x * x + b * x


def cusp_wells(a, b):
    roots = np.roots([1.0, 0.0, a, b])
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    return sorted([x for x in real_roots if 3.0 * x * x + a > 1e-9])


def phi_fp(alpha, tau=0.0, A0=-1.0, ref_axis=None):
    # Canonical polar coordinate: b is the NATURAL PAIRING <alpha, e_m> against a
    # configuration-determined imaginary axis e_m, NOT a basis argmax. On the seven
    # basis-aligned Fano lines the jet is single-axis, so e_m = e_argmax and the two
    # coincide exactly -- the only regime this contract exercises (hence its honest
    # `weak_covariance_witness` note). The pairing form is the G2-equivariant one:
    # under a generic / continuous automorphism the argmax readout is provably NOT
    # covariant while the pairing is. See
    #   scripts/research/functor_f_g2_equivariance_contract.py      (H_CHARACTERISED)
    #   scripts/research/functor_f_phi_fp_equivariant_contract.py   (E_GREEN)
    # Pass ref_axis=e_m to evaluate the equivariant b off the basis-aligned lines.
    norm = float(np.linalg.norm(alpha))
    alpha_im = alpha.copy()
    alpha_im[0] = 0.0
    idx = int(np.argmax(np.abs(alpha_im))) if ref_axis is None else int(ref_axis)
    coeff = float(alpha_im[idx])
    a = A0 + norm * norm / 4.0
    b = tau + coeff / 2.0
    return a, b, idx, coeff


def single_line_jet(line, off_unit, eps):
    i, j, k = line
    base = e(i) + eps * e(off_unit)
    return omul(omul(base, e(j)), e(k)) - omul(base, omul(e(j), e(k)))


def cross_line_jet(L1, L2, a1, a2, shared, z, delta):
    x = e(a1) + delta * e(a2)
    return omul(omul(x, e(shared)), e(z)) - omul(x, omul(e(shared), e(z)))


def deepest_well_sign(a, b):
    mins = cusp_wells(a, b)
    if not mins:
        return None
    vals = [cusp_value(x, a, b) for x in mins]
    return mins[int(np.argmin(vals))]


def all_critical_points(a, b):
    roots = np.roots([1.0, 0.0, a, b])
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    mins = sorted([x for x in real_roots if 3.0 * x * x + a > 1e-9])
    saddles = sorted([x for x in real_roots if 3.0 * x * x + a < -1e-9])
    return mins, saddles


def betti_zero(a, b):
    mins, saddles = all_critical_points(a, b)
    if not mins:
        return 0
    if len(mins) == 1:
        c = cusp_value(mins[0], a, b) + 1.0
    else:
        vals_min = [cusp_value(x, a, b) for x in mins]
        higher_min_val = max(vals_min)
        if not saddles:
            c = higher_min_val - 0.5
        else:
            saddle_val = cusp_value(saddles[0], a, b)
            c = 0.5 * (saddle_val + higher_min_val)
    R = max(3.0, 2.0 * (abs(b) + 1.0) / max(abs(a), 0.1) ** 0.5 if a != 0 else 3.0)
    xs = np.linspace(-R, R, 20001)
    below = cusp_value(xs, a, b) <= c
    comps = 0
    prev = False
    for flag in below:
        if flag and not prev:
            comps += 1
        prev = flag
    return comps


def check_G1_uniform_jet():
    ok = True
    lines = fano_lines()
    for line in lines:
        on_line = set(line)
        for u in range(1, 8):
            if u in on_line:
                continue
            alpha = single_line_jet(line, u, 1.0)
            norm = float(np.linalg.norm(alpha))
            alpha_im = alpha.copy()
            alpha_im[0] = 0.0
            support = np.sum(np.abs(alpha_im) > 1e-9)
            if abs(norm - 2.0) > 1e-9 or support != 1:
                print(f"G1_FAIL line={line} u={u} norm={norm} support={support}")
                ok = False
    print(f"G1_UNIFORM_JET {'PASS' if ok else 'FAIL'}")
    return ok


def check_G2_uniform_phi():
    ok = True
    lines = fano_lines()
    for line in lines:
        on_line = set(line)
        for u in range(1, 8):
            if u in on_line:
                continue
            alpha = single_line_jet(line, u, 1.0)
            a, b, _, _ = phi_fp(alpha, tau=0.0)
            if abs(a - 0.0) > 1e-9 or abs(abs(b) - 1.0) > 1e-9:
                print(f"G2_FAIL line={line} u={u} a={a} b={b}")
                ok = False
    print(f"G2_UNIFORM_PHI {'PASS' if ok else 'FAIL'}")
    return ok


def path_endstates(line, off_unit, eps_max=2.0):
    # Unit coefficient from eps=1; the jet is linear, so alpha(eps) = unit_coeff * eps.
    alpha1 = single_line_jet(line, off_unit, 1.0)
    _, _, _, unit_coeff = phi_fp(alpha1, tau=0.0)
    alpha = single_line_jet(line, off_unit, eps_max)
    # Path C: tau cancels the odd jet, so b == 0 for all eps.
    tau_c = -unit_coeff * eps_max / 2.0
    a_c, b_c, _, _ = phi_fp(alpha, tau=tau_c)
    x_c = deepest_well_sign(a_c, b_c)
    # Path D
    a_d, b_d, _, _ = phi_fp(alpha, tau=0.0)
    x_d = deepest_well_sign(a_d, b_d)
    # sign flip for D
    alpha_m = single_line_jet(line, off_unit, -eps_max)
    a_dm, b_dm, _, _ = phi_fp(alpha_m, tau=0.0)
    x_dm = deepest_well_sign(a_dm, b_dm)
    c_neutral = (x_c is not None and abs(x_c) < 0.2)
    d_polar = (x_d is not None and abs(x_d) > 0.5)
    d_flip = (x_d is not None and x_dm is not None and x_d * x_dm < -1e-6)
    return c_neutral, d_polar, d_flip


def check_G3_path_c_uniform():
    ok = True
    for line in fano_lines():
        on_line = set(line)
        for u in range(1, 8):
            if u in on_line:
                continue
            c, _, _ = path_endstates(line, u)
            if not c:
                print(f"G3_FAIL line={line} u={u}")
                ok = False
    print(f"G3_PATH_C_UNIFORM {'PASS' if ok else 'FAIL'}")
    return ok


def check_G4_path_d_uniform():
    ok = True
    for line in fano_lines():
        on_line = set(line)
        for u in range(1, 8):
            if u in on_line:
                continue
            _, p, f = path_endstates(line, u)
            if not (p and f):
                print(f"G4_FAIL line={line} u={u} polar={p} flip={f}")
                ok = False
    print(f"G4_PATH_D_UNIFORM {'PASS' if ok else 'FAIL'}")
    return ok


def check_G5_betti_uniform():
    ok = True
    for line in fano_lines():
        on_line = set(line)
        for u in range(1, 8):
            if u in on_line:
                continue
            b0_before = None
            b0_after = None
            for eps in np.linspace(0.0, 2.0, 41):
                alpha = single_line_jet(line, u, eps)
                a, b, _, _ = phi_fp(alpha, tau=0.0)
                n = len(cusp_wells(a, b))
                b0 = betti_zero(a, b)
                if n == 2 and b0_before is None:
                    b0_before = b0
                if n == 1 and b0_after is None:
                    b0_after = b0
            if b0_before != 2 or b0_after != 1:
                print(f"G5_FAIL line={line} u={u} before={b0_before} after={b0_after}")
                ok = False
    print(f"G5_BETTI_UNIFORM {'PASS' if ok else 'FAIL'}")
    return ok


def check_G6_cross_line_consistent():
    L1 = (1, 2, 3)
    L2 = (1, 4, 5)
    shared = 1
    a1 = 2
    a2 = 4
    z = 3
    eps_max = 2.0
    alpha_cross = cross_line_jet(L1, L2, a1, a2, shared, z, eps_max)
    _, _, _, coeff = phi_fp(alpha_cross, tau=0.0)
    a_c, b_c, _, _ = phi_fp(alpha_cross, tau=-coeff / 2.0)
    x_c = deepest_well_sign(a_c, b_c)
    a_d, b_d, _, _ = phi_fp(alpha_cross, tau=0.0)
    x_d = deepest_well_sign(a_d, b_d)
    c_neutral = (x_c is not None and abs(x_c) < 0.2)
    d_polar = (x_d is not None and abs(x_d) > 0.5)
    print(f"G6_CROSS_PATH_C deepest={x_c:.4f} neutral={c_neutral}")
    print(f"G6_CROSS_PATH_D deepest={x_d:.4f} polar={d_polar}")
    print(f"G6_CROSS_LINE_CONSISTENT {'PASS' if (c_neutral and d_polar) else 'FAIL'}")
    return c_neutral and d_polar


def main():
    results = []
    print("=" * 70)
    print("FUNCTOR F — G2-covariance across the seven Fano lines")
    print("=" * 70)
    results.append(("G1", check_G1_uniform_jet()))
    results.append(("G2", check_G2_uniform_phi()))
    results.append(("G3", check_G3_path_c_uniform()))
    results.append(("G4", check_G4_path_d_uniform()))
    results.append(("G5", check_G5_betti_uniform()))
    results.append(("G6", check_G6_cross_line_consistent()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"FUNCTOR_F_G2_VERDICT G_GREEN ({passed}/{total} clauses PASS)")
        print("FUNCTOR_F_G2_NOTE uniform_across_7_fano_lines; weak_covariance_witness; D3_forbidden")
        return 0
    else:
        print(f"FUNCTOR_F_G2_VERDICT G_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
