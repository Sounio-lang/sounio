#!/usr/bin/env python3
"""
R2 continuous tube law as theorem — executable verification of the proof sketch.

Companion to:
  docs/research/r2_continuous_law_theorem_spec_2026-07-25.md
  docs/research/r2_continuous_law_theorem_falsifiers_2026-07-25.md

Self-contained; re-implements the Cayley-Dickson sign law for auditability.
"""

import numpy as np

np.seterr(all='ignore')


def cds(a, b, bits=4):
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


def Lmatrix(x):
    n = 16
    L = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            L[k, j] = x[k ^ j] * cds(k ^ j, j)
    return L


def det_L(x):
    return float(np.linalg.det(Lmatrix(x)))


def D1(x):
    return float(np.dot(x, x))


def D2(x):
    x0 = x[0]
    x8 = x[8]
    u = x[1:8]
    w = x[9:16]
    A = float(np.dot(u, u))
    B = float(np.dot(w, w))
    gamma = float(np.dot(u, w))
    C = x0 * x0 + x8 * x8
    return (D1(x) ** 2 - 4.0 * (A * B - gamma * gamma))


def D2_sos(x):
    """D2 as sum of squares: C^2 + 2C(A+B) + (A-B)^2 + 4 gamma^2."""
    x0 = x[0]
    x8 = x[8]
    u = x[1:8]
    w = x[9:16]
    A = float(np.dot(u, u))
    B = float(np.dot(w, w))
    gamma = float(np.dot(u, w))
    C = x0 * x0 + x8 * x8
    return C * C + 2.0 * C * (A + B) + (A - B) ** 2 + 4.0 * gamma * gamma


def zd_conditions(x):
    x0 = x[0]
    x8 = x[8]
    u = x[1:8]
    w = x[9:16]
    A = float(np.dot(u, u))
    B = float(np.dot(w, w))
    gamma = float(np.dot(u, w))
    return x0, x8, A - B, gamma


def canonical_zd_pairs():
    """Return the 84 canonical 2-unit zero-divisor pairs."""
    n = 16
    pairs = []
    for i in range(1, n):
        for j in range(i + 1, n):
            for sgn in (1, -1):
                a = np.zeros(n)
                a[i] = 1.0
                a[j] = sgn
                L = Lmatrix(a)
                sv = np.linalg.svd(L, compute_uv=False)
                if sv.min() < 1e-9:
                    pairs.append((i, sgn, j))
    return pairs


def unit(i, n=16):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def check_T1_factorization():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        x = rng.standard_normal(16)
        x /= np.linalg.norm(x)
        det = det_L(x)
        d1 = D1(x)
        d2 = D2(x)
        rhs = (d1 ** 4) * (d2 ** 2)
        if abs(det) > 1e-300:
            rel = abs(det - rhs) / abs(det)
            max_err = max(max_err, rel)
    ok = max_err < 1e-12
    print(f"T1_FACTORIZATION max_rel_err={max_err:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def check_T2_d2_sum_of_squares():
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(200):
        x = rng.standard_normal(16)
        d2 = D2(x)
        sos = D2_sos(x)
        max_err = max(max_err, abs(d2 - sos))
    ok = max_err < 1e-12
    print(f"T2_D2_SUM_OF_SQUARES max_err={max_err:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def check_T3_zd_conditions():
    pairs = canonical_zd_pairs()
    ok = True
    for i, sgn, j in pairs[:20]:  # sample
        x = unit(i) + sgn * unit(j)
        x0, x8, ab, gamma = zd_conditions(x)
        if abs(x0) > 1e-9 or abs(x8) > 1e-9 or abs(ab) > 1e-9 or abs(gamma) > 1e-9:
            print(f"T3_FAIL on ({i},{sgn},{j}): x0={x0} x8={x8} A-B={ab} gamma={gamma}")
            ok = False
    print(f"T3_ZD_CONDITIONS {'PASS' if ok else 'FAIL'}")
    return ok


def check_T4_gradient_independence():
    # Jacobian of (x0, x8, A-B, gamma) with respect to x (4 x 16)
    pairs = canonical_zd_pairs()
    min_sv = float('inf')
    for i, sgn, j in pairs[:10]:
        x = unit(i) + sgn * unit(j)
        J = np.zeros((4, 16))
        # f1 = x0
        J[0, 0] = 1.0
        # f2 = x8
        J[1, 8] = 1.0
        # f3 = A - B = |u|^2 - |w|^2
        for k in range(1, 8):
            J[2, k] = 2.0 * x[k]
        for k in range(9, 16):
            J[2, k] = -2.0 * x[k]
        # f4 = gamma = <u,w>
        for k in range(1, 8):
            J[3, k] = x[k + 8]
        for k in range(9, 16):
            J[3, k] = x[k - 8]
        sv = np.linalg.svd(J, compute_uv=False)
        min_sv = min(min_sv, float(sv.min()))
    ok = min_sv > 1e-9
    print(f"T4_GRADIENT_INDEPENDENCE min_sv={min_sv:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def check_T5_quadratic_contact():
    # For canonical ZD points, perturb and check D2 / eps^2 bounded
    rng = np.random.default_rng(2)
    pairs = canonical_zd_pairs()
    ratios = []
    for i, sgn, j in pairs[:10]:
        x0 = unit(i) + sgn * unit(j)
        for _ in range(10):
            v = rng.standard_normal(16)
            v /= np.linalg.norm(v)
            for eps in (1e-3, 1e-2, 1e-1):
                x = x0 + eps * v
                d2 = D2(x)
                ratio = abs(d2) / (eps * eps)
                ratios.append(ratio)
    ratios = np.array(ratios)
    ok = np.all((ratios > 0.1) & (ratios < 10.0))
    print(f"T5_QUADRATIC_CONTACT min={ratios.min():.3f} max={ratios.max():.3f} {'PASS' if ok else 'FAIL'}")
    return ok


def check_T6_det_scaling():
    rng = np.random.default_rng(3)
    pairs = canonical_zd_pairs()
    log_eps = []
    log_det = []
    for i, sgn, j in pairs[:10]:
        x0 = unit(i) + sgn * unit(j)
        x0 = x0 / np.linalg.norm(x0)
        for _ in range(10):
            v = rng.standard_normal(16)
            v /= np.linalg.norm(v)
            for eps in (1e-3, 1e-2, 1e-1):
                x = x0 + eps * v
                x = x / np.linalg.norm(x)
                det = abs(det_L(x))
                if det > 1e-300:
                    log_eps.append(np.log(eps))
                    log_det.append(np.log(det))
    log_eps = np.array(log_eps)
    log_det = np.array(log_det)
    # linear fit: log(det) = m * log(eps) + c
    A = np.vstack([log_eps, np.ones(len(log_eps))]).T
    m, c = np.linalg.lstsq(A, log_det, rcond=None)[0]
    ok = (m > 3.5) and (m < 4.5)
    print(f"T6_DET_SCALING slope={m:.3f} (expect ~4) {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("R2 CONTINUOUS LAW THEOREM — executable verification")
    print("=" * 70)
    results.append(("T1", check_T1_factorization()))
    results.append(("T2", check_T2_d2_sum_of_squares()))
    results.append(("T3", check_T3_zd_conditions()))
    results.append(("T4", check_T4_gradient_independence()))
    results.append(("T5", check_T5_quadratic_contact()))
    results.append(("T6", check_T6_det_scaling()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"R2_THEOREM_VERDICT T_GREEN ({passed}/{total} clauses PASS)")
        print("R2_THEOREM_NOTE factorization_verified; codim4_complete_intersection; quadratic_contact; det_order4")
        return 0
    else:
        print(f"R2_THEOREM_VERDICT T_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
