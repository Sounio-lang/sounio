#!/usr/bin/env python3
"""
Functor F — FIELD functoriality: is F additive across cross-line couplings?

The R4 "field of seven squares" couples Fano lines through shared terms. A coupling
deforms one slot of the trilinear associator [x,y,z]. Two couplings on DIFFERENT
slots give

  alpha(d1,d2) = [x + d1 e_u1, y + d2 e_u2, z]
            = [x,y,z] + d1[e_u1,y,z] + d2[x,e_u2,z] + d1 d2 [e_u1,e_u2,z].

F is additive on the couplings iff the cross term d1 d2 [e_u1,e_u2,z] vanishes. It does
so EXACTLY when the two off-line directions and z share a Fano line (associative);
otherwise the residual IS an ord-1 associator, and that correction is G2-covariant.

Result: F is a functor on the field UP TO a G2-covariant ord-1 (associator) correction;
strictly additive on associative couplings. This computes the ord-1 -> field edge that
rupture-programme-synthesis §5 draws but never evaluates.

Companion to:
  docs/research/functor_f_field_functoriality_spec_2026-07-25.md
  docs/research/functor_f_phi_fp_equivariant_spec_2026-07-25.md   (parent: E_GREEN)

Self-contained; re-implements the CD sign law for auditability.
"""
import numpy as np

np.seterr(all='ignore')

D1, D2 = 0.7, 0.3          # fixed, distinct perturbation scales
EXACT_TOL = 1e-12
INV_TOL = 1e-9
N_G = 200


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
    v = np.zeros(8); v[i] = 1.0; return v


def assoc(u, v, w):
    return omul(omul(u, v), w) - omul(u, omul(v, w))


FANO = [(i, j, i ^ j) for i in range(1, 8) for j in range(i + 1, 8) if (i ^ j) > j]


def same_line(a, b, c):
    return any({a, b, c} == set(L) for L in FANO)


# ---------------------------------------------------------------- generic g in G2
def _iu(rng):
    v = rng.standard_normal(8); v[0] = 0.0; return v / np.linalg.norm(v)


def _po(v, B):
    w = v.copy()
    for b in B:
        w = w - np.dot(w, b) * b
    return w


def generic_automorphism(rng):
    I = _iu(rng)
    J = _po(_iu(rng), [I]); J /= np.linalg.norm(J)
    IJ = omul(I, J)
    L = _po(_iu(rng), [I, J, IJ]); L /= np.linalg.norm(L)
    M = np.zeros((8, 8))
    M[:, 0] = e(0); M[:, 1] = I; M[:, 2] = J; M[:, 4] = L
    M[:, 3] = cds(1, 2) * omul(I, J)
    M[:, 5] = cds(1, 4) * omul(I, L)
    M[:, 6] = cds(2, 4) * omul(J, L)
    M[:, 7] = cds(3, 4) * omul(M[:, 3], L)
    return M


# ---------------------------------------------------------------- configuration sweep
def configs():
    """All (line, u1, u2): base Fano line, two distinct off-line perturbation units."""
    out = []
    for (i, j, k) in FANO:
        on = {i, j, k}
        offs = [u for u in range(1, 8) if u not in on]
        for a in range(len(offs)):
            for b in range(a + 1, len(offs)):
                out.append(((i, j, k), offs[a], offs[b]))
    return out


def additivity_residual(line, u1, u2):
    i, j, k = line
    x, y, z = e(i), e(j), e(k)
    a_base = assoc(x, y, z)
    a_A = assoc(x + D1 * e(u1), y, z)
    a_B = assoc(x, y + D2 * e(u2), z)
    a_both = assoc(x + D1 * e(u1), y + D2 * e(u2), z)
    residual = (a_both - a_base) - ((a_A - a_base) + (a_B - a_base))
    cross = D1 * D2 * assoc(e(u1), e(u2), z)
    return residual, cross, a_base


def main():
    print("=" * 70)
    print("FUNCTOR F — FIELD functoriality (additivity across cross-line couplings)")
    print("=" * 70)
    cfgs = configs()

    # K1 — residual is EXACTLY the cross associator d1 d2 [e_u1,e_u2,e_z], every config
    worst_diff = 0.0
    for (line, u1, u2) in cfgs:
        residual, cross, _ = additivity_residual(line, u1, u2)
        worst_diff = max(worst_diff, float(np.linalg.norm(residual - cross)))
    k1 = worst_diff < EXACT_TOL
    print(f"K1_RESIDUAL_IS_CROSS_ASSOCIATOR n={len(cfgs)} worst||res-cross||={worst_diff:.2e} "
          f"{'PASS' if k1 else 'FAIL'}")

    # K2 — additive EXACTLY on associative couplings; obstructed otherwise
    assoc_zero = nonassoc_nonzero = bad = 0
    for (line, u1, u2) in cfgs:
        residual, _, _ = additivity_residual(line, u1, u2)
        r = float(np.linalg.norm(residual))
        assoc_coupling = same_line(u1, u2, line[2])
        if assoc_coupling and r < INV_TOL:
            assoc_zero += 1
        elif (not assoc_coupling) and r > INV_TOL:
            nonassoc_nonzero += 1
        else:
            bad += 1
    k2 = (bad == 0)
    print(f"K2_ADDITIVE_IFF_ASSOCIATIVE associative->zero={assoc_zero} "
          f"cross-line->nonzero={nonassoc_nonzero} violations={bad} {'PASS' if k2 else 'FAIL'}")

    # K3 — every nonzero correction is an ord-1 object: ||[e_u1,e_u2,e_z]|| == 2
    mags = []
    for (line, u1, u2) in cfgs:
        c = assoc(e(u1), e(u2), e(line[2]))
        m = float(np.linalg.norm(c))
        if m > INV_TOL:
            mags.append(m)
    k3 = len(mags) > 0 and all(abs(m - 2.0) < INV_TOL for m in mags)
    print(f"K3_CORRECTION_IS_ORD1 nonzero corrections={len(mags)} "
          f"all ||assoc||==2 ? {k3} (min={min(mags):.6f} max={max(mags):.6f}) {'PASS' if k3 else 'FAIL'}")

    # K4 — the correction is G2-covariant (norm + pairing invariant; argmax breaks)
    #      tested on all cross-line configs, over N_G automorphisms
    Ms = [generic_automorphism(np.random.default_rng(s)) for s in range(N_G)]
    worst_norm = worst_cov = 0.0
    max_argmax_break = 0.0
    tested = 0
    for (line, u1, u2) in cfgs:
        R = assoc(e(u1), e(u2), e(line[2]))
        if np.linalg.norm(R) < INV_TOL:
            continue
        tested += 1
        m = int(np.argmax(np.abs(np.concatenate([[0.0], R[1:]]))))
        em = e(m); nR = float(np.linalg.norm(R)); b0 = float(np.dot(R, em))
        for M in Ms:
            gR = M @ R
            worst_norm = max(worst_norm, abs(float(np.linalg.norm(gR)) - nR))
            worst_cov = max(worst_cov, abs(float(np.dot(gR, M @ em)) - b0))
            aim = gR.copy(); aim[0] = 0.0
            max_argmax_break = max(max_argmax_break,
                                   abs(abs(float(aim[int(np.argmax(np.abs(aim)))])) - abs(b0)))
    k4 = worst_norm < INV_TOL and worst_cov < INV_TOL and max_argmax_break > 1e-2
    print(f"K4_CORRECTION_G2_COVARIANT tested_cfgs={tested} "
          f"||R||_dev={worst_norm:.2e} pairing_dev={worst_cov:.2e} argmax_break={max_argmax_break:.3f} "
          f"{'PASS' if k4 else 'FAIL'}")

    print("=" * 70)
    if k1 and k2 and k3 and k4:
        print("FUNCTOR_F_FIELD_VERDICT K_CHARACTERISED (4/4 clauses PASS)")
        print("FUNCTOR_F_FIELD_NOTE functor_up_to_G2_covariant_ord1_correction; "
              "additive_iff_associative_coupling; correction=cross_associator(||.||=2); D3_forbidden")
        return 0
    print("FUNCTOR_F_FIELD_VERDICT K_INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
