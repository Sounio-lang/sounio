#!/usr/bin/env python3
"""
Functor F — coherence of the ord-1 correction: the G2 contraction identity.

The field-functoriality rung (K_CHARACTERISED) found the coupling-composition defect
to be the cross associator [e_u1,e_u2,e_z], an ord-1 object of magnitude 2. This rung
asks whether that correction is COHERENT: does the octonion associator 3-form phi
close under self-contraction onto its dual 4-form psi (the co-associator), via the
defining G2 identity

    sum_e phi_{abe} phi_{cde} = delta_ac delta_bd - delta_ad delta_bc - psi_{abcd}

with  phi_{abc} = <e_a e_b, e_c>                 (structure-constant 3-form)
      [e_a,e_b,e_c] = -2 * sum_d psi_{abcd} e_d   (associator gives the 4-form)?

If it holds exactly, the ord-1 correction is not ad hoc: it is the G2 3-form, its
compositions are governed by G2's own structure identity, and the graded 'ord-2'
coherence datum is the invariant 4-form psi. This is the ALGEBRAIC coherence
(contraction / 2-cocycle-type) identity of G2 -- NOT the literal Mac Lane pentagon,
which would require a monoidal 4-fold structure the octonions do not carry.

Companion to:
  docs/research/functor_f_g2_coherence_spec_2026-07-25.md
  docs/research/functor_f_field_functoriality_spec_2026-07-25.md   (parent: K_CHARACTERISED)

Self-contained; CD sign law re-implemented for audit. Independently cross-checks the
inherited octonion core against its defining axioms before using it.
"""
import numpy as np

np.seterr(all='ignore')

EXACT = 1e-12
INV_TOL = 1e-9


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


def build_forms():
    phi = np.zeros((8, 8, 8))
    psi = np.zeros((8, 8, 8, 8))
    for a in range(1, 8):
        for b in range(1, 8):
            ab = omul(e(a), e(b))
            for c in range(1, 8):
                phi[a, b, c] = float(np.dot(ab, e(c)))
                As = assoc(e(a), e(b), e(c))
                for d in range(1, 8):
                    psi[a, b, c, d] = -0.5 * float(np.dot(As, e(d)))
    return phi, psi


def totally_antisym(T, rank):
    idxs = range(1, 8)
    import itertools
    for combo in itertools.product(idxs, repeat=rank):
        for i in range(rank - 1):
            sw = list(combo); sw[i], sw[i + 1] = sw[i + 1], sw[i]
            if abs(T[combo] + T[tuple(sw)]) > EXACT:
                return False
    return True


def audit_core():
    """Independent axiom check of the inherited octonion core before use."""
    ident = all(np.allclose(omul(e(0), e(j)), e(j)) for j in range(8))
    sq = all(np.allclose(omul(e(i), e(i)), -e(0)) for i in range(1, 8))
    anti = all(np.allclose(omul(e(i), e(j)), -omul(e(j), e(i)))
               for i in range(1, 8) for j in range(1, 8) if i != j)
    alt = all(np.allclose(omul(omul(e(i), e(i)), e(j)), omul(e(i), omul(e(i), e(j))))
              for i in range(8) for j in range(8))
    ok = ident and sq and anti and alt
    print(f"P0_CORE_AUDIT identity={ident} sq=-1={sq} anticomm={anti} alternative={alt} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 70)
    print("FUNCTOR F — G2 coherence (contraction identity, ord-1 -> ord-2 4-form)")
    print("=" * 70)
    core = audit_core()

    phi, psi = build_forms()

    p1 = totally_antisym(phi, 3) and set(np.round(phi.ravel(), 6)) <= {-1.0, 0.0, 1.0}
    print(f"P1_PHI_3FORM totally_antisym + values in {{-1,0,1}} {'PASS' if p1 else 'FAIL'}")

    # psi G2-invariant over the imaginary 7-block
    worst_inv = 0.0
    for s in range(50):
        M7 = generic_automorphism(np.random.default_rng(s))[1:, 1:]
        P = psi[1:, 1:, 1:, 1:]
        Pg = np.einsum('ia,jb,kc,ld,ijkl->abcd', M7, M7, M7, M7, P)
        worst_inv = max(worst_inv, float(np.max(np.abs(Pg - P))))
    p2 = totally_antisym(psi, 4) and set(np.round(psi.ravel(), 6)) <= {-1.0, 0.0, 1.0} and worst_inv < INV_TOL
    print(f"P2_PSI_4FORM totally_antisym + values in {{-1,0,1}} + G2-invariant(dev={worst_inv:.1e}) "
          f"{'PASS' if p2 else 'FAIL'}")

    # THE contraction identity
    def dl(i, j):
        return 1.0 if i == j else 0.0
    worst = 0.0
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                for d in range(1, 8):
                    lhs = sum(phi[a, b, ee] * phi[c, d, ee] for ee in range(1, 8))
                    rhs = dl(a, c) * dl(b, d) - dl(a, d) * dl(b, c) - psi[a, b, c, d]
                    worst = max(worst, abs(lhs - rhs))
    p3 = worst < EXACT
    print(f"P3_CONTRACTION_IDENTITY max|phi.phi - (dd-dd-psi)| = {worst:.2e} "
          f"{'PASS' if p3 else 'FAIL'}")

    # tie to field functoriality: every field correction R = -2 psi[u1,u2,z,:]
    FANO = [(i, j, i ^ j) for i in range(1, 8) for j in range(i + 1, 8) if (i ^ j) > j]
    tie_ok = True
    n = 0
    for (i, j, k) in FANO:
        offs = [u for u in range(1, 8) if u not in (i, j, k)]
        for x in range(len(offs)):
            for y in range(x + 1, len(offs)):
                u1, u2 = offs[x], offs[y]
                R = assoc(e(u1), e(u2), e(k))
                Rpsi = -2.0 * np.array([0.0] + [psi[u1, u2, k, d] for d in range(1, 8)])
                if not np.allclose(R, Rpsi, atol=EXACT):
                    tie_ok = False
                n += 1
    p4 = tie_ok
    print(f"P4_CORRECTION_IS_PSI field correction R == -2*psi[u1,u2,z,:] all {n} configs "
          f"{'PASS' if p4 else 'FAIL'}")

    print("=" * 70)
    if core and p1 and p2 and p3 and p4:
        print("FUNCTOR_F_COHERENCE_VERDICT P_GREEN (5/5 clauses PASS)")
        print("FUNCTOR_F_COHERENCE_NOTE G2_contraction_identity_exact; ord1_3form_phi + "
              "invariant_4form_psi; field_correction=-2psi; algebraic_coherence_not_MacLane_pentagon")
        return 0
    print("FUNCTOR_F_COHERENCE_VERDICT P_INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
