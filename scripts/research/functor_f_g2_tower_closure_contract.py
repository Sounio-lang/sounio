#!/usr/bin/env python3
"""
Functor F — G2 form-tower closure: the rupture invariants terminate at ord-2.

The coherence rung (P_GREEN) proved phi.phi = dd - dd - psi. This rung contracts the
4-form psi with itself and with phi, and asks whether the graded rupture forms
{delta, phi (ord-1), psi (ord-2)} are CLOSED under contraction -- i.e. whether the
"order tower" generates any new invariant at ord-3, or terminates.

Measured closed forms (exact integer structure constants), indices 1..7 on Im(O):
  (A) psi_aefg psi_befg = 24 delta_ab
  (B) psi_abef psi_cdef = 4(d_ac d_bd - d_ad d_bc) - 2 psi_abcd
  (C) phi_aef  psi_bcef = -4 phi_abc
Together with phi_abe phi_cde = d_ac d_bd - d_ad d_bc - psi_abcd (P_GREEN), every
pairwise contraction of {phi,psi} returns {delta,phi,psi}: the tower closes, no ord-3
object appears. These are the standard G2 contraction identities; the content here is
that the programme's ord-1/ord-2 rupture forms ARE exactly this closed G2 algebra.

Companion to:
  docs/research/functor_f_g2_tower_closure_spec_2026-07-25.md
  docs/research/functor_f_g2_coherence_spec_2026-07-25.md   (parent: P_GREEN)

Self-contained; embeds an independent axiom-audit of the inherited octonion core.
"""
import numpy as np
import itertools

np.seterr(all='ignore')
EXACT = 1e-11


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


def audit_core():
    ident = all(np.allclose(omul(e(0), e(j)), e(j)) for j in range(8))
    sq = all(np.allclose(omul(e(i), e(i)), -e(0)) for i in range(1, 8))
    anti = all(np.allclose(omul(e(i), e(j)), -omul(e(j), e(i)))
               for i in range(1, 8) for j in range(1, 8) if i != j)
    alt = all(np.allclose(omul(omul(e(i), e(i)), e(j)), omul(e(i), omul(e(i), e(j))))
              for i in range(8) for j in range(8))
    ok = ident and sq and anti and alt
    print(f"Q0_CORE_AUDIT identity={ident} sq=-1={sq} anticomm={anti} alternative={alt} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def build_forms():
    phi = np.zeros((7, 7, 7))
    psi = np.zeros((7, 7, 7, 7))
    for a in range(7):
        for b in range(7):
            ab = omul(e(a + 1), e(b + 1))
            for c in range(7):
                phi[a, b, c] = float(np.dot(ab, e(c + 1)))
                As = assoc(e(a + 1), e(b + 1), e(c + 1))
                for d in range(7):
                    psi[a, b, c, d] = -0.5 * float(np.dot(As, e(d + 1)))
    return phi, psi


def main():
    print("=" * 70)
    print("FUNCTOR F — G2 form-tower closure (rupture invariants terminate at ord-2)")
    print("=" * 70)
    core = audit_core()
    phi, psi = build_forms()
    I7 = np.eye(7)
    DD = np.einsum('ac,bd->abcd', I7, I7) - np.einsum('ad,bc->abcd', I7, I7)

    # Q1: psi_aefg psi_befg = 24 delta
    A = np.einsum('aefg,befg->ab', psi, psi)
    q1 = np.allclose(A, 24.0 * I7, atol=EXACT)
    print(f"Q1_PSI_NORM psi_aefg psi_befg = 24*delta  max_off={np.max(np.abs(A-24*I7)):.1e} "
          f"{'PASS' if q1 else 'FAIL'}")

    # Q2: psi_abef psi_cdef = 4*DD - 2*psi
    B = np.einsum('abef,cdef->abcd', psi, psi)
    q2 = np.allclose(B, 4.0 * DD - 2.0 * psi, atol=EXACT)
    print(f"Q2_PSI_SELF psi_abef psi_cdef = 4*(dd) - 2*psi  "
          f"max_off={np.max(np.abs(B-(4*DD-2*psi))):.1e} {'PASS' if q2 else 'FAIL'}")

    # Q3: phi_aef psi_bcef = -4 phi
    C = np.einsum('aef,bcef->abc', phi, psi)
    q3 = np.allclose(C, -4.0 * phi, atol=EXACT)
    print(f"Q3_PHI_PSI_MIXED phi_aef psi_bcef = -4*phi  max_off={np.max(np.abs(C+4*phi)):.1e} "
          f"{'PASS' if q3 else 'FAIL'}")

    # Q4: phi.phi (P_GREEN base) + all above => tower closes on {delta,phi,psi}, no ord-3
    PP = np.einsum('abe,cde->abcd', phi, phi)
    base = np.allclose(PP, DD - psi, atol=EXACT)
    q4 = base and q1 and q2 and q3
    print(f"Q4_TOWER_CLOSES phi_abe phi_cde = dd - psi (base) held={base}; every {{phi,psi}} "
          f"contraction returns {{delta,phi,psi}} with integer coeffs {'PASS' if q4 else 'FAIL'}")

    print("=" * 70)
    if core and q1 and q2 and q3 and q4:
        print("FUNCTOR_F_TOWER_VERDICT Q_GREEN (5/5 clauses PASS)")
        print("FUNCTOR_F_TOWER_NOTE tower_closes_on{delta,phi,psi}; coeffs(24,4,-2,-4,-1); "
              "no_ord3_invariant_generated; standard_G2_identities_on_repo_core")
        return 0
    print("FUNCTOR_F_TOWER_VERDICT Q_INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
