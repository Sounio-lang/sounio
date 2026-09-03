#!/usr/bin/env python3
"""
Functor F — the exceptional bridge: the G2 3-form phi IS the E6/Albert cubic cross-term.

NOT the Petitot semantic conjecture (that stays quarantined). A concrete algebraic bridge:
the functor-F central object -- the G2 3-form phi -- is literally (the imaginary restriction
of) an exceptional-group invariant.

  E1  The associator [x,y,z]=(xy)z - x(yz) is PURELY IMAGINARY (Re[x,y,z]=0), so
      Re(x*y*z) is bracketing-INDEPENDENT: a well-defined trilinear form on O.
  E2  That real trilinear form is exactly the octonion cross-term 2*Re(x y z) of the CUBIC
      FORM (determinant) N of the exceptional Jordan / Albert algebra J3(O), whose
      automorphism group is F4 and which E6 preserves projectively. Verified: N is
      invariant under G2 = Aut(O) acting on the off-diagonal octonions (G2 subset F4).
  E3  For IMAGINARY x,y,z, Re(x*y*z) = -phi(x,y,z): the G2 3-form phi IS the imaginary
      restriction of that E6 cubic cross-term. So functor-F's phi sits INSIDE the E6 cubic.
  E4  CORRECTION of an earlier draft: the scalar 3-form phi and the VECTOR-valued associator
      [x,y,z] (the psi/4-form side) are DIFFERENT objects. phi = the cubic cross-term (scalar);
      the associator = the non-associative / psi part (vector). The first draft conflated them.

Verdict PHI_IS_THE_E6_CUBIC_CROSSTERM. Concrete algebra, not the semantic conjecture
(D3-quarantined, see functor_f_exceptional_frontier_note). Self-contained (octonions).
"""
import numpy as np

np.seterr(all='ignore')
TOL = 1e-9


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


def o(A, B):
    C = np.zeros(8)
    for i in range(8):
        if A[i] == 0.0:
            continue
        for j in range(8):
            if B[j] == 0.0:
                continue
            C[i ^ j] += cds(i, j) * A[i] * B[j]
    return C


def e(i):
    v = np.zeros(8); v[i] = 1.0; return v


def conj(x):
    c = -x.copy(); c[0] = x[0]; return c


def Re(x):
    return float(x[0])


def nrm2(x):
    return float(np.dot(x, x))


# --- G2 automorphism (generic, verified) for the invariance check ---
def _iu(rng):
    v = rng.standard_normal(8); v[0] = 0.0; return v / np.linalg.norm(v)


def _po(v, B):
    w = v.copy()
    for b in B:
        w = w - np.dot(w, b) * b
    return w


def g2auto(rng):
    I = _iu(rng); J = _po(_iu(rng), [I]); J /= np.linalg.norm(J); IJ = o(I, J)
    L = _po(_iu(rng), [I, J, IJ]); L /= np.linalg.norm(L)
    M = np.zeros((8, 8)); M[:, 0] = e(0); M[:, 1] = I; M[:, 2] = J; M[:, 4] = L
    M[:, 3] = cds(1, 2) * o(I, J); M[:, 5] = cds(1, 4) * o(I, L)
    M[:, 6] = cds(2, 4) * o(J, L); M[:, 7] = cds(3, 4) * o(M[:, 3], L)
    return M


# --- Albert-algebra cubic form (determinant) of a Hermitian 3x3 octonion matrix ---
# diag reals a,b,c; off-diagonal octonions x,y,z (positions (2,3),(1,3),(1,2)).
# N = a b c - a n(x) - b n(y) - c n(z) + 2 Re( z (x conj? ) ...) -- use the standard
# N = a b c - a n(x) - b n(y) - c n(z) + 2 Re( x y z )  (cyclic octonion cross-term).
def albert_N(a, b, c, x, y, z):
    return a * b * c - a * nrm2(x) - b * nrm2(y) - c * nrm2(z) + 2.0 * Re(o(o(x, y), z))


def main():
    print("=" * 70)
    print("FUNCTOR F — the associator is the G2 shadow of the E6/Albert cubic form")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # E1 — associator purely imaginary => Re(xyz) bracketing-independent
    worst_reassoc = 0.0; worst_reindep = 0.0
    for _ in range(300):
        x, y, z = rng.standard_normal(8), rng.standard_normal(8), rng.standard_normal(8)
        A = o(o(x, y), z) - o(x, o(y, z))
        worst_reassoc = max(worst_reassoc, abs(Re(A)))
        worst_reindep = max(worst_reindep, abs(Re(o(o(x, y), z)) - Re(o(x, o(y, z)))))
    e1 = worst_reassoc < 1e-9 and worst_reindep < 1e-9
    print(f"E1_RE_WELL_DEFINED Re[x,y,z]={worst_reassoc:.1e} (assoc purely imaginary); "
          f"Re((xy)z)-Re(x(yz))={worst_reindep:.1e} (bracketing-independent) {'PASS' if e1 else 'FAIL'}")

    # E2 — Re(xyz) is the octonion cross-term of the Albert cubic form N, and N is G2-invariant
    worst_Ninv = 0.0
    for _ in range(60):
        a, b, c = rng.standard_normal(3)
        x, y, z = rng.standard_normal(8), rng.standard_normal(8), rng.standard_normal(8)
        N0 = albert_N(a, b, c, x, y, z)
        g = g2auto(rng)
        Ng = albert_N(a, b, c, g @ x, g @ y, g @ z)          # G2 acts on the octonion entries
        worst_Ninv = max(worst_Ninv, abs(Ng - N0) / (abs(N0) + 1.0))
    e2 = worst_Ninv < 1e-9
    print(f"E2_G2_PRESERVES_ALBERT_CUBIC N(J3(O)) invariant under G2=Aut(O) on the octonion entries: "
          f"max rel dev = {worst_Ninv:.1e} (G2 subset F4=Aut(J3(O))) {'PASS' if e2 else 'FAIL'}")

    # E3 — the G2 3-form phi IS the E6 cubic cross-term restricted to imaginary octonions.
    #      For imaginary x,y,z:  Re(x*y*z) = -phi(x,y,z)  (phi = <xy,z>, the structure 3-form).
    def imag(r):
        v = r.standard_normal(8); v[0] = 0.0; return v
    def phi3(x, y, z):
        return float(np.dot(o(x, y), z))
    worst_phi = 0.0
    for _ in range(200):
        x, y, z = imag(rng), imag(rng), imag(rng)
        worst_phi = max(worst_phi, abs(Re(o(o(x, y), z)) + phi3(x, y, z)))
    e3 = worst_phi < 1e-9
    print(f"E3_PHI_IS_CUBIC_CROSSTERM Re(xyz)|imaginary == -phi(x,y,z): max dev {worst_phi:.1e} "
          f"=> the G2 3-form phi IS the imaginary restriction of the E6/Albert cubic cross-term "
          f"{'PASS' if e3 else 'FAIL'}")

    # E4 — the (vector-valued) associator is a SEPARATE object, NOT phi: it is the psi/4-form
    #      side. phi (scalar 3-form) = cubic cross-term; associator (vector) = non-associativity.
    x, y, z = imag(rng), imag(rng), imag(rng)
    assoc = o(o(x, y), z) - o(x, o(y, z))                     # VECTOR (octonion)
    phi_sc = phi3(x, y, z)                                    # SCALAR
    # associator relates to psi (the 4-form): [e_a,e_b,e_c] = -2 sum_d psi_abcd e_d
    e4 = (abs(Re(assoc)) < 1e-9 and np.linalg.norm(assoc) > 1e-9 and abs(phi_sc) > 1e-9)
    print(f"E4_ASSOCIATOR_IS_SEPARATE associator is VECTOR (||.||={np.linalg.norm(assoc):.2f}, the psi/"
          f"4-form side), phi is SCALAR ({phi_sc:+.2f}); they are DIFFERENT objects (correcting an earlier "
          f"conflation) {'PASS' if e4 else 'FAIL'}")

    # E5 — the bridge is CUBIC-SPECIFIC: Re(octonion word) is bracketing-independent at
    # length <=3 but NOT at length >=4, so the clean split does not lift to the E7 quartic.
    def all_brackets(f):
        if len(f) == 1:
            return [f[0]]
        r = []
        for k in range(1, len(f)):
            for Lf in all_brackets(f[:k]):
                for Rf in all_brackets(f[k:]):
                    r.append(o(Lf, Rf))
        return r
    spread = {}
    for nlen in (3, 4):
        w = 0.0
        for _ in range(40):
            f = [rng.standard_normal(8) for _ in range(nlen)]
            res = [Re(b) for b in all_brackets(f)]
            w = max(w, max(res) - min(res))
        spread[nlen] = w
    # and the mechanism: Re[x,y,z]=0 but Re([x,y,z]*w) != 0
    xx, yy, zz, ww = (rng.standard_normal(8) for _ in range(4))
    assoc4 = o(o(xx, yy), zz) - o(xx, o(yy, zz))
    e5 = spread[3] < 1e-9 and spread[4] > 1e-3 and abs(Re(o(assoc4, ww))) > 1e-6
    print(f"E5_CUBIC_SPECIFIC Re bracketing-spread: len3={spread[3]:.1e} (independent) len4={spread[4]:.1e} "
          f"(DEPENDENT); Re([x,y,z]*w)={Re(o(assoc4, ww)):+.2f}!=0 => the E6-cubic shadow is length-3 "
          f"(cubic) only, does NOT lift to the E7 quartic {'PASS' if e5 else 'FAIL'}")

    print("=" * 70)
    if e1 and e2 and e3 and e4 and e5:
        print("FUNCTOR_F_E6_VERDICT PHI_IS_THE_E6_CUBIC_CROSSTERM")
        print("FUNCTOR_F_E6_NOTE the G2 3-form phi (the functor-F central object) IS the imaginary "
              "restriction of the octonion cross-term Re(xyz) of the Albert-algebra cubic form N(J3(O)) "
              "(F4=Aut(J3(O)), E6 preserves N projectively; Re(xyz)|imaginary = -phi, verified; N is "
              "G2-invariant, G2 subset F4). So functor-F's phi sits INSIDE the E6 cubic invariant. NOTE: "
              "the scalar 3-form phi and the VECTOR associator [x,y,z] (the psi/4-form side) are DIFFERENT "
              "objects -- an earlier draft conflated them; corrected here (E3/E4). The bridge is "
              "CUBIC-SPECIFIC (E5): Re(word) bracketing-independent only at length<=3 (phi is 3-linear = "
              "the cubic degree), does NOT lift to the E7 quartic. CONCRETE algebra, NOT the Petitot "
              "semantic conjecture (D3-quarantined)")
        return 0
    print("FUNCTOR_F_E6_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
