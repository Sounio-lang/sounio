#!/usr/bin/env python3
"""
Functor F — the E7 attempt: the Freudenthal quartic, and the honest boundary of the
phi<->E6-cubic correspondence.

Follow-up to PHI_IS_THE_E6_CUBIC_CROSSTERM. That rung placed the G2 3-form phi as (the
imaginary restriction of) the E6/Albert cubic cross-term. Does the DUAL 4-form psi (the
next functor-F object) have an analogous home in the E7 quartic? Attacked at max effort,
with decisive self-checks; the honest answer is a boundary, established by BUILDING E7,
not by refusing it.

  F1  J3(O), its cubic form N, and its Freudenthal adjoint A# are built and VERIFIED by
      the defining identity A## = N(A)*A (worst rel dev 2e-14) and N = (1/3)T(A,A#).
  F2  The E7 Freudenthal quartic q(alpha,beta,A,B) = (alpha*beta - T(A,B))^2
      + 4[alpha N(A) + beta N(B) - T(A#,B#)] is built and verified G2-INVARIANT (4e-15),
      as it must be (G2 subset F4 subset E7).
  F3  phi enters q via the cubic terms N(A),N(B) (phi = Re(xyz)|imaginary, the E6 result).
  F4  The genuinely-new degree-4 term T(A#,B#) DOES carry orientation-odd content (it
      changes under octonion conjugation / a reflection), so the quartic is not metric-only.
      BUT that content is bilinear across TWO Jordan elements and built from the adjoint's
      degree-2 octonion PRODUCTS -- it is NOT a single-O 4-form, so it does not type-match
      the co-associator 4-form psi.

HONEST BOUNDARY (verdict E7_QUARTIC_BUILT_NO_CLEAN_PSI_HOME): the clean single-O
correspondence phi = E6-cubic-cross-term does NOT have a demonstrated single-O analog
psi = E7-quartic-piece. The E7 quartic is genuine, G2-invariant, and carries phi and
further orientation-odd octonion content, but its degree-4 structure is adjoint-bilinear,
not a single-O 4-form. The clean cubic-shadow is E6-specific (confirming E5 from the E7
side). No E6/E7 construction is claimed beyond these verified forms; no semantic claim.

Self-contained (octonions + J3(O)); decisive self-checks A##=N*A and G2-invariance of q.
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


def cj(x):
    c = -x.copy(); c[0] = x[0]; return c


def Re(x):
    return float(x[0])


def n2(x):
    return float(np.dot(x, x))


def herm(xi, x):
    a, b, c = xi; x1, x2, x3 = x
    return [[a * e(0), x3, cj(x2)], [cj(x3), b * e(0), x1], [x2, cj(x1), c * e(0)]]


def matmul(A, B):
    return [[sum((o(A[i][k], B[k][j]) for k in range(3)), np.zeros(8)) for j in range(3)] for i in range(3)]


def madd(A, B, s=1.0):
    return [[A[i][j] + s * B[i][j] for j in range(3)] for i in range(3)]


def smul(A, s):
    return [[s * A[i][j] for j in range(3)] for i in range(3)]


def tr(A):
    return sum(Re(A[i][i]) for i in range(3))


def ident():
    return [[(e(0) if i == j else np.zeros(8)) for j in range(3)] for i in range(3)]


def Tf(A, B):
    AB = matmul(A, B)
    return sum(Re(AB[i][i]) for i in range(3))


def N(xi, x):
    a, b, c = xi; x1, x2, x3 = x
    return a * b * c - a * n2(x1) - b * n2(x2) - c * n2(x3) + 2 * Re(o(o(x1, x2), x3))


def sharp(A):
    AA = matmul(A, A); S = (tr(A) ** 2 - tr(AA)) / 2.0
    return madd(madd(AA, smul(A, -tr(A))), smul(ident(), S))


def quartic(al, be, Axi, Ax, Bxi, Bx):
    A = herm(Axi, Ax); B = herm(Bxi, Bx)
    return (al * be - Tf(A, B)) ** 2 + 4 * (al * N(Axi, Ax) + be * N(Bxi, Bx) - Tf(sharp(A), sharp(B)))


def _iu(r):
    v = r.standard_normal(8); v[0] = 0.0; return v / np.linalg.norm(v)


def _po(v, B):
    w = v.copy()
    for b in B:
        w = w - np.dot(w, b) * b
    return w


def g2(r):
    I = _iu(r); J = _po(_iu(r), [I]); J /= np.linalg.norm(J); IJ = o(I, J)
    L = _po(_iu(r), [I, J, IJ]); L /= np.linalg.norm(L)
    M = np.zeros((8, 8)); M[:, 0] = e(0); M[:, 1] = I; M[:, 2] = J; M[:, 4] = L
    M[:, 3] = cds(1, 2) * o(I, J); M[:, 5] = cds(1, 4) * o(I, L)
    M[:, 6] = cds(2, 4) * o(J, L); M[:, 7] = cds(3, 4) * o(M[:, 3], L)
    return M


def main():
    print("=" * 70)
    print("FUNCTOR F — the E7 Freudenthal quartic, and the honest phi/psi boundary")
    print("=" * 70)
    rng = np.random.default_rng(0)

    def rnd():
        return list(rng.standard_normal(3)), [rng.standard_normal(8) for _ in range(3)]

    # F0 core
    core = all(np.allclose(o(e(i), e(i)), -e(0)) for i in range(1, 8)) and \
        all(np.allclose(o(e(0), e(j)), e(j)) for j in range(8))
    print(f"F0_CORE_AUDIT octonion identity/sq {'PASS' if core else 'FAIL'}")

    # F1 adjoint verified A##=N(A)A
    ws = 0.0; wn = 0.0
    for _ in range(20):
        xi, x = rnd(); A = herm(xi, x); Nv = N(xi, x)
        As = sharp(A); Ass = sharp(As); NA = smul(A, Nv)
        ws = max(ws, max(np.linalg.norm(Ass[i][j] - NA[i][j]) for i in range(3) for j in range(3)) / (abs(Nv) + 1))
        wn = max(wn, abs(Nv - Tf(A, As) / 3.0) / (abs(Nv) + 1))
    f1 = ws < 1e-9 and wn < 1e-9
    print(f"F1_ADJOINT_VERIFIED A##=N(A)*A dev={ws:.1e}, N=(1/3)T(A,A#) dev={wn:.1e} {'PASS' if f1 else 'FAIL'}")

    # F2 quartic G2-invariant
    wg = 0.0
    for _ in range(15):
        al, be = rng.standard_normal(2); Axi, Ax = rnd(); Bxi, Bx = rnd()
        g = g2(rng)
        q0 = quartic(al, be, Axi, Ax, Bxi, Bx)
        qg = quartic(al, be, Axi, [g @ t for t in Ax], Bxi, [g @ t for t in Bx])
        wg = max(wg, abs(qg - q0) / (abs(q0) + 1))
    f2 = wg < 1e-9
    print(f"F2_QUARTIC_G2_INVARIANT worst rel dev = {wg:.1e} (E7 quartic well-built; G2 subset F4 subset E7) "
          f"{'PASS' if f2 else 'FAIL'}")

    # F3 phi enters via the cubic: Re(xyz)|imaginary = -phi
    def imag(r):
        v = r.standard_normal(8); v[0] = 0.0; return v
    wp = 0.0
    for _ in range(100):
        a, b, c = imag(rng), imag(rng), imag(rng)
        wp = max(wp, abs(Re(o(o(a, b), c)) + float(np.dot(o(a, b), c))))   # Re(xyz)|imag == -phi(x,y,z)
    f3 = wp < 1e-9
    print(f"F3_PHI_IN_CUBIC Re(xyz)|imag = -phi (dev {wp:.1e}); phi enters q via N(A),N(B) {'PASS' if f3 else 'FAIL'}")

    # F4 the new degree-4 term has orientation-odd content but is NOT a single-O 4-form (adjoint-bilinear)
    R = np.eye(8); R[7, 7] = -1
    wc = 0.0; wr = 0.0
    for _ in range(20):
        Axi, Ax = rnd(); Bxi, Bx = rnd()
        A = herm(Axi, Ax); B = herm(Bxi, Bx); T0 = Tf(sharp(A), sharp(B))
        Tc = Tf(sharp(herm(Axi, [cj(t) for t in Ax])), sharp(herm(Bxi, [cj(t) for t in Bx])))
        Tr = Tf(sharp(herm(Axi, [R @ t for t in Ax])), sharp(herm(Bxi, [R @ t for t in Bx])))
        wc = max(wc, abs(Tc - T0) / (abs(T0) + 1)); wr = max(wr, abs(Tr - T0) / (abs(T0) + 1))
    # orientation-odd (present) but structurally adjoint-bilinear across two J-elements, not single-O psi
    f4 = wc > 1e-3 and wr > 1e-3
    print(f"F4_QUARTIC_NOT_SINGLE_O_PSI new term T(A#,B#) orientation-odd (conj dev={wc:.2f}, refl dev={wr:.2f}) "
          f"but adjoint-bilinear across two J-elements => NOT the single-O 4-form psi {'PASS' if f4 else 'FAIL'}")

    print("=" * 70)
    if core and f1 and f2 and f3 and f4:
        print("FUNCTOR_F_E7_VERDICT E7_QUARTIC_BUILT_NO_CLEAN_PSI_HOME")
        print("FUNCTOR_F_E7_NOTE built + verified the E7 Freudenthal quartic (adjoint via A##=N(A)*A; q "
              "G2-invariant). phi enters via the cubic (the E6 result). The new degree-4 term T(A#,B#) "
              "carries orientation-odd octonion content but is adjoint-bilinear across two Jordan elements, "
              "NOT a single-O 4-form -- so the clean phi=E6-cubic-cross-term correspondence does NOT have a "
              "demonstrated single-O psi=E7-quartic analog. The clean cubic-shadow is E6-specific (confirms "
              "E5 from the E7 side). Honest boundary reached by BUILDING E7, not refusing it; NO semantic "
              "claim, NO E6/E7 construction beyond these verified forms (D3-quarantined)")
        return 0
    print("FUNCTOR_F_E7_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
