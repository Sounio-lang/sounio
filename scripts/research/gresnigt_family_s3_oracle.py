#!/usr/bin/env python3
"""Oracle: Gresnigt's family S3 generator psi (non-monomial, over Q(sqrt3)) + the frame-relative
mechanism [phi,N]=0 vs [psi,N]!=0 (Frente B, vector 4/3 capstone). See gresnigt_family_s3.md.
Refs: Gresnigt arXiv:2306.13098 eq (56)-(76)."""
from fractions import Fraction as F


def cd_sigma(a, b, bits=4):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


# --- core over Q(sqrt3): number (p,q)=p+q√3 ---
radd = lambda x, y: (x[0] + y[0], x[1] + y[1])
rmul = lambda x, y: (x[0] * y[0] + 3 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])
rsc = lambda k, x: (k * x[0], k * x[1])
RZ = (F(0), F(0)); RO = (F(1), F(0))


def smul(u, v):
    w = [RZ] * 16
    for a in range(16):
        if u[a] == RZ:
            continue
        for b in range(16):
            if v[b] == RZ:
                continue
            k = a ^ b
            w[k] = radd(w[k], rsc(cd_sigma(a, b), rmul(u[a], v[b])))
    return w


def unit(i):
    e = [RZ] * 16; e[i] = RO; return e


def psi_unit(i):
    e = [RZ] * 16
    if i == 0 or i == 8:
        e[i] = RO
    elif 1 <= i <= 7:
        e[i] = (F(-1, 2), F(0)); e[i + 8] = (F(0), F(-1, 2))
    else:
        e[i - 8] = (F(0), F(1, 2)); e[i] = (F(-1, 2), F(0))
    return e


def psi_vec(u):
    out = [RZ] * 16
    for i in range(16):
        if u[i] == RZ:
            continue
        pu = psi_unit(i)
        for k in range(16):
            out[k] = radd(out[k], rmul(u[i], pu[k]))
    return out


def core():
    auto = 1 if all(psi_vec(smul(unit(j), unit(k))) == smul(psi_unit(j), psi_unit(k))
                    for j in range(16) for k in range(16)) else 0
    ord3 = 1 if (all(psi_vec(psi_vec(psi_unit(i))) == unit(i) for i in range(16))
                 and any(psi_unit(i) != unit(i) for i in range(16))) else 0
    nonmono = 1 if any(psi_unit(j)[k][1] != 0 for j in range(16) for k in range(16)) else 0
    a = (F(-1, 2), F(1, 2)); b = (F(-1, 2), F(-1, 2))
    Ad1r = [RZ] * 16; Ad1r[1] = RO; Ad1r[9] = RO
    Bd1r = [RZ] * 16; Bd1r[1] = a; Bd1r[9] = b
    Ad1i = [RZ] * 16; Ad1i[5] = RO; Ad1i[13] = RO
    Bd1i = [RZ] * 16; Bd1i[5] = a; Bd1i[13] = b
    mapok = 1 if (psi_vec(Ad1r) == Bd1r and psi_vec(Ad1i) == Bd1i) else 0
    return auto, ord3, nonmono, mapok


# --- commutators over Q(sqrt3,i) : 16x16 ---
cadd = lambda x, y: (radd(x[0], y[0]), radd(x[1], y[1]))
csub = lambda x, y: (radd(x[0], (-y[0][0], -y[0][1])), radd(x[1], (-y[1][0], -y[1][1])))
cmul = lambda x, y: ((rmul(x[0], y[0])[0] - rmul(x[1], y[1])[0], rmul(x[0], y[0])[1] - rmul(x[1], y[1])[1]),
                     (rmul(x[0], y[1])[0] + rmul(x[1], y[0])[0], rmul(x[0], y[1])[1] + rmul(x[1], y[0])[1]))
CZ = (RZ, RZ)
II = (RZ, RO)  # i


def zeros():
    return [[CZ] * 16 for _ in range(16)]


def Lunit(a):
    M = zeros()
    for k in range(16):
        M[a ^ k][k] = (((F(cd_sigma(a, k)), F(0)), RZ))
    return M


def mmul(A, B):
    C = zeros()
    for i in range(16):
        for j in range(16):
            s = CZ
            for t in range(16):
                if A[i][t] == CZ:
                    continue
                s = cadd(s, cmul(A[i][t], B[t][j]))
            C[i][j] = s
    return C


def madd(A, B):
    return [[cadd(A[i][j], B[i][j]) for j in range(16)] for i in range(16)]


def mscale(z, A):
    return [[cmul(z, A[i][j]) for j in range(16)] for i in range(16)]


def commutators():
    def Ldag(i):
        return madd(madd(Lunit(i), mscale(II, Lunit(i + 4))), madd(Lunit(i + 8), mscale(II, Lunit(i + 12))))

    def Llow(i):
        return madd(madd(mscale(((F(-1), F(0)), RZ), Lunit(i)), mscale(II, Lunit(i + 4))),
                    madd(mscale(((F(-1), F(0)), RZ), Lunit(i + 8)), mscale(II, Lunit(i + 12))))
    Nop = zeros()
    for i in [1, 2, 3]:
        Nop = madd(Nop, mmul(Ldag(i), Llow(i)))
    Psi = zeros()
    for j in range(16):
        pj = psi_unit(j)
        for k in range(16):
            Psi[k][j] = (pj[k], RZ)
    g = [0, 2, 3, 1, 4, 6, 7, 5, 8, 10, 11, 9, 12, 14, 15, 13]
    Phi = zeros()
    for j in range(16):
        Phi[g[j]][j] = (RO, RZ)

    def comm(X):
        A = mmul(X, Nop); B = mmul(Nop, X)
        return all(A[i][j] == B[i][j] for i in range(16) for j in range(16))
    phi_zero = 1 if comm(Phi) else 0
    psi_nonzero = 0 if comm(Psi) else 1
    return phi_zero, psi_nonzero


def main():
    auto, ord3, nonmono, mapok = core()
    phi_zero, psi_nonzero = commutators()
    print(f"PSI_AUTO {auto}")
    print(f"PSI_ORD3 {ord3}")
    print(f"PSI_NONMONO {nonmono}")
    print(f"PSI_MAPS_AB {mapok}")
    print(f"COMM_PHI_ZERO {phi_zero}")
    print(f"COMM_PSI_NONZERO {psi_nonzero}")
    ok = auto and ord3 and nonmono and mapok
    print(f"FAMILYS3 {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
