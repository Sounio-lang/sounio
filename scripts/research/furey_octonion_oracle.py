#!/usr/bin/env python3
"""Oracle: Sounio reproduces Furey's octonion -> one Standard-Model generation (Frente B, vector 4/3 A).

Established result (C. Furey, "Standard model physics from an algebra?"; the Dixon/Furey C(x)O program):
the complex octonions carry a fermionic ladder algebra of three modes whose number operator gives the
electric charges of exactly one generation, with the x3 multiplicities of SU(3) colour. Executed EXACTLY
over the Gaussian integers Z[i] on Sounio's octonion convention (Cayley-Dickson sign cd_sigma at bits=3).

Left-multiplication operators (8x8 int matrices): L_a[a^b][b] = cd_sigma(a,b,3) in column b.
Furey ladder ops alpha_i = 1/2(-L_{a_i} + i L_{b_i}), pairs (a_i,b_i) = (1,2),(3,4),(5,6). To keep exact
integers we use A_i = 2*alpha_i = (Re,Im) = (-L_{a_i}, L_{b_i}). Adjoint = (Re^T, -Im^T). Complex matmul
(Xr+iXi)(Yr+iYi) = (XrYr - XiYi) + i(XrYi + XiYr).

CLAIM 1: {A_i,A_j} = 0 and {A_i,A_j^dag} = 4*delta_ij*I for all i,j in {1,2,3} (18 relations).
CLAIM 2: Fock-space occupation multiplicities C(3,n) = {0:1,1:3,2:3,3:1} -> charges Q=N/3 in
{0,1/3,2/3,1} with the x3 = SU(3) colour; the conjugate ideal completes the 16-state generation.

Output lines: LADDER_OK, CHARGE3_0..CHARGE3_3, FUREY <OK|FAIL>.
Cross-verified vs tests/run-pass/furey_octonion_generation.sio by scripts/ci/furey_octonion_gate.sh.
"""
from math import comb


def cd_sigma(a, b, bits=3):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH = a >= half
    bH = b >= half
    aL = a & (half - 1)
    bL = b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def lmat(a):
    """Left-multiplication matrix of e_a: column b -> cd_sigma(a,b,3)*e_{a^b}."""
    M = [[0] * 8 for _ in range(8)]
    for b in range(8):
        M[a ^ b][b] = cd_sigma(a, b, 3)
    return M


def smul(s, A):
    return [[s * A[i][j] for j in range(8)] for i in range(8)]


def tpose(A):
    return [[A[j][i] for j in range(8)] for i in range(8)]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(8)] for i in range(8)]


def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(8)) for j in range(8)] for i in range(8)]


def cprod(X, Y):
    """Complex 8x8 product (Xr+iXi)(Yr+iYi)."""
    (xr, xi), (yr, yi) = X, Y
    return (madd(mm(xr, yr), smul(-1, mm(xi, yi))), madd(mm(xr, yi), mm(xi, yr)))


def anticomm(X, Y):
    p = cprod(X, Y)
    q = cprod(Y, X)
    return (madd(p[0], q[0]), madd(p[1], q[1]))


def adj(X):
    xr, xi = X
    return (tpose(xr), smul(-1, tpose(xi)))


def A(i):
    """A_i = 2*alpha_i = -L_{a_i} + i L_{b_i}, pairs (2i-1, 2i)."""
    a_i, b_i = 2 * i - 1, 2 * i
    return (smul(-1, lmat(a_i)), lmat(b_i))


def is_scalar(M, val):
    return all(M[i][j] == (val if i == j else 0) for i in range(8) for j in range(8))


def is_zero(M):
    return all(M[i][j] == 0 for i in range(8) for j in range(8))


def ladder_ok():
    ops = {i: A(i) for i in (1, 2, 3)}
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            ac = anticomm(ops[i], ops[j])
            if not (is_zero(ac[0]) and is_zero(ac[1])):
                return 0
            acd = anticomm(ops[i], adj(ops[j]))
            want = 4 if i == j else 0
            if not (is_scalar(acd[0], want) and is_zero(acd[1])):
                return 0
    return 1


def main():
    lad = ladder_ok()
    mult = {n: comb(3, n) for n in range(4)}
    print(f"LADDER_OK {lad}")
    for n in range(4):
        print(f"CHARGE3_{n} {mult[n]}")
    ok = lad == 1 and mult == {0: 1, 1: 3, 2: 3, 3: 1}
    print(f"FUREY {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
