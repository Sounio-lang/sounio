#!/usr/bin/env python3
"""Oracle: Furey Cl(6) charge operator + the decisive test that the G2 automorphism phi does not
preserve charge (Frente B, vector 4/3). Exact Gaussian-integer 8x8 matrices. See furey_charge_g2.md."""
N = 8


def cd_sigma(a, b, bits=3):
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


cadd = lambda x, y: (x[0] + y[0], x[1] + y[1])
cmul = lambda x, y: (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])
cconj = lambda x: (x[0], -x[1])
Z = (0, 0)


def zeros():
    return [[Z] * N for _ in range(N)]


def L(a):
    M = zeros()
    for k in range(N):
        M[a ^ k][k] = (cd_sigma(a, k), 0)
    return M


def mm(A, B):
    return [[tuple(sum(v) for v in zip(*(cmul(A[i][t], B[t][j]) for t in range(N))))
             for j in range(N)] for i in range(N)]


def madd(A, B):
    return [[cadd(A[i][j], B[i][j]) for j in range(N)] for i in range(N)]


def scale(c, A):
    return [[cmul(c, A[i][j]) for j in range(N)] for i in range(N)]


def dag(A):
    return [[cconj(A[j][i]) for j in range(N)] for i in range(N)]


def main():
    pairs = [(1, 2), (3, 4), (5, 6)]
    M = [madd(scale((-1, 0), L(a)), scale((0, 1), L(b))) for a, b in pairs]
    Md = [dag(x) for x in M]
    I4 = [[(4 if i == j else 0, 0) for j in range(N)] for i in range(N)]
    witt = 1
    for i in range(3):
        w = madd(mm(M[i], Md[i]), mm(Md[i], M[i]))
        if w != I4:
            witt = 0
    D = zeros()
    for i in range(3):
        D = madd(D, mm(Md[i], M[i]))
    g = [0, 2, 3, 1, 4, 6, 7, 5]
    ginv = [0, 3, 1, 2, 4, 7, 5, 6]
    comm = 0
    for r in range(N):
        for c in range(N):
            if D[ginv[r]][c] != D[r][g[c]]:
                comm = 1
    cm = {0: 0, 1: 0, 2: 0, 3: 0}
    for occ in range(8):
        cm[(occ & 1) + ((occ >> 1) & 1) + ((occ >> 2) & 1)] += 1
    charge = 1 if cm == {0: 1, 1: 3, 2: 3, 3: 1} else 0
    print(f"WITT_OK {witt}")
    print(f"COMM_NONZERO {comm}")
    print(f"CHARGE_OK {charge}")
    print(f"FUREYCHARGE {'OK' if witt and comm and charge else 'FAIL'}")


if __name__ == "__main__":
    main()
