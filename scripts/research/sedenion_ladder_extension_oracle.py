#!/usr/bin/env python3
"""Oracle: the SEDENION EXTENSION of the Furey ladder (Frente B, vector 4/3 Part B).

Clean re-implementation of the Part B facts. Over the sedenion LEFT-multiplication matrices
L_a (16x16, L_a[a^b][b] = cd_sigma(a,b,4)), a single-pair ladder op is the complex 16x16 matrix
A = alpha(a,b) = (-L_a, L_b) (Re, Im) — this is Furey's A = 2*alpha, hence the factor 4. adjoint of
(Re,Im) is (Re^T, -Im^T); anticommutator {X,Y} = XY + YX.

  B1 (octonion generation persists): the three octonion ladder ops from the disjoint pairs
     (1,2),(3,4),(5,6), now as 16x16 matrices, still satisfy {A_i,A_j}=0 and {A_i,A_j†}=4*delta_ij*I.
  B2 (maximal fermionic rank is 4, not 6): the greedy maximal set of mutually-fermionic single-pair
     ladder ops is 3 in the octonion (indices 1..7) and 4 in the sedenion (indices 1..15). The doubling
     adds EXACTLY ONE more fermionic mode (rank 3->4) — NOT a clean second generation (which needs 6).
  B3 (honest scope): the basis UNITS never multiply to zero; the sedenion zero divisors are 2-support
     elements, so the ZD geometry lives at the STATE/spinor level, not the ladder-generator level.

Output lines: B1_OK, OCT_RANK, SED_RANK, SEDEXT <OK|FAIL>.
Cross-verified vs tests/run-pass/sedenion_ladder_extension.sio by scripts/ci/sedenion_ladder_extension_gate.sh.
"""

N = 16


def cd_sigma(a, b, bits=4):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH = a >= half; bH = b >= half; aL = a & (half - 1); bL = b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def Lmat(a):
    M = [[0] * N for _ in range(N)]
    for b in range(N):
        M[a ^ b][b] = cd_sigma(a, b, 4)
    return M


def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(N)) for j in range(N)] for i in range(N)]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(N)] for i in range(N)]


def smul(s, A):
    return [[s * A[i][j] for j in range(N)] for i in range(N)]


def tpose(A):
    return [[A[j][i] for j in range(N)] for i in range(N)]


def alpha(a, b):
    return (smul(-1, Lmat(a)), Lmat(b))  # A = (-L_a, L_b)


def adj(X):
    xr, xi = X
    return (tpose(xr), smul(-1, tpose(xi)))


def anticomm(X, Y):
    (xr, xi), (yr, yi) = X, Y

    def prod(P):
        (ar, ai), (br, bi) = P
        return (madd(mm(ar, br), smul(-1, mm(ai, bi))), madd(mm(ar, bi), mm(ai, br)))

    pr = prod((X, Y)); pl = prod((Y, X))
    return (madd(pr[0], pl[0]), madd(pr[1], pl[1]))


def is_zero(M):
    return all(M[i][j] == 0 for i in range(N) for j in range(N))


def is_scalar(M, v):
    return all(M[i][j] == (v if i == j else 0) for i in range(N) for j in range(N))


def self_fermionic(a, b):
    X = alpha(a, b)
    s = anticomm(X, adj(X)); s2 = anticomm(X, X)
    return is_scalar(s[0], 4) and is_zero(s[1]) and is_zero(s2[0]) and is_zero(s2[1])


def cross_fermionic(a, b, c, d):
    X = alpha(a, b); Y = alpha(c, d)
    ac = anticomm(X, Y); acd = anticomm(X, adj(Y))
    return is_zero(ac[0]) and is_zero(ac[1]) and is_zero(acd[0]) and is_zero(acd[1])


def greedy_rank(hi):
    chosen = []
    for a in range(1, hi + 1):
        for b in range(a + 1, hi + 1):
            if not self_fermionic(a, b):
                continue
            if all(cross_fermionic(a, b, c, d) for (c, d) in chosen):
                chosen.append((a, b))
    return len(chosen)


def b1_octonion_persists():
    tri = [(1, 2), (3, 4), (5, 6)]
    A = [alpha(a, b) for (a, b) in tri]
    for i in range(3):
        for j in range(3):
            ac = anticomm(A[i], A[j])
            if not (is_zero(ac[0]) and is_zero(ac[1])):
                return 0
            acd = anticomm(A[i], adj(A[j]))
            want = 4 if i == j else 0
            if not (is_scalar(acd[0], want) and is_zero(acd[1])):
                return 0
    return 1


def main():
    b1 = b1_octonion_persists()
    oct_rank = greedy_rank(7)
    sed_rank = greedy_rank(15)
    print(f"B1_OK {b1}")
    print(f"OCT_RANK {oct_rank}")
    print(f"SED_RANK {sed_rank}")
    ok = (b1 == 1 and oct_rank == 3 and sed_rank == 4)
    print(f"SEDEXT {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
