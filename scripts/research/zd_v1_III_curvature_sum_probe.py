#!/usr/bin/env python3
"""The level-independent curvature tensor K_j, and its sum, computed for j = 3,4,5,6.

§37 reduced (III)'s open content to one number per j: the curvature sum over the classes at and
below the seam. This computes it. Partition the vertices by their low j+1 bits (M = 2^(j+1)
classes, each of size cls = 2^(n-j-2)); with A the seam's graph and Bp = Phi*A_(tau W),

    K_j(u,v,w) := [ tr(A_uv A_vw A_wu) - tr(Bp_uv Bp_vw Bp_wu) ] / cls^3

Measured:
  * K_j is LEVEL-INDEPENDENT -- the whole tensor, not just its sum (max abs diff 0 across n);
  * K_j takes only the values 0 and -2, i.e. every contributing class is a single sign flip;
  * sum(K_j) = -1728 * [j choose 3]_2  exactly, i.e. 864*[j,3]_2 flipped classes;
    j = 3,4,5,6 -> [j,3]_2 = 1, 15, 155, 1395;
  * every nonzero entry has (u^v, v^w) linearly independent.

Hence delta(n,j) = cls^3 * sum(K_j) = (2^(n-j-2))^3 * (-1728) * [j choose 3]_2, which is the
closed form of §33.3 -- now assembled from a finite object rather than fitted.
"""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, lsb, sign_table_fast  # noqa: E402


def tau(j, x):
    return x if (x & 1) == ((x >> j) & 1) else x ^ (1 | (1 << j))


def qb3(j):
    return (2**j - 1) * (2**(j-1) - 1) * (2**(j-2) - 1) // 21


def K_tensor(n, W):
    """Blocked by residue mod 2^(j+1). Vertex 0 is added as an isolated vertex so the reshape is
    uniform; it contributes nothing. einsum over the block indices -- M^3*cls^3 work, which is
    what makes j = 6 (M = 128) reachable."""
    j = lsb(W)
    H = 1 << (n - 1)
    M = 1 << (j + 1)
    cls = H // M
    S = sign_table_fast(n)
    A = np.zeros((H, H)); A[1:, 1:] = A_sig_fast(n, W, S)
    B = np.zeros((H, H)); B[1:, 1:] = A_sig_fast(n, tau(j, W), S)
    p = np.array([0] + [tau(j, a) for a in range(1, H)])
    Bp = B[np.ix_(p, p)]

    def cube(X):
        Y = X.reshape(cls, M, cls, M)
        return np.einsum('puqv,qvrw,rwpu->uvw', Y, Y, Y, optimize=True)

    K = (cube(A) - cube(Bp)) / cls**3
    Kr = np.round(K)
    assert np.allclose(K, Kr)
    return Kr, cls


def main():
    for j, W in ((3, 8), (4, 16), (5, 32), (6, 64)):
        prev = None
        for n in (7, 8, 9, 10):
            if W >= 1 << (n - 1):
                continue
            K, cls = K_tensor(n, W)
            hist = dict(Counter(K.ravel().tolist()))
            s = K.sum()
            want = -1728 * qb3(j)
            M = 1 << (j + 1)
            uu, vv, ww = np.meshgrid(np.arange(M), np.arange(M), np.arange(M), indexing='ij')
            x, y = uu ^ vv, vv ^ ww
            good = (x != 0) & (y != 0) & (x != y)
            nz = np.flatnonzero(K.ravel() != 0)
            indep = bool(((K != 0) & ~good).sum() == 0)
            same = "n/a" if prev is None else str(np.array_equal(K, prev))
            print(f"j={j} n={n}: sum(K)={s:.0f} want {want} {'OK' if s == want else 'MISMATCH'}"
                  f" | values {hist} | flipped classes {len(nz)} = 864*[{j},3]_2 ->"
                  f" {len(nz) == 864 * qb3(j)} | (u^v,v^w) independent on support: {indep}"
                  f" | tensor equals previous level: {same}")
            print(f"        delta(n,j) = cls^3 * sum(K) = {cls}^3 * {want} = "
                  f"{cls**3 * want}  (= -27*8^(n-j)*[j,3]_2 = {-27*8**(n-j)*qb3(j)})")
            prev = K
            sys.stdout.flush()


if __name__ == "__main__":
    main()
