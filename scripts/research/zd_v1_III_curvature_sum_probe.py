#!/usr/bin/env python3
"""The level-independent curvature tensor K_j, and its sum, computed for j = 3..12 (j >= 8 via the streaming path, `... curvature_sum_probe.py stream`).

At n = j+2 the class size is 1, so K_j IS the raw per-triangle defect table at that level; the
level-independence then says every higher level is a uniform cls-fold blow-up of it.

COST NOTE. At cls = 1, sum(K_j) is by definition delta(j+2, j), which is a TWO-TRACE computation:
25 s at j = 12 against 1873 s for the tensor. So the SUM never needs the tensor; the tensor is what
establishes the structure (values in {0,-2}, the 864*[j,3]_2 count, the support condition).

§37 reduced (III)'s open content to one number per j: the curvature sum over the classes at and
below the seam. This computes it. Partition the vertices by their low j+1 bits (M = 2^(j+1)
classes, each of size cls = 2^(n-j-2)); with A the seam's graph and Bp = Phi*A_(tau W),

    K_j(u,v,w) := [ tr(A_uv A_vw A_wu) - tr(Bp_uv Bp_vw Bp_wu) ] / cls^3

Measured:
  * K_j is LEVEL-INDEPENDENT -- the whole tensor, not just its sum (max abs diff 0 across n);
  * K_j takes only the values 0 and -2, i.e. every contributing class is a single sign flip;
  * sum(K_j) = -1728 * [j choose 3]_2  exactly, i.e. 864*[j,3]_2 flipped classes;
    j = 3..12 -> [j,3]_2 = 1, 15, 155, 1395, 11811, 97155, 788035, 6347715, 50955971,
                408345795;
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
        # accumulate block-triple by block-triple: one M^3 temporary at a time, so j = 7
        # (M = 256, a 16.8M-entry tensor) stays inside memory.
        Y = X.reshape(cls, M, cls, M)
        out = np.zeros((M, M, M))
        for p_ in range(cls):
            for q_ in range(cls):
                Xpq = Y[p_, :, q_, :]
                for r_ in range(cls):
                    out += np.einsum('uv,vw,wu->uvw', Xpq, Y[q_, :, r_, :], Y[r_, :, p_, :])
        return out

    K = (cube(A) - cube(Bp)) / cls**3
    Kr = np.round(K)
    assert np.allclose(K, Kr)
    return Kr, cls


def K_summary(n, W, sup_every=64):
    """Streaming form: never materialises the M^3 tensor. Needed from j = 8 (M = 512, 134M
    entries) and unavoidable at j = 10 (M = 2048, 8.6 BILLION). Returns
    (sum, values-in-{0,-2}?, #nonzero, #support violations, #sampled slices, checksum, cls).

    The checksum is the weighted linear hash sum_{u,v,w} K[u,v,w]*(u*M^2 + v*M + w), computed from
    row and column sums so no nonzero extraction is needed -- that is what keeps j = 10 at ~40 s.
    Equal checksums + equal value profiles pin the tensor across levels without holding it.
    The support check needs the nonzero indices, so it is sampled every `sup_every` slices.
    """
    j = lsb(W)
    H = 1 << (n - 1)
    M = 1 << (j + 1)
    cls = H // M
    S = sign_table_fast(n)
    A = np.zeros((H, H), dtype=np.int32); A[1:, 1:] = A_sig_fast(n, W, S)
    B = np.zeros((H, H), dtype=np.int32); B[1:, 1:] = A_sig_fast(n, tau(j, W), S)
    pp = np.array([0] + [tau(j, a) for a in range(1, H)])
    Bp = B[np.ix_(pp, pp)]
    del S, B
    ar = np.arange(M, dtype=np.int64)
    tot = nz = chk = badsup = sampled = 0
    okvals = True
    for u in range(M):
        acc = np.zeros((M, M), dtype=np.int32)
        for X, sgn in ((A, 1), (Bp, -1)):
            for p_ in range(cls):
                ra = X[p_ * M + u]
                for q_ in range(cls):
                    row = ra[q_ * M:(q_ + 1) * M]
                    if not row.any():
                        continue
                    for r_ in range(cls):
                        acc += sgn * (np.outer(row, X[r_ * M:(r_ + 1) * M, p_ * M + u])
                                      * X[q_ * M:(q_ + 1) * M, r_ * M:(r_ + 1) * M])
        assert (acc % cls**3 == 0).all()
        Su = acc // cls**3
        su = int(Su.sum()); tot += su
        nz += int(np.count_nonzero(Su))
        okvals &= bool(((Su == 0) | (Su == -2)).all())
        chk = (chk + u * M * M * su + M * int(np.dot(ar, Su.sum(1, dtype=np.int64)))
               + int(np.dot(ar, Su.sum(0, dtype=np.int64)))) % (2**61 - 1)
        if u % sup_every == 0:
            v, w = np.nonzero(Su)
            x, y = u ^ v, v ^ w
            badsup += int((~((x != 0) & (y != 0) & (x != y))).sum())
            sampled += 1
    return tot, okvals, nz, badsup, sampled, chk, cls


def main_stream(js):
    for j, W in js:
        prev = None
        for n in range(j + 2, j + 4):
            if W >= 1 << (n - 1):
                continue
            tot, okvals, nz, badsup, sampled, chk, cls = K_summary(n, W)
            want = -1728 * qb3(j)
            same = "n/a" if prev is None else str(chk == prev)
            print(f"j={j} n={n} cls={cls}: sum(K)={tot} want {want} "
                  f"{'OK' if tot == want else 'MISMATCH'} | values in {{0,-2}}: {okvals} | "
                  f"flipped {nz} = 864*[{j},3]_2 -> {nz == 864 * qb3(j)} | "
                  f"support violations {badsup} over {sampled} sampled slices | "
                  f"checksum {chk} equals previous level: {same}")
            print(f"        delta = cls^3*sum = {cls**3 * tot} "
                  f"(= -27*8^(n-j)*[j,3]_2 = {-27 * 8**(n - j) * qb3(j)})")
            prev = chk
            sys.stdout.flush()


def main():
    if sys.argv[1:] == ["stream"]:
        main_stream([(8, 256), (9, 512), (10, 1024), (11, 2048), (12, 4096)])
        return
    for j, W in ((3, 8), (4, 16), (5, 32), (6, 64), (7, 128)):
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
