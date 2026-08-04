#!/usr/bin/env python3
"""Is the lane's annihilation graph the Cayley-Dickson zero-divisor / orthogonality graph?

The lane's A_sig is indexed by l in [1, 2^(n-1)) with hi(l) = l ^ L, L = Llo | 2^(n-1). So a vertex
l stands for the PENCIL element x_l = e_l + eps_l * e_(l^L). Multiplying two of them:

  x_l x_y = [ sigma(l,y)    + eps_l eps_y sigma(l^L, y^L) ] e_(l^y)
          + [ eps_y sigma(l, y^L) + eps_l sigma(l^L, y)   ] e_(l^y^L)

Both components vanish iff  eps_l eps_y = -P1(l,y)  AND  eps_l eps_y = -P3(l,y), i.e. iff

        P1 = P3      (annihilation)      with      eps_l eps_y = -P1 = A_sig(l,y).

A_sig additionally demands the two symmetry clauses P1 = P1^T and P3 = P3^T -- and those are
P1_symm / P3_symm, proven for all n in SounioZDFiberAntisym.lean. So the identification is a
theorem given lemmas already in the tree, and the clauses below are its numerical confirmation.

  C1  sign_table_fast really is the CD basis multiplication (independent doubling recursion)
  C2a n=4 (sedenions): x_l x_y = 0 tested with REAL CD multiplication, against A_sig
  C2b n=5..8: annihilation <=> P1 = P3, and A_sig = -P1 on the support
  C3  n=4 shape: every label gives K_6 minus a perfect matching (the octahedron), whose
      non-adjacency is exactly the coset pairing l <-> l ^ Llo -- the shape of the "double
      hexagon" of Guterman & Zhilina, Zap. Nauchn. Sem. POMI 496 (2020) 61-86.
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402


def cd_conj(x):
    n = len(x)
    if n == 1:
        return x.copy()
    h = n // 2
    o = np.empty_like(x)
    o[:h] = cd_conj(x[:h])
    o[h:] = -x[h:]
    return o


def cd_mul(x, y):
    n = len(x)
    if n == 1:
        return np.array([x[0] * y[0]], dtype=np.int64)
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([cd_mul(a, c) - cd_mul(cd_conj(d), b),
                           cd_mul(d, a) + cd_mul(b, cd_conj(c))])


def pencil(n, l, L, e):
    v = np.zeros(1 << n, dtype=np.int64)
    v[l] += 1
    v[l ^ L] += e
    return v


def main():
    # C1
    for n in (4, 5, 6):
        S = sign_table_fast(n)
        bad = sum(1 for i in range(1 << n) for j in range(1 << n)
                  if (lambda p: len(np.nonzero(p)[0]) != 1 or np.nonzero(p)[0][0] != (i ^ j)
                      or p[i ^ j] != S[i, j])(
                          cd_mul(np.eye(1 << n, dtype=np.int64)[i],
                                 np.eye(1 << n, dtype=np.int64)[j])))
        print(f"C1  n={n}: sign_table_fast vs independent CD multiplication -- {bad} mismatches")

    # C2a
    n = 4
    H = 1 << (n - 1)
    S = sign_table_fast(n)
    tot = bs = bg = both = 0
    for Llo in range(1, H):
        L = Llo | H
        A = A_sig_fast(n, Llo, S).astype(int)
        for l in range(1, H):
            for y in range(1, H):
                if l == y:
                    continue
                ann = {el * ey for el in (1, -1) for ey in (1, -1)
                       if not cd_mul(pencil(n, l, L, el), pencil(n, y, L, ey)).any()}
                tot += 1
                edge = A[l - 1, y - 1] != 0
                if bool(ann) != edge:
                    bs += 1
                elif edge and ann != {int(A[l - 1, y - 1])}:
                    bg += 1
                both += len(ann) > 1
    print(f"C2a n=4, REAL CD multiplication: {tot} ordered pairs over {H-1} labels -- "
          f"support mismatches {bs}, sign mismatches {bg}, both-signs-annihilate {both}")

    # C2b
    for n in range(5, 9):
        H = 1 << (n - 1)
        S = sign_table_fast(n)
        bs = bg = tot = 0
        for Llo in range(1, H):
            L = Llo | H
            los = np.arange(1, H)
            hi = los ^ L
            P1 = S[np.ix_(los, los)].astype(np.int16) * S[np.ix_(hi, hi)].astype(np.int16)
            P3 = S[np.ix_(los, hi)].astype(np.int16) * S[np.ix_(hi, los)].astype(np.int16)
            ann = P1 == P3
            np.fill_diagonal(ann, False)
            A = A_sig_fast(n, Llo, S).astype(int)
            supp = A != 0
            bs += int((ann != supp).sum())
            tot += ann.size
            m = supp & ann
            bg += int((A[m] != (-P1)[m]).sum())
        print(f"C2b n={n}: {tot} entries over {H-1} labels -- support != {{P1=P3}} : {bs} | "
              f"A_sig != -P1 on the support : {bg}")

    # C3
    n = 4
    H = 1 << (n - 1)
    S = sign_table_fast(n)
    ok = 0
    for Llo in range(1, H):
        A = A_sig_fast(n, Llo, S).astype(int)
        deg = [int(np.count_nonzero(A[i])) for i in range(H - 1)]
        iso = [i + 1 for i in range(H - 1) if deg[i] == 0]
        nonadj = all(A[l - 1, (l ^ Llo) - 1] == 0
                     for l in range(1, H) if 1 <= (l ^ Llo) < H and (l ^ Llo) != l)
        ok += (iso == [Llo] and sorted(deg) == [0] + [4] * 6 and nonadj)
    print(f"C3  n=4: labels whose graph is K_6 minus the coset matching, plus the isolated "
          f"vertex l = Llo (the octahedron): {ok} of {H-1}")


if __name__ == "__main__":
    main()
