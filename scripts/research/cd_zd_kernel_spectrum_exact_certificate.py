"""Exact certification of dim ker(L_u) at n = 64 and n = 128, without ever
doing rational elimination on a 128x128 matrix.

Why not just redo the RREF with Fraction: at n = 128 there are 4032 index pairs
and each elimination blows the numerators up. And a single-prime modular result
can never be made safe by a bound here -- the Hadamard bound for a 128x128
matrix with |a_ij| <= 2 is astronomically beyond any word-sized prime.

The certificate instead uses a fact that holds for EVERY prime:

    rank over F_P  <=  rank over Q          (a Q-dependency stays dependent mod P;
                                             the converse can fail)

so                dim ker over Q  <=  dim ker over F_P.

The modular kernel dimension is therefore an UPPER bound, for free and with no
assumption about the entries. To pin the value exactly it is enough to exhibit
that many kernel vectors and verify them over the integers:

  1. take the modular kernel basis,
  2. lift each vector to integers in the symmetric range and clear denominators,
  3. verify  M . v == 0  EXACTLY over Z  -- no modulus,
  4. verify the lifted vectors are independent over Q, by integer-only
     Bareiss elimination on the 4 x n stack.

(3) gives dim ker_Q >= (number of verified vectors); (1) gives <=. Equality
follows, and every step is integer arithmetic.
"""
import importlib.util
spec = importlib.util.spec_from_file_location("oracle", "/tmp/oracle.py")
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
P = o.P

def sym(x):
    x %= P
    return x - P if x > P // 2 else x

def bareiss_rank(rows, n):
    """Exact rank over Q of an integer matrix, integer arithmetic only."""
    M = [r[:] for r in rows]; m = len(M)
    if m == 0: return 0
    prev = 1; r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if M[i][c] != 0: p = i; break
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(r + 1, m):
            for j in range(c + 1, n):
                M[i][j] = (M[i][j] * M[r][c] - M[i][c] * M[r][j]) // prev
            M[i][c] = 0
        prev = M[r][c]; r += 1
        if r == m: break
    return r

def certify(K, verbose=False):
    tab = o.build_fast(K); n = 1 << K; h = n // 2
    counts = {}; bad = []
    for a in range(1, h):
        for b in range(h, n):
            M = [[0]*n for _ in range(n)]
            for j in range(n):
                s, k = tab[a][j]; M[k][j] += s
                s, k = tab[b][j]; M[k][j] += s
            Mm = [[x % P for x in row] for row in M]
            R, piv = o._rref(Mm, n)
            free = [c for c in range(n) if c not in piv]
            d_mod = len(free)
            # lift the modular kernel basis to integers
            vecs = []
            for f in free:
                v = [0]*n; v[f] = 1
                for ri, c in enumerate(piv): v[c] = sym(-R[ri][f])
                vecs.append(v)
            # (3) M . v == 0 exactly over Z
            ok = 0
            for v in vecs:
                if all(sum(M[i][j]*v[j] for j in range(n)) == 0 for i in range(n)):
                    ok += 1
            # (4) independence over Q by integer Bareiss
            indep = bareiss_rank(vecs, n) if vecs else 0
            d_low = min(ok, indep)
            if d_low != d_mod:
                bad.append((a, b, d_mod, d_low))
            counts[d_mod] = counts.get(d_mod, 0) + 1
    return n, counts, bad

import sys
for K in (3, 4, 5, 6, 7):
    n, counts, bad = certify(K)
    print(f"  n={n:4d}  dim ker (modular, = upper bound): {dict(sorted(counts.items()))}")
    if bad:
        print(f"          NAO CERTIFICADO em {len(bad)} pares: {bad[:5]}")
    else:
        print(f"          CERTIFICADO sobre Z/Q: cada dim atingida por vetores")
        print(f"          verificados exatamente e independentes. limite superior = inferior.")
    sys.stdout.flush()
