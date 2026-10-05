import collections, sys

# prime modulus for rank computations. NOTE (2026-08-23 correction,
# see docs/audit/KERNEL_SPECTRUM_ORACLE_DOMAIN_2026-08-23.md): entries being
# 0,+-1 does NOT imply rank over F_P equals rank over Q — a minor can be
# nonzero over Q and divisible by P (16x16 Hadamard bound 2^32 = 2.00x P).
# rank_{F_P} <= rank_Q holds for every prime, so this modular path yields a
# genuine UPPER bound on the rank, i.e. a lower bound on dim ker. The exact
# authority is cd_zd_kernel_spectrum_exact_certificate.py (integer-only,
# Bareiss independence + M.v = 0 verification, valid for every prime).
P = (1 << 31) - 1
def build_fast(K):
    """basis table for dim 2^K via CD recursion on tables. tab[i][j]=(sign,k)."""
    tab=[[(1,0)]]                      # dim 1: e0*e0 = e0
    for lev in range(K):
        n=len(tab); N=2*n
        new=[[None]*N for _ in range(N)]
        def cj(i):  # conj sign for basis e_i in dim n
            return 1 if i==0 else -1
        for i in range(N):
            for j in range(N):
                a,al = i%n, i//n     # i = a + al*n
                c,be = j%n, j//n
                if al==0 and be==0:
                    s,k = tab[a][c]; new[i][j]=(s,k)
                elif al==0 and be==1:
                    # (a,0)(0,d) = (0, d a)
                    s,k = tab[c][a]; new[i][j]=(s,k+n)
                elif al==1 and be==0:
                    # (0,b)(c,0) = (0, b conj(c))
                    s,k = tab[a][c]; new[i][j]=(s*cj(c), k+n)
                else:
                    # (0,b)(0,d) = (-conj(d) b, 0)
                    s,k = tab[c][a]; new[i][j]=(-s*cj(c), k)
        tab=new
    return tab
def cplus_values(K):
    tab=build_fast(K); n=1<<K; half=n//2
    vals=collections.Counter()
    for a in range(1,half):
        for b in range(half,n):
            perm=[0]*n; sgn=[0]*n
            for j in range(n):
                s1,k1=tab[b][j]; s2,k2=tab[a][k1]
                perm[j]=k2; sgn[j]=-s1*s2
            seen=bytearray(n); c=0; bad=False
            for st in range(n):
                if seen[st]: continue
                L=0;pr=1;x=st
                while not seen[x]:
                    seen[x]=1; pr*=sgn[x]; x=perm[x]; L+=1
                if L!=2: bad=True; break
                if pr==1: c+=1
            vals[(-1 if bad else c)]+=1
    return n, vals
# sanity vs earlier exhaustive results, then push higher
for K in (3,4,5,6,7):
    n,v = cplus_values(K)
    print(f"dim {n:4d}: c+ values {sorted(v)}  counts {dict(v)}", flush=True)


# ---------------------------------------------------------------------------
# Structural claims of the accompanying document, reproduced here so that every
# statement in it is checkable from this one file.
#   - degeneracy rule:  c+ = 0  <=>  b' = 0 or a = b'      (exhaustive)
#   - distinct 4-dim kernels, mutual-annihilation clique, independent kernels
# ---------------------------------------------------------------------------
def _rref(M, n):
    R = [r[:] for r in M]; r = 0; piv = []
    for c in range(n):
        q = None
        for i in range(r, len(R)):
            if R[i][c] % P: q = i; break
        if q is None: continue
        R[r], R[q] = R[q], R[r]
        inv = pow(R[r][c], P - 2, P); R[r] = [(x * inv) % P for x in R[r]]
        for i in range(len(R)):
            if i != r and R[i][c] % P:
                f = R[i][c]; R[i] = [(a - f * b) % P for a, b in zip(R[i], R[r])]
        piv.append(c); r += 1
    return R[:r], piv

def check_degeneracy(K):
    """c+ = 0 exactly on b'=0 or a=b'. Exhaustive over all primitives."""
    tab = build_fast(K); n = 1 << K; h = n // 2; bad = 0; tot = 0
    for a in range(1, h):
        for b in range(h, n):
            perm = [0]*n; sgn = [0]*n
            for j in range(n):
                s1, k1 = tab[b][j]; s2, k2 = tab[a][k1]
                perm[j] = k2; sgn[j] = -s1*s2
            seen = bytearray(n); c = 0
            for st in range(n):
                if seen[st]: continue
                pr = 1; x = st
                while not seen[x]:
                    seen[x] = 1; pr *= sgn[x]; x = perm[x]
                if pr == 1: c += 1
            tot += 1
            if (c == 0) != (b - h == 0 or a == b - h): bad += 1
    return tot, bad

def structure(K):
    """Distinct dim-4 kernels, mutual-annihilation clique, max independent set.

    NOTE: this enumerates index PAIRS (a,b). Each pair yields two primitives,
    e_a + e_b and e_a - e_b, which share the same kernel (dim ker does not
    depend on the sign s -- see Theorem A). So `pairs` here is half the
    primitive count quoted in the document: 42 pairs <-> 84 primitives at n=16.
    """
    tab = build_fast(K); n = 1 << K; h = n // 2
    prims = {}; kernels = {}
    for a in range(1, h):
        for b in range(h, n):
            M = [[0]*n for _ in range(n)]
            for j in range(n):
                s, k = tab[a][j]; M[k][j] = (M[k][j] + s) % P
                s, k = tab[b][j]; M[k][j] = (M[k][j] + s) % P
            R, piv = _rref(M, n); free = [c for c in range(n) if c not in piv]
            if len(free) != 4: continue
            B = []
            for f in free:
                v = [0]*n; v[f] = 1
                for ri, c in enumerate(piv): v[c] = (-R[ri][f]) % P
                B.append(v)
            C, _ = _rref(B, n); key = tuple(tuple(x % P for x in r) for r in C)
            kernels[key] = 1; prims[(a, b)] = key
    KL = list(kernels)
    # mutual annihilation: u*v = 0 AND v*u = 0, over primitive index pairs
    def zero_pair(p, q):
        (a1, b1), (a2, b2) = p, q
        vec = [0]*n
        for (x, y, s) in ((a1, a2, 1), (a1, b2, 1), (b1, a2, 1), (b1, b2, 1)):
            sg, k = tab[x][y]; vec[k] = (vec[k] + s*sg) % P
        return all(v % P == 0 for v in vec)
    pl = list(prims)
    adj = {p: {q for q in pl if q != p and zero_pair(p, q)} for p in pl}
    clique = 1 if pl else 0
    for p in pl:
        for q in adj[p]:
            if clique < 2: clique = 2
            if any((r in adj[p]) and (r in adj[q]) for r in adj[p]): clique = max(clique, 3)
    # maximum independent kernels, exhaustive backtracking
    best = [0]
    def bt(start, k, rows):
        if k > best[0]: best[0] = k
        if k == n // 4: return True
        for i in range(start, len(KL)):
            if k + (len(KL) - i) <= best[0]: return False
            t = rows + [list(r) for r in KL[i]]
            if len(_rref(t, n)[0]) == 4*(k+1):
                if bt(i+1, k+1, t): return True
        return False
    bt(0, 0, [])
    return len(prims), len(KL), clique, best[0]

if __name__ == "__main__":
    print("\n--- degeneracy rule (exhaustive) ---")
    for K in (4, 5, 6):
        tot, bad = check_degeneracy(K)
        print(f"dim {1<<K:4d}: {tot} index pairs (= {2*tot} primitives) checked, "
              f"mismatches {bad}")
    print("\n--- structure at n = 16 (exhaustive) ---")
    np_, nk, clq, ind = structure(4)
    print(f"dim 16: {np_} ZD index pairs (= {2*np_} primitives), "
          f"{nk} distinct dim-4 kernels, "
          f"max mutual-annihilation clique {clq}, max independent kernels {ind}")
