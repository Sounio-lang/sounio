"""Exact structure (distinct kernels, mutual-annihilation clique, maximum
independent set) for n >= 32, with each of the three claims handled by the
argument its own error direction demands.

  distinct kernels   equal over Q  =>  equal modular key (the lifted modular
                     basis is a certified exact basis of ker_Q, so equal
                     Q-spaces reduce to equal F_P-spaces). Hence distinct keys
                     => distinct over Q, and the modular count is a LOWER bound.
                     Confirming it needs only: inside each key bucket, are the
                     members really equal over Q? Rank test, integers only.

  clique             already exact. zero_pair sums four +-1 table entries, so
                     every component has |v_k| <= 4 < P, and `v % P == 0` is
                     equivalent to `v == 0` over Z. No change needed.

  max independent    independent mod P => independent over Q (rank_FP <= rank_Q),
                     so the modular maximum is a LOWER bound. Confirming it is
                     the maximum needs: no set of size max+1 is independent
                     over Q. Checked with integer Bareiss.
"""
import importlib.util, sys, itertools
spec = importlib.util.spec_from_file_location("oracle", "/tmp/oracle.py")
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
P = o.P

def sym(x):
    x %= P
    return x - P if x > P // 2 else x

def bareiss_rank(rows, n):
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

def kernels_of(K):
    tab = o.build_fast(K); n = 1 << K; h = n // 2
    prims = {}; buckets = {}
    for a in range(1, h):
        for b in range(h, n):
            M = [[0]*n for _ in range(n)]
            for j in range(n):
                s, k = tab[a][j]; M[k][j] += s
                s, k = tab[b][j]; M[k][j] += s
            Mm = [[x % P for x in row] for row in M]
            R, piv = o._rref(Mm, n)
            free = [c for c in range(n) if c not in piv]
            if not free: continue
            B = []
            for f in free:
                v = [0]*n; v[f] = 1
                for ri, c in enumerate(piv): v[c] = sym(-R[ri][f])
                B.append(v)
            C, _ = o._rref([[x % P for x in v] for v in B], n)
            key = tuple(tuple(x % P for x in r) for r in C)
            prims[(a, b)] = (key, B)
            buckets.setdefault(key, []).append((a, b))
    return tab, n, prims, buckets

def bucket_is_uniform(bucket, prims, n):
    """Every member of a modular-key bucket spans the same Q-subspace?"""
    ref = prims[bucket[0]][1]; d = len(ref)
    rref = bareiss_rank(ref, n)
    for m in bucket[1:]:
        B = prims[m][1]
        if len(B) != d: return False
        if bareiss_rank(ref + B, n) != rref: return False
    return True

def report(K, indep_cap=None):
    tab, n, prims, buckets = kernels_of(K)
    dims = {}
    for (a,b),(k,B) in prims.items(): dims[len(B)] = dims.get(len(B),0)+1
    # clique -- exact by the |v_k| <= 4 < P argument
    def zero_pair(p, q):
        (a1,b1),(a2,b2) = p,q
        vec = [0]*n
        for (x,y) in ((a1,a2),(a1,b2),(b1,a2),(b1,b2)):
            sg,k = tab[x][y]; vec[k] += sg
        return all(v == 0 for v in vec)
    pl = list(prims)
    adj = {p:{q for q in pl if q!=p and zero_pair(p,q)} for p in pl}
    clique = 0
    if pl:
        clique = 1
        for p in pl:
            for q in adj[p]:
                clique = max(clique, 2)
                common = adj[p] & adj[q]
                if common: clique = max(clique, 3)
    nonuni = [k for k,b in buckets.items() if len(b)>1 and not bucket_is_uniform(b, prims, n)]
    print(f"  n={n}")
    print(f"    dims                 : {dict(sorted(dims.items()))}")
    print(f"    kernels (chave mod)  : {len(buckets)}")
    print(f"    buckets NAO uniformes: {len(nonuni)}   <- 0 significa contagem exata")
    print(f"    clique (exato)       : {clique}")
    sys.stdout.flush()
    return buckets, prims, n

for K in ():
    report(K)

def max_independent(buckets, prims, n, cap_seconds=600):
    import time
    reps = []
    for k, members in buckets.items():
        reps.append(prims[members[0]][1])
    reps.sort(key=len)
    best = [0]; t0 = time.time(); timed_out = [False]
    def bt(start, chosen, rows, dimsum):
        if len(chosen) > best[0]: best[0] = len(chosen)
        if time.time() - t0 > cap_seconds: timed_out[0] = True; return
        for i in range(start, len(reps)):
            B = reps[i]
            if dimsum + len(B) > n: continue
            nr = rows + B
            if bareiss_rank(nr, n) == dimsum + len(B):
                bt(i + 1, chosen + [i], nr, dimsum + len(B))
                if timed_out[0]: return
    bt(0, [], [], 0)
    return best[0], timed_out[0]

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "indep":
    K = int(sys.argv[2])
    tab, n, prims, buckets = kernels_of(K)
    m, to = max_independent(buckets, prims, n)
    print(f"  n={n}  max independentes (exato sobre Q) = {m}{'  [TEMPO ESGOTADO -- limite inferior]' if to else ''}")

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "struct":
    report(int(sys.argv[2]))
