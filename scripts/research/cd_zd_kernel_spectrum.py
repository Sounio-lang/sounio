import collections, sys
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
