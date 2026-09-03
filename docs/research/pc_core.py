# Shared analysis core v2: fields, sweep, permuted null WITH empirical p, positive control (sensitivity floor).
import numpy as np
from sklearn.neighbors import NearestNeighbors
def mutual_knn(E,k):
    _,idx=NearestNeighbors(n_neighbors=k+1,metric='cosine').fit(E).kneighbors(E)
    nbr=[set(r[1:]) for r in idx]; adj=[[] for _ in range(len(E))]
    for i in range(len(E)):
        for j in nbr[i]:
            if i in nbr[j]: adj[i].append(j)
    return adj
def sweep(order,adj):
    n=len(order); par=list(range(n)); sz=[1]*n; pres=np.zeros(n,bool)
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    largest=0; rho=np.empty(n)
    for m,v in enumerate(order,1):
        pres[v]=True
        for u in adj[v]:
            if pres[u]:
                a,b=find(u),find(v)
                if a!=b:
                    if sz[a]<sz[b]: a,b=b,a
                    par[b]=a; sz[a]+=sz[b]
        largest=max(largest,sz[find(v)]); rho[m-1]=largest/m
    return rho
def _perm_bank(adjv,n,rng,B):
    P=np.empty((B,n))
    for b in range(B): P[b]=sweep(rng.permutation(n),adjv)
    return P
def analyze_field(s,adj,rng,B=200):
    """Returns dict incl empirical p (min over c of frac perms as-or-more disconnected), longest below-band run."""
    vi=np.where(~np.isnan(s))[0]; remap={v:i for i,v in enumerate(vi)}
    adjv=[[remap[u] for u in adj[v] if u in remap] for v in vi]; sv=s[vi]; n=len(vi)
    rho=sweep(np.argsort(sv,kind='stable'),adjv)
    P=_perm_bank(adjv,n,rng,B); lo=np.percentile(P,2.5,0); med=np.percentile(P,50,0)
    below=rho<lo; start=max(1,int(0.05*n)); best=cur=0
    for m in range(start,n):
        cur=cur+1 if below[m] else 0
        if cur>best: best=cur
    T=lambda r: float(np.maximum(0.0, med[start:]-r[start:]).sum())          # integrated deficit below null median
    Tobs=T(rho); Tp=np.array([T(P[b]) for b in range(B)])
    p_corr=float((1+np.sum(Tp>=Tobs))/(B+1))                                  # c-sweep-corrected permutation p
    return dict(n=int(n),delta_min=float((rho-med)[start:].min()),longest_below_frac=float(best/max(1,n-start)),
                p_corrected=p_corr, rho=rho, null_lo=lo)
def fiedler_separator(adj,n):
    """Spectral bisection -> vertex separator (R-side boundary): removing it disconnects the graph into 2 large comps."""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    rows=[];cols=[]
    for i in range(n):
        for j in adj[i]:
            rows.append(i);cols.append(j)
    if not rows: return None,None,None
    A=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(float)
    deg=np.asarray(A.sum(1)).ravel(); L=sp.diags(deg)-A
    try:
        vals,vecs=spla.eigsh(L,k=2,which='SM',maxiter=5000)
        f=vecs[:,1]
    except Exception:
        return None,None,None
    Lset=set(np.where(f<=np.median(f))[0].tolist()); Rset=set(range(n))-Lset
    sep=[i for i in Rset if any(j in Lset for j in adj[i])]     # R nodes adjacent to L = vertex cut
    Lsize=len(Lset); Rint=len(Rset)-len(sep)
    return sep, Lsize, Rint
def positive_control(s_base,adj,rng,deltas=(4,3,2,1.5,1.0,0.75,0.5,0.35,0.25,0.15),B=200):
    """Inject barrier: raise s by delta*std on a Fiedler vertex separator -> {s<=c} splits in 2. Sweep delta -> delta*.
    Detection = longest below-null run >= 0.10 (an order above the trivial ~0.03 floor)."""
    vi=np.where(~np.isnan(s_base))[0]; remap={v:i for i,v in enumerate(vi)}
    adjv=[[remap[u] for u in adj[v] if u in remap] for v in vi]; n=len(vi); sb=s_base[vi].astype(float)
    sep,Lsz,Rint=fiedler_separator(adjv,n)
    if sep is None or Lsz<0.15*n or Rint<0.15*n or len(sep)==0:
        return dict(ok=False,reason=f"no balanced vertex separator (L={Lsz},Rint={Rint},|sep|={len(sep) if sep else 0})")
    sd=sb.std()+1e-9; out=[]; dstar=None
    for d in deltas:
        s2=sb.copy(); s2[sep]+=d*sd
        r=analyze_field_arr(s2,adjv,rng,B)
        det=r['longest_below_frac']>=0.10
        out.append((float(d),r['longest_below_frac'],float(r['p_min']),det))
        if det: dstar=float(d)                      # smallest detected so far (list descends)
    return dict(ok=True, sep_frac=len(sep)/n, L_frac=Lsz/n, Rint_frac=Rint/n, sweep=out, delta_star=dstar)
def analyze_field_arr(s,adjv,rng,B):
    n=len(s); rho=sweep(np.argsort(s,kind='stable'),adjv)
    P=_perm_bank(adjv,n,rng,B); lo=np.percentile(P,2.5,0); med=np.percentile(P,50,0)
    below=rho<lo; start=max(1,int(0.05*n)); best=cur=0
    for m in range(start,n):
        cur=cur+1 if below[m] else 0; best=max(best,cur)
    T=lambda r: float(np.maximum(0.0, med[start:]-r[start:]).sum())
    Tobs=T(rho); Tp=np.array([T(P[b]) for b in range(B)]); p=float((1+np.sum(Tp>=Tobs))/(B+1))
    return dict(longest_below_frac=float(best/max(1,n-start)), p_min=p)
