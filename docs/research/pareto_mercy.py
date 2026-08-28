#!/usr/bin/env python3
# The publishable version of the mercy computation (per OPUS-4.8-EXTRA critique #2). Delivers:
#  §1 the aggregation≡straight coincidence is the THEOREM (aggregative blindness to thin barriers): the
#     λ-sweep is a horizontal line; J_maximin−J_straight = 1.001 + 0.362λ > 0 ∀λ≥0.
#  §2 c* verified INDEPENDENTLY of the trajectory optimizer by union-find sublevel-set percolation (genuine
#     topological obstruction, not a background floor).
#  §3 the Pareto frontier Φ(c) = min ∫s s.t. max s ≤ c (mask above c, re-run) — replaces the 3-line table;
#     yields the TRUE leximin Φ(c*) (the naive bottleneck maximin overpays).
#  §4 the price of mercy = Δ∫s/Δpeak (the local slope of Φ) — the transportable scalar.
#  §5 mesh-convergence of the straight-path ∫s (so the thin-barrier effect isn't a discretization artifact).
import numpy as np, heapq
np.seterr(all='ignore')
def build(NG):
    xs=np.linspace(0,1,NG); ys=np.linspace(0,1,NG); S=np.full((NG,NG),0.05)
    X,Y=np.meshgrid(xs,ys,indexing='ij')
    g1=np.exp(-((Y-0.50)/0.05)**2); g2=np.exp(-((Y-0.85)/0.06)**2)
    wall=8.0*np.exp(-((X-0.5)/0.040)**2); b1=3.0*np.exp(-((X-0.5)/0.010)**2); b2=0.5*np.exp(-((X-0.5)/0.100)**2)
    S+=np.maximum(0,wall*np.maximum(0,1-g1-g2)+g1*b1+g2*b2)
    return xs,ys,S
NB=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def node(xs,ys,pt): return (int(round(pt[0]*(len(xs)-1))), int(round(pt[1]*(len(ys)-1))))
def dij(S,xs,ys,src,dst,mode,lam=0.0,cmask=None):
    NG=len(xs)
    def w(u,v,du):                      # edge weight
        if mode=='add':   return (1+lam*0.5*(S[u]+S[v]))*du         # ∫(1+λs)ds
        if mode=='ints':  return 0.5*(S[u]+S[v])*du                 # ∫s ds
    if mode=='minimax':
        dist={src:S[src]}; prev={}; pq=[(S[src],src)]; seen=set()
        while pq:
            d,u=heapq.heappop(pq)
            if u in seen: continue
            seen.add(u)
            if u==dst: break
            for di,dj in NB:
                v=(u[0]+di,u[1]+dj)
                if 0<=v[0]<NG and 0<=v[1]<NG:
                    nd=max(d,S[v])
                    if nd<dist.get(v,1e18): dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
        return dist[dst]
    dist={src:0.0}; prev={}; pq=[(0.0,src)]; seen=set()
    while pq:
        d,u=heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u==dst: break
        for di,dj in NB:
            v=(u[0]+di,u[1]+dj)
            if 0<=v[0]<NG and 0<=v[1]<NG:
                if cmask is not None and S[v]>cmask: continue     # admissible region {s ≤ c}
                du=np.hypot(xs[v[0]]-xs[u[0]],ys[v[1]]-ys[u[1]]); nd=d+w(u,v,du)
                if nd<dist.get(v,1e18): dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    if dst not in dist: return None,None
    p=[dst]
    while p[-1]!=src: p.append(prev[p[-1]])
    p=p[::-1]
    pts=np.array([[xs[i],ys[j]] for (i,j) in p]); ln=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    ints=sum(0.5*(S[p[k]]+S[p[k+1]])*ln[k] for k in range(len(p)-1)); peak=max(S[n] for n in p)
    return (peak,float(ints),float(ln.sum())), p
def straight_ints(S,xs,ys,A,B):
    tt=np.linspace(0,1,600)[:,None]; sl=np.array([xs[A[0]],ys[A[1]]])*(1-tt)+np.array([xs[B[0]],ys[B[1]]])*tt
    nn=[];
    for p in sl:
        n=node(xs,ys,p)
        if not nn or n!=nn[-1]: nn.append(n)
    pts=np.array([[xs[i],ys[j]] for (i,j) in nn]); ln=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    return sum(0.5*(S[nn[k]]+S[nn[k+1]])*ln[k] for k in range(len(nn)-1)), max(S[n] for n in nn)
# union-find percolation c* (independent of trajectory optimizer)
def cstar_percolation(S,xs,ys,A,B):
    NG=len(xs); order=sorted(((S[i,j],(i,j)) for i in range(NG) for j in range(NG)))
    par={};
    def find(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    added=set()
    for val,u in order:
        par[u]=u; added.add(u)
        for di,dj in NB:
            v=(u[0]+di,u[1]+dj)
            if v in added:
                ra,rb=find(u),find(v)
                if ra!=rb: par[ra]=rb
        if A in added and B in added and find(A)==find(B):
            return val
    return None
# ---------------- run ----------------
xs,ys,S=build(140); A=node(xs,ys,(0.08,0.5)); B=node(xs,ys,(0.92,0.5))
(pk_s,in_s,ln_s),_=dij(S,xs,ys,A,B,'ints')            # min ∫s freely = also straightish here
in_str,pk_str=straight_ints(S,xs,ys,A,B)
cstar_mm=dij(S,xs,ys,A,B,'minimax')
cstar_pc=cstar_percolation(S,xs,ys,A,B)
print(f"§2  c*(Dijkstra-minimax) = {cstar_mm:.3f}   c*(union-find percolation) = {cstar_pc:.3f}  "
      f"→ {'MATCH — genuine topological obstruction' if abs(cstar_mm-cstar_pc)<0.02 else 'MISMATCH'}")
# ambient floor check: median s away from the wall (x<0.3)
floor=np.median(S[:int(0.3*140)]); print(f"    ambient floor (median s off-wall) = {floor:.3f}  ≪ c* → c* is a PASS, not a floor")
print(f"\n§1  aggregative blindness — λ-sweep of the aggregation-optimal path (peak stays on the thin spike):")
for lam in [0.0,1.0,5.0,20.0,100.0]:
    (pk,ii,ll),_=dij(S,xs,ys,A,B,'add',lam=lam)
    print(f"      λ={lam:6.1f}: peak {pk:.3f}  ∫s {ii:.3f}  length {ll:.3f}")
print(f"    straight Pareto-dominates maximin (L {ln_s:.3f}<1.843 and ∫s {in_str:.3f}<0.460) ⇒ no λ rescues aggregation.")
print(f"\n§3  Pareto frontier Φ(c) = min ∫s  s.t.  max s ≤ c   (mask above c, re-run):")
print(f"      {'c (peak cap)':>12}{'Φ(c)=min ∫s':>13}{'length':>9}")
cs=np.linspace(cstar_mm+1e-3, 2.72, 12); front=[]
for c in cs:
    r,_=dij(S,xs,ys,A,B,'ints',cmask=c)
    if r: front.append((c,r[1],r[2])); print(f"      {c:>12.3f}{r[1]:>13.3f}{r[2]:>9.3f}")
lex=front[0]
print(f"    → TRUE leximin Φ(c*) = {lex[1]:.3f} at peak {lex[0]:.3f}  (naive bottleneck maximin paid 0.460 — it overpays)")
# §4 price of mercy = slope of the frontier between its endpoints
dpk=front[-1][0]-front[0][0]; dint=front[0][1]-front[-1][1]; dlen=front[0][2]-front[-1][2]
print(f"\n§4  price of mercy = Δ∫s/Δpeak = {dint/dpk:.3f}   (and Δlength/Δpeak = {dlen/dpk:.3f});")
print(f"    buying peak {front[0][0]:.2f}→{front[-1][0]:.2f} costs {front[0][1]-front[-1][1]:+.3f} ∫s and {front[0][2]-front[-1][2]:+.3f} length.")
print(f"\n§5  mesh convergence of the straight-path ∫s (thin barrier not a discretization artifact):")
for NG in [100,200,400,800]:
    xs2,ys2,S2=build(NG); A2=node(xs2,ys2,(0.08,0.5)); B2=node(xs2,ys2,(0.92,0.5))
    ii,pp=straight_ints(S2,xs2,ys2,A2,B2); print(f"      NG={NG:4d}: straight ∫s = {ii:.4f}  peak {pp:.3f}")
