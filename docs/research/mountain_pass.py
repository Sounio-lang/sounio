#!/usr/bin/env python3
# The mountain-pass definition of mercy (per the OPUS-4.8-EXTRA critique §4), made visible.
# On a suffering field s(x) where start and goal are separated by a ridge, EVERY admissible path crosses
# it, and c* = min_γ max_t s(γ(t)) is the mountain-pass / transition-state level (Ambrosetti–Rabinowitz).
#   necessary suffering := c*            (a property of the GEOMETRY, not of policy)
#   gratuitous suffering(γ) := max_t s(γ) − c*   (excess imputable to the chosen trajectory)
#   mercy := achieving c*
# The ethics is the CHOICE OF FUNCTIONAL, not the algebra:
#   • aggregation (utilitarian): minimize ∫(1+λs) ds   — a sharp peak can be bought with calm travel
#   • maximin (Rawlsian):        minimize max_t s(γ)    — no instant of agony bought with comfort
#   • leximin: minimize the peak first (→ c*), then ∫s among peak-optimal paths (recommended)
# Synthetic field: a wall between A and B with TWO gaps — a NEAR/high one (short) and a FAR/low one (the
# true pass). Aggregation is tempted by the near gap; maximin insists on the low pass. Same field.
import numpy as np, heapq
np.seterr(all='ignore')
NG=140; xs=np.linspace(0,1,NG); ys=np.linspace(0,1,NG)
def field():
    # near pass (y≈0.5): THIN + TALL spike → brief acute agony (low ∫s, high peak), on the short route.
    # far pass  (y≈0.85): WIDE + LOW hump → prolonged mild discomfort (high ∫s, low peak), via a detour.
    # elsewhere: an impassable wall. This is the utilitarian-vs-Rawlsian tension on one field.
    S=np.full((NG,NG),0.05)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            g1=np.exp(-((y-0.50)/0.05)**2); g2=np.exp(-((y-0.85)/0.06)**2)
            wall=8.0*np.exp(-((x-0.5)/0.040)**2)
            b1 =3.0*np.exp(-((x-0.5)/0.010)**2)          # thin tall
            b2 =0.5*np.exp(-((x-0.5)/0.100)**2)          # wide low
            w=wall*max(0.0,1-g1-g2)+g1*b1+g2*b2
            S[i,j]+=w
    return S
S=field()
def node(pt): return (int(round(pt[0]*(NG-1))), int(round(pt[1]*(NG-1))))
A=node((0.08,0.5)); B=node((0.92,0.5))
NB=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def coord(n): return np.array([xs[n[0]],ys[n[1]]])
def dijkstra_add(src,dst,lam):                     # minimize ∫(1+λs)ds  (aggregation / utilitarian)
    dist={src:0.0}; prev={}; pq=[(0.0,src)]; seen=set()
    while pq:
        d,u=heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u==dst: break
        i,j=u
        for di,dj in NB:
            v=(i+di,j+dj)
            if 0<=v[0]<NG and 0<=v[1]<NG:
                w=(1+lam*0.5*(S[u]+S[v]))*np.linalg.norm(coord(v)-coord(u))
                nd=d+w
                if nd<dist.get(v,1e18): dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    p=[dst]
    while p[-1]!=src: p.append(prev[p[-1]])
    return p[::-1]
def dijkstra_minimax(src,dst):                      # minimize max_t s(γ)  (maximin / Rawlsian) → c*
    dist={src:S[src]}; prev={}; pq=[(S[src],src)]; seen=set()
    while pq:
        d,u=heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u==dst: break
        i,j=u
        for di,dj in NB:
            v=(i+di,j+dj)
            if 0<=v[0]<NG and 0<=v[1]<NG:
                nd=max(d,S[v])
                if nd<dist.get(v,1e18): dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    p=[dst]
    while p[-1]!=src: p.append(prev[p[-1]])
    return p[::-1], dist[dst]
def metrics(path):
    pts=np.array([coord(n) for n in path]); ln=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    peak=max(S[n] for n in path)
    integ=sum(0.5*(S[path[k]]+S[path[k+1]])*ln[k] for k in range(len(path)-1))
    return peak, float(integ), float(ln.sum())
# straight line (reward/efficiency reference)
tt=np.linspace(0,1,300)[:,None]; sl=coord(A)*(1-tt)+coord(B)*tt
sl_nodes=[];
for p in sl:
    n=node(p)
    if not sl_nodes or n!=sl_nodes[-1]: sl_nodes.append(n)
agg=dijkstra_add(A,B,lam=2.0)
mm,cstar=dijkstra_minimax(A,B)
pk_s,in_s,ln_s=metrics(sl_nodes); pk_a,in_a,ln_a=metrics(agg); pk_m,in_m,ln_m=metrics(mm)
print(f"c*  (mountain-pass / NECESSARY suffering) = {cstar:.3f}   [the lowest saddle any path must cross]")
print(f"\n{'path':<26}{'peak (max s)':>13}{'∫s ds':>10}{'length':>9}{'gratuitous = peak−c*':>22}")
print(f"{'STRAIGHT (reward)':<26}{pk_s:>13.3f}{in_s:>10.3f}{ln_s:>9.3f}{pk_s-cstar:>22.3f}")
print(f"{'AGGREGATION min∫(1+λs)':<26}{pk_a:>13.3f}{in_a:>10.3f}{ln_a:>9.3f}{pk_a-cstar:>22.3f}")
print(f"{'MAXIMIN min max s (→c*)':<26}{pk_m:>13.3f}{in_m:>10.3f}{ln_m:>9.3f}{pk_m-cstar:>22.3f}")
# leximin: among near-c* paths minimize ∫s — approximate by minimax then note it already achieves c*
print(f"\nReading:")
print(f"  • necessary suffering c* = {cstar:.3f} is fixed by the GEOMETRY (the ridge), not by the ethics.")
print(f"  • the AGGREGATION path takes the near/high gap: peak {pk_a:.2f} — it BUYS a shorter/cheaper trip")
print(f"    with a higher agony peak → gratuitous suffering {pk_a-cstar:.2f} by the maximin standard.")
print(f"  • the MAXIMIN path insists on the low pass: peak = c* = {cstar:.3f}, gratuitous 0 — at a longer")
print(f"    trip (∫s {in_m:.2f} vs {in_a:.2f}). Same field s; the divergence IS the ethical choice.")
print(f"  • Dabrowski without contradiction: positive disintegration = crossing the pass; mercy is not")
print(f"    avoiding it (impossible — start & goal are separated) but finding the LOWEST saddle (c*).")
print(f"  • leximin (recommended): minimize the peak first (→ c*), then ∫s among peak-optimal paths —")
print(f"    anti-aggregationist (no agony bought with comfort) yet still duration-sensitive.")
