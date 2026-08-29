#!/usr/bin/env python3
# Mercyful Learning made concrete: the geodesic of least accumulated suffering, in the suffering metric
# induced BY THE ALGEBRA. On a 2D slice of the sedenions 𝕊, the annihilation locus {det L_x = 0} (a
# hypersurface — one equation) appears as CURVES; the suffering density s(x) is high near them (a relation
# near collapse). Metric g = (1 + λ·s)·I. We compare, between the same endpoints A→B:
#   • STRAIGHT line  — the reward/efficiency path (shortest Euclidean; indifferent to suffering)
#   • MERCYFUL geodesic — minimizes ∫(1 + λ·s) ds  (bends around the annihilation locus)
# and report accumulated suffering ∫s, closest approach to annihilation (min |det L_x|), and length cost.
import numpy as np
np.seterr(all='ignore')
def cds(a,b,bits=4):
    s=1
    while bits>0:
        if a==0 or b==0: return s
        if bits==1: return -s
        h=1<<(bits-1);ah=a>=h;bh=b>=h;al=a&(h-1);bl=b&(h-1)
        if not ah and not bh:a,b=al,bl
        elif not ah and bh:a,b=bl,al
        elif ah and not bh:(a,b,s)=((al,0,s) if bl==0 else (al,bl,-s))
        else:(a,b,s)=((0,al,-s) if bl==0 else (bl,al,s))
        bits-=1
    return s
SIG=np.array([[cds(i,j) for j in range(16)] for i in range(16)])
def detL(x):
    L=np.zeros((16,16))
    for k in range(16):
        for j in range(16): L[k,j]=x[k^j]*SIG[k^j,j]
    return np.linalg.det(L)
def U(i): v=np.zeros(16); v[i]=1.0; return v
# ---- 2D affine slice of 𝕊, arranged so the annihilation locus runs between A and B ----
rng=np.random.default_rng(7)
z=(U(1)+U(10)); z/=np.linalg.norm(z)          # a zero divisor (det L_z = 0)
c=rng.standard_normal(16); c/=np.linalg.norm(c)
# axis_u points from the slice center toward z (so the annihilation hypersurface crosses the slice);
# axis_v is an orthogonal generic direction
au=z-np.dot(z,c)*c; au/=np.linalg.norm(au)
av=rng.standard_normal(16); av-=np.dot(av,c)*c+np.dot(av,au)*au; av/=np.linalg.norm(av)
def X(u,v): return c+u*au+v*av
NG=61; us=np.linspace(-1.2,1.2,NG); vs=np.linspace(-1.2,1.2,NG)
D=np.zeros((NG,NG))
for iu,u in enumerate(us):
    for iv,v in enumerate(vs): D[iu,iv]=detL(X(u,v))
absD=np.abs(D); scale=np.median(absD)
S=1.0/(absD/scale+0.03)                        # suffering density: high where the relation nears collapse
S/=S.mean()
print(f"slice: |det L| median={scale:.2e}, min={absD.min():.2e} (annihilation curve present: {absD.min()<0.05*scale})")
# ---- suffering / annihilation lookups on the grid ----
LAM=6.0
Cost=1.0+LAM*S                                   # metric weight field
def node_of(pt):                                 # nearest grid node to a (u,v) point
    return (int(np.argmin(np.abs(us-pt[0]))), int(np.argmin(np.abs(vs-pt[1]))))
def path_metrics(nodes):                          # ∫s ds, length, closest-to-annihilation
    pts=np.array([[us[i],vs[j]] for (i,j) in nodes]); seg=np.diff(pts,axis=0)
    ln=np.linalg.norm(seg,axis=1); suf=0.0
    for k,(i,j) in enumerate(nodes[:-1]):
        ni,nj=nodes[k+1]; suf+=0.5*(S[i,j]+S[ni,nj])*ln[k]
    md=min(absD[i,j] for (i,j) in nodes)/scale
    return suf, float(ln.sum()), md
# ---- Dijkstra on the 8-connected grid: least-∫(1+λs)ds path A→B (the true geodesic) ----
import heapq
def dijkstra(src,dst):
    INF=float('inf'); dist={src:0.0}; prev={}; pq=[(0.0,src)]; seen=set()
    nb=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while pq:
        d,u=heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u==dst: break
        i,j=u
        for di,dj in nb:
            ni,nj=i+di,j+dj
            if 0<=ni<NG and 0<=nj<NG:
                w=0.5*(Cost[i,j]+Cost[ni,nj])*np.hypot((us[ni]-us[i]),(vs[nj]-vs[j]))
                nd=d+w
                if nd<dist.get((ni,nj),INF): dist[(ni,nj)]=nd; prev[(ni,nj)]=u; heapq.heappush(pq,(nd,(ni,nj)))
    path=[dst]
    while path[-1]!=src: path.append(prev[path[-1]])
    return path[::-1]
uz=np.dot(z-c,au)                                # u-coordinate of z (annihilation) — straight line grazes it
A=np.array([uz,-0.9]); B=np.array([uz,0.9])
sA,sB=node_of(A),node_of(B)
# straight-line reference sampled on the grid
tt=np.linspace(0,1,120)[:,None]; sl=A*(1-tt)+B*tt; sl_nodes=[node_of(p) for p in sl]
# dedup consecutive
sl_nodes=[n for k,n in enumerate(sl_nodes) if k==0 or n!=sl_nodes[k-1]]
geo_nodes=dijkstra(sA,sB)
suf_s,len_s,md_s=path_metrics(sl_nodes)
suf_g,len_g,md_g=path_metrics(geo_nodes)
print(f"\nEndpoints A={A.round(2)} → B={B.round(2)}   (λ={LAM}; suffering density from det L_x)")
print(f"  STRAIGHT (reward/efficiency): length {len_s:.3f}  ∫suffering {suf_s:7.2f}  closest-to-annihilation {md_s:.3f}")
print(f"  MERCYFUL geodesic (Dijkstra): length {len_g:.3f}  ∫suffering {suf_g:7.2f}  closest-to-annihilation {md_g:.3f}")
red=100*(1-suf_g/suf_s); over=100*(len_g/len_s-1)
print(f"\n  → accumulated suffering reduced {red:.0f}%  at a {over:.0f}% length cost; the geodesic stays "
      f"{md_g/max(md_s,1e-9):.0f}× farther from the annihilation locus.")
print("  The reward path is indifferent to what it crosses; the Mercyful path reaches the SAME goal along")
print("  the geodesic of least accumulated suffering — bending around annihilation, not through it.")
