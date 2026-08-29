#!/usr/bin/env python3
# THE FAITHFUL BRIDGE: an octonion-valued path signature. The naive static octonion associator of 3
# net increments failed to see the Borromean/Massey signal (BORROMEAN_AINFINITY.md, 48.9%). Here the
# non-associativity is put into the TEMPORAL iterated structure: embed the path increments as imaginary
# octonions g_t (units e1,e2,e4 — a NON-Fano triple, [e1,e2,e4]!=0), and build the octonion signature by
# ordered octonion products. Depth-3 has TWO bracketings whose difference is a genuine iterated associator
#   D = sum_{r<s<t} ((g_r g_s) g_t) - (g_r (g_s g_t)) = sum_{r<s<t} [g_r,g_s,g_t]   (8-dim octonion)
# Question: does the temporal octonion signature capture the level-3 Massey invariant mu_k=∫A_ij dX^k that
# the static associator missed? Honest either way. Tested on the same pairwise-trivial (Borromean) slice.
import numpy as np
np.seterr(all='ignore')
def cds(a,b,bits=3):
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
SIG=np.array([[cds(i,j) for j in range(8)] for i in range(8)],float)
def omul(A,B):                       # (...,8)x(...,8)->(...,8)
    C=np.zeros(np.broadcast_shapes(A.shape,B.shape))
    for i in range(8):
        for j in range(8): C[...,i^j]+=SIG[i,j]*A[...,i]*B[...,j]
    return C
# sanity: [e1,e2,e4] != 0 (non-Fano), [e1,e2,e3]==0 (quaternion triple)
def unit(k): v=np.zeros(8);v[k]=1.0;return v
assoc=lambda a,b,c: omul(omul(a,b),c)-omul(a,omul(b,c))
print(f"non-Fano [e1,e2,e4] norm = {np.linalg.norm(assoc(unit(1),unit(2),unit(4))):.2f} (expect !=0); "
      f"Fano [e1,e2,e3] norm = {np.linalg.norm(assoc(unit(1),unit(2),unit(3))):.2f} (expect 0)")
# ---- Levy areas + Massey mu (the label), same as BORROMEAN_AINFINITY ----
def invariants(X):
    dX=np.diff(X,0);P=np.zeros(3);a12=a13=a23=0.0;mu1=mu2=mu3=0.0
    for t in range(dX.shape[0]):
        d=dX[t];mu1+=a23*d[0];mu2+=a13*d[1];mu3+=a12*d[2]
        a12+=0.5*(P[0]*d[1]-P[1]*d[0]);a13+=0.5*(P[0]*d[2]-P[2]*d[0]);a23+=0.5*(P[1]*d[2]-P[2]*d[1]);P+=d
    return np.array([a12,a13,a23]),np.array([mu1,mu2,mu3])
# ---- octonion-valued path signature to depth 3 (both bracketings) ----
def oct_signature(X):
    dX=np.diff(X,0); g=np.zeros((dX.shape[0],8))
    g[:,1]=dX[:,0]; g[:,2]=dX[:,1]; g[:,4]=dX[:,2]         # embed R^3 -> Im octonion (e1,e2,e4)
    T=g.shape[0]
    P=np.zeros((T,8)); acc=np.zeros(8)                     # P[t]=sum_{s<t} g_s
    for t in range(T): P[t]=acc; acc=acc+g[t]
    S1=g.sum(0)
    pg=omul(P,g)                                           # P[t]·g[t]
    S2=pg.sum(0)                                           # sum_{s<t} g_s g_t  (one octonion)
    Q=np.zeros((T,8)); acc=np.zeros(8)                     # Q[t]=sum_{s<t}(P[s]·g[s])
    for t in range(T): Q[t]=acc; acc=acc+pg[t]
    S3L=omul(Q,g).sum(0)                                   # sum_{r<s<t} (g_r g_s) g_t
    # S3R = sum_t sum_{s<t} P[s]·(g[s]·g[t])   (O(T^2), vectorized over s per t)
    S3R=np.zeros(8)
    for t in range(1,T):
        gg=omul(g[:t], g[t][None,:])                       # (t,8) = g_s·g_t
        S3R+=omul(P[:t], gg).sum(0)
    D=S3L-S3R                                              # iterated associator sum_{r<s<t}[g_r,g_s,g_t]
    return S1,S2,S3L,S3R,D
# ---- dataset (same generator/slice/label as the A∞ experiment) ----
def make_path(rng,T=96):
    t=np.linspace(0,1,T)[:,None];X=np.zeros((T,3))
    for i in range(3):
        for f in (1,2,3): X[:,i]+=rng.standard_normal()*np.sin(2*np.pi*f*t[:,0]+rng.uniform(0,2*np.pi))
    return X
rng=np.random.default_rng(20260719); N=5000
paths=[make_path(rng) for _ in range(N)]
INV=[invariants(X) for X in paths]
areas=np.array([a for a,_ in INV]); mu=np.array([m for _,m in INV])
amag=np.abs(areas).max(1); slice_idx=np.argsort(amag)[:N//3]
mutot=mu.sum(1); y=(mutot[slice_idx]>np.median(mutot[slice_idx])).astype(float)
print(f"slice {len(slice_idx)}/{N}  max|area|={amag[slice_idx].max():.3f}  mu std={mutot[slice_idx].std():.3f}")
print("computing octonion signatures on the slice ...",flush=True)
OS=[oct_signature(paths[i]) for i in slice_idx]
S1=np.array([o[0] for o in OS]); S2=np.array([o[1] for o in OS])
S3L=np.array([o[2] for o in OS]); S3R=np.array([o[3] for o in OS]); D=np.array([o[4] for o in OS])
print(f"iterated-associator D: std={D.std():.3e} max|D|={np.abs(D).max():.3e}  (nonzero => genuine depth-3 term)")
# correlation of the associator-signature with the Massey label direction
from numpy.linalg import lstsq
def corr_with_mu(F):
    Fc=F-F.mean(0); m=mutot[slice_idx]-mutot[slice_idx].mean()
    w=lstsq(Fc,m,rcond=None)[0]; pred=Fc@w;
    return np.corrcoef(pred,m)[0,1]
print(f"corr(D , mu) = {corr_with_mu(D):.3f}   corr([S1,S2,S3L,S3R,D] , mu) = {corr_with_mu(np.column_stack([S1,S2,S3L,S3R,D])):.3f}")
ns=len(slice_idx);perm=rng.permutation(ns);tr,te=perm[:ns*7//10],perm[ns*7//10:]
def logistic(X,epochs=1200,lr=0.3,l2=1e-3):
    Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te];mu_=Xtr.mean(0);sd=Xtr.std(0)+1e-9;Xtr=(Xtr-mu_)/sd;Xte=(Xte-mu_)/sd
    w=np.zeros(Xtr.shape[1]);b=0.0
    for _ in range(epochs):
        p=1/(1+np.exp(-(Xtr@w+b)));g=p-ytr;w-=lr*(Xtr.T@g/len(ytr)+l2*w);b-=lr*g.mean()
    return (((Xte@w+b)>0).astype(float)==yte).mean()
print("Faithful octonion path-signature vs the Borromean/Massey label (chance=50%):")
print(f"  OCT-DEV   [S1,S2] (depth<=2 octonion)              : {100*logistic(np.column_stack([S1,S2])):.1f}%")
print(f"  OCT-ASSOC D = iterated associator (depth-3, faithful): {100*logistic(D):.1f}%")
print(f"  OCT-FULL  [S1,S2,S3L,S3R,D]                        : {100*logistic(np.column_stack([S1,S2,S3L,S3R,D])):.1f}%")
