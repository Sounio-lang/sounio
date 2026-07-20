#!/usr/bin/env python3
# align(k) as a CURVE — the shoulder distinguishes annihilation from low effective rank (OPUS-4.8-EXTRA).
# Alignment ≈1 at one k is ambiguous: shared DEAD subspace (annihilation, small dead dim + healthy bulk) OR
# low effective rank (large shared weak complement). Shuffled-weights & untrained-init nulls are full-rank
# → they miss the low-rank confounder. The fix needs no new null: sweep k, and read the SHOULDER position.
#   annihilation  : high for k ≤ m (small), DROPS for k > m — a shoulder at small k, healthy bulk above
#   low eff. rank : high up to k = d − r (LARGE), shoulder only at large k
#   nothing       : flat at the baseline √(k/d)
# The shoulder POSITION is the datum (the dead-subspace dimension), not a chosen k.
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
SIG=np.array([[cds(i,j) for j in range(16)] for i in range(16)]);M=np.zeros((16,16,16))
for k in range(16):
    for b in range(16): M[k,k^b,b]=SIG[k,b]
def Lx(x): return np.tensordot(x,M,axes=(0,0))
def normed(A): return A/np.linalg.svd(A,compute_uv=False)[0]
rng=np.random.default_rng(1); D=12; dim=16
def botV(A,k): return np.linalg.svd(A)[2][-k:]
def align_curve(mats,ks):
    out=[]
    for k in ks:
        cs=[np.linalg.svd(botV(mats[l],k)@botV(mats[l+1],k).T,compute_uv=False).mean() for l in range(len(mats)-1)]
        out.append(np.mean(cs))
    return np.array(out)
# (1) genuine annihilation: aligned near-ZD sedenions — dead dim 4, healthy bulk 12
z=np.zeros(16); z[1]=1; z[10]=1; z/=np.linalg.norm(z)
def near(delta):
    r=rng.standard_normal(16); r-=(r@z)*z; r/=np.linalg.norm(r); x=z+delta*r; return x/np.linalg.norm(x)
sed=[normed(Lx(near(0.15))) for _ in range(D)]
# (2) low effective rank: signal in r rotating dims, a SHARED dead complement Q_dead of dim d−r
r=6; Qd=np.linalg.qr(rng.standard_normal((dim,dim)))[0][:,r:]     # shared dead subspace (dim 10)
def lowrank():
    Usig=np.linalg.qr(rng.standard_normal((dim,dim)))[0][:,:r]     # rotating signal dirs
    Usig=Usig-Qd@(Qd.T@Usig); Usig=np.linalg.qr(Usig)[0][:,:r]     # orthogonal to the shared dead subspace
    sig=np.diag(np.abs(rng.standard_normal(r))+0.5)
    A=Usig@sig@Usig.T + 1e-4*Qd@rng.standard_normal((dim-r,dim))   # rank-r signal + tiny on Q_dead
    return normed(A)
low=[lowrank() for _ in range(D)]
# (3) nothing: real Gaussian (full rank)
gau=[normed(rng.standard_normal((16,16))) for _ in range(D)]
ks=list(range(1,16))
base=np.sqrt(np.array(ks)/dim)
print("align(k) — mean cos(principal angle) of the bottom-k subspaces between consecutive Jacobians:")
print(f"  {'k':>3} " + " ".join(f"{k:>4}" for k in ks))
for name,mats in [('ANNIHILATION (dead≈4)',sed),('LOW-RANK (r=6, dead≈10)',low),('GAUSSIAN (nothing)',gau)]:
    a=align_curve(mats,ks)
    print(f"  {name:22s} " + " ".join(f"{v:4.2f}" for v in a))
print(f"  {'baseline √(k/d)':22s} " + " ".join(f"{v:4.2f}" for v in base))
# shoulder detection: largest drop in align(k) above baseline
def shoulder(mats):
    a=align_curve(mats,ks); excess=a-base; drop=excess[:-1]-excess[1:]; ki=int(np.argmax(drop))
    return ks[ki], round(float(a[ki]),2), round(float(a[ki+1]),2)
print("\nshoulder (k where align drops most, above baseline) — the dead-subspace dimension:")
for name,mats in [('ANNIHILATION',sed),('LOW-RANK',low),('GAUSSIAN',gau)]:
    k,hi,lo=shoulder(mats); print(f"  {name:14s}: shoulder at k={k}  (align {hi}→{lo})")
print("\n→ a shoulder at SMALL k with a healthy bulk above = annihilation; shoulder only at LARGE k = low")
print("  effective rank; no shoulder (flat on baseline) = nothing. The confounder is resolved by the CURVE.")
