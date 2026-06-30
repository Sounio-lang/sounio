#!/usr/bin/env python3
"""Apply the VALIDATED fair ablation to ABIDE FC: does the brain carry non-associative
(octonionic-associator) structure the matched associative model can't capture?

Both algebras get IDENTICAL features: per-fold PCA-64 of FC (linear signal, ~65% floor)
reshaped to 8 octonions -> overlapping triples -> both bracketings L,R per triple
+ quadratic over each triple's [L,R]. Toggle ONLY the product (octonion vs H(+)H).
Neither handed the associator. LOSO over 20 sites. PRE-COMMITTED:
  oct-assoc >= 10 -> non-assoc structure in brain ;  3..10 suggestive ;  <3 -> none (under this projection).
"""
import os, numpy as np
CACHE="/workspace/.tmp/claude-1000/-workspace-sounio/b70e058e-f1c5-424f-a527-da432d125564/scratchpad"
X=np.load(os.path.join(CACHE,"X.npy")); y=np.load(os.path.join(CACHE,"y.npy"))
sites=np.load(os.path.join(CACHE,"sites.npy"),allow_pickle=True)
print(f"loaded FC: {X.shape}, ASD={int((y==1).sum())} ctrl={int((y==0).sum())}",flush=True)

def oct_mul(a,b):
    r=np.empty(8)
    r[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4]-a[5]*b[5]-a[6]*b[6]-a[7]*b[7]
    r[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]+a[4]*b[5]-a[5]*b[4]-a[6]*b[7]+a[7]*b[6]
    r[2]=a[0]*b[2]+a[2]*b[0]-a[1]*b[3]+a[3]*b[1]+a[4]*b[6]-a[6]*b[4]+a[5]*b[7]-a[7]*b[5]
    r[3]=a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]+a[4]*b[7]-a[7]*b[4]-a[5]*b[6]+a[6]*b[5]
    r[4]=a[0]*b[4]+a[4]*b[0]-a[1]*b[5]+a[5]*b[1]-a[2]*b[6]+a[6]*b[2]-a[3]*b[7]+a[7]*b[3]
    r[5]=a[0]*b[5]+a[5]*b[0]+a[1]*b[4]-a[4]*b[1]-a[2]*b[7]+a[7]*b[2]+a[3]*b[6]-a[6]*b[3]
    r[6]=a[0]*b[6]+a[6]*b[0]+a[1]*b[7]-a[7]*b[1]+a[2]*b[4]-a[4]*b[2]-a[3]*b[5]+a[5]*b[3]
    r[7]=a[0]*b[7]+a[7]*b[0]-a[1]*b[6]+a[6]*b[1]-a[2]*b[5]+a[5]*b[2]+a[3]*b[4]-a[4]*b[3]
    return r
def quat_mul(a,b):
    return np.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
                     a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                     a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                     a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
def hh_mul(a,b): return np.concatenate([quat_mul(a[:4],b[:4]),quat_mul(a[4:],b[4:])])

TRIPLES=[(0,1,2),(2,3,4),(4,5,6),(6,7,0),(1,3,5),(2,4,6)]
def quad_self(M):
    d=M.shape[1]; cols=[M]
    for i in range(d): cols.append(M[:,i:i+1]*M[:,i:])
    return np.concatenate(cols,1)
def make_feats(P, mul):     # P: (n,64) -> 8 octonions
    n=len(P); O=P.reshape(n,8,8)
    raw=P
    LRs=[]; quads=[]
    for (i,j,k) in TRIPLES:
        L=np.empty((n,8)); R=np.empty((n,8))
        for s in range(n):
            L[s]=mul(mul(O[s,i],O[s,j]),O[s,k]); R[s]=mul(O[s,i],mul(O[s,j],O[s,k]))
        LRs.append(L); LRs.append(R)
        quads.append(quad_self(np.concatenate([L,R],1)))
    return np.concatenate([raw]+LRs+quads,1)
def ridge(Xtr,ytr,Xte,lam=20.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; t=np.where(ytr==1,1.,-1.)
    w=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]),Xtr.T@t); return (Xte@w>=0).astype(int)
def lin_ridge(Xtr,ytr,Xte,lam=1.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; t=np.where(ytr==1,1.,-1.); n=Xtr.shape[0]
    a=np.linalg.solve(Xtr@Xtr.T+lam*n*np.eye(n),t); return ((Xte@Xtr.T)@a>=0).astype(int)
def bal(p,yt):
    tpr=((p==1)&(yt==1)).sum()/max(1,(yt==1).sum()); tnr=((p==0)&(yt==0)).sum()/max(1,(yt==0).sum())
    return 50*(tpr+tnr)

usites=sorted(set(sites.tolist()))
res={"linear PCA64":[], "octonion":[], "H(+)H assoc":[]}
for s in usites:
    te=sites==s; tr=~te
    if te.sum()==0 or len(set(y[tr].tolist()))<2 or len(set(y[te].tolist()))<2: continue
    Xtr,Xte=X[tr],X[te]
    mu=Xtr.mean(0,keepdims=True); Xc=Xtr-mu
    _,_,Vt=np.linalg.svd(Xc,full_matrices=False); B=Vt[:64].T
    Ptr=Xc@B; Pte=(Xte-mu)@B
    pmu=Ptr.mean(0,keepdims=True); psd=Ptr.std(0,keepdims=True); psd[psd<1e-8]=1
    Ptr=(Ptr-pmu)/psd; Pte=(Pte-pmu)/psd
    res["linear PCA64"].append(bal(lin_ridge(Ptr,y[tr],Pte),y[te]))
    for name,mul in [("octonion",oct_mul),("H(+)H assoc",hh_mul)]:
        Ftr=make_feats(Ptr,mul); Fte=make_feats(Pte,mul)
        res[name].append(bal(ridge(Ftr,y[tr],Fte),y[te]))
    print(f"site {s:12s} lin={res['linear PCA64'][-1]:.1f} oct={res['octonion'][-1]:.1f} assoc={res['H(+)H assoc'][-1]:.1f}",flush=True)

print("\n==== LOSO mean balanced accuracy (ABIDE FC, fair ablation) ====")
for k in ["linear PCA64","octonion","H(+)H assoc"]:
    a=np.array(res[k]); print(f"  {k:16s} {a.mean():.2f} +/- {a.std():.2f}")
o=np.array(res["octonion"]).mean(); h=np.array(res["H(+)H assoc"]).mean()
print(f"\n  oct - assoc = {o-h:+.2f}")
print("  PRE-COMMITTED: >=10 non-assoc structure | 3..10 suggestive | <3 none (under PCA64 projection)")
