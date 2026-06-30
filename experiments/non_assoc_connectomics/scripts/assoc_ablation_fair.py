#!/usr/bin/env python3
"""FAIR retest: identical quadratic readout giving BOTH algebras degree-3 access,
neither handed the associator. Toggle ONLY the product (octonion vs H(+)H assoc).

Recipe (identical for both algebras): for triple (x0,x1,x2) compute the TWO
bracketings L=(x0*x1)*x2 and R=x0*(x1*x2). Features = linear[x0,x1,x2,L,R]
PLUS quadratic terms over the [L,R] block (so ||L-R||^2 is *expressible* but never
handed). For octonion L!=R (non-assoc) -> material exists; for H(+)H L=R exactly
(assoc) -> the extra block is redundant. Any octonion win => from non-associativity.

PRE-COMMITTED (unchanged): non-assoc helps <=> NA: oct-assoc>=10 AND A/neutral |oct-assoc|<=3.
"""
import numpy as np
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
def assoc_norm(a,b,c): return np.linalg.norm(oct_mul(oct_mul(a,b),c)-oct_mul(a,oct_mul(b,c)))
def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-9 else v

N=2000; TRIALS=5
def gen(seed):
    rg=np.random.default_rng(seed)
    X=np.stack([rg.standard_normal((N,8)) for _ in range(3)],1)
    for i in range(N):
        for t in range(3): X[i,t]=unit(X[i,t])
    an=np.array([assoc_norm(X[i,0],X[i,1],X[i,2]) for i in range(N)])
    y_na=(an>np.median(an)).astype(int)
    w=rg.standard_normal(8); y_a=(((X[:,0]+X[:,1]+X[:,2])@w)>0).astype(int)
    y_a=(((X[:,0]+X[:,1]+X[:,2])@w)>np.median((X[:,0]+X[:,1]+X[:,2])@w)).astype(int)
    W2=rg.standard_normal((8,8)); q=np.einsum('ni,ij,nj->n',X[:,0],W2,X[:,1])
    y_neu=(q>np.median(q)).astype(int)
    return X,{"NA (non-assoc)":y_na,"A (assoc/linear)":y_a,"neutral (bilinear)":y_neu}

def quad_block(M):  # quadratic (incl. squares) over columns of M
    n,d=M.shape; cols=[M]
    for i in range(d):
        cols.append(M[:,i:i+1]*M[:,i:])
    return np.concatenate(cols,1)

def features(X,mul):
    n=len(X); L=np.empty((n,8)); R=np.empty((n,8))
    for i in range(n):
        x0,x1,x2=X[i,0],X[i,1],X[i,2]
        L[i]=mul(mul(x0,x1),x2); R[i]=mul(x0,mul(x1,x2))
    lin=np.concatenate([X[:,0],X[:,1],X[:,2],L,R],1)      # 40 (identical recipe)
    quad=quad_block(np.concatenate([L,R],1))             # quadratic over [L,R] (16->152)
    return np.concatenate([lin,quad],1)

def ridge(Xtr,ytr,Xte,lam=5.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; t=np.where(ytr==1,1.,-1.)
    w=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]),Xtr.T@t); return (Xte@w>=0).astype(int)
def bal(p,y):
    tpr=((p==1)&(y==1)).sum()/max(1,(y==1).sum()); tnr=((p==0)&(y==0)).sum()/max(1,(y==0).sum())
    return 50*(tpr+tnr)

algos={"octonion (NON-assoc)":oct_mul,"H(+)H (assoc ctrl)":hh_mul}
tasks=["NA (non-assoc)","A (assoc/linear)","neutral (bilinear)"]
res={a:{t:[] for t in tasks} for a in algos}
for trial in range(TRIALS):
    X,ys=gen(100+trial); idx=np.random.default_rng(trial).permutation(N); cut=int(.7*N); tr,te=idx[:cut],idx[cut:]
    F={a:features(X,m) for a,m in algos.items()}
    for a in algos:
        for t in tasks:
            res[a][t].append(bal(ridge(F[a][tr],ys[t][tr],F[a][te]),ys[t][te]))
    print(f"trial {trial+1}/{TRIALS} done",flush=True)

print(f"\nFAIR ablation (quadratic readout, both can express deg-3; neither handed assoc).")
print(f"N={N}, {TRIALS} trials. Chance=50.\n")
print(f"{'task':22s} {'octonion':>15s} {'H(+)H assoc':>15s} {'oct-assoc':>10s}")
for t in tasks:
    o=np.array(res['octonion (NON-assoc)'][t]); h=np.array(res['H(+)H (assoc ctrl)'][t])
    print(f"{t:22s} {o.mean():6.2f}+/-{o.std():4.2f}  {h.mean():6.2f}+/-{h.std():4.2f}  {o.mean()-h.mean():+8.2f}")
print("\nPRE-COMMITTED RULE: non-assoc helps <=> NA: oct-assoc>=10  AND  A&neutral: |oct-assoc|<=3")
