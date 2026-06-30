#!/usr/bin/env python3
"""HONEST RE-TEST: was the octonion 99.5% vs real 55.7% a polynomial-DEGREE
artifact, not evidence for the specific non-associative algebra?

The label = ||assoc(A,B,C)|| is degree-3 in the 24 inputs (norm^2 is degree-6).
My earlier 'real reservoir' was degree<=2 ([tanh(P), P*P]) -> structurally cannot
fit a degree-6 target. Here: give the GENERIC REAL baseline matched/higher degree
and see if it closes the gap. If it does, octonion non-associativity per se added
little; if it doesn't, the specific algebra is doing real work.
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
def assoc_norm(a,b,c): return np.linalg.norm(oct_mul(oct_mul(a,b),c)-oct_mul(a,oct_mul(b,c)))

N=4000; TRIALS=5
def gen(seed):
    rg=np.random.default_rng(seed)
    A=rg.standard_normal((N,8)); A/=np.linalg.norm(A,axis=1,keepdims=True)
    B=rg.standard_normal((N,8)); B/=np.linalg.norm(B,axis=1,keepdims=True)
    C=rg.standard_normal((N,8)); C/=np.linalg.norm(C,axis=1,keepdims=True)
    X=np.concatenate([A,B,C],1)
    an=np.array([assoc_norm(A[i],B[i],C[i]) for i in range(N)])
    y=(an>np.median(an)).astype(int)
    return X,y,A,B,C
def balacc(p,y):
    tpr=((p==1)&(y==1)).sum()/max(1,(y==1).sum()); tnr=((p==0)&(y==0)).sum()/max(1,(y==0).sum())
    return 50*(tpr+tnr)
def ridge(Xtr,ytr,Xte,lam=1.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; t=np.where(ytr==1,1.,-1.)
    w=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]),Xtr.T@t); return (Xte@w>=0).astype(int)
def poly_feats(X,seed,K,deg):
    rg=np.random.default_rng(seed); W=rg.standard_normal((X.shape[1],K))/np.sqrt(X.shape[1])
    P=X@W; return np.concatenate([P**k for k in range(1,deg+1)],1)

models={"real deg2 (old)":(2,128),"real deg3":(3,128),"real deg4":(4,256),"real deg6":(6,256),
        "real deg6 wide":(6,512)}
res={k:[] for k in models}; res["octonion (assoc feat)"]=[]
for t in range(TRIALS):
    X,y,A,B,C=gen(100+t); idx=np.random.default_rng(t).permutation(N); cut=int(.7*N); tr,te=idx[:cut],idx[cut:]
    OF=np.array([assoc_norm(A[i],B[i],C[i]) for i in range(N)]).reshape(-1,1)
    res["octonion (assoc feat)"].append(balacc(ridge(OF[tr],y[tr],OF[te]),y[te]))
    for name,(deg,K) in models.items():
        F=poly_feats(X,7+t,K,deg); res[name].append(balacc(ridge(F[tr],y[tr],F[te]),y[te]))

print(f"Target = octonion associator-norm (degree-3 form). N={N}, {TRIALS} trials. Chance=50.\n")
order=["octonion (assoc feat)","real deg2 (old)","real deg3","real deg4","real deg6","real deg6 wide"]
for k in order:
    a=np.array(res[k]); print(f"  {k:24s} {a.mean():6.2f} +/- {a.std():4.2f}")
print("\nIf real deg>=3 approaches octonion -> earlier 99.5-vs-55.7 was a DEGREE artifact,")
print("not evidence the specific non-associative algebra beats generic nonlinearity.")
