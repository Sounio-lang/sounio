#!/usr/bin/env python3
"""DECISIVE TEST of "octonion restriction = feature, in its proper domain".

Removes the domain-mismatch confound by GENERATING data whose class label IS
the octonion associator. If the octonion inductive bias is ever useful, it must
win HERE, where the data literally has octonionic non-associative structure.

Two tasks (double dissociation):
  A) OCTONIONIC target: label = (||[a,b,c]|| > median).  [a,b,c]=(ab)c - a(bc),
     a degree-3 octonion invariant. Linear/quadratic real features are ~blind.
  B) ASSOCIATIVE/linear target: label = (<w, a+b+c> > median). Pure linear signal.

Models (all get the SAME raw 24 real inputs = 3 octonions x 8 comps):
  - linear        : logistic/ridge on raw 24                (real baseline)
  - real-reservoir: K random FIXED quadratic features + linear readout
                    (matched-capacity GENERIC nonlinearity, real-valued)
  - octonion      : fixed octonion product -> ||associator|| feature + linear
                    (the O-SSM inductive bias: the specific Cayley-Dickson rule)

Thesis prediction if octonion non-associativity is a real inductive bias:
  Task A: octonion >> linear AND octonion > real-reservoir (specific algebra beats
          generic nonlinearity at matched capacity).
  Task B: linear >= octonion (no spurious octonion advantage on associative signal).
"""
import numpy as np

rng = np.random.default_rng(0)
N = 4000
N_TRIALS = 5

def oct_mul(a, b):
    r = np.empty(8)
    r[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4]-a[5]*b[5]-a[6]*b[6]-a[7]*b[7]
    r[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]+a[4]*b[5]-a[5]*b[4]-a[6]*b[7]+a[7]*b[6]
    r[2]=a[0]*b[2]+a[2]*b[0]-a[1]*b[3]+a[3]*b[1]+a[4]*b[6]-a[6]*b[4]+a[5]*b[7]-a[7]*b[5]
    r[3]=a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]+a[4]*b[7]-a[7]*b[4]-a[5]*b[6]+a[6]*b[5]
    r[4]=a[0]*b[4]+a[4]*b[0]-a[1]*b[5]+a[5]*b[1]-a[2]*b[6]+a[6]*b[2]-a[3]*b[7]+a[7]*b[3]
    r[5]=a[0]*b[5]+a[5]*b[0]+a[1]*b[4]-a[4]*b[1]-a[2]*b[7]+a[7]*b[2]+a[3]*b[6]-a[6]*b[3]
    r[6]=a[0]*b[6]+a[6]*b[0]+a[1]*b[7]-a[7]*b[1]+a[2]*b[4]-a[4]*b[2]-a[3]*b[5]+a[5]*b[3]
    r[7]=a[0]*b[7]+a[7]*b[0]-a[1]*b[6]+a[6]*b[1]-a[2]*b[5]+a[5]*b[2]+a[3]*b[4]-a[4]*b[3]
    return r

def assoc_norm(a, b, c):
    return np.linalg.norm(oct_mul(oct_mul(a, b), c) - oct_mul(a, oct_mul(b, c)))

def gen(seed):
    rg = np.random.default_rng(seed)
    A = rg.standard_normal((N, 8)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = rg.standard_normal((N, 8)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    C = rg.standard_normal((N, 8)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    X = np.concatenate([A, B, C], axis=1)                 # (N,24) raw inputs
    an = np.array([assoc_norm(A[i], B[i], C[i]) for i in range(N)])
    yA = (an > np.median(an)).astype(int)                 # octonionic target
    w = rg.standard_normal(8)
    lin = (A + B + C) @ w
    yB = (lin > np.median(lin)).astype(int)               # associative/linear target
    return X, yA, yB, A, B, C

def balacc(pred, y):
    tpr = ((pred==1)&(y==1)).sum()/max(1,(y==1).sum())
    tnr = ((pred==0)&(y==0)).sum()/max(1,(y==0).sum())
    return 50.0*(tpr+tnr)

def fit_linear(Xtr, ytr, Xte, lam=1.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1.0
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    t=np.where(ytr==1,1.0,-1.0)
    w=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]), Xtr.T@t)
    return (Xte@w>=0).astype(int)

def real_reservoir_feats(X, seed, K=64):
    # generic real fixed nonlinearity of matched capacity: random quadratic features
    rg=np.random.default_rng(seed)
    W1=rg.standard_normal((X.shape[1],K))/np.sqrt(X.shape[1])
    P=X@W1
    return np.concatenate([np.tanh(P), P*P], axis=1)       # 2K nonlinear feats

def octonion_feat(A,B,C):
    an=np.array([assoc_norm(A[i],B[i],C[i]) for i in range(len(A))])
    return an.reshape(-1,1)                                 # the inductive-bias feature

res={k:{"A":[],"B":[]} for k in ["linear","real_reservoir","octonion"]}
for t in range(N_TRIALS):
    X,yA,yB,A,B,C = gen(100+t)
    idx=np.random.default_rng(t).permutation(N); cut=int(0.7*N)
    tr,te=idx[:cut],idx[cut:]
    for tag,y in [("A",yA),("B",yB)]:
        # linear on raw
        res["linear"][tag].append(balacc(fit_linear(X[tr],y[tr],X[te]),y[te]))
        # real reservoir (generic nonlinearity, matched-ish capacity)
        RF=real_reservoir_feats(X,seed=7+t)
        res["real_reservoir"][tag].append(balacc(fit_linear(RF[tr],y[tr],RF[te]),y[te]))
        # octonion inductive bias: associator-norm feature + linear
        OF=octonion_feat(A,B,C)
        res["octonion"][tag].append(balacc(fit_linear(OF[tr],y[tr],OF[te]),y[te]))

print(f"N={N} per task, {N_TRIALS} trials, 70/30 split. Chance=50.\n")
print(f"{'model':16s} {'Task A (octonionic assoc)':28s} {'Task B (linear/associative)'}")
for k in ["linear","real_reservoir","octonion"]:
    a=np.array(res[k]['A']); b=np.array(res[k]['B'])
    print(f"{k:16s} {a.mean():6.2f} +/- {a.std():4.2f}            {b.mean():6.2f} +/- {b.std():4.2f}")
print("\nThesis needs: Task A octonion >> linear AND octonion > real_reservoir;")
print("              Task B linear >= octonion (no spurious octonion edge).")
