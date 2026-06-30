#!/usr/bin/env python3
"""DECISIVE ABLATION: does NON-ASSOCIATIVITY itself help, isolated from capacity?

Same architecture, same param count, both TRAINED, NEITHER handed the generative
feature. Toggle ONLY the algebra product:
  - octonion  : 8-dim, NON-associative (division algebra)
  - H (+) H   : 8-dim, ASSOCIATIVE control (two quaternions; same dim, same #params)

Model = tuned 8-dim recurrence (reservoir): state h_t = unit((A * h_{t-1}) * x_t),
A is a learnable 8-dim algebra element (8 params, identical for both). Readout =
linear/ridge on FINAL STATE ONLY (8 features). 'Training' = optimize A by random
search + local refine on TRAIN balanced-acc, then fit readout. No associator feature
is ever given to either model.

PRE-COMMITTED decision rule:
  non-associativity helps  <=>  on task NA: oct - assoc >= 10 pts
                                AND on tasks A/neutral: |oct - assoc| <= 3 pts.
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
def hh_mul(a,b):  # H (+) H : associative, 8-dim
    return np.concatenate([quat_mul(a[:4],b[:4]), quat_mul(a[4:],b[4:])])
def assoc_norm(a,b,c): return np.linalg.norm(oct_mul(oct_mul(a,b),c)-oct_mul(a,oct_mul(b,c)))
def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-9 else v

N=2000; TRIALS=5; SEQ=3
def gen(seed):
    rg=np.random.default_rng(seed)
    S=[unit(rg.standard_normal(8)) for _ in range(SEQ)]  # placeholder
    Xs=np.stack([rg.standard_normal((N,8)) for _ in range(SEQ)],1)  # (N,SEQ,8)
    for i in range(N):
        for t in range(SEQ): Xs[i,t]=unit(Xs[i,t])
    # task labels
    an=np.array([assoc_norm(Xs[i,0],Xs[i,1],Xs[i,2]) for i in range(N)])
    y_na=(an>np.median(an)).astype(int)                       # NON-associative structure
    w=rg.standard_normal(8)
    lin=(Xs[:,0]+Xs[:,1]+Xs[:,2])@w
    y_a=(lin>np.median(lin)).astype(int)                      # associative/linear
    W2=rg.standard_normal((8,8))
    quad=np.einsum('ni,ij,nj->n',Xs[:,0],W2,Xs[:,1])          # generic bilinear (neutral)
    y_neu=(quad>np.median(quad)).astype(int)
    return Xs,{"NA (non-assoc)":y_na,"A (assoc/linear)":y_a,"neutral (bilinear)":y_neu}

def feats(Xs,A,mul):
    F=np.empty((len(Xs),8))
    for i in range(len(Xs)):
        h=Xs[i,0].copy()
        for t in range(1,SEQ): h=unit(mul(mul(A,h),Xs[i,t]))
        F[i]=h
    return F
def ridge(Xtr,ytr,Xte,lam=1.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; t=np.where(ytr==1,1.,-1.)
    w=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]),Xtr.T@t)
    return Xtr@w, Xte@w
def bal(p,y):
    tpr=((p==1)&(y==1)).sum()/max(1,(y==1).sum()); tnr=((p==0)&(y==0)).sum()/max(1,(y==0).sum())
    return 50*(tpr+tnr)

def train_eval(Xs,y,mul,tr,te,seed,n_cand=200):
    rg=np.random.default_rng(seed); best=None; bestA=None
    Ftr_cache={}
    for c in range(n_cand):
        A=unit(rg.standard_normal(8))
        F=feats(Xs,A,mul)
        s_tr,_=ridge(F[tr],y[tr],F[te]); acc=bal((s_tr>=0).astype(int),y[tr])
        if best is None or acc>best: best=acc; bestA=A
    F=feats(Xs,bestA,mul)
    _,s_te=ridge(F[tr],y[tr],F[te])
    return bal((s_te>=0).astype(int),y[te])

algos={"octonion (NON-assoc)":oct_mul, "H(+)H (assoc ctrl)":hh_mul}
tasks=["NA (non-assoc)","A (assoc/linear)","neutral (bilinear)"]
res={a:{t:[] for t in tasks} for a in algos}
for trial in range(TRIALS):
    Xs,ys=gen(100+trial)
    idx=np.random.default_rng(trial).permutation(N); cut=int(.7*N); tr,te=idx[:cut],idx[cut:]
    for aname,mul in algos.items():
        for t in tasks:
            res[aname][t].append(train_eval(Xs,ys[t],mul,tr,te,seed=trial*10))
    print(f"trial {trial+1}/{TRIALS} done",flush=True)

print(f"\nTrained matched-capacity ablation. N={N}, seq={SEQ}, {TRIALS} trials. Chance=50.\n")
print(f"{'task':22s} {'octonion':>16s} {'H(+)H assoc':>16s} {'oct-assoc':>10s}")
for t in tasks:
    o=np.array(res['octonion (NON-assoc)'][t]); h=np.array(res['H(+)H (assoc ctrl)'][t])
    print(f"{t:22s} {o.mean():6.2f}+/-{o.std():4.2f}   {h.mean():6.2f}+/-{h.std():4.2f}   {o.mean()-h.mean():+8.2f}")
print("\nPRE-COMMITTED RULE: non-assoc helps <=> NA: oct-assoc>=10  AND  A&neutral: |oct-assoc|<=3")
