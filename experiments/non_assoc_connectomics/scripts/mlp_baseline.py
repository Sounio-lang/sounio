#!/usr/bin/env python3
"""Fair generic-nonlinearity baseline: a TRAINED MLP (vs random features).
Can a generic learnable nonlinearity discover the octonion associator target,
and at what capacity? This is the honest test the random-feature baseline missed.
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

N=4000
def gen(seed):
    rg=np.random.default_rng(seed)
    A=rg.standard_normal((N,8)); A/=np.linalg.norm(A,axis=1,keepdims=True)
    B=rg.standard_normal((N,8)); B/=np.linalg.norm(B,axis=1,keepdims=True)
    C=rg.standard_normal((N,8)); C/=np.linalg.norm(C,axis=1,keepdims=True)
    X=np.concatenate([A,B,C],1)
    an=np.array([assoc_norm(A[i],B[i],C[i]) for i in range(N)])
    return X,(an>np.median(an)).astype(int)
def balacc(p,y):
    tpr=((p==1)&(y==1)).sum()/max(1,(y==1).sum()); tnr=((p==0)&(y==0)).sum()/max(1,(y==0).sum())
    return 50*(tpr+tnr)

def mlp(Xtr,ytr,Xte,yte,H=(256,256),epochs=300,lr=2e-3,seed=0):
    rg=np.random.default_rng(seed)
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    d=Xtr.shape[1]; sizes=[d]+list(H)+[1]
    Ws=[rg.standard_normal((sizes[i],sizes[i+1]))*np.sqrt(2/sizes[i]) for i in range(len(sizes)-1)]
    bs=[np.zeros(s) for s in sizes[1:]]
    mW=[np.zeros_like(w) for w in Ws]; vW=[np.zeros_like(w) for w in Ws]
    mb=[np.zeros_like(b) for b in bs]; vb=[np.zeros_like(b) for b in bs]
    t01=ytr.astype(float); b1=0.9; b2=0.999; eps=1e-8; step=0
    n=len(Xtr); bs_sz=256
    for ep in range(epochs):
        perm=rg.permutation(n)
        for s in range(0,n,bs_sz):
            idx=perm[s:s+bs_sz]; x=Xtr[idx]; yt=t01[idx]
            acts=[x];
            for i,(W,b) in enumerate(zip(Ws,bs)):
                z=acts[-1]@W+b
                acts.append(np.maximum(z,0) if i<len(Ws)-1 else z)
            logit=acts[-1][:,0]; p=1/(1+np.exp(-np.clip(logit,-30,30)))
            g=(p-yt)/len(idx); grad=g[:,None]
            step+=1
            for i in reversed(range(len(Ws))):
                gW=acts[i].T@grad; gb=grad.sum(0)
                if i>0: grad=(grad@Ws[i].T)*(acts[i]>0)
                mW[i]=b1*mW[i]+(1-b1)*gW; vW[i]=b2*vW[i]+(1-b2)*gW*gW
                mb[i]=b1*mb[i]+(1-b1)*gb; vb[i]=b2*vb[i]+(1-b2)*gb*gb
                Ws[i]-=lr*(mW[i]/(1-b1**step))/(np.sqrt(vW[i]/(1-b2**step))+eps)
                bs[i]-=lr*(mb[i]/(1-b1**step))/(np.sqrt(vb[i]/(1-b2**step))+eps)
    a=Xte
    for i,(W,b) in enumerate(zip(Ws,bs)):
        z=a@W+b; a=np.maximum(z,0) if i<len(Ws)-1 else z
    return balacc((a[:,0]>=0).astype(int),yte)

X,y=gen(100); idx=np.random.default_rng(0).permutation(N); cut=int(.7*N); tr,te=idx[:cut],idx[cut:]
print(f"Target=octonion associator-norm. N={N}, train={cut}. Chance=50.\n")
for H in [(64,),(256,256),(512,512,512)]:
    params=sum(a*b for a,b in zip([24]+list(H),list(H)+[1]))
    b=mlp(X[tr],y[tr],X[te],y[te],H=H,epochs=300)
    print(f"  trained MLP {str(H):16s} (~{params:6d} params): balacc={b:.2f}")
print(f"\n  octonion assoc-feat (reference):              balacc~99.45")
print("\nIf even a large trained MLP lags octonion -> the specific algebra is a strong,")
print("hard-to-learn inductive bias. If a modest MLP matches -> generic nonlinearity suffices.")
