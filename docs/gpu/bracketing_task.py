#!/usr/bin/env python3
# A dataset where non-associativity IS the label — evaluation-order (bracketing) discrimination.
# Inputs are a realistic symbolic distribution (Zipfian vocabulary + length-4 token sequences, like
# language); the label of a sequence s=(s1,s2,s3,s4) is  y = 1[<w*, r1 - r2> > 0]  where
#   r1 = ((s1·s2)·(s3·s4))   (balanced bracketing)
#   r2 = (((s1·s2)·s3)·s4)   (left bracketing)
# and · is octonion (non-associative) multiplication of fixed teacher embeddings. Because r1 - r2 is a
# *bracketing associator*, it is identically 0 for ANY associative algebra — so the label is undecidable
# to an associative model, and decidable to a non-associative one. Ablation (all trained, same arch):
#   OCT  : learnable octonion embeddings, non-assoc mult, logit=<w, r1-r2>  (uses the product VJP)
#   QUAT : identical architecture but the associative 4-dim subalgebra  -> r1==r2 -> logit==0 -> blind
#   LINEAR / MLP : associative real models on the concatenated learnable real token embeddings
# Honest: semi-synthetic (real symbolic input distribution, label from octonion bracketing). This is the
# "does non-associativity ever matter on non-toy inputs" test the novelty map (#6.1) calls for.
import numpy as np
np.seterr(all='ignore')
# ---- Cayley-Dickson octonion multiply (bits=3) + bilinear VJPs ----
def cds(a,b,bits=3):
    s=1
    while bits>0:
        if a==0 or b==0: return s
        if bits==1: return -s
        h=1<<(bits-1); ah=a>=h; bh=b>=h; al=a&(h-1); bl=b&(h-1)
        if not ah and not bh: a,b=al,bl
        elif not ah and bh: a,b=bl,al
        elif ah and not bh: (a,b)=(al,0) if bl==0 else (al,bl); s=s if bl==0 else -s
        else: (a,b)=(0,al) if bl==0 else (bl,al); s=-s if bl==0 else s
        bits-=1
    return s
SIG=np.array([[cds(i,j) for j in range(8)] for i in range(8)],float)
XOR=np.array([[i^j for j in range(8)] for i in range(8)])
def omul(A,B):                       # (...,8),(...,8)->(...,8)  C[m]=sum_{i^j=m} sig(i,j)A[i]B[j]
    C=np.zeros(np.broadcast(A[...,0],B[...,0]).shape+(8,))
    for i in range(8):
        for j in range(8): C[...,i^j]+=SIG[i,j]*A[...,i]*B[...,j]
    return C
def omul_vjp(A,B,dC):                # dA[i]=sum_j sig(i,j)B[j]dC[i^j]; dB[j]=sum_i sig(i,j)A[i]dC[i^j]
    dA=np.zeros_like(A); dB=np.zeros_like(B)
    for i in range(8):
        for j in range(8):
            dA[...,i]+=SIG[i,j]*B[...,j]*dC[...,i^j]
            dB[...,j]+=SIG[i,j]*A[...,i]*dC[...,i^j]
    return dA,dB
# ---- bracketing forward/backward for a batch of 4 embeddings (each (N,8)) ----
def bracket_fwd(e1,e2,e3,e4):
    p=omul(e1,e2); q=omul(e3,e4); r1=omul(p,q)          # balanced
    u=omul(p,e3);  r2=omul(u,e4)                        # left
    return r1,r2,(p,q,u)
def bracket_bwd(e1,e2,e3,e4,cache,dr1,dr2):
    p,q,u=cache
    dp1,dq=omul_vjp(p,q,dr1)
    du,de4=omul_vjp(u,e4,dr2)
    dp2,de3=omul_vjp(p,e3,du)
    dp=dp1+dp2
    de1a,de2a=omul_vjp(e1,e2,dp)
    de3b,de4b=omul_vjp(e3,e4,dq); de3=de3+de3b; de4=de4+de4b
    return de1a,de2a,de3,de4
# ---- gradient check (finite differences) ----
def grad_check():
    rng=np.random.default_rng(1)
    e=[rng.standard_normal((1,8)) for _ in range(4)]; w=rng.standard_normal(8)
    r1,r2,c=bracket_fwd(*e); L=(w*(r1-r2)).sum()
    dr1=w[None]; dr2=-w[None]
    g=bracket_bwd(*e,c,dr1,dr2)
    eps=1e-6; ok=True
    for t in range(4):
        for k in range(8):
            ep=[x.copy() for x in e]; ep[t][0,k]+=eps
            r1p,r2p,_=bracket_fwd(*ep); Lp=(w*(r1p-r2p)).sum()
            num=(Lp-L)/eps
            if abs(num-g[t][0,k])>1e-3: ok=False; print("  grad mismatch",t,k,num,g[t][0,k])
    print("gradient check:", "PASS" if ok else "FAIL")
    return ok
assert grad_check()
# ---- data: Zipfian vocabulary + length-4 sequences ----
rng=np.random.default_rng(20260719); V=64; L=4; Ntr,Nte=6000,2000
probs=1.0/np.arange(1,V+1); probs/=probs.sum()          # Zipf
def sample_seqs(n): return rng.choice(V,size=(n,L),p=probs)
Str=sample_seqs(Ntr); Ste=sample_seqs(Nte)
Estar=rng.standard_normal((V,8)); Estar/=np.linalg.norm(Estar,axis=1,keepdims=True)  # teacher octonions
wstar=rng.standard_normal(8)
def teacher_proj(S):
    e=[Estar[S[:,i]] for i in range(4)]; r1,r2,_=bracket_fwd(*e); return (wstar*(r1-r2)).sum(1)
ptr=teacher_proj(Str); pte=teacher_proj(Ste)
thr=np.median(np.concatenate([ptr,pte]))               # balance the label ~50/50
ytr=(ptr>thr).astype(float); yte=(pte>thr).astype(float)
print(f"vocab={V} seqs: train={Ntr} test={Nte}  label balance train={ytr.mean():.2f} test={yte.mean():.2f}")
def acc(logit,y): return ((logit>0).astype(float)==y).mean()
# ---- Adam helper ----
class Adam:
    def __init__(s,shape,lr): s.m=np.zeros(shape);s.v=np.zeros(shape);s.lr=lr;s.t=0
    def step(s,p,g):
        s.t+=1; s.m=.9*s.m+.1*g; s.v=.999*s.v+.001*g*g
        p-=s.lr*(s.m/(1-.9**s.t))/(np.sqrt(s.v/(1-.999**s.t))+1e-8); return p
# Embeddings are FIXED and shared across all models (the same symbolic representation). Each model learns
# only its HEAD — this isolates the scientific question (does the algebra let you read the bracketing
# label?) from the separate, hard non-convex problem of recovering an embedding table from sign labels
# (which stalls every model and only tests the optimizer, not the hypothesis).
def bracket_diff(S,E,mask4=False):
    Em=E.copy()
    if mask4: Em=Em.copy(); Em[:,4:]=0
    e=[Em[S[:,i]] for i in range(4)]; r1,r2,_=bracket_fwd(*e); return r1-r2      # (N,8) bracketing associator
def logistic(Xtr,ytr,Xte,yte,epochs=800,lr=0.3,l2=1e-3):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    w=np.zeros(Xtr.shape[1]); b=0.0
    for _ in range(epochs):
        p=1/(1+np.exp(-(Xtr@w+b))); g=p-ytr
        w-=lr*(Xtr.T@g/len(ytr)+l2*w); b-=lr*g.mean()
    return acc(Xte@w+b,yte)
def mlp(Xtr,ytr,Xte,yte,hidden=64,epochs=800,lr=0.05):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    D=Xtr.shape[1]; W1=rng.standard_normal((D,hidden))*0.2;b1=np.zeros(hidden);W2=rng.standard_normal(hidden)*0.2;b2=0.0
    o=[Adam(W1.shape,lr),Adam(b1.shape,lr),Adam(W2.shape,lr),Adam((1,),lr)]
    for _ in range(epochs):
        h=np.tanh(Xtr@W1+b1); logit=h@W2+b2; p=1/(1+np.exp(-logit)); dL=p-ytr
        gW2=h.T@dL/len(ytr); gb2=dL.mean(); dh=(dL[:,None]*W2[None])*(1-h*h)
        gW1=Xtr.T@dh/len(ytr); gb1=dh.mean(0)
        W1=o[0].step(W1,gW1); b1=o[1].step(b1,gb1); W2=o[2].step(W2,gW2); b2=float(o[3].step(np.array([b2]),np.array([gb2]))[0])
    h=np.tanh(Xte@W1+b1); return acc(h@W2+b2,yte)
Foct_tr,Foct_te=bracket_diff(Str,Estar),bracket_diff(Ste,Estar)               # non-assoc bracketing feature
Fq_tr,Fq_te   =bracket_diff(Str,Estar,True),bracket_diff(Ste,Estar,True)      # associative subalgebra -> ~0
Xraw_tr=np.concatenate([Estar[Str[:,i]] for i in range(4)],1)                  # raw token embeddings (N,32)
Xraw_te=np.concatenate([Estar[Ste[:,i]] for i in range(4)],1)
print("Bracketing (evaluation-order) task — non-associativity IS the label. Test accuracy (chance=50%):")
print(f"  OCT bracketing associator r1-r2 (non-assoc) + logistic head : {100*logistic(Foct_tr,ytr,Foct_te,yte):.1f}%")
print(f"  QUAT same feature in the associative 4-dim subalgebra (=0)   : {100*logistic(Fq_tr,ytr,Fq_te,yte):.1f}%  (||r1-r2||={np.abs(Fq_te).max():.1e})")
print(f"  LINEAR logistic on raw concatenated token embeddings        : {100*logistic(Xraw_tr,ytr,Xraw_te,yte):.1f}%")
print(f"  MLP 32->64->1 on raw concatenated token embeddings          : {100*mlp(Xraw_tr,ytr,Xraw_te,yte):.1f}%")
