#!/usr/bin/env python3
# Where the ALGEBRA (not the metaphor) enters the training claim. Two corpora, same tokens, differing only
# in the generating process; an order-AWARE model (MLP on the ORDERED concatenation of token embeddings)
# vs an order-BLIND model (MLP on the SUM — informationally what shuffling reduces the data to). Claim:
# for a NON-ASSOCIATIVE generating process (label = projection of the left-to-right octonion product of the
# tokens), order is CONTENT — the order-blind model cannot represent the label (chance), so shuffling
# destroys signal. For an ASSOCIATIVE process (label = projection of the SUM), order is nuisance — both
# succeed. This is the clean, honest form of "order is content": it holds iff the algebra is non-associative.
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
def omul(A,B):
    C=np.zeros(8)
    for i in range(8):
        for j in range(8): C[i^j]+=SIG[i,j]*A[i]*B[j]
    return C
def gen(rng,N,L,V,E,w,mode):
    S=rng.integers(0,V,(N,L)); y=np.zeros(N)
    for n in range(N):
        if mode=='nonassoc':
            acc=E[S[n,0]].copy()
            for t in range(1,L): acc=omul(acc,E[S[n,t]])     # left-to-right octonion product (order matters)
        else:
            acc=E[S[n]].sum(0)                                # sum (order-invariant)
        y[n]=1.0 if w@acc>0 else 0.0
    return S,y
class Adam:
    def __init__(s,sh,lr):s.m=np.zeros(sh);s.v=np.zeros(sh);s.lr=lr;s.t=0
    def step(s,p,g):s.t+=1;s.m=.9*s.m+.1*g;s.v=.999*s.v+.001*g*g;p-=s.lr*(s.m/(1-.9**s.t))/(np.sqrt(s.v/(1-.999**s.t))+1e-8);return p
def train_model(S,y,Ste,yte,V,L,ed,order_aware,rng,H=64,iters=400,bs=256):
    Emb=rng.standard_normal((V,ed))*0.3
    D=L*ed if order_aware else ed
    W1=rng.standard_normal((D,H))*0.2;b1=np.zeros(H);W2=rng.standard_normal(H)*0.2;b2=0.0
    o={n:Adam(v.shape if n!='b2' else (1,),3e-3) for n,v in [('E',Emb),('W1',W1),('b1',b1),('W2',W2),('b2',0)]}
    def feats(Sb,Emb):
        if order_aware: return np.concatenate([Emb[Sb[:,t]] for t in range(L)],1)   # ordered concat
        return sum(Emb[Sb[:,t]] for t in range(L))                                  # order-blind sum
    N=len(y)
    for it in range(iters):
        bi=rng.integers(0,N,bs); Sb=S[bi]; yb=y[bi]; X=feats(Sb,Emb)
        h=np.tanh(X@W1+b1);lg=h@W2+b2;p=1/(1+np.exp(-lg));dL=p-yb
        gW2=h.T@dL/bs;gb2=dL.mean();dh=(dL[:,None]*W2[None])*(1-h*h);gW1=X.T@dh/bs;gb1=dh.mean(0);gX=dh@W1.T
        gE=np.zeros_like(Emb)
        for t in range(L):
            seg=gX[:,t*ed:(t+1)*ed] if order_aware else gX
            np.add.at(gE,Sb[:,t],seg)
        Emb=o['E'].step(Emb,gE/bs);W1=o['W1'].step(W1,gW1);b1=o['b1'].step(b1,gb1);W2=o['W2'].step(W2,gW2);b2=float(o['b2'].step(np.array([b2]),np.array([gb2]))[0])
    X=feats(Ste,Emb);h=np.tanh(X@W1+b1);pred=(h@W2+b2)>0;return (pred==yte).mean()
rng=np.random.default_rng(0); V=12;L=4;ed=8
print("Does shuffling destroy signal? order-AWARE (ordered concat) vs order-BLIND (sum) model, test acc (chance=50%):\n")
for mode in ['nonassoc','assoc']:
    accs={'order-aware':[],'order-blind':[]}
    for seed in range(6):
        r=np.random.default_rng(seed+100)
        E=r.standard_normal((V,8)); E/=np.linalg.norm(E,axis=1,keepdims=True); w=r.standard_normal(8)
        Str,ytr=gen(r,6000,L,V,E,w,mode); Ste,yte=gen(r,2000,L,V,E,w,mode)
        accs['order-aware'].append(train_model(Str,ytr,Ste,yte,V,L,ed,True,r))
        accs['order-blind'].append(train_model(Str,ytr,Ste,yte,V,L,ed,False,r))
    oa=np.array(accs['order-aware'])*100; ob=np.array(accs['order-blind'])*100
    lab='NON-ASSOCIATIVE (label = ordered octonion product)' if mode=='nonassoc' else 'ASSOCIATIVE (label = sum)'
    print(f"  {lab}")
    print(f"      order-AWARE (ordered concat): {oa.mean():5.1f}% ± {oa.std():.1f}")
    print(f"      order-BLIND (sum = shuffled): {ob.mean():5.1f}% ± {ob.std():.1f}\n")
print("→ signal survives shuffling iff order-blind ≈ order-aware. If non-assoc kills the blind model but")
print("  assoc does not, the algebra — not a curriculum heuristic — is what makes order CONTENT.")
