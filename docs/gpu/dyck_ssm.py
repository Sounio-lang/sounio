#!/usr/bin/env python3
# Proof-of-concept: is the non-associative octonion SSM trainable for next-token prediction, and does
# non-associativity help on NESTING structure (the #1222 bracketing hypothesis)? Task = Dyck-k (balanced
# nested brackets), the canonical language where predicting the CLOSING bracket type requires tracking the
# nesting stack. Models share everything except the state-mixing algebra:
#   OCT  : h_t = tanh(A ⊗ h_{t-1} + B x_t)     octonion mult (NON-associative), C channels, state 8C
#   QUAT : same with quaternion mult            (associative hypercomplex control), 2C channels, state 8C
#   REAL : h_t = tanh(M h_{t-1} + B x_t)        real matrix M (associative linear SSM), state 8C
# All trained with BPTT + Adam; report next-token accuracy overall and on CLOSING brackets (where the
# stack matters). Honest PoC scale (tiny); the point is trainability + the nesting ablation, not an LM.
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
def multable(bits):
    n=1<<bits; S=np.zeros((n,n)); X=np.zeros((n,n),int)
    for i in range(n):
        for j in range(n): S[i,j]=cds(i,j,bits); X[i,j]=i^j
    return S,X
SIG8,XOR8=multable(3); SIG4,XOR4=multable(2)
def mk_mul(SIG,XOR,n):
    def omul(A,B):                       # (...,n)x(...,n)->(...,n)
        C=np.zeros(np.broadcast_shapes(A.shape,B.shape))
        for i in range(n):
            for j in range(n): C[...,XOR[i,j]]+=SIG[i,j]*A[...,i]*B[...,j]
        return C
    def vjp(A,B,dC):                     # dA[i]=Σ_j S[i,j]B[j]dC[i^j]; dB[j]=Σ_i S[i,j]A[i]dC[i^j]
        dA=np.zeros(np.broadcast_shapes(A.shape,dC.shape)); dB=np.zeros(np.broadcast_shapes(B.shape,dC.shape))
        for i in range(n):
            for j in range(n):
                dA[...,i]+=SIG[i,j]*B[...,j]*dC[...,XOR[i,j]]
                dB[...,j]+=SIG[i,j]*A[...,i]*dC[...,XOR[i,j]]
        return dA,dB
    return omul,vjp
omul8,vjp8=mk_mul(SIG8,XOR8,8); omul4,vjp4=mk_mul(SIG4,XOR4,4)
# ---------- Dyck-k data ----------
def gen_dyck(rng,k=3,maxdepth=6,minlen=20,maxlen=40,n=4000):
    # tokens: open i -> 2i, close i -> 2i+1  (i=0..k-1); EOS = 2k
    seqs=[]
    for _ in range(n):
        s=[]; stack=[]
        while len(s)<maxlen:
            if stack and (len(stack)>=maxdepth or (len(s)>=minlen and rng.random()<0.5) or rng.random()<0.45):
                i=stack.pop(); s.append(2*i+1)
            else:
                i=rng.integers(k); stack.append(i); s.append(2*i)
        while stack: s.append(2*stack.pop()+1)      # close remainder
        s=s[:maxlen]; s.append(2*k)                 # EOS
        seqs.append(s)
    L=max(len(s) for s in seqs); V=2*k+1
    arr=np.full((n,L),2*k,int)
    for r,s in enumerate(seqs): arr[r,:len(s)]=s
    return arr,V
# ---------- models ----------
class Adam:
    def __init__(s,sh,lr):s.m=np.zeros(sh);s.v=np.zeros(sh);s.lr=lr;s.t=0
    def step(s,p,g):s.t+=1;s.m=.9*s.m+.1*g;s.v=.999*s.v+.001*g*g;p-=s.lr*(s.m/(1-.9**s.t))/(np.sqrt(s.v/(1-.999**s.t))+1e-8);return p
def softmax_ce(logits,tgt):                          # logits (B,V), tgt (B,)
    z=logits-logits.max(1,keepdims=True); e=np.exp(z); p=e/e.sum(1,keepdims=True)
    B=len(tgt); loss=-np.log(p[np.arange(B),tgt]+1e-12).mean(); dl=p.copy(); dl[np.arange(B),tgt]-=1; dl/=B
    return loss,dl,p
class AlgSSM:                                         # octonion (dim8) or quaternion (dim4) channel SSM
    def __init__(s,V,C,dim,omul,vjp,rng):
        s.dim=dim;s.C=C;s.omul=omul;s.vjp=vjp;s.V=V
        s.E=rng.standard_normal((V,C,dim))*0.3        # input embedding -> B x_t per channel
        s.A=rng.standard_normal((C,dim))*0.3          # state-mixing algebra element per channel
        s.W=rng.standard_normal((C*dim,V))*0.1; s.c=np.zeros(V)
        s.opt={n:Adam(getattr(s,n).shape if n!='c' else (V,),3e-3) for n in['E','A','W','c']}
    def run(s,X,train=True):                          # X (B,L) tokens -> loss, acc, closeacc  (+ update)
        B,L=X.shape; C,dim=s.C,s.dim; h=np.zeros((B,C,dim))
        Hs=[];PRs=[];toks=[]
        gE=np.zeros_like(s.E);gA=np.zeros_like(s.A);gW=np.zeros_like(s.W);gc=np.zeros_like(s.c)
        loss=0.0; correct=0;total=0;cclose=0;tclose=0
        # forward
        logits_all=[];p_all=[]
        for t in range(L-1):
            xt=s.E[X[:,t]]                            # (B,C,dim)
            Ah=s.omul(s.A[None],h)                    # (B,C,dim)
            pre=Ah+xt; hn=np.tanh(pre)
            Hs.append((h,hn,pre,X[:,t])); h=hn
            logit=hn.reshape(B,-1)@s.W+s.c
            lo,dl,p=softmax_ce(logit,X[:,t+1]); loss+=lo
            logits_all.append((hn.reshape(B,-1),dl));
            pred=p.argmax(1); tgt=X[:,t+1]
            correct+=(pred==tgt).sum(); total+=B
            cl=(tgt%2==1)&(tgt<s.V-1)                 # closing-bracket targets
            cclose+=((pred==tgt)&cl).sum(); tclose+=cl.sum()
        loss/=(L-1)
        acc=correct/total; caccu=cclose/max(tclose,1)
        if not train: return loss,acc,caccu
        # backward (BPTT)
        dh=np.zeros((B,C,dim))
        for t in range(L-2,-1,-1):
            hflat,dl=logits_all[t]
            gW+=hflat.T@dl/1; gc+=dl.sum(0)
            dhn=(dl@s.W.T).reshape(B,C,dim)+dh
            h_prev,hn,pre,xtok=Hs[t]
            dpre=dhn*(1-hn*hn)
            # xt = E[xtok]; Ah=omul(A,h_prev)
            np.add.at(gE,xtok,dpre)
            dA,dhprev=s.vjp(s.A[None],h_prev,dpre)
            gA+=dA.sum(0); dh=dhprev
        for n,g in [('E',gE),('A',gA),('W',gW),('c',gc)]:
            setattr(s,n,s.opt[n].step(getattr(s,n),g))
        return loss,acc,caccu
class RealSSM:                                        # associative linear-matrix SSM, matched state dim
    def __init__(s,V,H,rng):
        s.H=H;s.V=V
        s.E=rng.standard_normal((V,H))*0.3; s.M=rng.standard_normal((H,H))*(1/np.sqrt(H))
        s.W=rng.standard_normal((H,V))*0.1; s.c=np.zeros(V)
        s.opt={n:Adam(getattr(s,n).shape if n!='c' else (V,),3e-3) for n in['E','M','W','c']}
    def run(s,X,train=True):
        B,L=X.shape;H=s.H;h=np.zeros((B,H));Hs=[];logits_all=[]
        gE=np.zeros_like(s.E);gM=np.zeros_like(s.M);gW=np.zeros_like(s.W);gc=np.zeros_like(s.c)
        loss=0.0;correct=0;total=0;cclose=0;tclose=0
        for t in range(L-1):
            xt=s.E[X[:,t]];pre=h@s.M+xt;hn=np.tanh(pre);Hs.append((h,hn,X[:,t]));h=hn
            logit=hn@s.W+s.c;lo,dl,p=softmax_ce(logit,X[:,t+1]);loss+=lo
            logits_all.append((hn,dl));pred=p.argmax(1);tgt=X[:,t+1]
            correct+=(pred==tgt).sum();total+=B;cl=(tgt%2==1)&(tgt<s.V-1)
            cclose+=((pred==tgt)&cl).sum();tclose+=cl.sum()
        loss/=(L-1);acc=correct/total;caccu=cclose/max(tclose,1)
        if not train:return loss,acc,caccu
        dh=np.zeros((B,H))
        for t in range(L-2,-1,-1):
            hn_prev,dl=logits_all[t];gW+=hn_prev.T@dl;gc+=dl.sum(0)
            dhn=dl@s.W.T+dh;h_prev,hn,xtok=Hs[t];dpre=dhn*(1-hn*hn)
            np.add.at(gE,xtok,dpre);gM+=h_prev.T@dpre;dh=dpre@s.M.T
        for n,g in[('E',gE),('M',gM),('W',gW),('c',gc)]:setattr(s,n,s.opt[n].step(getattr(s,n),g))
        return loss,acc,caccu
# ---------- gradient check (octonion SSM, tiny) ----------
def gcheck():
    rng=np.random.default_rng(0);m=AlgSSM(5,1,8,omul8,vjp8,rng);X=rng.integers(0,5,(2,5))
    def L():
        h=np.zeros((2,1,8));tot=0.0
        for t in range(4):
            Ah=m.omul(m.A[None],h);hn=np.tanh(Ah+m.E[X[:,t]]);lg=hn.reshape(2,-1)@m.W+m.c
            lo,_,_=softmax_ce(lg,X[:,t+1]);tot+=lo;h=hn
        return tot/4
    l0=L();eps=1e-6;A0=m.A.copy();num=np.zeros_like(m.A)
    for i in range(8):
        m.A[0,i]+=eps;lp=L();m.A[0,i]=A0[0,i];num[0,i]=(lp-l0)/eps
    # analytic dA via one train step's grad (rerun forward+backward without updating)
    import copy;m2=copy.deepcopy(m)
    # monkey: capture gA by temporarily setting lr 0
    for n in m2.opt:m2.opt[n].lr=0.0
    # extract gA: replicate backward
    B,Lx=X.shape;h=np.zeros((2,1,8));Hs=[];logits_all=[]
    for t in range(4):
        xt=m2.E[X[:,t]];Ah=m2.omul(m2.A[None],h);pre=Ah+xt;hn=np.tanh(pre);Hs.append((h,hn,pre,X[:,t]));h=hn
        lo,dl,p=softmax_ce(hn.reshape(2,-1)@m2.W+m2.c,X[:,t+1]);logits_all.append((hn.reshape(2,-1),dl))
    gA=np.zeros_like(m2.A);dh=np.zeros((2,1,8))
    for t in range(3,-1,-1):
        hflat,dl=logits_all[t];dhn=(dl@m2.W.T).reshape(2,1,8)+dh;h_prev,hn,pre,_=Hs[t];dpre=dhn*(1-hn*hn)
        dA,dhp=m2.vjp(m2.A[None],h_prev,dpre);gA+=dA.sum(0);dh=dhp
    err=np.abs(gA/4-num).max()
    print(f"octonion-SSM BPTT gradient check: max err = {err:.2e}", "PASS" if err<1e-4 else "FAIL")
    return err<1e-4
assert gcheck()
# ---------- experiment ----------
rng=np.random.default_rng(20260719); k=3
data,V=gen_dyck(rng,k=k,n=5000); tr,te=data[:4000],data[4000:]
print(f"Dyck-{k}: vocab={V}, seq len={data.shape[1]}, train={len(tr)} test={len(te)}")
def train(model,name,iters=400,bs=128):
    for it in range(iters):
        bi=rng.integers(0,len(tr),bs); model.run(tr[bi],train=True)
    # eval on test
    l,a,c=0,0,0;nb=0
    for s in range(0,len(te),256):
        L_,A_,C_=model.run(te[s:s+256],train=False);l+=L_;a+=A_;c+=C_;nb+=1
    print(f"  {name:26s} test: loss {l/nb:.3f}  next-tok acc {100*a/nb:.1f}%  CLOSING-bracket acc {100*c/nb:.1f}%")
    return a/nb,c/nb
print("Training (BPTT + Adam, 400 iters). Non-assoc octonion vs associative controls at matched state dim 32:")
train(AlgSSM(V,4,8,omul8,vjp8,np.random.default_rng(1)),"OCT  octonion (non-assoc)")
train(AlgSSM(V,8,4,omul4,vjp4,np.random.default_rng(2)),"QUAT quaternion (assoc)")
train(RealSSM(V,32,np.random.default_rng(3)),"REAL linear matrix (assoc)")
