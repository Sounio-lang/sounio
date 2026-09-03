#!/usr/bin/env python3
# Erosion (Hebbian consolidation) test — the missing mechanism the ordering-null localized (critique #6 §4).
# The river's flow RESHAPES THE BED (reciprocal coupling); standard training has a fixed landscape, so
# coherent ordering was mere blocking → forgetting → it lost. Add erosion — the traversed path deepens a
# consolidated 'bed' and protects carved channels (local, reward-free, trajectory-dependent = Hebb) — and
# ask: does coherent ordering THEN beat shuffle? Prediction: erosion helps coherent MORE than shuffled
# (coherent is the forgetter); the gap narrows/reverses. Same data/init/seeds as affective_ordering.py.
import numpy as np
np.seterr(all='ignore')
def make_data(rng,N,d,Wm):
    a=rng.uniform(-1,1,(N,3)); X=rng.standard_normal((N,d))
    W=np.tanh(np.concatenate([a,np.ones((N,1))],1)@Wm); y=(np.sum(W*X,1)>0).astype(float); return X,a,y
def coherent_order(a):
    N=len(a); used=np.zeros(N,bool); order=[0]; used[0]=True
    for _ in range(N-1):
        d=np.sum((a-a[order[-1]])**2,1); d[used]=1e9; order.append(int(np.argmin(d))); used[order[-1]]=True
    return np.array(order)
class Net:
    def __init__(s,D,H,rng,erosion=False,mu=0.0,alpha=0.0):
        s.W1=rng.standard_normal((D,H))*0.3; s.b1=np.zeros(H); s.W2=rng.standard_normal(H)*0.3; s.b2=0.0
        s.erosion=erosion; s.mu=mu; s.alpha=alpha
        # consolidated 'bed' (EMA) + Hebbian importance Ω (accumulated squared gradient) per weight
        s.cW1=s.W1.copy(); s.cW2=s.W2.copy()
        s.OW1=np.zeros_like(s.W1); s.OW2=np.zeros_like(s.W2)
    def step(s,x,y,lr):
        h=np.tanh(x@s.W1+s.b1); logit=h@s.W2+s.b2; p=1/(1+np.exp(-logit)); dL=p-y
        gW2=h*dL; gb2=dL; dh=dL*s.W2*(1-h*h); gW1=np.outer(x,dh); gb1=dh
        if s.erosion:
            # Hebbian channel depth (where flow was strong) + anchor to the consolidated bed
            s.OW1=0.999*s.OW1+gW1**2; s.OW2=0.999*s.OW2+gW2**2
            aW1=s.mu*(s.OW1/(s.OW1.mean()+1e-8))*(s.W1-s.cW1)
            aW2=s.mu*(s.OW2/(s.OW2.mean()+1e-8))*(s.W2-s.cW2)
            s.W1-=lr*(gW1+aW1); s.W2-=lr*(gW2+aW2); s.b1-=lr*gb1; s.b2-=lr*gb2
            s.cW1+=s.alpha*(s.W1-s.cW1); s.cW2+=s.alpha*(s.W2-s.cW2)   # bed slowly follows (consolidation)
        else:
            s.W1-=lr*gW1; s.W2-=lr*gW2; s.b1-=lr*gb1; s.b2-=lr*gb2
    def acc(s,X,Y):
        h=np.tanh(X@s.W1+s.b1); return (((h@s.W2+s.b2)>0)==Y).mean()
def run(seed,erosion,mu=0.4,alpha=0.01):
    rng=np.random.default_rng(seed); d=16; H=64; N=4000; lr=0.05
    Wm=rng.standard_normal((4,d)); Xtr,atr,ytr=make_data(rng,N,d,Wm); Xte,ate,yte=make_data(rng,1000,d,Wm)
    Ftr=np.concatenate([Xtr,atr],1); Fte=np.concatenate([Xte,ate],1); D=d+3
    W1_0=rng.standard_normal((D,H))*0.3; W2_0=rng.standard_normal(H)*0.3
    out={}
    for name,order in {'coherent':coherent_order(atr),'shuffled':rng.permutation(N)}.items():
        m=Net(D,H,rng,erosion,mu,alpha); m.W1=W1_0.copy(); m.b1=np.zeros(H); m.W2=W2_0.copy(); m.b2=0.0
        m.cW1=m.W1.copy(); m.cW2=m.W2.copy()
        for i in order: m.step(Ftr[i],ytr[i],lr)
        out[name]=m.acc(Fte,yte)
    return out
print("Erosion (Hebbian consolidation) — does the reciprocal flow↔bed coupling rescue coherent ordering?")
print("Test accuracy, 10 seeds, single-pass online SGD (chance=50%):\n")
for ero,lab in [(False,'NO erosion (baseline)'),(True,'WITH erosion (Hebbian bed)')]:
    co=[];sh=[]
    for s in range(10):
        r=run(s,ero); co.append(r['coherent']); sh.append(r['shuffled'])
    co=np.array(co)*100; sh=np.array(sh)*100; diff=co-sh
    print(f"  {lab:26s}: coherent {co.mean():5.2f}±{co.std():4.2f}   shuffled {sh.mean():5.2f}±{sh.std():4.2f}   "
          f"Δ(coh−shuf) {diff.mean():+.2f}pp (wins {int((diff>0).sum())}/10)")
print("\n→ erosion supports the thesis iff Δ(coh−shuf) moves UP vs baseline (coherent helped more than shuffled).")
print("  if Δ stays negative / erosion helps both equally: the ordering advantage is absent even with erosion.")
