#!/usr/bin/env python3
# The order-is-content test (OPUS-4.8-EXTRA critique #6, §5). Standard training SHUFFLES data — it treats
# order as noise to remove. The non-associativity thesis says order is content. Testable and cheap:
#   Does affectively-COHERENT ordering (a continuous VAD trajectory, no abrupt affective jumps between
#   consecutive examples) beat SHUFFLING — same data, same compute, same seeds?
# Setup: each example carries an affective coordinate a∈[-1,1]³ (valence/arousal/dominance); the target rule
# drifts SMOOTHLY with mood, y = sign(w(a)·x), w(a) a smooth mood-dependent weight. The model sees [x;a].
# Single-pass online SGD (the regime where order matters most). Coherent = greedy nearest-neighbour VAD
# path; shuffled = random; anti-coherent = maximally jumpy (control). Honest either way.
import numpy as np
np.seterr(all='ignore')
def make_data(rng,N,d,Wm):
    a=rng.uniform(-1,1,(N,3)); X=rng.standard_normal((N,d))
    A1=np.concatenate([a,np.ones((N,1))],1)          # [a;1]
    W=np.tanh(A1@Wm)                                  # (N,d) mood-dependent weight
    y=(np.sum(W*X,1)>0).astype(float)
    return X,a,y
def coherent_order(a):                                # greedy nearest-neighbour path in VAD space
    N=len(a); used=np.zeros(N,bool); order=[0]; used[0]=True
    for _ in range(N-1):
        last=a[order[-1]]; dist=np.sum((a-last)**2,1); dist[used]=1e9
        nxt=int(np.argmin(dist)); order.append(nxt); used[nxt]=True
    return np.array(order)
def anti_order(a):                                    # maximally-jumpy: always go to the FARTHEST unvisited
    N=len(a); used=np.zeros(N,bool); order=[0]; used[0]=True
    for _ in range(N-1):
        last=a[order[-1]]; dist=np.sum((a-last)**2,1); dist[used]=-1
        order.append(int(np.argmax(dist))); used[order[-1]]=True
    return np.array(order)
class MLP:
    def __init__(s,D,H,rng):
        s.W1=rng.standard_normal((D,H))*0.3; s.b1=np.zeros(H); s.W2=rng.standard_normal(H)*0.3; s.b2=0.0
    def step(s,x,y,lr):
        h=np.tanh(x@s.W1+s.b1); logit=h@s.W2+s.b2; p=1/(1+np.exp(-logit)); dL=p-y
        gW2=h*dL; gb2=dL; dh=dL*s.W2*(1-h*h); gW1=np.outer(x,dh); gb1=dh
        s.W2-=lr*gW2; s.b2-=lr*gb2; s.W1-=lr*gW1; s.b1-=lr*gb1
    def acc(s,X,Y):
        h=np.tanh(X@s.W1+s.b1); p=(h@s.W2+s.b2)>0; return (p==Y).mean()
def run(seed):
    rng=np.random.default_rng(seed); d=16; H=64; N=4000; lr=0.05
    Wm=rng.standard_normal((4,d))                     # the smooth mood→weight map (shared across conditions)
    Xtr,atr,ytr=make_data(rng,N,d,Wm); Xte,ate,yte=make_data(rng,1000,d,Wm)
    Ftr=np.concatenate([Xtr,atr],1); Fte=np.concatenate([Xte,ate],1); D=d+3
    W1_0=rng.standard_normal((D,H))*0.3; b1_0=np.zeros(H); W2_0=rng.standard_normal(H)*0.3  # shared init
    orders={'coherent':coherent_order(atr),'shuffled':rng.permutation(N),'anti-coherent':anti_order(atr)}
    res={}
    for name,order in orders.items():
        m=MLP(D,H,rng); m.W1=W1_0.copy(); m.b1=b1_0.copy(); m.W2=W2_0.copy(); m.b2=0.0   # identical start
        for i in order: m.step(Ftr[i],ytr[i],lr)      # single pass, in the given order
        res[name]=m.acc(Fte,yte)
    return res
print("Order-is-content: single-pass online SGD, identical data/init/steps — only the ORDER differs.")
print("Target rule drifts smoothly with affective coordinate (VAD). Test accuracy (chance=50%):")
seeds=range(8); R={k:[] for k in ['coherent','shuffled','anti-coherent']}
for s in seeds:
    r=run(s)
    for k in r: R[k].append(r[k])
for k in ['coherent','shuffled','anti-coherent']:
    v=np.array(R[k])*100; print(f"  {k:15s}: {v.mean():5.2f}% ± {v.std():4.2f}   (per-seed {np.round(v,1)})")
co=np.array(R['coherent']); sh=np.array(R['shuffled'])
diff=(co-sh)*100
print(f"\ncoherent − shuffled: {diff.mean():+.2f} pp ± {diff.std():.2f}   wins {int((diff>0).sum())}/8 seeds")
print("→ if coherent > shuffled: training composition is ORDER-DEPENDENT in a way current practice (shuffle) discards.")
print("  if not: affective non-associativity does not manifest in small-scale single-pass training — an honest null.")
