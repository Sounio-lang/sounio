# Confound-free target: a deep RESIDUAL feedforward net with DISTINCT weights per layer, probing the
# BRANCH Jacobians F'_l = J_l − I (not J_l — the residual anchors J_l to I and would inflate alignment).
# Distinct per-layer weights ⇒ NO shared backbone (unlike the LSTM's shared W_hh), so the init control
# should sit at BASELINE — a clean target where any trained alignment above baseline is a real signal.
# Branch: F_l(x)=V_l·tanh(U_l·x);  F'_l(x)=V_l·diag(1−tanh²(U_l x))·U_l  (analytic, numpy).
import numpy as np
np.seterr(all='ignore')
def make_net(L,d,h,seed):
    rng=np.random.default_rng(seed)
    return [(rng.standard_normal((h,d))/np.sqrt(d), rng.standard_normal((d,h))/np.sqrt(h)*0.4) for _ in range(L)]
def branch_jacs(net, x):                 # propagate x through the residual stack; return per-layer F'_l
    Js=[]
    for U,V in net:
        pre=U@x; a=np.tanh(pre); Fp=V@(( 1-a*a)[:,None]*U); Js.append(Fp); x=x+V@a
    return Js
def botV(A,k): return np.linalg.svd(A)[2][-k:]
def align_curve(mats,ks):
    Vt=[np.linalg.svd(A)[2] for A in mats]
    return np.array([np.mean([np.linalg.svd(Vt[l][-k:]@Vt[l+1][-k:].T,compute_uv=False).mean()
                    for l in range(len(Vt)-1)]) for k in ks])
L,d,h=24,64,128; ks=list(range(1,40)); base=np.sqrt(np.array(ks)/d)
net=make_net(L,d,h,seed=0); rng=np.random.default_rng(9)
HH=[align_curve(branch_jacs(net, rng.standard_normal(d)*0.3), ks) for _ in range(12)]
hh=np.array(HH).mean(0)
show=[1,2,4,8,16,32]
print(f"INIT (untrained) deep residual FFN, distinct weights/layer — branch F'_l align(k):")
print(f"  k          "+" ".join(f"{k:>5}" for k in show))
print(f"  baseline   "+" ".join(f"{np.sqrt(k/d):5.2f}" for k in show))
print(f"  INIT F'_l  "+" ".join(f"{hh[ks.index(k)]:5.2f}" for k in show))
excess=(hh-base); print(f"\n  max excess over baseline: {excess.max():.2f}  at k={ks[int(np.argmax(excess))]}")
clean = excess.max()<0.15
print(f"  → target is {'CLEAN (init at baseline — distinct weights, no architectural alignment) → any trained signal is real' if clean else 'STILL CONFOUNDED (init above baseline)'}")
print("DONE",flush=True)
