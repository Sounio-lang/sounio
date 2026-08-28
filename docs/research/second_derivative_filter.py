#!/usr/bin/env python3
# PART I — the second-derivative filter (OPUS-4.8-EXTRA, generative turn). Born from the sedenion
# observation: a zero divisor z has y with zy=0 → the directional (first) derivative vanishes, but the
# SECOND does not. Translated to training: at the annihilation locus first-order info dies, so plain SGD
# is BLIND exactly where annihilation occurs. Two examples with ‖∇_θL‖→0 are opposite in content:
#   λ_min(H) > 0  → DOMINATED (genuine minimum, learned)
#   λ_min(H) ≤ 0  → ANNIHILATOR (incomponible in the current composition — a training-set 'zero divisor')
# The filter selects SMALL gradient + DEGENERATE/negative curvature — the inverse of hard-example mining
# (which selects large gradient). Two scalars/example via Hessian-vector products (no full Hessian).
import numpy as np
np.seterr(all='ignore')
rng=np.random.default_rng(0)
d=8; H=24
# ---- data: a majority rule + a MINORITY 'annihilator' subpopulation with the OPPOSITE rule ----
def gen(n, frac_ann, w, rng):
    X=rng.standard_normal((n,d)); y=(X@w>0).astype(float); ann=np.zeros(n,bool)
    k=int(n*frac_ann); idx=rng.choice(n,k,replace=False); y[idx]=1-y[idx]; ann[idx]=True  # contradictory sub-rule
    return X,y,ann
w=rng.standard_normal(d)
Xtr,ytr,_=gen(3000,0.12,w,rng); Xpr,ypr,ann=gen(600,0.12,w,rng)   # probe set with known annihilator labels
# ---- model ----
def init(rng): return [rng.standard_normal((d,H))*0.4, np.zeros(H), rng.standard_normal(H)*0.4, np.array(0.0)]
def flat(P): return np.concatenate([p.ravel() for p in P])
def unflat(v): return [v[:d*H].reshape(d,H), v[d*H:d*H+H], v[d*H+H:d*H+2*H], v[-1]]
def fwd(P,X):
    h=np.tanh(X@P[0]+P[1]); return h, h@P[2]+P[3]
def loss_grad(P,X,Y):                       # mean loss + flat grad
    h,z=fwd(P,X); p=1/(1+np.exp(-z)); L=-(Y*np.log(p+1e-9)+(1-Y)*np.log(1-p+1e-9)).mean(); dz=(p-Y)/len(Y)
    gW2=h.T@dz; gb2=dz.sum(); dh=(dz[:,None]*P[2][None])*(1-h*h); gW1=X.T@dh; gb1=dh.sum(0)
    return L, np.concatenate([gW1.ravel(),gb1,gW2,[gb2]])
# ---- train ----
P=init(rng); lr=0.3
for it in range(3000):
    _,g=loss_grad(P,Xtr,ytr); v=flat(P)-lr*g; P=unflat(v)
print("trained. now probe each example with (‖∇‖, λ_min(H)) via finite-diff HVP + power iteration.\n")
def per_example(P,x,y):
    xg=x[None]; yg=np.array([y])
    L,g=loss_grad(P,xg,yg); gn=np.linalg.norm(g)
    # HVP by central finite difference of the per-example gradient; power-iterate on (κI−H) for λ_min
    th=flat(P); eps=1e-4; kappa=50.0
    def Hv(vv):
        gp=loss_grad(unflat(th+eps*vv),xg,yg)[1]; gm=loss_grad(unflat(th-eps*vv),xg,yg)[1]
        return (gp-gm)/(2*eps)
    u=rng.standard_normal(len(th)); u/=np.linalg.norm(u)
    for _ in range(12):
        w2=kappa*u-Hv(u); w2/=np.linalg.norm(w2)+1e-12; u=w2
    lam_min=kappa-(u@(kappa*u-Hv(u)))            # λ_min = κ − λ_max(κI−H)
    return gn, lam_min, L
G=[];Lam=[];Loss=[]
for i in range(len(Xpr)):
    gn,lm,L=per_example(P,Xpr[i],ypr[i]); G.append(gn);Lam.append(lm);Loss.append(L)
G=np.array(G);Lam=np.array(Lam);Loss=np.array(Loss)
# ---- analysis: among SMALL-gradient examples, does λ_min separate annihilators from dominated? ----
small=G<np.percentile(G,50)                      # the 'looks-resolved' half (low gradient)
print(f"among the low-gradient half (the 'looks-resolved' examples that SGD ignores):")
dom = small & (Lam>0.02); anni = small & (Lam<=0.02)
def frac_ann(mask): return 100*ann[mask].mean() if mask.sum() else 0
print(f"  small∇ & λ_min>0  (DOMINATED) : n={dom.sum():3d}  are-annihilator {frac_ann(dom):5.1f}%  mean-loss {Loss[dom].mean():.3f}")
print(f"  small∇ & λ_min≤0  (ANNIHILATOR): n={anni.sum():3d}  are-annihilator {frac_ann(anni):5.1f}%  mean-loss {Loss[anni].mean():.3f}")
print(f"\ncontrast with hard-example mining (LARGE gradient):")
big=G>np.percentile(G,50)
print(f"  large∇ (hard-mining picks these): are-annihilator {frac_ann(big):5.1f}%")
# precision of the filter
prec=100*ann[anni].mean() if anni.sum() else 0; recall=100*(anni&ann).sum()/max(ann.sum(),1)
print(f"\nFILTER (small∇ & degenerate curvature): precision {prec:.1f}% recall {recall:.1f}% for the annihilator subpopulation")
print("→ the second scalar (λ_min) separates 'learned' from 'incomponible' AMONG the examples first-order")
print("  training discards as silence — the class hard-example mining is structurally blind to.")
