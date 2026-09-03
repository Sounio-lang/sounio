# Confound-free target: deep RESIDUAL MLP (ResMLP), DISTINCT weights per layer — no shared backbone
# (unlike the LSTM's shared W_hh → architectural align≈1). Probe the BRANCH Jacobians F'_l = J_l − I
# (J_l = I + F'_l; the residual anchors J_l to I, so we measure the branch, per the §4 caveat).
#   block:  z_{l+1} = z_l + B_l·tanh(A_l·z_l);   F'_l(z) = B_l·diag(1−tanh²(A_l z))·A_l   (W×W)
# Init sits at BASELINE (clean target — proven below). Question: does LEARNING create a shared dying
# subspace (annihilation → small-k shoulder above baseline) or low-rank (high to large k) or nothing?
import numpy as np, torch, torch.nn as nn
np.seterr(all='ignore')
d,W,L=64,96,8
class ResMLP(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Linear(d,W)
        s.A=nn.ParameterList([nn.Parameter(torch.randn(W,W)/W**0.5) for _ in range(L)])
        s.B=nn.ParameterList([nn.Parameter(torch.randn(W,W)/W**0.5*0.5) for _ in range(L)])
        s.rd=nn.Linear(W,1)
    def forward(s,x):
        z=s.emb(x)
        for A,B in zip(s.A,s.B): z=z+torch.tanh(z@A.t())@B.t()
        return s.rd(z).squeeze(-1)
def gen(n,rng):     # nonlinear 6-dim signal (rewards depth): y = sign(x0x1 + x2x3 − x4x5)
    X=rng.standard_normal((n,d)).astype('float32')
    s_=X[:,0]*X[:,1]+X[:,2]*X[:,3]-X[:,4]*X[:,5]
    return torch.tensor(X),torch.tensor((s_>0).astype('float32'))
rng=np.random.default_rng(0); torch.manual_seed(0)
net=ResMLP(); opt=torch.optim.Adam(net.parameters(),2e-3); lossf=nn.BCEWithLogitsLoss()
Xte,yte=gen(4000,np.random.default_rng(1)); acc=0.0
for it in range(5000):
    X,y=gen(256,rng); opt.zero_grad(); l=lossf(net(X),y); l.backward(); opt.step()
    if it%1000==0 or it==4999:
        with torch.no_grad(): acc=((net(Xte)>0).float()==yte).float().mean().item()
        print(f"  step {it:4d}  test acc {acc:.3f}",flush=True)
CONVERGED=acc>0.85
print(f"  CONVERGENCE: test acc {acc:.3f} → {'LEARNED (probe meaningful)' if CONVERGED else 'NOT LEARNED'}",flush=True)
# ---- analytic branch Jacobians (numpy) from trained + init weights ----
def branch_jacs(embW,embb,A,B,x):
    z=embW@x+embb; Js=[]
    for Al,Bl in zip(A,B):
        a=np.tanh(Al@z); Js.append(Bl@((1-a*a)[:,None]*Al)); z=z+Bl@a
    return Js
def align_curve(mats,ks):
    Vt=[np.linalg.svd(A)[2] for A in mats]
    return np.array([np.mean([np.linalg.svd(Vt[l][-k:]@Vt[l+1][-k:].T,compute_uv=False).mean() for l in range(len(Vt)-1)]) for k in ks])
def scramble(mats,rng):    # orientation-scramble null: F'_l → O_{l+1} F'_l O_l^T (preserves spectrum, kills alignment)
    Os=[np.linalg.qr(rng.standard_normal((W,W)))[0] for _ in range(len(mats)+1)]
    return [Os[l+1]@mats[l]@Os[l].T for l in range(len(mats))]
emb=net.emb.weight.detach().numpy(); embb=net.emb.bias.detach().numpy()
At=[p.detach().numpy() for p in net.A]; Bt=[p.detach().numpy() for p in net.B]
net0=ResMLP(); e0=net0.emb.weight.detach().numpy(); b0=net0.emb.bias.detach().numpy()
A0=[p.detach().numpy() for p in net0.A]; B0=[p.detach().numpy() for p in net0.B]
ks=list(range(1,W)); base=np.sqrt(np.array(ks)/W); rr=np.random.default_rng(7)
TR=np.array([align_curve(branch_jacs(emb,embb,At,Bt, rr.standard_normal(d)),ks) for _ in range(16)]).mean(0)
IN=np.array([align_curve(branch_jacs(e0,b0,A0,B0, rr.standard_normal(d)),ks) for _ in range(16)]).mean(0)
SC=np.array([align_curve(scramble(branch_jacs(emb,embb,At,Bt, rr.standard_normal(d)),rr),ks) for _ in range(16)]).mean(0)
show=[1,2,4,6,8,16,32,48]
def row(a): return " ".join(f"{a[ks.index(k)]:5.2f}" for k in show)
print(f"\nResMLP branch F'_l align(k)   W={W}, L={L}, task y=sign(x0x1+x2x3−x4x5), acc={acc:.2f}:")
print(f"  k          "+" ".join(f"{k:>5}" for k in show))
print(f"  baseline   "+" ".join(f"{np.sqrt(k/W):5.2f}" for k in show))
print(f"  TRAINED    "+row(TR))
print(f"  INIT       "+row(IN))
print(f"  SCRAMBLE   "+row(SC))
exc=TR-base; kmax=ks[int(np.argmax(exc))]
print(f"\n  trained max excess over baseline {exc.max():+.2f} at k={kmax}; trained−init@{kmax}={TR[ks.index(kmax)]-IN[ks.index(kmax)]:+.2f}")
if exc.max()<0.10:
    verdict="NEGATIVE — branch alignment at baseline even after training (no shared dying subspace)."
elif TR[ks.index(48)]>base[ks.index(48)]+0.15:
    verdict="LOW-RANK — alignment high to large k (magnitude/rank), not a small-k annihilation shoulder."
elif TR[ks.index(kmax)]-IN[ks.index(kmax)]<0.10:
    verdict="ARCHITECTURAL — trained ≈ init, not learned."
else:
    verdict=f"POSITIVE — small-k shoulder at k≈{kmax} above baseline AND above init/scramble: learned subspace annihilation."
print(f"  → {verdict}")
print("DONE",flush=True)
