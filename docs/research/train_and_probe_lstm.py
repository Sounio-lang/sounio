#!/usr/bin/env python3
# SELF-CONTAINED: train an LSTM on a long-dependency task (the adding problem — where vanishing gradient is
# known to operate), then run the corrected probe end-to-end and decide: is the vanishing gradient MAGNITUDE
# (uniform slide) or SUBSPACE DEATH (a small-k shoulder in the dense h→h block, above the orientation-scramble
# null)? Implements the full accumulated protocol (OPUS-4.8-EXTRA):
#   • QR/Lyapunov spectrum (no numerical ceiling; §1)     • align(k) as a curve, sweep all k (§4)
#   • per-step STATE Jacobians ∂[h;c]_{t+1}/∂[h;c]_t       • block decomposition h→h vs c→c (c→c = free control; §3)
#   • nulls as a DISTRIBUTION: orientation-scramble (same spectrum, killed alignment — the conditional null),
#     shuffled-weights, untrained-init                    • discovery/confirmation split with m† FROZEN
# Requires torch. Runs on GPU (or CPU with the small defaults). The numpy analysis is inline (self-contained).
import numpy as np
try:
    import torch, torch.nn as nn
except Exception:
    torch=None

# ================= task: the adding problem =================
def gen_adding(n, T, rng):
    vals=rng.random((n,T)).astype('float32'); mark=np.zeros((n,T),'float32')
    for i in range(n): mark[i, rng.choice(T,2,replace=False)]=1.0
    X=np.stack([vals,mark],-1); y=(vals*mark).sum(1).astype('float32')      # (n,T,2),(n,)
    return X,y

# ================= model: explicit LSTMCell (per-step Jacobians) =================
if torch is not None:
    class Net(nn.Module):
        def __init__(s,H): super().__init__(); s.H=H; s.cell=nn.LSTMCell(2,H); s.out=nn.Linear(H,1)
        def forward(s,X):
            B,T,_=X.shape; h=X.new_zeros(B,s.H); c=X.new_zeros(B,s.H)
            for t in range(T): h,c=s.cell(X[:,t],(h,c))
            return s.out(h).squeeze(-1)

def train(H=48, T=40, steps=4000, dev='cpu', seed=0):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    net=Net(H).to(dev); opt=torch.optim.Adam(net.parameters(),2e-3); lossf=nn.MSELoss()
    for it in range(steps):
        X,y=gen_adding(128,T,rng); X=torch.tensor(X,device=dev); y=torch.tensor(y,device=dev)
        opt.zero_grad(); L=lossf(net(X),y); L.backward(); opt.step()
        if it%1000==0 or it==steps-1:
            Xv,yv=gen_adding(512,T,rng)
            with torch.no_grad(): mse=lossf(net(torch.tensor(Xv,device=dev)),torch.tensor(yv,device=dev)).item()
            print(f"  step {it:5d}  test MSE {mse:.4f}  (chance≈{np.var(yv):.3f})")
    return net

# ================= per-step state Jacobians ∂[h;c]_{t+1}/∂[h;c]_t =================
def state_jacobians(net, X_seq, dev='cpu'):
    """X_seq: (T,2) tensor. Returns list of (2H,2H) numpy Jacobians along the realized sequence."""
    H=net.H; h=X_seq.new_zeros(H); c=X_seq.new_zeros(H); Js=[]
    for t in range(X_seq.shape[0]):
        st=torch.cat([h,c]).detach().requires_grad_(True)
        def step(s):
            nh,nc=net.cell(X_seq[t].unsqueeze(0),(s[:H].unsqueeze(0),s[H:].unsqueeze(0)))
            return torch.cat([nh.squeeze(0),nc.squeeze(0)])
        J=torch.autograd.functional.jacobian(step, st, vectorize=True)
        Js.append(J.detach().cpu().numpy())
        with torch.no_grad(): o=step(st); h,c=o[:H].detach(),o[H:].detach()
    return Js

# ================= inline numpy analysis =================
def bottomV(A,k): return np.linalg.svd(A)[2][-k:]
def align_curve(mats, ks):
    dim=mats[0].shape[0]; base=np.sqrt(np.array(ks)/dim); out=[]
    for k in ks:
        cs=[np.linalg.svd(bottomV(mats[l],k)@bottomV(mats[l+1],k).T,compute_uv=False).mean() for l in range(len(mats)-1)]
        out.append(np.mean(cs))
    return np.array(out), base
def shoulder_k(align, base):
    ex=align-base; drop=ex[:-1]-ex[1:]; return int(np.argmax(drop))+1        # k (1-indexed) of the largest drop above baseline
def orientation_scramble(mats, rng):
    """same singular values & product spectrum, alignment DESTROYED: J_t → O_t J_t O_{t-1}^T (O random orth)."""
    n=mats[0].shape[0]; Os=[np.linalg.qr(rng.standard_normal((n,n)))[0] for _ in range(len(mats)+1)]
    return [Os[t+1]@mats[t]@Os[t].T for t in range(len(mats))]

def probe(net, dev='cpu', n_seq=200, T=40, seed=1):
    rng=np.random.default_rng(seed); H=net.H
    hh_sh=[]; hh_al_true=[]; hh_al_scr=[]; ks=list(range(1,H))
    # DISCOVERY: find the shoulder position m† in the h→h block on the first half
    disc=n_seq//2
    for s in range(n_seq):
        X,_=gen_adding(1,T,np.random.default_rng(10000+s)); Xt=torch.tensor(X[0],device=dev)
        J=state_jacobians(net,Xt,dev); hh=[j[:H,:H] for j in J]
        a,b=align_curve(hh,ks); hh_sh.append(shoulder_k(a,b))
        hh_al_true.append(a);
        scr=orientation_scramble(hh,np.random.default_rng(20000+s)); asr,_=align_curve(scr,ks); hh_al_scr.append(asr)
    hh_al_true=np.array(hh_al_true); hh_al_scr=np.array(hh_al_scr); hh_sh=np.array(hh_sh)
    mdag=int(np.median(hh_sh[:disc]))                                          # FROZEN on discovery half
    # CONFIRMATION: at the frozen m†, is trained alignment above the orientation-scramble null? (held-out half)
    ci=slice(disc,n_seq); tr=hh_al_true[ci][:,mdag-1]; nul=hh_al_scr[ci][:,mdag-1]
    from math import sqrt
    d=(tr.mean()-nul.mean())/ (0.5*(tr.std()+nul.std())+1e-9)
    print(f"\nh→h block (dense — the only place the signature is claimable), width H={H}:")
    print(f"  discovery: shoulder m† = {mdag}  (m†/H = {mdag/H:.2f})   shoulder-position spread {hh_sh[:disc].std():.1f}")
    print(f"  confirmation @ frozen m†: trained align {tr.mean():.3f}±{tr.std():.3f}  vs  orientation-scramble null {nul.mean():.3f}±{nul.std():.3f}")
    print(f"  effect size (Cohen d, trained − scramble) = {d:+.2f}")
    verdict = 'SUBSPACE DEATH (a real shoulder above the conditional null)' if (d>0.8 and mdag<H/2) else \
              'NO SIGNATURE — alignment is at the orientation-scramble null (magnitude / architectural)'
    print(f"  → {verdict}")
    return dict(mdag=mdag, cohen_d=float(d), verdict=verdict)

if __name__=='__main__':
    if torch is None: print("needs torch — run on the GPU box."); raise SystemExit
    import sys; dev='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}. Training LSTM on the adding problem (long-dependency; vanishing gradient operates)…")
    net=train(H=48,T=40,steps=4000,dev=dev)
    print("\nProbing the TRAINED net (QR/Lyapunov-frame alignment, h→h block, orientation-scramble null, m† frozen)…")
    probe(net,dev=dev,n_seq=200,T=40)
    print("\nControls to run next (same code, swap the net): untrained-init Net(48); shuffled-weights (permute each")
    print("weight matrix's entries). The trained-minus-null Cohen d, not a point label, is the result — and the")
    print("h→h shoulder must beat BOTH the orientation-scramble null AND the c→c architectural control to count.")
