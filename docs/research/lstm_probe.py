#!/usr/bin/env python3
# LSTM probe (torch) — the target the corrected protocol points at. Dense, input-dependent transition, so
# the per-step Jacobians are genuinely distinct and nothing forces alignment (unlike S4/Mamba, whose
# diagonal Ā shares eigenvectors → alignment=1 by architecture; and unlike the S-SSM, whose L_x is 4/8/4 by
# algebra). Vanishing gradient in LSTMs is known and conceptually unresolved: is it magnitude, or subspace
# death? This measures which. Use align_curve (from align_curve.py) on the per-step state Jacobians and read
# the SHOULDER position: small-k shoulder + healthy bulk = structural annihilation; large-k = low rank; none
# = magnitude. Run over hundreds of sequences and report the distribution of the shoulder position.
import numpy as np
def lstm_step_jacobians(cell, x_seq, h0=None, c0=None):
    """cell: torch.nn.LSTMCell. x_seq: (T, input_size) tensor. Returns list of per-step STATE Jacobians
    J_t = ∂[h_{t+1};c_{t+1}] / ∂[h_t;c_t] (dim 2H×2H) at the realized inputs — the genuine factors of ∂h_T/∂h_0."""
    import torch
    H=cell.hidden_size; T=x_seq.shape[0]
    h=torch.zeros(H) if h0 is None else h0; c=torch.zeros(H) if c0 is None else c0
    Js=[]
    for t in range(T):
        st=torch.cat([h,c]).detach().requires_grad_(True)
        def step(s):
            hh,cc=s[:H],s[H:]; nh,nc=cell(x_seq[t].unsqueeze(0),(hh.unsqueeze(0),cc.unsqueeze(0)))
            return torch.cat([nh.squeeze(0),nc.squeeze(0)])
        J=torch.autograd.functional.jacobian(step, st, vectorize=True)
        Js.append(J.detach().cpu().numpy())
        with torch.no_grad(): out=step(st); h,c=out[:H],out[H:]
    return Js
# NOTE — residual caveat (§4): if you probe a ResNet/transformer instead, J_l = I + F'_l and the residual
# anchors every layer in the same basis → alignment inflates by architecture. Measure the alignment of the
# BRANCHES F'_l (= J_l − I), not the full J_l, or you are measuring the identity.
if __name__=='__main__':
    print("import lstm_step_jacobians, build/load a trained LSTMCell, run over many fixed sequences,")
    print("feed the per-step Jacobians to align_curve(...) and record the SHOULDER position per sequence.")
# closed system is (h,c) — compute the FULL Jacobians (lstm_step_jacobians), then report alignment BY BLOCK:
#   h→h : dense — the only place subspace death can live; the signature is claimable ONLY here.
#   c→c : diagonal by architecture (gates depend on h_{t-1},x_t, never on c_{t-1}) — a FREE internal
#         positive control: it will read align≈1 trivially; if the FULL-state alignment matches c→c while
#         h→h sits at the null, the alignment is entirely architectural, found in the same computation.
import numpy as np
def bottomV(A,k): return np.linalg.svd(A)[2][-k:]
def align_curve_of(mats,ks,dim):
    base=np.sqrt(np.array(ks)/dim); out=[]
    for k in ks:
        cs=[np.linalg.svd(bottomV(mats[l],k)@bottomV(mats[l+1],k).T,compute_uv=False).mean() for l in range(len(mats)-1)]
        out.append(np.mean(cs))
    return np.array(out), base
def block_report(state_jacs, H, ks=None):
    """state_jacs: list of (2H,2H) per-step Jacobians. Reports align(k) for the h→h block, the c→c block,
    and the full state. c→c is the free positive control; the signature is claimable only in h→h."""
    ks=ks or list(range(1,2*H))
    hh=[J[:H,:H] for J in state_jacs]; cc=[J[H:,H:] for J in state_jacs]
    ah,_=align_curve_of(hh,[k for k in ks if k<H],H)
    ac,_=align_curve_of(cc,[k for k in ks if k<H],H)
    af,_=align_curve_of(state_jacs,ks,2*H)
    return dict(hh=ah, cc=ac, full=af)   # look for a SMALL-k shoulder in hh; cc≈1 is architectural
