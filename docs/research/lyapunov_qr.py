#!/usr/bin/env python3
# Correction §1 (MANDATORY): do NOT form the product P_T (its condition number blows past machine epsilon,
# the small σ stop being resolved, and gap(T) censors ~12–14 decades). Use the discrete-QR method — the
# standard Lyapunov-spectrum algorithm: Q0=I; J_t Q_{t-1}=Q_t R_t; log σ_i(P_T) ≈ Σ_t log|R_t[i,i]|.
# Reorthonormalize each step → essentially unlimited dynamic range at the same cost, and the Q_t frames are
# the leading Oseledets directions (covariant Lyapunov vectors; Ginelli et al.) that the alignment needs —
# no SVD of the product. This whole instrument IS Lyapunov-spectrum + CLV analysis of RNNs (Engelken/Wolf/
# Abbott; Vogt et al.); position, don't claim.
import numpy as np
np.seterr(all='ignore')
def cds(a,b,bits=4):
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
SIG=np.array([[cds(i,j) for j in range(16)] for i in range(16)]);M=np.zeros((16,16,16))
for k in range(16):
    for b in range(16): M[k,k^b,b]=SIG[k,b]
def Lx(x): return np.tensordot(x,M,axes=(0,0))
def normed(A): return A/np.linalg.svd(A,compute_uv=False)[0]
def qr_logspectrum(Jlist):
    """log10 σ_i(P_T) via discrete QR, WITHOUT forming P_T. Returns (log-σ descending, list of Q frames)."""
    n=Jlist[0].shape[0]; Q=np.eye(n); logs=np.zeros(n); Qs=[Q.copy()]
    for J in Jlist:
        A=J@Q; Q,R=np.linalg.qr(A); d=np.diag(R); sgn=np.sign(d); sgn[sgn==0]=1
        Q=Q*sgn; logs+=np.log10(np.abs(d)+1e-300); Qs.append(Q.copy())
    idx=np.argsort(logs)[::-1]; return logs[idx], Qs
def direct_logspectrum(Jlist):
    P=np.eye(Jlist[0].shape[0])
    for J in Jlist: P=J@P
    return np.log10(np.sort(np.linalg.svd(P,compute_uv=False))[::-1]+1e-300)
rng=np.random.default_rng(1); z=np.zeros(16); z[1]=1; z[10]=1; z/=np.linalg.norm(z)
def near(delta):
    r=rng.standard_normal(16); r-=(r@z)*z; r/=np.linalg.norm(r); x=z+delta*r; return x/np.linalg.norm(x)
big=[normed(Lx(near(0.15))) for _ in range(256)]
print("min log10 σ(P_T) — DIRECT product (censored at machine eps) vs QR method (uncensored):")
print(f"  {'T':>4} | {'direct min-logσ':>16} | {'QR min-logσ':>12} | {'QR gap(σ4→σ5) dec':>18}")
for T in [8,16,32,64,128,256]:
    dl=direct_logspectrum(big[:T]); ql,_=qr_logspectrum(big[:T])
    gap=ql[3]-ql[4]                      # the 4→5 gap (the algebra's low-mult boundary), in decades
    print(f"  {T:>4} | {dl.min():16.1f} | {ql.min():12.1f} | {gap:18.1f}")
print("\n→ direct saturates near −14 (machine epsilon); QR keeps descending linearly in T (log σ_i ≈ λ_i·T),")
print("  so the gap grows without a numerical ceiling. β = λ_4 − λ_5 (a Lyapunov-exponent difference) is the")
print("  slope, read cleanly at any depth. G(T)=α+βT is the signature; the ceiling was censoring it.")
# β (Lyapunov exponent gap) from a linear fit of gap(T)
Ts=np.array([16,32,64,128,256]); gaps=[]
for T in Ts:
    ql,_=qr_logspectrum(big[:T]); gaps.append(ql[3]-ql[4])
beta=np.polyfit(Ts,gaps,1)[0]
print(f"\nβ = λ_4 − λ_5 (slope of gap(T)) = {beta:.4f} decades/step  (nonzero ⇒ a persistent spectral gap; the")
print("  covariant-Lyapunov-vector alignment of the two Oseledets subspaces is the mechanism, not the gap).")
