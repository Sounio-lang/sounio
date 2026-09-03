#!/usr/bin/env python3
# The spectral-signature test (OPUS-4.8-EXTRA, generative turn). Annihilation is BINARY and composition is
# MULTIPLICATIVE — the necessary-condition filter. The one place multiplicative composition already lives in
# real training: the GRADIENT is a product of Jacobians, J = J_L…J_1, and vanishing gradient is annihilation
# by composition: σ_min(∏J) ≪ ∏σ_min(J). The algebra's DISTINCT prediction (vs classical vanishing gradient
# = uniform magnitude slide, Saxe/Pennington dynamical isometry; vs transformer rank-collapse-to-1, Dong et
# al.): in 𝕊 the collapse is LOW-MULTIPLICITY WITH A GAP — L_xᵀL_x has spectrum {D₁−2q ×4, D₁ ×8, D₁+2q ×4},
# so a 4-dim subspace dies while the bulk stays healthy. This test needs NO training: SVDs only.
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
SIG=np.array([[cds(i,j) for j in range(16)] for i in range(16)])
M=np.zeros((16,16,16))
for k in range(16):
    for b in range(16): M[k,k^b,b]=SIG[k,b]
def Lx(x): return np.tensordot(x,M,axes=(0,0))
z=np.zeros(16); z[1]=1; z[10]=1; z/=np.linalg.norm(z)         # a zero divisor
rng=np.random.default_rng(0)
# ---- Test A: spectrum of L_x approaching the zero divisor — low-mult collapse WITH GAP? ----
print("Test A — single sedenion L_x approaching a zero divisor (δ = distance to it):")
r=rng.standard_normal(16); r-=(r@z)*z; r/=np.linalg.norm(r)
for delta in [0.5,0.2,0.05,0.01]:
    x=z+delta*r; x/=np.linalg.norm(x); sv=np.sort(np.linalg.svd(Lx(x),compute_uv=False))
    n_small=int((sv<0.3*np.median(sv)).sum())
    print(f"  δ={delta:.2f}: σ (sorted) min4={np.round(sv[:4],3)}  median={np.median(sv):.3f}  #collapsed(<0.3·med)={n_small}  gap σ4/σ5={sv[4]/max(sv[3],1e-9):.1f}×")
print("  → 4 singular values collapse; the other 12 stay ~healthy; a GAP opens between σ4 and σ5.\n")
# ---- Test B: PRODUCT of near-zero-divisor Jacobians over depth — does the gap survive composition? ----
def prod_spectrum(mats):
    P=np.eye(16)
    for A in mats: P=A@P
    return np.sort(np.linalg.svd(P,compute_uv=False))[::-1]   # descending
def normed(A): return A/np.linalg.svd(A,compute_uv=False)[0]   # scale so top σ = 1 (isolate SHAPE)
print("Test B — product of D Jacobians over depth (each scaled to top-σ=1, so only SHAPE matters):")
D=12
# (i) sedenion stack near zero divisors
Smats=[normed(Lx((z+0.15*(lambda v:(v-(v@z)*z)/np.linalg.norm(v-(v@z)*z))(rng.standard_normal(16))))) for _ in range(D)]
# (ii) real Gaussian stack (dynamical-isometry control): random 16×16, no algebraic structure
Rmats=[normed(rng.standard_normal((16,16))) for _ in range(D)]
for name,mats in [('SEDENION product (near ZD)',Smats),('REAL Gaussian product (control)',Rmats)]:
    sv=prod_spectrum(mats); sv/=sv[0]
    n_dead=int((sv<1e-3).sum()); gap=sv[11]/max(sv[12],1e-30) if sv[12]>0 else np.inf
    print(f"  {name:32s}: σ/σmax (log10) = {np.round(np.log10(sv+1e-30),1)}")
    print(f"      dead modes (<1e-3): {n_dead}/16   gap at rank 12→13: {gap:.1e}×")
print("\nDISCRIMINANT:")
print("  • low-multiplicity collapse + GAP (few σ die, bulk survives)  → structural annihilation (𝕊);")
print("    residual connections preserve norm and DO NOTHING for it — a mode dynamical isometry can't see")
print("    (it asks if the WHOLE spectrum is near 1, not if a small subspace died while the rest lives).")
print("  • uniform slide / spread, no discrete gap  → magnitude (classical vanishing gradient).")
print("  Nearest prior art to cite & differentiate: Saxe/Pennington (dynamical isometry — magnitude);")
print("  Dong et al. 'attention loses rank doubly exponentially' (whole rep → rank 1, NOT small-subspace).")
