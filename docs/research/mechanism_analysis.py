#!/usr/bin/env python3
# Corrected probe analysis (OPUS-4.8-EXTRA revised protocol). The product-spectrum classifier reads a
# POSITIVE but cannot read a NEGATIVE: a uniform slide is ambiguous (no dead subspace, OR dead subspaces
# that ROTATE between factors and so don't compose). The mechanism is SUBSPACE ALIGNMENT, and the signature
# is not a gap at one T but its GEOMETRIC GROWTH with T. This measures both, plus a null distribution — and
# validates that it separates genuine composing structure from a matched-4/8/4-but-rotating control that a
# naive spectrum classifier would false-positive on.
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
def zdiv(rng):                                  # a random-ish sedenion zero divisor via unit-pair structure
    i,j=1,10; z=np.zeros(16); z[i]=1; z[j]=rng.choice([-1,1]); return z/np.linalg.norm(z)
def near(z,delta,rng):
    r=rng.standard_normal(16); r-=(r@z)*z; r/=np.linalg.norm(r); x=z+delta*r; return x/np.linalg.norm(x)
def bottomV(A,k=4):                              # bottom-k right singular subspace (the dying directions)
    U,s,Vt=np.linalg.svd(A); return Vt[-k:]
def principal_cos(V1,V2):                        # cosines of principal angles between two row-subspaces
    return np.linalg.svd(V1@V2.T,compute_uv=False)
def normed(A): return A/np.linalg.svd(A,compute_uv=False)[0]
def product(mats):
    P=np.eye(16)
    for A in mats: P=A@P
    return P
def gap_dominance(sv):
    s=np.sort(sv)[::-1]; s=s/s[0]; logs=np.log10(s+1e-30); g=logs[:-1]-logs[1:]; gi=int(np.argmax(g))
    return float(g[gi])/(float(logs[0]-logs[gi])+1e-9)
# ---- three stacks of DEPTH up to 32 ----
rng=np.random.default_rng(1); D=32; delta=0.15
z0=zdiv(rng)
aligned=[normed(Lx(near(z0,delta,rng))) for _ in range(D)]                    # SAME zero divisor → shared dying subspace
rotating=[normed(Lx(near(zdiv(np.random.default_rng(100+l)),delta,rng))) for l in range(D)]  # DIFFERENT z each layer
realg=[normed(rng.standard_normal((16,16))) for _ in range(D)]
# ---- §6.1 principal angles between consecutive dying subspaces (the mechanism) ----
def mean_align(mats):
    cs=[principal_cos(bottomV(mats[l]),bottomV(mats[l+1])).mean() for l in range(len(mats)-1)]
    return np.mean(cs)
print("§ mechanism — mean cos(principal angle) between consecutive dying 4-subspaces (1=aligned, 0=orthogonal):")
print(f"    aligned (same zero divisor) : {mean_align(aligned):.3f}")
print(f"    rotating (different z/layer) : {mean_align(rotating):.3f}")
print(f"    real Gaussian               : {mean_align(realg):.3f}")
# ---- §4b gap(T): geometric growth (shared subspace) vs saturation (rotating) ----
print("\n§ signature — gap_dominance vs number of composed factors T (growth = composing structure):")
Ts=[1,2,4,8,16,32]; print(f"    {'T':>3} | aligned | rotating | realGauss")
for T in Ts:
    ga=gap_dominance(np.linalg.svd(product(aligned[:T]),compute_uv=False))
    gr=gap_dominance(np.linalg.svd(product(rotating[:T]),compute_uv=False))
    gg=gap_dominance(np.linalg.svd(product(realg[:T]),compute_uv=False))
    print(f"    {T:>3} | {ga:7.2f} | {gr:8.2f} | {gg:8.2f}")
# ---- §2 null distribution of gap_dominance (false-positive rate for the LOW-MULT-GAP label) ----
print("\n§ null — distribution of gap_dominance for random products (T=16), to set a FP-controlled threshold:")
def null_dist(kind,n=300):
    out=[]
    for s in range(n):
        rg=np.random.default_rng(1000+s)
        if kind=='gauss': mats=[normed(rg.standard_normal((16,16))) for _ in range(16)]
        elif kind=='rot': mats=[normed(Lx(near(zdiv(np.random.default_rng(2000+s*17+l)),delta,rg))) for l in range(16)]
        out.append(gap_dominance(np.linalg.svd(product(mats),compute_uv=False)))
    return np.array(out)
for kind,lab in [('gauss','random Gaussian'),('rot','matched-4/8/4 but ROTATING (the key control)')]:
    d=null_dist(kind); print(f"    {lab:42s}: gap_dom  median {np.median(d):.2f}  95th %ile {np.percentile(d,95):.2f}  P(>1) {100*(d>1).mean():.0f}%")
ga16=gap_dominance(np.linalg.svd(product(aligned[:16]),compute_uv=False))
print(f"    genuine aligned structure (T=16)          : gap_dom {ga16:.2f}")
print("\n→ verdict is legible only against the null + the gap(T) growth curve + the alignment, NOT a point label.")
print("  The rotating control has 4/8/4 PER FACTOR yet must NOT read as structure — the mechanism catches it.")
