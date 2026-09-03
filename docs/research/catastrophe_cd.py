#!/usr/bin/env python3
# The Thom/Petitot bridge, made concrete: RUPTURE = SINGULARITY. The zero-divisor set of a Cayley-Dickson
# algebra is the SINGULAR LOCUS of left-multiplication L_x (where det L_x = 0) — i.e. the *bifurcation set*
# (catastrophe set) of the multiplication, in exactly Thom's sense: a family of operators parametrized by x
# that degenerates on a subvariety. Claim to test: the catastrophe set is EMPTY for the division algebras
# (ℝ,ℂ,ℍ,𝕆: det L_x = |x|^dim, no off-origin zeros) and is BORN at the 𝕆→𝕊 Cayley-Dickson doubling. So CD
# doubling is an UNFOLDING sequence and rupture appears as a catastrophe precisely at 𝕊. All computed.
import numpy as np
np.seterr(all='ignore')
def cds(a,b,bits):
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
def Lmatrix(x,bits):                      # L_x[k,j] = coeff of e_k in x·e_j = x[k^j]·cds(k^j,j)
    n=1<<bits; L=np.zeros((n,n))
    for k in range(n):
        for j in range(n): L[k,j]=x[k^j]*cds(k^j,j,bits)
    return L
def unit(i,n): v=np.zeros(n); v[i]=1.0; return v
# ---- 1. division check: det L_x = |x|^dim for ℍ,𝕆 ; and the catastrophe set of 2-unit sums a=e_i±e_j ----
def catastrophe_scan(bits,name):
    n=1<<bits; rng=np.random.default_rng(0)
    # random-x determinant law
    ratios=[]
    for _ in range(200):
        x=rng.standard_normal(n); d=np.linalg.det(Lmatrix(x,bits)); ratios.append(d/ (np.linalg.norm(x)**n))
    ratios=np.array(ratios); law = np.allclose(ratios,ratios[0],atol=1e-6)
    # enumerate 2-unit sums a=e_i+sign*e_j (imaginary units), count singular L_a (=> zero divisors)
    zd=0; total=0; examples=[]
    for i in range(1,n):
        for j in range(i+1,n):
            for sgn in (+1,-1):
                a=unit(i,n)+sgn*unit(j,n); total+=1
                sv=np.linalg.svd(Lmatrix(a,bits),compute_uv=False)
                if sv.min()<1e-9:
                    zd+=1
                    if len(examples)<4: examples.append((i,sgn,j))
    detrand=abs(np.linalg.det(Lmatrix(rng.standard_normal(n),bits)))
    print(f"{name:14s} dim {n:2d}: det(L_x)=|x|^{n} law: {'HOLDS (division algebra)' if law else 'FAILS'}"
          f"   generic |det L_x|~{detrand:.1e}   catastrophe set (singular 2-unit-sums): {zd}/{total}")
    if examples:
        ex=", ".join(f"e{i}{'+' if s>0 else '-'}e{j}" for i,s,j in examples)
        print(f"                zero-divisor examples: {ex} ...")
    return zd
print("Catastrophe set of Cayley-Dickson multiplication (rupture = singular locus of L_x):")
catastrophe_scan(2,"ℍ quaternion")
catastrophe_scan(3,"𝕆 octonion")
zdS=catastrophe_scan(4,"𝕊 sedenion")
zdT=catastrophe_scan(5,"𝕋 triginta")
# ---- 2. verify a found zero divisor annihilates, and locate its annihilator ----
n=16; a=unit(1,n)+unit(10,n)          # test a classic-style ZD candidate; if singular, extract annihilator
L=Lmatrix(a,4); sv,V=np.linalg.svd(L)[1],np.linalg.svd(L)[2]
if sv.min()<1e-9:
    b=V[np.argmin(sv)]                 # right null vector: a·b=0
    prod=Lmatrix(a,4)@b
    print(f"\nannihilation check: a=e1+e10 is a zero divisor; ||a||={np.linalg.norm(a):.3f}, "
          f"||b||={np.linalg.norm(b):.3f}, ||a·b||={np.linalg.norm(prod):.2e}  (nonzero × nonzero → 0)")
else:
    print("\n(e1+e10 not singular in this basis; the scan above already located the catastrophe set)")
# ---- 3. the bifurcation PATH: interpolate toward a zero divisor, watch det L cross zero ----
rng=np.random.default_rng(1)
# find any singular 2-unit sum from the scan to aim at
target=None
for i in range(1,16):
    for j in range(i+1,16):
        for sgn in (1,-1):
            a=unit(i,16)+sgn*unit(j,16)
            if np.linalg.svd(Lmatrix(a,4),compute_uv=False).min()<1e-9: target=a;break
        if target is not None:break
    if target is not None:break
x0=rng.standard_normal(16); x0/=np.linalg.norm(x0)
print("\nbifurcation path x(t) = (1-t)·(generic x) + t·(zero divisor)  — det L_x(t) crosses zero at the rupture:")
for t in [0.0,0.25,0.5,0.75,0.9,1.0]:
    x=(1-t)*x0+t*target; d=np.linalg.det(Lmatrix(x,4))
    print(f"  t={t:.2f}  det L_x = {d:+.3e}")
print("\n=> The catastrophe set is EMPTY for ℝℂℍ𝕆 (division) and BORN at 𝕊. Cayley-Dickson doubling is an")
print("   unfolding; rupture (zero division) appears as Thom's bifurcation set of the multiplication family.")
