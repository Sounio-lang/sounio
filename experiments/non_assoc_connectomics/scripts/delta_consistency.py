#!/usr/bin/env python3
"""DECISIVE physics test: the octonion exceptional-Jordan program FIXES one algebraic
constant, delta^2 = 3/8  (delta = 0.612372). Does the real fermion mass data WANT that
value? For each Table-I sqrt-mass-ratio (DIFFERENT functional forms, DIFFERENT sectors),
invert the closed form to solve for the delta that exactly reproduces the observed ratio,
propagate PDG uncertainties (Monte Carlo), and see whether the implied deltas CONVERGE
on sqrt(3/8) across independent sectors. Convergence across different functional forms =
the constant is real; wide scatter = post-hoc fitting.
"""
import numpy as np
rng=np.random.default_rng(0)
DELTA_TH=np.sqrt(3/8)   # 0.6123724

# PDG masses (MeV). Leptons pole (scale-stable). Quarks: give BOTH pole-ish and note RG.
# central, 1-sigma
lep={"e":(0.51099895,1e-7),"mu":(105.6583755,2e-6),"tau":(1776.86,0.12)}
# quark MSbar (PDG): u,d,s @2GeV ; c=mc(mc); b=mb(mb); t pole
qk ={"u":(2.16,0.49),"d":(4.67,0.48),"s":(93.4,3.4),"c":(1270.,20.),"b":(4180.,30.),"t":(172570.,290.)}

def sample(d):
    return {k:max(1e-9,rng.normal(m,s)) for k,(m,s) in d.items()}

# inverse formulas: given observed R = sqrt(mass-ratio), return implied delta
def inv_A(R):        # R = (1+d)/(1-d)
    return (R-1)/(R+1)
def inv_Asq(R):      # R = (1+d)^2/(1-d)  -> solve numerically
    ds=np.linspace(0.01,0.95,9401); vals=(1+ds)**2/(1-ds)
    return ds[np.argmin(np.abs(vals-R))]
def inv_C1(R):       # R = (2/3+d)/(2/3-d)
    return (2/3)*(R-1)/(R+1)
def inv_C2(R):       # R = (2/3)/(2/3-d)
    return (2/3)*(1-1/R)

# Each ratio: (label, sector, function to get observed sqrt-ratio from masses, inverse)
ratios=[
 ("sqrt(m_tau/m_mu)","lepton", lambda m: np.sqrt(m["tau"]/m["mu"]), inv_A),
 ("sqrt(m_s/m_d)",   "down-qk", lambda m: np.sqrt(m["s"]/m["d"]),   inv_A),
 ("sqrt(m_b/m_s)",   "down-qk", lambda m: np.sqrt(m["b"]/m["s"]),   inv_Asq),
 ("sqrt(m_c/m_u)",   "up-qk",   lambda m: np.sqrt(m["c"]/m["u"]),   inv_C1),
 ("sqrt(m_t/m_c)",   "up-qk",   lambda m: np.sqrt(m["t"]/m["c"]),   inv_C2),
]

print(f"Algebraic prediction: delta = sqrt(3/8) = {DELTA_TH:.5f}\n")
print(f"{'ratio':18s} {'sector':9s} {'obs sqrt-ratio':>15s} {'implied delta':>18s} {'(th-impl)/sig':>13s}")
print("-"*78)
all_means=[]; all_sigs=[]
for lab,sec,f,inv in ratios:
    ds=[]
    for _ in range(20000):
        m=sample(lep if sec=="lepton" else qk)
        R=f(m); ds.append(inv(R))
    ds=np.array(ds); mu=ds.mean(); sg=ds.std()
    all_means.append(mu); all_sigs.append(sg)
    z=(DELTA_TH-mu)/sg if sg>0 else 0
    print(f"{lab:18s} {sec:9s} {f({k:v[0] for k,v in (lep if sec=='lepton' else qk).items()}):15.4f} {mu:11.4f} +/-{sg:6.4f} {z:+13.2f}")

all_means=np.array(all_means)
print("\n  Implied-delta spread across 5 ratios (3 functional forms, 3 sectors):")
print(f"    mean = {all_means.mean():.4f}   std = {all_means.std():.4f}   range = [{all_means.min():.4f}, {all_means.max():.4f}]")
print(f"    algebraic sqrt(3/8) = {DELTA_TH:.4f}")
print(f"    => all 5 implied deltas within {100*max(abs(all_means-DELTA_TH))/DELTA_TH:.1f}% of sqrt(3/8)")
# Null: if formulas were arbitrary, implied delta would scatter over (0,1). Quantify clustering.
print(f"\n  Clustering: std/mean = {all_means.std()/all_means.mean():.3f}  (tight cluster near one value = constant is real)")
