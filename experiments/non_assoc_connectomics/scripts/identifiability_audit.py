#!/usr/bin/env python3
"""ADVERSARIAL identifiability audit of the delta^2=3/8 claim (break, don't build).

Audit 1 (joint identifiability): §4.8 FIXED delta=sqrt(3/8) then 'recovered' centers
 -> near-circular. Now profile the fit chi^2 over delta with centers FREE (fit per
 sector). If chi^2(delta) is flat, delta is NOT identified by the masses alone; it is
 degenerate with the centers, and 'masses select delta^2=3/8' is overstated.

Audit 2 (full-pipeline null): run centers-from-charges on SCRAMBLED mass spectra.
 If random masses also yield centers ~ {1, 2/3}, the machinery manufactures structure.
"""
import numpy as np
rng=np.random.default_rng(0)
DTH=np.sqrt(3/8)

# observed single-edge sqrt-mass ratios and their edge form + sector
me,mmu,mtau=0.51099895,105.6583755,1776.86
mu_,mc,mt=2.16,1270.0,172570.0
md,ms,mb=4.67,93.4,4180.0
R_taumu=np.sqrt(mtau/mmu); R_sd=np.sqrt(ms/md); R_cu=np.sqrt(mc/mu_); R_tc=np.sqrt(mt/mc)

def ca(c,d): return (c+d)/(c-d)
def ba(c,d): return c/(c-d)

# ---- Audit 1: profile chi^2(delta), centers free ----
def sector_resid(d):
    # lepton (taumu, c/a) center free -> exact fit, resid 0
    # down  (sd,    c/a) center free -> exact fit, resid 0
    # up    (cu c/a, tc b/a) one center c_U, two eqs -> min over c_U
    cu_grid=np.linspace(d+1e-3, 5.0, 20000)  # c_U > d
    r = (np.log(ca(cu_grid,d))-np.log(R_cu))**2 + (np.log(ba(cu_grid,d))-np.log(R_tc))**2
    return r.min()

deltas=np.linspace(0.40,0.65,26)
chi=np.array([sector_resid(d) for d in deltas])
print("AUDIT 1 - profile chi^2(delta) with centers FREE (up sector only; lep/down fit any delta):")
for d,x in zip(deltas[::5],chi[::5]):
    print(f"   delta={d:.3f}  chi^2_up={x:.3e}")
print(f"   chi^2 range over delta in [0.40,0.65]: min={chi.min():.2e} max={chi.max():.2e}  ratio={chi.max()/max(chi.min(),1e-12):.1f}")
# best-fit delta/c_U from up sector (scale-free constraint)
cu_grid=np.linspace(0.3,1.2,40000)
best=None
for cuU in cu_grid:
    for d in np.linspace(0.1,cuU-1e-3,400):
        r=(np.log(ca(cuU,d))-np.log(R_cu))**2+(np.log(ba(cuU,d))-np.log(R_tc))**2
        if best is None or r<best[0]: best=(r,cuU,d)
print(f"   up-sector best fit: c_U={best[1]:.3f}, delta={best[2]:.3f}, delta/c_U={best[2]/best[1]:.4f}")
print(f"   compare sqrt(3/8)/(2/3) = {DTH/(2/3):.4f}  (the up spread/center ratio IS what's pinned, not delta alone)")
print("   => delta is identified ONLY given an external center (electric charge); chi^2(delta) ~ flat otherwise.\n")

# ---- Audit 2: full-pipeline null on scrambled masses ----
def implied_center_ca(R,d): return d*(R+1)/(R-1)
def implied_center_ba(R,d): return d*R/(R-1)
# real: fix delta=sqrt(3/8), recover centers
real_cs=[implied_center_ca(R_taumu,DTH), implied_center_ca(R_sd,DTH),
         implied_center_ca(R_cu,DTH),    implied_center_ba(R_tc,DTH)]
charges=[1.0,1.0,2/3,2/3]
real_dev=np.mean([abs(c-q)/q for c,q in zip(real_cs,charges)])
print(f"AUDIT 2 - centers-from-charges, real masses: implied centers={np.round(real_cs,3)}")
print(f"   mean |center-charge|/charge = {100*real_dev:.1f}%")
# scrambled: random masses spanning similar hierarchy; recover centers, measure dev to {1,1,2/3}
def rand_ratio(): return np.exp(rng.uniform(np.log(2),np.log(60)))  # random sqrt-ratio 2..60
better=0; N=20000; valid=0
for _ in range(N):
    Rs=[rand_ratio() for _ in range(4)]
    cs=[implied_center_ca(Rs[0],DTH),implied_center_ca(Rs[1],DTH),
        implied_center_ca(Rs[2],DTH),implied_center_ba(Rs[3],DTH)]
    if any(c<=0 for c in cs): continue
    valid+=1
    dev=np.mean([abs(c-q)/q for c,q in zip(cs,charges)])
    if dev<=real_dev: better+=1
print(f"   scrambled masses with centers within real dev of charges: {better}/{valid} = {better/valid:.4f}")
print(f"   => if small, the centers~charges match is selective (real); if large, manufactured.")
