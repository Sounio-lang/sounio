#!/usr/bin/env python3
"""KILL THE SELECTION CONFOUND with a genuine held-out test.
The paper's group theory (Dynkin swap, trace split) forces relations with ZERO free
parameters and ties sectors together. Test predictivity ACROSS held-out sectors:
  (1) Pure group-forced EQUALITIES (no delta at all):
        - Dynkin swap: sqrt(m_tau/m_mu) == sqrt(m_s/m_d)
        - trace split: sqrt(m_e):sqrt(m_u):sqrt(m_d) == 1:2:3
  (2) Cross-sector HELD-OUT: fit the single delta on QUARKS only, then PREDICT the
      lepton ratios (forced by the swap) with NO lepton data used. And vice-versa.
If quark-fit delta predicts leptons to a few %, the structure is predictive, not fitted.
"""
import numpy as np
rng=np.random.default_rng(0); DTH=np.sqrt(3/8)
lep={"e":(0.51099895,1e-7),"mu":(105.6583755,2e-6),"tau":(1776.86,0.12)}
qk ={"u":(2.16,0.49),"d":(4.67,0.48),"s":(93.4,3.4),"c":(1270.,20.),"b":(4180.,30.),"t":(172570.,290.)}
def smp(d): return {k:max(1e-9,rng.normal(m,s)) for k,(m,s) in d.items()}

# forms
A   =lambda d:(1+d)/(1-d)
Asq =lambda d:(1+d)**2/(1-d)
C1  =lambda d:(2/3+d)/(2/3-d)
C2  =lambda d:(2/3)/(2/3-d)
def fit_delta(targets):  # targets: list of (form, observed sqrt-ratio); LS in delta
    ds=np.linspace(1/3+1e-3, 2/3-1e-3, 60001)   # valid range: all forms positive
    err=np.zeros_like(ds)
    for form,obs in targets:
        v=form(ds); err+=np.where(v>0,(np.log(np.maximum(v,1e-12))-np.log(obs))**2,1e9)
    return ds[np.argmin(err)]

print("="*70)
print("(1) PURE GROUP-FORCED EQUALITIES (zero free parameters)")
print("="*70)
# Monte Carlo
sd_eq=[]; gen1=[]
for _ in range(20000):
    L=smp(lep); Q=smp(qk)
    sd_eq.append((np.sqrt(L["tau"]/L["mu"]), np.sqrt(Q["s"]/Q["d"])))
    gen1.append((np.sqrt(Q["u"]/L["e"]), np.sqrt(Q["d"]/L["e"])))
sd_eq=np.array(sd_eq); gen1=np.array(gen1)
print(f"  Dynkin swap  sqrt(m_tau/m_mu) == sqrt(m_s/m_d):")
print(f"     {sd_eq[:,0].mean():.3f} vs {sd_eq[:,1].mean():.3f}  -> off by {100*abs(sd_eq[:,0].mean()-sd_eq[:,1].mean())/sd_eq[:,1].mean():.1f}%")
print(f"  trace split  sqrt(m_e):sqrt(m_u):sqrt(m_d) == 1:2:3:")
print(f"     1 : {gen1[:,0].mean():.3f} : {gen1[:,1].mean():.3f}   (predict 1:2:3, off {100*abs(gen1[:,0].mean()-2)/2:.0f}% / {100*abs(gen1[:,1].mean()-3)/3:.0f}%)")

print("\n"+"="*70)
print("(2) CROSS-SECTOR HELD-OUT: fit delta on QUARKS, predict LEPTONS (zero lepton freedom)")
print("="*70)
tau_pred=[]; mu_pred=[]; dq=[]
for _ in range(20000):
    L=smp(lep); Q=smp(qk)
    d=fit_delta([(A,np.sqrt(Q["s"]/Q["d"])),(Asq,np.sqrt(Q["b"]/Q["s"])),
                 (C1,np.sqrt(Q["c"]/Q["u"])),(C2,np.sqrt(Q["t"]/Q["c"]))])
    dq.append(d)
    pred_tau=A(d); obs_tau=np.sqrt(L["tau"]/L["mu"])
    pred_mu =A(d)*(d+1/3)/(d-1/3); obs_mu=np.sqrt(L["mu"]/L["e"])
    tau_pred.append((pred_tau,obs_tau)); mu_pred.append((pred_mu,obs_mu))
dq=np.array(dq); tau_pred=np.array(tau_pred); mu_pred=np.array(mu_pred)
print(f"  delta fit on quarks only: {dq.mean():.4f} +/- {dq.std():.4f}   (algebraic sqrt(3/8)={DTH:.4f})")
print(f"  PREDICT sqrt(m_tau/m_mu): {tau_pred[:,0].mean():.3f}  vs observed {tau_pred[:,1].mean():.3f}  -> {100*(tau_pred[:,0].mean()/tau_pred[:,1].mean()-1):+.1f}%")
print(f"  PREDICT sqrt(m_mu/m_e):   {mu_pred[:,0].mean():.3f} vs observed {mu_pred[:,1].mean():.3f}  -> {100*(mu_pred[:,0].mean()/mu_pred[:,1].mean()-1):+.1f}%")

print("\n  Reverse: fit delta on LEPTONS, predict QUARKS")
cu=[]; tc=[]; dl=[]
for _ in range(20000):
    L=smp(lep); Q=smp(qk)
    d=fit_delta([(A,np.sqrt(L["tau"]/L["mu"])),(lambda x:A(x)*(x+1/3)/(x-1/3),np.sqrt(L["mu"]/L["e"]))])
    dl.append(d)
    cu.append((C1(d),np.sqrt(Q["c"]/Q["u"]))); tc.append((C2(d),np.sqrt(Q["t"]/Q["c"])))
dl=np.array(dl); cu=np.array(cu); tc=np.array(tc)
print(f"  delta fit on leptons only: {dl.mean():.4f} +/- {dl.std():.4f}")
print(f"  PREDICT sqrt(m_c/m_u): {cu[:,0].mean():.2f} vs observed {cu[:,1].mean():.2f}  -> {100*(cu[:,0].mean()/cu[:,1].mean()-1):+.1f}%")
print(f"  PREDICT sqrt(m_t/m_c): {tc[:,0].mean():.2f} vs observed {tc[:,1].mean():.2f}  -> {100*(tc[:,0].mean()/tc[:,1].mean()-1):+.1f}%")
print("\n  (pole-scale masses; honest test of cross-sector predictivity with one shared delta)")
