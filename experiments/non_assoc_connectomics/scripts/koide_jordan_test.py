#!/usr/bin/env python3
"""Falsification test on REAL PDG data of the octonionic exceptional-Jordan
J3(O_C) mass-relation lineage (Singh 2508.10131; Todorov; Dubois-Violette).

The paper's parameter-free content: eigenvalue spectrum (q-d, q, q+d) with d^2=3/8,
sqrt(m) ~ monomial in eigenvalues. I show this is exactly Koide's relation, then
test Koide (the established, falsifiable relation this program reproduces) on data.

KEY BRIDGE (derived here): for sqrt(m) vector u=(sqrt m1,sqrt m2,sqrt m3),
  Koide Q = (sum m)/(sum sqrt m)^2 = 1/(3 cos^2 phi), phi=angle(u, (1,1,1)).
  Q=2/3  <=>  cos^2 phi = 1/2  <=>  phi = 45 deg  <=>  (for arith. spectrum
  u=(q-d,q,q+d))  3q^2 = 2 d^2  <=>  with q=1/2,  d^2 = 3/8.
So d^2=3/8 is literally the Koide 2/3 point. Testing Koide tests the law's core.
"""
import numpy as np

def koide(m):
    m = np.array(m, float)
    s = np.sqrt(m)
    Q = m.sum() / (s.sum()**2)
    # angle of sqrt-m vector to democratic axis (1,1,1)
    n = np.ones(3)/np.sqrt(3)
    cos2 = (s@n)**2 / (s@s)
    phi = np.degrees(np.arccos(np.sqrt(cos2)))
    return Q, phi

def arith_test(m):
    """Literal (q-d,q,q+d) reading: are sqrt(m) in arithmetic progression?
       i.e. 2*sqrt(m2) == sqrt(m1)+sqrt(m3) ?"""
    s = np.sqrt(sorted(m))
    return 2*s[1], s[0]+s[2]

# ---- PDG 2024 masses (MeV). Leptons: pole. Quarks: MS-bar (u,d,s @2GeV; c,b at own
# scale; t pole) — scheme/scale dependent, flagged. ----
leptons = {"e":0.51099895000, "mu":105.6583755, "tau":1776.86}
up      = {"u":2.16,  "c":1270.0,   "t":172570.0}     # u MSbar2GeV, c=mc(mc), t pole
down    = {"d":4.67,  "s":93.4,     "b":4180.0}       # MSbar
# uncertainties (approx, MeV) for a quick sensitivity band on quarks
up_err   = {"u":0.49, "c":20.0,  "t":290.0}
down_err = {"d":0.48, "s":3.4,   "b":30.0}

print("="*64)
print("KOIDE Q = (m1+m2+m3)/(sqrt m1+sqrt m2+sqrt m3)^2   [predict 2/3=0.66667]")
print("  equivalently: angle(sqrt-m, democratic axis) = 45.000 deg")
print("="*64)
for name, fam in [("charged leptons (e,mu,tau)", leptons),
                  ("up quarks (u,c,t)", up),
                  ("down quarks (d,s,b)", down)]:
    Q, phi = koide(list(fam.values()))
    print(f"\n{name}")
    print(f"  Q   = {Q:.6f}   (2/3 = 0.666667,  deviation = {(Q-2/3):+.6f})")
    print(f"  phi = {phi:.4f} deg   (predict 45.0000)")

# Monte-Carlo band for quarks (scheme/uncertainty sensitivity)
rng = np.random.default_rng(0)
for name, fam, err in [("up quarks", up, up_err), ("down quarks", down, down_err)]:
    qs = []
    for _ in range(20000):
        m = [max(1e-6, rng.normal(fam[k], err[k])) for k in fam]
        qs.append(koide(m)[0])
    qs = np.array(qs)
    print(f"\n{name}: Koide Q = {qs.mean():.4f} +/- {qs.std():.4f}  (1-sigma mass band)")

print("\n" + "="*64)
print("LITERAL arithmetic-spectrum test (q-d,q,q+d): is 2*sqrt(m2)=sqrt(m1)+sqrt(m3)?")
print("="*64)
for name, fam in [("charged leptons", leptons), ("up quarks", up), ("down quarks", down)]:
    lhs, rhs = arith_test(list(fam.values()))
    print(f"  {name:16s}: 2*sqrt(m2)={lhs:9.3f}   sqrt(m1)+sqrt(m3)={rhs:9.3f}   ratio={lhs/rhs:.3f}")

# Verify the d^2=3/8 <-> Koide bridge numerically
print("\n" + "="*64)
print("BRIDGE CHECK: arithmetic spectrum (1/2 - d, 1/2, 1/2 + d), d^2=3/8")
d = np.sqrt(3/8); spec = np.array([0.5-d, 0.5, 0.5+d])
Q = (spec**2).sum()/ (spec.sum()**2)
print(f"  eigenvalues = {spec.round(4)}   ->  Koide-form Q = {Q:.6f}  (=2/3 confirms bridge)")
print(f"  NOTE eigenvalue 1 = {spec[0]:.4f} < 0  => bare eigenvalue != sqrt(m);")
print(f"       the monomial weights a^p b^q c^r are REQUIRED (sqrt m must be > 0).")
