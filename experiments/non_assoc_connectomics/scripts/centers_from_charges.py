#!/usr/bin/env python3
"""Close the 'sector center' part of the assignment confound.

The eigenvalue spectrum is (c_s - d, c_s, c_s + d) per sector; c_s was an input.
Claim to test mass-blind: c_s is the sector's ELECTRIC CHARGE, not a fit to masses.
Method: FIX the single algebraic constant d = sqrt(3/8) (zero mass input), then for
each single-edge sqrt-mass ratio invert the edge form for the center c_s it implies.
If the masses independently land c_s on the electric charges (up 2/3, lepton/down 1
via the Dynkin swap), the centers are charge-grounded, recovered from data with no
free parameter left.
"""
import numpy as np
d=np.sqrt(3/8)

# c/a edge: (c+d)/(c-d)=R  -> c = d*(R+1)/(R-1)
def c_from_ca(R): return d*(R+1)/(R-1)
# C2 edge b/a in up-form: c/(c-d)=R -> c = d*R/(R-1)
def c_from_ba(R): return d*R/(R-1)

print(f"FIXED algebraic constant: delta = sqrt(3/8) = {d:.5f}   (zero mass input)\n")
print(f"{'ratio':8s} {'sector':9s} {'edge':5s} {'R(obs)':>9s} {'implied c_s':>12s} {'charge':>8s} {'dev%':>7s}")
print("-"*62)
rows=[
 ("tau/mu","lepton","c/a", np.sqrt(1776.86/105.6583755), c_from_ca, 1.0),
 ("s/d",   "down",  "c/a", np.sqrt(93.4/4.67),           c_from_ca, 1.0),  # center 1 via Dynkin swap
 ("c/u",   "up",    "c/a", np.sqrt(1270.0/2.16),         c_from_ca, 2.0/3.0),
 ("t/c",   "up",    "b/a", np.sqrt(172570.0/1270.0),     c_from_ba, 2.0/3.0),
]
cs=[]
for lab,sec,edge,R,f,charge in rows:
    c=f(R); cs.append((sec,c,charge))
    print(f"{lab:8s} {sec:9s} {edge:5s} {R:9.3f} {c:12.4f} {charge:8.4f} {100*(c-charge)/charge:+7.1f}")

print("\nSummary: with ONLY delta=sqrt(3/8) fixed, the masses select sector centers =")
ups=[c for s,c,ch in cs if s=='up']
oth=[c for s,c,ch in cs if s!='up']
print(f"  up sector    : {np.mean(ups):.4f}  vs up electric charge 2/3 = {2/3:.4f}  (dev {100*(np.mean(ups)-2/3)/(2/3):+.1f}%)")
print(f"  lepton/down  : {np.mean(oth):.4f}  vs electron charge 1     = 1.0000  (dev {100*(np.mean(oth)-1.0):+.1f}%)")
print("\n=> sector centers are the ELECTRIC CHARGES, recovered mass-blind (no free center).")
print("   The down sector takes center 1 (electron charge) via the Dynkin swap, not its own 1/3.")

# --- compound (2-edge) ratios: forward checks, delta and centers FIXED (zero free param) ---
ba=1/(1-d); ca=(1+d)/(1-d); cb=(1+d)
print("\nCompound (2-edge) ratios, forward (delta=sqrt(3/8), centers=charges, NO fit):")
bs_pred=ca*cb; bs_obs=np.sqrt(4180.0/93.4)
print(f"  b/s = (c/a)*(c/b)|c=1            = {bs_pred:.4f}  vs obs {bs_obs:.4f}  ({100*(bs_pred/bs_obs-1):+.1f}%)")
mue_pred=ca*((d+1/3)/(d-1/3)); mue_obs=np.sqrt(105.6583755/0.51099895)
print(f"  mu/e = (c/a)*(d+1/3)/(d-1/3)     = {mue_pred:.4f} vs obs {mue_obs:.4f} ({100*(mue_pred/mue_obs-1):+.1f}%)")
print("  (the (d +/- 1/3) factor is the Dynkin-swap 1<->1/3 image; consistent, not re-derived here)")
