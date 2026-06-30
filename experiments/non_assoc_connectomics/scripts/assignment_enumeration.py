#!/usr/bin/env python3
"""Quantify the formula-ASSIGNMENT freedom (the residual selection confound).

Triality forces each adjacent-generation sqrt-mass ratio to be one of the three
ascending "edge types" on the Sym^3(3) weight triangle, with eigenvalues
(c_s - d, c_s, c_s + d) for a sector center c_s:
    b/a = c_s/(c_s-d)        c/a = (c_s+d)/(c_s-d)        c/b = (c_s+d)/c_s
For each single-edge ratio we DON'T fix which edge Singh chose; we enumerate ALL
edge assignments (3 per ratio) and ask: does demanding ONE shared delta across
sectors single out an assignment, and does it land on sqrt(3/8)? If only one
assignment is cross-sector-consistent, the "assignment freedom" is illusory.
"""
import numpy as np
DTH=np.sqrt(3/8)

# (label, sector center c_s, observed sqrt-mass ratio)  -- single-edge ratios only
ratios=[("tau/mu",1.0, np.sqrt(1776.86/105.6583755)),
        ("s/d",   1.0, np.sqrt(93.4/4.67)),
        ("c/u",   2.0/3.0, np.sqrt(1270.0/2.16)),
        ("t/c",   2.0/3.0, np.sqrt(172570.0/1270.0))]

def delta_for(edge, cs, R):
    if edge=="b/a": d=cs*(1.0-1.0/R)
    elif edge=="c/a": d=cs*(R-1.0)/(R+1.0)
    else:            d=cs*(R-1.0)            # c/b
    return d

edges=["b/a","c/a","c/b"]
print(f"algebraic delta = sqrt(3/8) = {DTH:.4f}\n")
print("per-ratio implied delta for each edge choice (valid 0<d<c_s):")
for lab,cs,R in ratios:
    row=[]
    for e in edges:
        d=delta_for(e,cs,R)
        ok = 0.0<d<cs
        row.append(f"{e}={d:.4f}{'' if ok else '*'}")
    print(f"  {lab:7s} (c_s={cs:.3f}, R={R:.3f}): " + "   ".join(row))
print("  (* = outside valid range 0<delta<c_s)\n")

# enumerate all 3^4 edge assignments; for each, delta per ratio, spread, distance to sqrt(3/8)
best=[]
import itertools
for combo in itertools.product(edges,repeat=4):
    ds=[]
    valid=True
    for (lab,cs,R),e in zip(ratios,combo):
        d=delta_for(e,cs,R)
        if not (0.05<d<cs): valid=False; break
        ds.append(d)
    if not valid: continue
    ds=np.array(ds); spread=ds.std(); mean=ds.mean()
    best.append((spread, mean, combo, ds))
best.sort(key=lambda x:x[0])

print(f"valid edge-assignments: {len(best)} of {3**4}")
print(f"\n{'rank':4s} {'spread(std)':>11s} {'mean delta':>10s} {'|mean-sqrt(3/8)|':>16s}  assignment")
for i,(sp,mn,combo,ds) in enumerate(best[:8]):
    print(f"{i+1:4d} {sp:11.4f} {mn:10.4f} {abs(mn-DTH):16.4f}  {combo}")

tol=0.02
consistent=[b for b in best if b[0]<=tol]
print(f"\nassignments with delta-spread <= {tol}: {len(consistent)}")
for sp,mn,combo,ds in consistent:
    print(f"   spread={sp:.4f} mean={mn:.4f} ({'MATCHES sqrt(3/8)' if abs(mn-DTH)<0.01 else 'off'})  {combo}  deltas={np.round(ds,4)}")
print(f"\n=> if exactly one cross-sector-consistent assignment exists and it is c/a everywhere")
print(f"   landing on sqrt(3/8), the 'assignment freedom' that inflates the confound is illusory.")
