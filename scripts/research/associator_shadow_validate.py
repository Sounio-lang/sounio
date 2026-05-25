#!/usr/bin/env python3
"""Associator-shadow experiment. For a ZD pair u·v=0, [p,u,v]=(p·u)·v.
Question: does any associator shadow ESCAPE the 7 fixed hyperovals that the
linear UNLEARN surgery is trapped in (i.e. produce a non-arc, a new arc, or a
configuration outside the 7 line-complements)?"""
import io, contextlib, importlib.util, itertools
from collections import Counter
spec = importlib.util.spec_from_file_location("fv", "/workspace/sounio/scripts/research/fano_arcs_blocking_validate.py")
fv = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(fv)
sedSigma = fv.sedSigma
validPrims = fv.validPrims; ozd = fv.orderedZDPairs

# ---- general sedenion multiplication on 16-dim integer vectors ----
def smul(a, b):
    c = [0]*16
    for i in range(16):
        if a[i]==0: continue
        for j in range(16):
            if b[j]==0: continue
            c[i ^ j] += a[i]*b[j]*sedSigma(i, j)
    return c

def vec(u):  # primitive -> 16-vec
    w=[0]*16; w[u.lo]=1; w[u.hi] = -1 if u.neg else 1; return w

def support(w): return [k for k in range(16) if w[k]!=0]
def loshadow(w):  # Fano point-shadow: lo-class of each support index, in {1..7}
    return sorted({(k & 7) for k in support(w) if (k & 7) in range(1,8)})

# ---- Fano lines / hyperovals ----
fanoLines=[(1,2,3),(1,4,5),(1,6,7),(2,4,6),(2,5,7),(3,4,7),(3,5,6)]
lineSets=[frozenset(L) for L in fanoLines]
def contains_line(s): ps=frozenset(s); return any(L<=ps for L in lineSets)
def is_arc(s): return not contains_line(s)
hyperovals = {frozenset(range(1,8))-L for L in lineSets}  # the 7 line-complements

# ZD pairs as vectors
zd_pairs = [(u,v) for (u,v) in ozd]  # u·v = 0 (verified)
print("ZD ordered pairs:", len(zd_pairs))

# sanity: u·v == 0 as full sedenion product
bad = sum(1 for (u,v) in zd_pairs if any(x!=0 for x in smul(vec(u),vec(v))))
print("ZD pairs with nonzero full product (should be 0):", bad)

# ---------- baseline: linear UNLEARN shadow family (for comparison) ----------
def annihilators(u): return [v for (a,v) in ozd if a==u]
lin_shadows = set()
for u in validPrims:
    lin_shadows.add(frozenset(a.lo for a in annihilators(u)))
print("\nLINEAR (UNLEARN) shadows: distinct =", len(lin_shadows),
      " all hyperovals? ", lin_shadows == hyperovals)

# ---------- associator shadows over probes p ----------
def run(probe_name, probes):
    shads = Counter()
    nonarc = 0; new_arc = 0; outside_hyperoval = 0; sizes = Counter()
    examples_nonarc = []; examples_outside = []
    for p in probes:
        pv = vec(p) if hasattr(p,'lo') else p
        for (u,v) in zd_pairs:
            w = smul(smul(pv, vec(u)), vec(v))   # (p·u)·v = associator (since u·v=0)
            sh = frozenset(loshadow(w))
            shads[sh]+=1
            sizes[len(sh)]+=1
            if not is_arc(sh):
                nonarc+=1
                if len(examples_nonarc)<4: examples_nonarc.append((p_repr(p),u,v,sorted(sh)))
            if sh not in hyperovals:
                outside_hyperoval+=1
                if len(examples_outside)<4: examples_outside.append((p_repr(p),u,v,sorted(sh)))
            if is_arc(sh) and sh not in hyperovals and len(sh)>=4:
                new_arc+=1
    print(f"\n=== probes = {probe_name} ({len(probes)}) ===")
    print("  distinct shadows:", len(shads))
    print("  shadow-size distribution:", dict(sorted(sizes.items())))
    print("  shadows that are NOT arcs (contain a Fano line):", nonarc)
    print("  shadows OUTSIDE the 7 hyperovals:", outside_hyperoval)
    print("  arcs outside the 7 hyperovals (size>=4):", new_arc)
    print("  distinct shadow SETS (sorted):")
    for sh,ct in sorted(shads.items(), key=lambda x:(len(x[0]),sorted(x[0]))):
        tag = "HYPEROVAL" if sh in hyperovals else ("arc" if is_arc(sh) else "NON-ARC")
        print(f"     {sorted(sh)!s:24} x{ct:<5} {tag}")
    if examples_nonarc:
        print("  example NON-ARC shadows:", examples_nonarc[:3])
    if examples_outside:
        print("  example OUTSIDE-hyperoval shadows:", examples_outside[:3])

def p_repr(p):
    return repr(p) if hasattr(p,'lo') else ("e"+str([k for k in range(16) if p[k]][0]))

# probe set 1: the 16 basis vectors e_0..e_15
basis = []
for i in range(16):
    w=[0]*16; w[i]=1; basis.append(w)
run("basis e_0..e_15", basis)

# probe set 2: the 84 valid primitives as base points
run("84 validPrims", validPrims)

print("\n=== one explicit witness triple per Fano line (shadow == that line) ===")
want = [frozenset(L) for L in fanoLines]
found = {}
for p in validPrims:
    for (u,v) in zd_pairs:
        sh = frozenset(loshadow(smul(smul(vec(p),vec(u)),vec(v))))
        if sh in want and sh not in found:
            found[sh] = (p, u, v)
    if len(found)==7: break
for L in fanoLines:
    p,u,v = found[frozenset(L)]
    print(f"  line {L}: p={p!r}, u={u!r}, v={v!r}  (u·v=0? {all(x==0 for x in smul(vec(u),vec(v)))})")

# confirm full characterization: distinct shadows == all subsets of {1..7} with |S|<=3
allsmall = set()
for p in validPrims:
    for (u,v) in zd_pairs:
        allsmall.add(frozenset(loshadow(smul(smul(vec(p),vec(u)),vec(v)))))
import itertools as it
target = set()
for k in range(4):
    for c in it.combinations(range(1,8),k): target.add(frozenset(c))
print("\n  distinct assoc shadows == {all S⊆{1..7}, |S|≤3}? ", allsmall==target, " (|fam|=",len(allsmall),")")
print("  max shadow size:", max(len(s) for s in allsmall))
