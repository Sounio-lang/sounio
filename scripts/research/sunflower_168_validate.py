#!/usr/bin/env python3
"""Independent census mirror for SounioSunflower.lean (Erdős [20] sunflower
structure on the verified 168/ZD/Surgical set system).

Reuses the validated cdSigma/PrimSed/ZD mirror in fano_arcs_blocking_validate.py.
Pre-validates every native_decide theorem in formal/lean4/SounioSunflower.lean.
"""
import itertools, io, contextlib, importlib.util, os
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "fv", os.path.join(_HERE, "fano_arcs_blocking_validate.py"))
fv = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(fv)

validPrims, orderedZDPairs, xorLabel = fv.validPrims, fv.orderedZDPairs, fv.xorLabel
idx = {p.key(): i for i, p in enumerate(validPrims)}
def pid(p): return idx[p.key()]
def annset(u): return frozenset(pid(v) for (a, v) in orderedZDPairs if a == u)

def is_sunflower(sets):
    if len(sets) < 2: return True
    core = sets[0] & sets[1]
    return all((a & b) == core for a, b in itertools.combinations(sets, 2))

ok = True
def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")

print("=== edge system: every point is a 4-petal star sunflower core ===")
check("every primitive has exactly 4 distinct annihilators (!= self)",
      all(len(annset(u)) == 4 and pid(u) not in annset(u) for u in validPrims))

print("=== neighborhood system (84 annihilator 4-sets) ===")
F = {pid(u): annset(u) for u in validPrims}
check("each neighborhood lies in one fiber",
      all(all(xorLabel(validPrims[a]) == xorLabel(u) for a in annset(u)) for u in validPrims))

fibers = {}
for u in validPrims: fibers.setdefault(xorLabel(u), set()).add(pid(u))
labels = sorted(fibers)
check("7 fibers, pairwise disjoint (canonical 7-petal sunflower)",
      labels == [9,10,11,12,13,14,15] and
      all(not (fibers[a] & fibers[b]) for a, b in itertools.combinations(labels, 2)))

# fiber-wise: no 3 DISTINCT neighborhoods form a sunflower; no 3 pairwise disjoint
sf_free = disj_le2 = True
for L in labels:
    fp = [u for u in validPrims if xorLabel(u) == L]
    sets = [annset(u) for u in fp]
    for i, j, k in itertools.combinations(range(len(fp)), 3):
        a, b, c = sets[i], sets[j], sets[k]
        distinct = a != b and a != c and b != c
        if distinct and is_sunflower([a, b, c]): sf_free = False
        if distinct and not (a & b) and not (a & c) and not (b & c): disj_le2 = False
check("fiber-wise 3-sunflower-free (max 2 petals)", sf_free)
check("at most 2 pairwise-disjoint neighborhoods per fiber", disj_le2)

print("=== 14-petal empty-core witness ===")
W = [(2,11,False),(2,11,True),(1,11,False),(1,11,True),(1,10,False),(1,10,True),
     (1,13,False),(1,13,True),(1,12,False),(1,12,True),(1,15,False),(1,15,True),
     (1,14,False),(1,14,True)]
def find(lo,hi,neg):
    for u in validPrims:
        if (u.lo,u.hi,u.neg)==(lo,hi,neg): return u
    return None
wit = [find(*w) for w in W]
check("all 14 witness elements valid & distinct", all(wit) and len(set(map(pid,wit)))==14)
wsets = [annset(u) for u in wit]
check("14 witness neighborhoods pairwise disjoint (empty-core 14-petal sunflower)",
      all(not (a & b) for a, b in itertools.combinations(wsets, 2)))

print("=== surgical-op cardinalities (dictionary) ===")
def gate(u): return [v for v in validPrims if u!=v and not fv.isZeroPair(u,v)]
check("UNLEARN=4, EDIT=12, GATE=79 for all",
      all(len(annset(u))==4 and len(fibers[xorLabel(u)])==12 and len(gate(u))==79
          for u in validPrims))

print("\nALL CHECKS PASS" if ok else "\nSOME CHECK FAILED")
raise SystemExit(0 if ok else 1)
