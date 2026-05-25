#!/usr/bin/env python3
"""Mirror of the Lean defs in SounioCayleyDickson / SounioZeroDivisorBridge,
used to PRE-VALIDATE every native_decide theorem before writing Lean
(lean toolchain is not installed locally)."""

import sys
from functools import lru_cache

# ---- cdSigma (mirror of SounioCayleyDickson.cdSigma) ----
@lru_cache(maxsize=None)
def cdSigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aHi = a >= half
    bHi = b >= half
    aLo = a & (half - 1)
    bLo = b & (half - 1)
    if (not aHi) and (not bHi):
        return cdSigma(aLo, bLo, bits - 1)
    elif (not aHi) and bHi:
        return cdSigma(bLo, aLo, bits - 1)
    elif aHi and (not bHi):
        return cdSigma(aLo, bLo, bits - 1) if bLo == 0 else -cdSigma(aLo, bLo, bits - 1)
    else:
        return -cdSigma(bLo, aLo, bits - 1) if bLo == 0 else cdSigma(bLo, aLo, bits - 1)

def octSigma(a, b): return cdSigma(a, b, 3)
def sedSigma(a, b): return cdSigma(a, b, 4)

# ---- 168 theorem (octonions) ----
def alphaSign(i, j, k): return octSigma(i, j) * octSigma(i ^ j, k)
def betaSign(i, j, k):  return octSigma(j, k) * octSigma(i, j ^ k)
def waveFunc(i, j, k):  return alphaSign(i, j, k) - betaSign(i, j, k)
def isFano(i, j, k):    return alphaSign(i, j, k) == betaSign(i, j, k)

allImagTriples = [(i, j, k) for i in range(1, 8) for j in range(1, 8) for k in range(1, 8)]
fanoCount    = sum(1 for (i, j, k) in allImagTriples if isFano(i, j, k))
nonFanoCount = sum(1 for (i, j, k) in allImagTriples if not isFano(i, j, k))
print("triple_count      :", len(allImagTriples), "(expect 343)")
print("fano_count        :", fanoCount, "(expect 175)")
print("non_fano_count    :", nonFanoCount, "(expect 168)")

# ---- PrimSed / ZD (mirror of SounioZeroDivisorBridge) ----
class PrimSed:
    __slots__ = ("lo", "hi", "neg")
    def __init__(self, lo, hi, neg): self.lo, self.hi, self.neg = lo, hi, neg
    def key(self): return (self.lo, self.hi, self.neg)
    def __eq__(self, o): return self.key() == o.key()
    def __hash__(self): return hash(self.key())
    def __repr__(self):
        return f"e{self.lo}{'-' if self.neg else '+'}e{self.hi}"

def isPrimValid(v):
    return (1 <= v.lo <= 7) and (9 <= v.hi <= 15) and ((v.lo ^ v.hi) != 8)

allPrims = [PrimSed(lo, hi, neg)
            for lo in range(1, 8)
            for hi in range(9, 16)
            for neg in (False, True)]
validPrims = [v for v in allPrims if isPrimValid(v)]
print("validPrims        :", len(validPrims), "(expect 84)")

def primProd(u, v, k):
    c_ll = sedSigma(u.lo, v.lo) if (u.lo ^ v.lo) == k else 0
    s_v = -1 if v.neg else 1
    s_u = -1 if u.neg else 1
    c_lh = (s_v * sedSigma(u.lo, v.hi)) if (u.lo ^ v.hi) == k else 0
    c_hl = (s_u * sedSigma(u.hi, v.lo)) if (u.hi ^ v.lo) == k else 0
    c_hh = (s_u * s_v * sedSigma(u.hi, v.hi)) if (u.hi ^ v.hi) == k else 0
    return c_ll + c_lh + c_hl + c_hh

def isZeroPair(u, v):
    return all(primProd(u, v, k) == 0 for k in range(16))

def xorLabel(v): return v.lo ^ v.hi

orderedZDPairs = [(u, v) for u in validPrims for v in validPrims
                  if u != v and isZeroPair(u, v)]
print("orderedZDPairs    :", len(orderedZDPairs), "(expect 336)")

def annihilators(u):
    return [v for (a, v) in orderedZDPairs if a == u]

# every primitive has exactly 4 right-annihilators
print("all-degree-4      :", all(len(annihilators(u)) == 4 for u in validPrims))

# ---- Fano plane lines on points {1..7} (XOR-zero triples) ----
fanoLines = [(1,2,3),(1,4,5),(1,6,7),(2,4,6),(2,5,7),(3,4,7),(3,5,6)]
fanoLineSets = [frozenset(L) for L in fanoLines]

def contains_a_line(pointset):
    ps = frozenset(pointset)
    return any(L <= ps for L in fanoLineSets)

def is_arc(pointset):
    # arc = no 3 collinear = contains no full Fano line (lines have 3 pts)
    return not contains_a_line(pointset)

# ================================================================
# CONJECTURE 1: the 4 annihilator lo-indices form a 4-arc (hyperoval)
# CONJECTURE 2: that lo-set is the COMPLEMENT of a Fano line
# ================================================================
print("\n--- hyperoval bridge across all 84 primitives ---")
all_arc = True
all_complement = True
all_size4 = True
examples = []
for u in validPrims:
    anns = annihilators(u)
    los = sorted(set(a.lo for a in anns))
    if len(los) != 4:
        all_size4 = False
    if not is_arc(los):
        all_arc = False
    comp = frozenset(range(1,8)) - frozenset(los)  # the 3 missing points
    is_comp_of_line = frozenset(comp) in [frozenset(L) for L in fanoLines]
    if not is_comp_of_line:
        all_complement = False
    if len(examples) < 8:
        examples.append((repr(u), xorLabel(u), los, sorted(comp), is_comp_of_line))

print("all 4 annihilator lo-indices distinct (size 4):", all_size4)
print("annihilator lo-set is a 4-arc (no Fano line)  :", all_arc)
print("annihilator lo-set = complement of a Fano line:", all_complement)
print("\nexamples (prim, fiberLabel, ann-lo-set, missing-triple, missing-is-line):")
for e in examples:
    print("  ", e)

# Which line is the missing triple, relative to fiber label and prim lo?
print("\n--- structure of the missing line vs (fiber, prim.lo) ---")
for u in validPrims[:14]:
    anns = annihilators(u)
    los = sorted(set(a.lo for a in anns))
    comp = sorted(frozenset(range(1,8)) - frozenset(los))
    print(f"  prim={u!r:9} fiber={xorLabel(u)} loP={u.lo}  missingLine={comp}  loP in missing? {u.lo in comp}")

print("\n=== sharper: missing line = unique Fano line through {lo, fiber XOR 8} ===")
def line_through(p, q):
    cands = [L for L in fanoLines if p in L and q in L]
    return cands[0] if len(cands)==1 else None
ok_char = True
for u in validPrims:
    anns = annihilators(u)
    los = sorted(set(a.lo for a in anns))
    missing = sorted(frozenset(range(1,8)) - frozenset(los))
    q = xorLabel(u) ^ 8
    L = line_through(u.lo, q)
    if L is None or sorted(L) != missing:
        ok_char = False
        print("  FAIL", u, "loP", u.lo, "q", q, "L", L, "missing", missing)
print("missing line == line_through(lo, fiber^8) for all 84:", ok_char)
# note lo != q always?
print("lo == fiber^8 ever? :", any(u.lo == (xorLabel(u)^8) for u in validPrims))

print("\n=== EDIT(u): fiber lo-multiset and arc/blocking status ===")
def fiberPrims(L): return [v for v in validPrims if xorLabel(v)==L]
for L in [9,10,11,12,13,14,15]:
    fp = fiberPrims(L)
    los = sorted(v.lo for v in fp)
    print(f"  fiber {L}: 12 prims, lo-multiset={los}, distinct-lo={sorted(set(los))}")

print("\n=== Fano plane decidable facts (over 2^7 subsets of points) ===")
from itertools import combinations
pts = list(range(1,8))
# max arc size
max_arc = max(len(s) for r in range(8) for s in combinations(pts,r) if is_arc(s))
num_max_arcs = sum(1 for s in combinations(pts,max_arc) if is_arc(s))
print("max arc size (hyperoval):", max_arc, " count:", num_max_arcs)
# nontrivial blocking set: meets every line, contains no line
def meets_all_lines(s):
    ss=frozenset(s); return all(len(ss & L)>0 for L in fanoLineSets)
def is_blocking(s):
    return meets_all_lines(s) and not contains_a_line(s)
num_blocking = sum(1 for r in range(8) for s in combinations(pts,r) if is_blocking(s))
print("nontrivial blocking sets in PG(2,2):", num_blocking, "(expect 0 — Fano has none)")
# minimum set meeting all lines that may contain lines (transversal) -> a line works, size 3
min_transversal = min(len(s) for r in range(1,8) for s in combinations(pts,r) if meets_all_lines(s))
print("min transversal (line-blocking allowed) size:", min_transversal)

print("\n=== UNLEARN kernel hyperoval: is it the SAME for all 4-in-fiber sharing a line? ===")
# how many distinct hyperovals arise as annihilator lo-sets
hovals=set()
for u in validPrims:
    hovals.add(frozenset(a.lo for a in annihilators(u)))
print("distinct annihilator-hyperovals:", len(hovals), "(7 complements of 7 lines?)", sorted(map(sorted,hovals)))

print("\n=== op point-shadows (lo-index images) ===")
def nub(xs):
    out=[]
    for x in xs:
        if x not in out: out.append(x)
    return out
def gate(u): return [v for v in validPrims if u!=v and not isZeroPair(u,v)]
edit_ok=True; gate_ok=True; edit_missing_ok=True
for u in validPrims:
    eshadow=sorted(nub([v.lo for v in fiberPrims(xorLabel(u))]))
    gshadow=sorted(nub([v.lo for v in gate(u)]))
    if len(eshadow)!=6: edit_ok=False
    if sorted(set(range(1,8))-set(eshadow))!=[xorLabel(u)^8]: edit_missing_ok=False
    if gshadow!=[1,2,3,4,5,6,7]: gate_ok=False
print("EDIT shadow = 6 points for all:", edit_ok)
print("EDIT missing point == fiber^8 for all:", edit_missing_ok)
print("GATE shadow = all 7 points for all:", gate_ok)
# all 7 hyperovals realized as annihilator complements
realized=set(frozenset(a.lo for a in annihilators(u)) for u in validPrims)
print("distinct UNLEARN hyperovals:", len(realized))
