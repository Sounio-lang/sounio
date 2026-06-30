#!/usr/bin/env python3
"""Symbolic verification (SymPy) of every algebraic identity stated in
formal/DynkinSwapMassLadder.lean. Grounds the (not-machine-checked) Lean file.
"""
import sympy as sp
c, d, q = sp.symbols('c delta q', positive=True)

def caEdge(center, off): return (center + off) / (center - off)
def cbEdge(center, off): return (center + off) / center
def koideQ(center, dd):
    sm = [center - dd, center, center + dd]
    return sum(s**2 for s in sm) / (sum(sm))**2

checks = []

# §1  koideQ_eq : koideQ c δ = 2δ²/(9c²) + 1/3
checks.append(("koideQ_eq",
    sp.simplify(koideQ(c, d) - (2*d**2/(9*c**2) + sp.Rational(1,3))) == 0))

# koide_two_thirds_iff : koideQ = 2/3  ⟺  δ² = (3/2) c²
sol = sp.solve(sp.Eq(koideQ(c, d), sp.Rational(2,3)), d**2)
checks.append(("koide_two_thirds_iff (δ²=3/2·c²)",
    any(sp.simplify(s - sp.Rational(3,2)*c**2) == 0 for s in sol)))

# delta_sq_eq_three_eighths : at c=1/2, ⟺ δ²=3/8
sol_half = sp.solve(sp.Eq(koideQ(sp.Rational(1,2), d), sp.Rational(2,3)), d**2)
checks.append(("delta_sq_eq_three_eighths (δ²=3/8 at c=1/2)",
    any(sp.simplify(s - sp.Rational(3,8)) == 0 for s in sol_half)))

# §2  swap_factor_is_center_spread_exchange : (δ+q)/(δ−q) = caEdge(δ, q)
checks.append(("swap_factor_is_center_spread_exchange",
    sp.simplify((d + q)/(d - q) - caEdge(d, q)) == 0))

# muOverE_expand : caEdge(1,δ)·caEdge(δ,1/3) = ((1+δ)/(1−δ))·((δ+1/3)/(δ−1/3))
muOverE = caEdge(1, d) * caEdge(d, sp.Rational(1,3))
checks.append(("muOverE_expand",
    sp.simplify(muOverE - ((1+d)/(1-d))*((d+sp.Rational(1,3))/(d-sp.Rational(1,3)))) == 0))

# bOverS_expand : caEdge(1,δ)·cbEdge(1,δ) = ((1+δ)/(1−δ))·(1+δ)
bOverS = caEdge(1, d) * cbEdge(1, d)
checks.append(("bOverS_expand",
    sp.simplify(bOverS - ((1+d)/(1-d))*(1+d)) == 0))

# reconciliation: ONE δ=√(3/8) gives center-1 edge ≈ τ/μ AND center-1/2 Koide = 2/3
dval = sp.sqrt(sp.Rational(3,8))
checks.append(("reconcile: Koide(c=1/2, δ=√(3/8)) = 2/3",
    sp.simplify(koideQ(sp.Rational(1,2), dval) - sp.Rational(2,3)) == 0))

print("Symbolic verification of formal/DynkinSwapMassLadder.lean identities:\n")
allok = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    allok = allok and bool(ok)
print(f"\n{'ALL IDENTITIES VERIFIED' if allok else 'SOME FAILED'}")
print(f"center-1 c/a edge at δ=√(3/8): (1+δ)/(1−δ) = {float(caEdge(1,dval)):.4f}  (obs √(mτ/mμ)≈4.10)")
