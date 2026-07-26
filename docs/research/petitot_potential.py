#!/usr/bin/env python3
# Petitot's morphodynamic semantics, made concrete, and its algebraic (octonion) counterpart.
# Petitot realizes the Greimas semiotic square as the critical-point structure of a POTENTIAL: the
# semantic positions = wells (attractors), the oppositions = the bifurcation set, and his IMPOSSIBILITY
# THEOREM says the square is NOT Booleanizable (contrariety and contradiction are distinct *topological*
# relations, not one Boolean complement). We (1) reproduce the cusp (binary contrariety) and butterfly
# (the mediating "complex term"), (2) exhibit why the two opposition-types are topologically distinct
# (non-Booleanizability), (3) give a concrete octonion/Fano model of a SYSTEM of oppositions and state
# honestly where it meets and diverges from Petitot.
import numpy as np
np.seterr(all='ignore')
def wells(coeffs_Vprime):
    r=np.roots(coeffs_Vprime); r=r[np.abs(r.imag)<1e-9].real
    return sorted(r.tolist())
def n_minima(dV, ddV_coeffs):
    xs=wells(dV); mins=[x for x in xs if np.polyval(ddV_coeffs,x)>1e-9]; return len(mins),mins
# ---------- 1. CUSP: V = x^4/4 + a x^2/2 + b x  → binary contrariety (2 wells) ----------
print("CUSP  V=x⁴/4 + a·x²/2 + b·x   (binary contrariety A/B):")
def cusp_minima(a,b):
    dV=[1,0,a,b]; ddV=[3,0,a]; return n_minima(dV,ddV)
for (a,b,lab) in [(-1.0,0.0,"a=-1,b=0  symmetric"),(-1.0,0.35,"tilted"),(0.5,0.0,"a>0 monostable")]:
    n,ms=cusp_minima(a,b); print(f"  {lab:22s}: {n} well(s)  at {[round(x,2) for x in ms]}")
# bifurcation set: 2 wells (bistable) iff discriminant of x³+ax+b > 0  ⇔ 4a³+27b²<0
grid=0
A=np.linspace(-2,1,120); B=np.linspace(-1.5,1.5,120); bist=0; tot=0
for a in A:
    for b in B:
        tot+=1; n,_=cusp_minima(a,b); bist+= (n==2)
print(f"  bistable (2-well = both contraries coexist) region: {100*bist/tot:.0f}% of control plane; "
      f"the fold curve 4a³+27b²=0 is the bifurcation set (the semantic 'jump').")
# ---------- 2. BUTTERFLY: V = x^6/6 + t x^4/4 + v x^2/2 + w x  → the mediating 3rd well ----------
print("\nBUTTERFLY  V=x⁶/6 + t·x⁴/4 + v·x²/2 + w·x   (the 'complex/neutral term' = a 3rd well):")
def bfly_minima(t,v,w):
    dV=[1,0,t,0,v,w]; ddV=[5,0,3*t,0,v]; return n_minima(dV,ddV)
# scan a (t,v) slice at w=0, count max wells; locate the 3-well "pocket"
maxw=0; pocket=0; tot=0
for t in np.linspace(-4,0,80):
    for v in np.linspace(-3,1,80):
        tot+=1; n,_=bfly_minima(t,v,0.0); maxw=max(maxw,n); pocket+=(n>=3)
print(f"  max coexisting wells in the (t,v) slice: {maxw}   (3 = two contraries + the mediation)")
print(f"  '3-well pocket' (Petitot's complex term): {100*pocket/tot:.0f}% of the scanned slice — "
      f"it exists, and it is bounded by butterfly cusp lines, not a Boolean corner.")
# ---------- 3. non-Booleanizability: two DISTINCT opposition moves ----------
print("\nNON-BOOLEANIZABILITY (Petitot's impossibility theorem, illustrated):")
print("  contrariety A|B  = two wells that can BOTH vanish (merge over the cusp point) → both-false possible")
n0,_=cusp_minima(-1.0,0.0); n1,_=cusp_minima(0.5,0.0)
print(f"     cusp a:-1→+0.5 : {n0} wells → {n1} well  (A and B both dissolve into a single neutral state)")
print("  contradiction A|¬A = antipodal: one appears exactly as the other vanishes → both-false IMPOSSIBLE")
print("     (a well and its absence; no control value gives 'neither' — structurally different move)")
print("  A Boolean lattice 2² has ONE complement ⇒ cannot host two distinct opposition-types ⇒ the square")
print("  is not Booleanizable. The distinction is carried by the *topology* of the strata, not by logic.")
# ---------- 4. octonion / Fano model of a SYSTEM of oppositions ----------
def cds(a,b,bits=3):
    s=1
    while bits>0:
        if a==0 or b==0: return s
        if bits==1: return -s
        h=1<<(bits-1);ah=a>=h;bh=b>=h;al=a&(h-1);bl=b&(h-1)
        if not ah and not bh:a,b=al,bl
        elif not ah and bh:a,b=bl,al
        elif ah and not bh:(a,b,s)=((al,0,s) if bl==0 else (al,bl,-s))
        else:(a,b,s)=((0,al,-s) if bl==0 else (bl,al,s))
        bits-=1
    return s
def omul(A,B):
    C=np.zeros(8)
    for i in range(8):
        for j in range(8): C[i^j]+=cds(i,j)*A[i]*B[j]
    return C
def e(i): v=np.zeros(8); v[i]=1.0; return v
def assoc_norm(i,j,k):
    z=omul(omul(e(i),e(j)),e(k))-omul(e(i),omul(e(j),e(k))); return np.linalg.norm(z)
lines=[(i,j,i^j) for i in range(1,8) for j in range(i+1,8) if i^j>j]   # Fano lines (associative triples)
print(f"\nOCTONION / FANO model of a SYSTEM of semantic oppositions:")
print(f"  Fano lines (associative triples {{i,j,i⊕j}}) = quaternion subalgebras = 'Booleanizable squares': {len(lines)}")
print(f"     {lines}")
# each Fano triple is associative (a Booleanizable square); a cross-line triple is not
fano_assoc=max(assoc_norm(i,j,k) for (i,j,k) in lines)
noncol=[(i,j,k) for i in range(1,8) for j in range(i+1,8) for k in range(j+1,8) if k!=(i^j) and (i^j)!=k]
noncol_assoc=min(assoc_norm(i,j,k) for (i,j,k) in noncol)
print(f"  within a line (a single square): max associator = {fano_assoc:.1e}  → associative ⇒ Booleanizable")
print(f"  across lines (combining squares): min associator = {noncol_assoc:.1f}  → NON-associative ⇒ the")
print(f"  system of squares is not globally Booleanizable. Two Fano lines meet in exactly one unit (incidence).")
print("\n  READING (honest): each single opposition closes into a quaternion (associative, Booleanizable) —")
print("  which DIVERGES from Petitot (who denies Booleanizability of the single square); the algebraic")
print("  non-Booleanizability appears at the level of the FIELD of oppositions (the octonion), carried by")
print("  the associator. Two formalizations of one intuition; they agree that the exceptional/non-assoc")
print("  structure is irreducible, and differ on WHERE the obstruction sits (single square vs the field).")
