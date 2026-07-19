#!/usr/bin/env python3
# The geometry of RELATIONAL ANNIHILATION in the sedenions 𝕊 — which "two-subject" configurations can go
# to zero, and how SPECIFIC that is. Two nonzero subjects x,y with x·y=0 (a zero-divisor pair): the
# subjects remain nonzero; their RELATION annihilates. Clinically motivated (Joiner's interpersonal theory:
# annihilation is a CONJUNCTION, not generic pain). The claim to test: annihilation is not generic — only a
# small, highly structured set of configurations (de Marrais's 7 box-kites / 42 assessors) permits it.
import numpy as np
np.seterr(all='ignore')
def cds(a,b,bits=4):
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
def Lmat(x):
    L=np.zeros((16,16))
    for k in range(16):
        for j in range(16): L[k,j]=x[k^16 if False else k^j]*cds(k^j,j)  # L[k,j]=x[k^j]*cds(k^j,j)
    return L
def unit(i): v=np.zeros(16); v[i]=1.0; return v
def prod(x,y):
    z=np.zeros(16)
    for i in range(16):
        for j in range(16): z[i^j]+=cds(i,j)*x[i]*y[j]
    return z
# --- 1. all zero-divisor 2-unit sums a = e_i + s·e_j, and their annihilators ---
ZDs=[]                              # (i, s, j, [annihilator units as (k,sgn,l)])
for i in range(1,16):
    for j in range(i+1,16):
        for s in (+1,-1):
            a=unit(i)+s*unit(j); L=Lmat(a); sv,_,V=np.linalg.svd(L,full_matrices=True);
            svals=np.linalg.svd(L,compute_uv=False)
            if svals.min()<1e-9:
                # annihilator = right null space
                _,_,Vt=np.linalg.svd(L); null=Vt[svals.size-np.sum(svals<1e-9):]
                ZDs.append((i,s,j,null))
print(f"zero-divisor 2-unit sums a=e_i±e_j in 𝕊: {len(ZDs)}  (de Marrais/Cawagas: 84)")
# --- 2. specificity: how many partners does one subject annihilate with? ---
a=unit(1)+unit(10); L=Lmat(a); svals=np.linalg.svd(L,compute_uv=False)
_,_,Vt=np.linalg.svd(L); ker=Vt[np.where(svals<1e-9)[0]]
print(f"\nSPECIFICITY — the subject a=e1+e10 (‖a‖={np.linalg.norm(a):.3f}):")
print(f"  its annihilator space has dimension {ker.shape[0]} / 16  → it annihilates with a {ker.shape[0]}-dim")
print(f"  sliver of partners, not with 'anyone'. Generic relation a·y is invertible (no annihilation).")
b=ker[0]; nz=[(k,round(b[k],3)) for k in range(16) if abs(b[k])>1e-6]
print(f"  a specific annihilating partner b: units {nz}   ‖a·b‖={np.linalg.norm(prod(a,b)):.1e}")
# --- 3. the 7 box-kites: partition the 42 assessor planes by strut constant ---
# assessor plane {i,j} (unordered) is a zero-divisor plane; group by the box-kite invariant.
assessors=set()
for (i,s,j,_) in ZDs: assessors.add((i,j))
assessors=sorted(assessors)
print(f"\nassessor planes (distinct {{i,j}} zero-divisor planes): {len(assessors)}  (expect 42)")
# de Marrais: each assessor pairs a low unit (1..7, an octonion unit) with a high unit (8..15). strut
# constant S groups them. Try S = (i XOR j) restricted, or the octonion-index of the low member.
def classify(i,j):
    lo,hi=min(i,j),max(i,j)
    # low is the octonion unit (1..7); the box-kite strut constant = lo XOR (hi AND 7)
    return lo ^ (hi & 7)
from collections import defaultdict
kites=defaultdict(list)
for (i,j) in assessors: kites[classify(i,j)].append((i,j))
print(f"box-kites (assessors grouped by strut constant S = lo ⊕ (hi&7)): {len(kites)} groups")
for S in sorted(kites): print(f"  S={S}: {len(kites[S])} assessors  {kites[S]}")
sizes=sorted(len(v) for v in kites.values())
print(f"  group sizes: {sizes}   (de Marrais: 7 box-kites × 6 assessors = 42)")
print(f"\n=> Annihilation is RARE and STRUCTURED: {len(ZDs)}/{15*14} of unit-pairs, partitioned into a fixed")
print(f"   combinatorial skeleton. Relational annihilation requires a specific configuration, not generic")
print(f"   distress — the algebraic echo of Joiner's conjunction (belonging × burden × capability).")
