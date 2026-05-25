#!/usr/bin/env python3
"""Exact unit-distance certifier over K = Q(√3, √11), demonstrated on the
Moser spindle (7 vertices, 11 unit edges, chi=4). Numbers are 4-tuples
(q0 + q1·√3 + q2·√11 + q3·√33) of rationals; √33 = √3·√11.
This is the EXACT arithmetic that float-based unit-distance searches get wrong."""
from fractions import Fraction as Q
from itertools import combinations, product

# element of K = q0 + q1√3 + q2√11 + q3√33
def K(a=0,b=0,c=0,d=0): return (Q(a),Q(b),Q(c),Q(d))
def add(x,y): return tuple(x[i]+y[i] for i in range(4))
def sub(x,y): return tuple(x[i]-y[i] for i in range(4))
def mul(x,y):
    a0,a1,a2,a3=x; b0,b1,b2,b3=y
    # basis products: 1,√3,√11,√33 ; √3√3=3, √11√11=11, √33√33=33,
    # √3√11=√33, √3√33=3√11, √11√33=11√3
    r0 = a0*b0 + 3*a1*b1 + 11*a2*b2 + 33*a3*b3
    r1 = a0*b1 + a1*b0 + 11*a2*b3 + 11*a3*b2
    r2 = a0*b2 + a2*b0 + 3*a1*b3 + 3*a3*b1
    r3 = a0*b3 + a3*b0 + a1*b2 + a2*b1
    return (r0,r1,r2,r3)
def is_one(x): return x==(Q(1),Q(0),Q(0),Q(0))

s3 = K(0,1,0,0); s11 = K(0,0,1,0); s33 = K(0,0,0,1)
half=K(Q(1,2));
def sc(frac): return K(frac)  # rational scalar

# cosθ=5/6, sinθ=√11/6
cos = K(Q(5,6)); sin = (Q(1,6),Q(0),Q(0),Q(0)); sin = mul(K(Q(1,6)), s11)  # √11/6

def P(x,y): return (x,y)
def rot(p):  # R(θ)·(x,y)
    x,y=p
    nx = sub(mul(cos,x), mul(sin,y))
    ny = add(mul(sin,x), mul(cos,y))
    return (nx,ny)

# Rhombus 1 (along x-axis)
A=(K(0),K(0))
B=(mul(K(Q(1,2)),s3), K(Q(1,2)))     # (√3/2, 1/2)
C=(mul(K(Q(1,2)),s3), K(Q(-1,2)))    # (√3/2,-1/2)
D=(s3, K(0))                          # (√3,0)
# Rhombus 2 = rhombus 1 rotated by θ about A
E=rot(B); F=rot(C); G=rot(D)
V={'A':A,'B':B,'C':C,'D':D,'E':E,'F':F,'G':G}
names=list(V)

def d2(p,q):
    dx=sub(p[0],q[0]); dy=sub(p[1],q[1])
    return add(mul(dx,dx),mul(dy,dy))

edges_claimed=[('A','B'),('A','C'),('B','C'),('B','D'),('C','D'),
               ('A','E'),('A','F'),('E','F'),('E','G'),('F','G'),('D','G')]

print("=== exact squared distances (should be (1,0,0,0) on edges) ===")
unit_pairs=[]
for u,w in combinations(names,2):
    dd=d2(V[u],V[w])
    if is_one(dd): unit_pairs.append((u,w))
print("unit-distance pairs (exact ==1):", sorted(unit_pairs))
print("count:", len(unit_pairs))
print("claimed edges:", len(edges_claimed))
print("MATCH (exact unit-distance graph == claimed spindle):",
      set(map(frozenset,unit_pairs))==set(map(frozenset,edges_claimed)))

# show a few non-unit (irrational) distances to prove exactness matters
print("\n=== sample NON-edge squared distances (irrational, ≠1) ===")
cnt=0
for u,w in combinations(names,2):
    if frozenset((u,w)) not in set(map(frozenset,edges_claimed)):
        print(f"  {u}{w}: {d2(V[u],V[w])}")
        cnt+=1
        if cnt>=4: break

# chromatic number of the abstract graph (brute force)
adj={n:set() for n in names}
for u,w in unit_pairs: adj[u].add(w); adj[w].add(u)
def kcolorable(k):
    for asg in product(range(k),repeat=len(names)):
        col=dict(zip(names,asg))
        if all(col[u]!=col[w] for u,w in unit_pairs): return True
    return False
chi=next(k for k in range(1,8) if kcolorable(k))
print("\n=== chromatic number ===")
print("chi(Moser spindle) =", chi, "(expect 4)")
