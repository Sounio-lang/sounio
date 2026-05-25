# Exact arithmetic over K = Q(sqrt3,sqrt5,sqrt7,sqrt11), degree 16.
# Basis indexed by 4-bit mask over primes (bit0=3,bit1=5,bit2=7,bit3=11);
# index m <-> sqrt(product of primes whose bit is set). sqrt(a)*sqrt(b) =
# (product of shared primes) * sqrt(symmetric-difference set).
from fractions import Fraction as Q
from itertools import combinations
PRIMES=[3,5,7,11]
def prime_product(mask):
    p=1
    for b in range(4):
        if mask>>b & 1: p*=PRIMES[b]
    return p
def radicand(mask):  # the squarefree integer sqrt(mask-index) represents
    return prime_product(mask)
# KVec = dict index->Fraction (sparse)
def kadd(x,y):
    r=dict(x)
    for i,c in y.items(): r[i]=r.get(i,Q(0))+c
    return {i:c for i,c in r.items() if c!=0}
def ksub(x,y): return kadd(x,{i:-c for i,c in y.items()})
def kmul(x,y):
    r={}
    for i,a in x.items():
        for j,b in y.items():
            idx=i^j; fac=prime_product(i&j)
            r[idx]=r.get(idx,Q(0))+a*b*fac
    return {i:c for i,c in r.items() if c!=0}
def K(d=0): return {0:Q(d)} if d else {}
def root(prime):  # sqrt(prime) as KVec
    b=PRIMES.index(prime); return {1<<b:Q(1)}
def is_scalar(x,val):  # x == val*1 exactly (all other 15 comps zero)
    return x=={0:Q(val)} or (val==0 and x=={})

print("=== field multiplication sanity ===")
for p in PRIMES:
    print(f"  (sqrt{p})^2 == {p}:", is_scalar(kmul(root(p),root(p)), p))
print("  sqrt3*sqrt5 == sqrt15 (index 0b0011=3):", kmul(root(3),root(5))=={3:Q(1)})
# reviewer's example: sqrt15 * sqrt21 = 3 sqrt35
s15={0b0011:Q(1)}; s21={0b0101:Q(1)}  # 15=3*5 ->bits0,1 ; 21=3*7 ->bits0,2
prod=kmul(s15,s21)
# 3*sqrt35: 35=5*7 -> bits1,2 = 0b0110=6, coeff 3
print("  sqrt15*sqrt21 == 3*sqrt35:", prod=={0b0110:Q(3)})

def d2(P,Pq):
    dx=ksub(P[0],Pq[0]); dy=ksub(P[1],Pq[1])
    return kadd(kmul(dx,dx),kmul(dy,dy))

print("\n=== per-axis unit-edge witnesses (each certifies a unit dist living in that axis) ===")
O=(K(0),K(0))
# sqrt5 axis: rotation 2*arcsin(1/4): cos=7/8,sin=sqrt15/8. R*(1,0)=(7/8,sqrt15/8); |.|=1.
# scaled x8: (7, sqrt15); d^2 = 49+15 = 64 = 8^2
Q5=(K(7), {0b0011:Q(1)})   # (7, sqrt15)
print("  sqrt5 witness  O-(7,sqrt15): d^2==64:", is_scalar(d2(O,Q5),64), " uses sqrt15 (3*5)")
# sqrt7 axis: rotation arcsin(1/8): cos=sqrt63/8=3sqrt7/8, sin=1/8. R*(1,0)=(3sqrt7/8,1/8)
# scaled x8: (3 sqrt7, 1); d^2 = 9*7 + 1 = 64
Q7=({0b0100:Q(3)}, K(1)) # (3 sqrt7, 1)
print("  sqrt7 witness  O-(3sqrt7,1): d^2==64:", is_scalar(d2(O,Q7),64), " uses sqrt7")
# sqrt11 axis (from spindle M-stage): (7/6,sqrt11/6) is at distance 1 from origin? 49/36+11/36=60/36 no.
# use de Grey S-point style: sin(arcsin(1/sqrt12)) gives cos=sqrt11/sqrt12. R*(1,0)=(sqrt11/sqrt12,1/sqrt12)
# scaled x sqrt12: (sqrt11,1); d^2 = 11+1 = 12 = (sqrt12)^2 -> unit edge scaled to 12
Q11=({0b1000:Q(1)}, K(1))  # (sqrt11, 1)
print("  sqrt11 witness O-(sqrt11,1): d^2==12:", is_scalar(d2(O,Q11),12), " uses sqrt11")
print("\nfield K basis size:", 16, " radicands:", [radicand(m) for m in range(16)])
