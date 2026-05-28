# de Grey's base graph H: centre + 6 vertices of a UNIT-side regular hexagon
# (circumradius = 1). Exact over Q(√3); coords scaled ×2. Unit edge -> sq.dist 4.
from itertools import combinations
def mul(x,y):
    a0,a1,a2,a3=x; b0,b1,b2,b3=y
    return (a0*b0+3*a1*b1+11*a2*b2+33*a3*b3, a0*b1+a1*b0+11*a2*b3+11*a3*b2,
            a0*b2+a2*b0+3*a1*b3+3*a3*b1, a0*b3+a3*b0+a1*b2+a2*b1)
def sub(x,y): return tuple(x[i]-y[i] for i in range(4))
def add(x,y): return tuple(x[i]+y[i] for i in range(4))
def I(n): return (n,0,0,0)
def S3(n): return (0,n,0,0)   # n√3
# scaled ×2: O=(0,0); hexagon vertices radius 1 -> (2cos,2sin): (2,0),(1,√3),(-1,√3),(-2,0),(-1,-√3),(1,-√3)
V=[(I(0),I(0)),(I(2),I(0)),(I(1),S3(1)),(I(-1),S3(1)),(I(-2),I(0)),(I(-1),S3(-1)),(I(1),S3(-1))]
def d2(p,q):
    dx=sub(p[0],q[0]); dy=sub(p[1],q[1]); return add(mul(dx,dx),mul(dy,dy))
unit=[]
for i,j in combinations(range(7),2):
    if d2(V[i],V[j])==(4,0,0,0): unit.append((i,j))
print("exact unit edges:", unit)
print("edge count:", len(unit), "(de Grey H has 12)")
# converse audit already inherent: 'unit' = ALL pairs at exact dist 1, nothing extra.
# chromatic number
from itertools import product
def chi():
    for k in range(1,8):
        for asg in product(range(k),repeat=7):
            if all(asg[i]!=asg[j] for i,j in unit): return k
    return 7
print("chi(H) =", chi(), "(expect 3)")
# sample non-edges are sqrt3 or 2 (not 1): show O..opposite and 2-apart
print("dist^2 (1,√3)..(-1,√3) [2-apart]:", d2(V[2],V[3]))  # should be 4? these are adjacent actually
print("dist^2 hub..none; (2,0)..(-2,0) opposite:", d2(V[1],V[4]))  # =16 ->4 scaled? =16 means 4 unscaled
