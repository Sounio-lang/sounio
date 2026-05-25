import math, sys
from fractions import Fraction as Fr
exec(open('certify510.py').read().split('verts=[]')[0])  # F, parse, sqrt_rat, tokenize, Pr
# load 510 vertices
V=[]
for line in open('cnp510.vtx'):
    line=line.strip()
    if not line: continue
    inner=line[1:-1]; depth=0; pos=None
    for i,ch in enumerate(inner):
        if ch in '([': depth+=1
        elif ch in ')]': depth-=1
        elif ch==',' and depth==0: pos=i; break
    V.append((parse(inner[:pos]),parse(inner[pos+1:])))
N0=len(V)
# numeric eval
RAD={0:1.0,1:3.0,2:5.0,3:15.0,4:11.0,5:33.0,6:55.0,7:165.0}
def numv(comp): return sum(float(c)*math.sqrt(RAD[i]) for i,c in comp.c.items())
def key(P): return (tuple(sorted(P[0].c.items())),tuple(sorted(P[1].c.items())))
def num2(P): return (numv(P[0]),numv(P[1]))

# moves = unit-edge difference vectors from the certified edges (both signs)
E=[]
for line in open('cnp510.edge'):
    if line.startswith('e '): _,a,b=line.split(); E.append((int(a)-1,int(b)-1))
moves={}
for a,b in E:
    d=(V[a][0]-V[b][0], V[a][1]-V[b][1])
    moves[key(d)]=d
    dn=(V[b][0]-V[a][0], V[b][1]-V[a][1]); moves[key(dn)]=dn
MOVES=list(moves.values())
print(f"seed vertices {N0}, distinct unit moves {len(MOVES)}", flush=True)

# closure within radius cap, vertex cap
import collections
seen={key(P):P for P in V}
order=list(V)
RCAP=9.0; MAXV=6000
qi=0
while qi<len(order) and len(seen)<MAXV:
    cur=order[qi]; qi+=1
    for m in MOVES:
        Q=(cur[0]+m[0], cur[1]+m[1]); k=key(Q)
        if k in seen: continue
        x,y=num2(Q)
        if x*x+y*y>RCAP*RCAP+1e-6: continue
        seen[k]=Q; order.append(Q)
        if len(seen)>=MAXV: break
VV=order
n=len(VV); print(f"closure vertices: {n} (cap {MAXV}, radius {RCAP})", flush=True)

# edges: numeric prefilter (grid hash) then exact verify
def is_one(v): return v.c=={0:Fr(1)}
nums=[num2(P) for P in VV]
grid=collections.defaultdict(list)
for i,(x,y) in enumerate(nums): grid[(round(x),round(y))].append(i)
edges=[]
import itertools
for i in range(n):
    xi,yi=nums[i]
    cand=set()
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            cand.update(grid.get((round(xi)+dx,round(yi)+dy),[]))
    for j in cand:
        if j<=i: continue
        d=(VV[i][0]-VV[j][0], VV[i][1]-VV[j][1])
        if is_one(d[0]*d[0]+d[1]*d[1]): edges.append((i,j))
print(f"exact unit-distance edges in closure: {len(edges)}", flush=True)

# SAT 5-colorability
from pysat.formula import CNF
from pysat.solvers import Solver
k=5
cnf=CNF()
for v in range(n):
    cnf.append([v*k+c+1 for c in range(k)])
    for c in range(k):
        for d in range(c+1,k): cnf.append([-(v*k+c+1),-(v*k+d+1)])
for a,b in edges:
    for c in range(k): cnf.append([-(a*k+c+1),-(b*k+c+1)])
with Solver(name='cadical153', bootstrap_with=cnf) as s:
    sat=s.solve()
print(f"\n=== closure ({n} vtx, {len(edges)} edges): 5-colorable? {sat} ===", flush=True)
if sat:
    print("NEGATIVE: chunk is 5-colorable. No 6-chromatic graph in this explored region.")
else:
    print("*** POSITIVE: NOT 5-colorable -> chi >= 6 !! (must drat-trim verify) ***")
