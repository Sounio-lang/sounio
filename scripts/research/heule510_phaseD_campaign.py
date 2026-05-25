import math, sys, time, json
from fractions import Fraction as Fr
exec(open('certify510.py').read().split('verts=[]')[0])  # F, parse, sqrt_rat
RAD={0:1.0,1:3.0,2:5.0,3:15.0,4:11.0,5:33.0,6:55.0,7:165.0}
def numv(c): return sum(float(v)*math.sqrt(RAD[i]) for i,v in c.c.items())
def key(P): return (tuple(sorted(P[0].c.items())),tuple(sorted(P[1].c.items())))
def is_one(v): return v.c=={0:Fr(1)}

def load():
    Vv=[]
    for line in open('cnp510.vtx'):
        line=line.strip()
        if not line: continue
        inner=line[1:-1]; d=0; pos=None
        for i,ch in enumerate(inner):
            if ch in '([': d+=1
            elif ch in ')]': d-=1
            elif ch==',' and d==0: pos=i; break
        Vv.append((parse(inner[:pos]),parse(inner[pos+1:])))
    E=[]
    for line in open('cnp510.edge'):
        if line.startswith('e '): _,a,b=line.split(); E.append((int(a)-1,int(b)-1))
    mv={}
    for a,b in E:
        for d in [(Vv[a][0]-Vv[b][0],Vv[a][1]-Vv[b][1]),(Vv[b][0]-Vv[a][0],Vv[b][1]-Vv[a][1])]:
            mv[key(d)]=d
    return Vv, list(mv.values())

def closure(V0, MOVES, maxv, rcap):
    seen={key(P):0 for P in V0}; order=list(V0); qi=0
    while qi<len(order) and len(order)<maxv:
        cur=order[qi]; qi+=1
        for m in MOVES:
            Q=(cur[0]+m[0],cur[1]+m[1]); k=key(Q)
            if k in seen: continue
            x=numv(Q[0]); y=numv(Q[1])
            if x*x+y*y>rcap*rcap+1e-6: continue
            seen[k]=1; order.append(Q)
            if len(order)>=maxv: break
    return order

def complete_edges(VV):
    # complete + fast: floor-grid candidates -> cheap float screen -> exact verify.
    # float screen tol 1e-6 >> float error (~1e-12) so NO true unit pair is skipped;
    # exact ‖·‖²=1 in Q(√3,√5,√11) confirms. Validated == the all-exact method (11553 @2600).
    nums=[(numv(P[0]),numv(P[1])) for P in VV]
    import collections
    grid=collections.defaultdict(list)
    for i,(x,y) in enumerate(nums): grid[(math.floor(x),math.floor(y))].append(i)
    edges=[]
    for i in range(len(VV)):
        xi,yi=nums[i]; cand=set()
        for dx in(-1,0,1):
            for dy in(-1,0,1): cand.update(grid.get((math.floor(xi)+dx,math.floor(yi)+dy),[]))
        for j in cand:
            if j<=i: continue
            fx=xi-nums[j][0]; fy=yi-nums[j][1]
            if abs(fx*fx+fy*fy-1.0)>1e-6: continue
            dd=(VV[i][0]-VV[j][0],VV[i][1]-VV[j][1])
            if is_one(dd[0]*dd[0]+dd[1]*dd[1]): edges.append((i,j))
    return edges

def five_colorable(n, edges):
    from pysat.formula import CNF; from pysat.solvers import Solver
    k=5; cnf=CNF()
    for v in range(n):
        cnf.append([v*k+c+1 for c in range(k)])
        for c in range(k):
            for d in range(c+1,k): cnf.append([-(v*k+c+1),-(v*k+d+1)])
    for a,b in edges:
        for c in range(k): cnf.append([-(a*k+c+1),-(b*k+c+1)])
    with Solver(name='cadical153',bootstrap_with=cnf) as s:
        return s.solve()

def main():
    V0,MOVES=load()
    log=open('/tmp/campaign.log','a')
    def emit(s): log.write(s+'\n'); log.flush(); print(s,flush=True)
    emit(f"# campaign start {time.strftime('%Y-%m-%dT%H:%M:%S')}  seed={len(V0)} moves={len(MOVES)}")
    for (maxv,rcap) in [(2600,6.0),(5000,8.0),(10000,12.0),(20000,18.0),(40000,28.0),(70000,40.0)]:
        t0=time.time()
        VV=closure(V0,MOVES,maxv,rcap); n=len(VV)
        E=complete_edges(VV); te=time.time()
        sat=five_colorable(n,E); t1=time.time()
        emit(f"size={n:6d} edges={len(E):7d} 5col={sat}  (edges {te-t0:.0f}s, sat {t1-te:.0f}s)")
        if not sat:
            emit(f"*** POSITIVE chi>=6 at size {n}! dumping CNF for drat-trim/bv_decide verification ***")
            json.dump({'n':n,'edges':E}, open(f'/tmp/HIT_{n}.json','w'))
            break
    emit("# campaign sweep complete")
main()
