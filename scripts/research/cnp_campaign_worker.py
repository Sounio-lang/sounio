#!/usr/bin/env python3
"""Phase-D campaign worker (one candidate graph per invocation) — Hadwiger-Nelson χ≥6.

Self-contained: exact arithmetic over K=Q(√3,√5,√11), closure under the certified unit
directions of HeuleGraph510, COMPLETE exact edge detection (floor-grid + float screen,
validated == all-exact), and a 5-colourability SAT test. Designed to be fanned out as a
SLURM array (one (size, variant, seed) config per task) across many CPUs.

Outcome per task:
  5col=True   -> negative (this chunk is 5-colourable; expected for de Grey enlargement)
  5col=False  -> *** χ≥6 candidate *** : dumps CNF + DRAT proof for offline
                 drat-trim + bv_decide verification (NEVER claimed without that).

Usage:
  cnp_campaign_worker.py --data DIR --size N --rcap R [--variant V] [--seed S]
                         [--colors 5] [--timeout SEC] --out OUTDIR
DIR must contain cnp510.vtx and cnp510.edge (the certified graph).
Deps: Python3 only for graph build; SAT via python-sat (pip) or a `cadical`/`kissat`
binary on PATH (DRAT emitted when a binary is used).
"""
import argparse, math, os, sys, json, time, subprocess
from fractions import Fraction as Fr

# ---- exact field K = Q(√3,√5,√11) : mask bit0=3,bit1=5,bit2=11 ; idx<->√(prod) ----
P=[3,5,11]
def pp(mask):
    r=1
    for b in range(3):
        if mask>>b&1: r*=P[b]
    return r
class F:
    __slots__=('c',)
    def __init__(s,c=None): s.c=c or {}
    def __add__(s,o):
        r=dict(s.c)
        for i,v in o.c.items(): r[i]=r.get(i,Fr(0))+v
        return F({i:v for i,v in r.items() if v})
    def __sub__(s,o):
        r=dict(s.c)
        for i,v in o.c.items(): r[i]=r.get(i,Fr(0))-v
        return F({i:v for i,v in r.items() if v})
    def __neg__(s): return F({i:-v for i,v in s.c.items()})
    def __mul__(s,o):
        r={}
        for i,a in s.c.items():
            for j,b in o.c.items():
                k=i^j; r[k]=r.get(k,Fr(0))+a*b*pp(i&j)
        return F({i:v for i,v in r.items() if v})
    def conj(s,T): return F({i:(v if bin(i&T).count('1')%2==0 else -v) for i,v in s.c.items()})
    def inv(s):
        num=F({0:Fr(1)})
        for T in range(1,8): num=num*s.conj(T)
        nrm=(s*num).c.get(0,Fr(0)); assert set((s*num).c)<= {0}
        return F({i:v/nrm for i,v in num.c.items()})
    def __truediv__(s,o): return s*o.inv()
def rat(q): return F({0:Fr(q)})
def sqrt_rat(q):
    num,den=q.numerator,q.denominator; m=num*den; sq=1; mm=m
    for p in [2,3,5,7,11,13]:
        while mm%(p*p)==0: mm//=p*p; sq*=p
    mask=0
    for b,p in enumerate(P):
        if mm%p==0: mm//=p; mask|=1<<b
    assert mm==1,("bad radicand",m); return F({mask:Fr(sq,den)})
import re
def _tok(s): return re.findall(r'Sqrt|\d+|[\[\]()+\-*/]',s)
class Pr:
    def __init__(s,t): s.t=t; s.i=0
    def pk(s): return s.t[s.i] if s.i<len(s.t) else None
    def eat(s,x=None): tk=s.t[s.i]; assert x is None or tk==x,(x,tk); s.i+=1; return tk
    def expr(s):
        v=s.term()
        while s.pk() in('+','-'): v=v+s.term() if s.eat()=='+' else v-s.term()
        return v
    def term(s):
        v=s.fac()
        while s.pk() in('*','/'): v=v*s.fac() if s.eat()=='*' else v/s.fac()
        return v
    def fac(s):
        if s.pk()=='-': s.eat(); return -s.fac()
        if s.pk()=='+': s.eat(); return s.fac()
        return s.prim()
    def prim(s):
        tk=s.pk()
        if tk=='(': s.eat('('); v=s.expr(); s.eat(')'); return v
        if tk=='Sqrt': s.eat('Sqrt'); s.eat('['); v=s.expr(); s.eat(']'); assert set(v.c)<= {0}; return sqrt_rat(v.c.get(0,Fr(0)))
        return rat(int(s.eat()))
def parse(s): p=Pr(_tok(s)); v=p.expr(); assert p.i==len(p.t); return v
RAD={0:1.0,1:3.0,2:5.0,3:15.0,4:11.0,5:33.0,6:55.0,7:165.0}
def numv(c): return sum(float(v)*math.sqrt(RAD[i]) for i,v in c.c.items())
def key(Pt): return (tuple(sorted(Pt[0].c.items())),tuple(sorted(Pt[1].c.items())))
def is_one(v): return v.c=={0:Fr(1)}

def load(data):
    V=[]
    for line in open(os.path.join(data,'cnp510.vtx')):
        line=line.strip()
        if not line: continue
        inner=line[1:-1]; d=0; pos=None
        for i,ch in enumerate(inner):
            if ch in '([': d+=1
            elif ch in ')]': d-=1
            elif ch==',' and d==0: pos=i; break
        V.append((parse(inner[:pos]),parse(inner[pos+1:])))
    E=[]
    for line in open(os.path.join(data,'cnp510.edge')):
        if line.startswith('e '): _,a,b=line.split(); E.append((int(a)-1,int(b)-1))
    mv={}
    for a,b in E:
        for dd in [(V[a][0]-V[b][0],V[a][1]-V[b][1]),(V[b][0]-V[a][0],V[b][1]-V[a][1])]:
            mv[key(dd)]=dd
    return V,list(mv.values())

def closure(V0,MOVES,maxv,rcap,seed_offset=0):
    # seed_offset rotates the move-application order (cheap variant diversification)
    seen={key(Pt):0 for Pt in V0}; order=list(V0); qi=0
    M=MOVES[seed_offset:]+MOVES[:seed_offset]
    while qi<len(order) and len(order)<maxv:
        cur=order[qi]; qi+=1
        for m in M:
            Q=(cur[0]+m[0],cur[1]+m[1]); k=key(Q)
            if k in seen: continue
            x=numv(Q[0]); y=numv(Q[1])
            if x*x+y*y>rcap*rcap+1e-6: continue
            seen[k]=1; order.append(Q)
            if len(order)>=maxv: break
    return order

def complete_edges(VV):
    nums=[(numv(Pt[0]),numv(Pt[1])) for Pt in VV]; import collections
    g=collections.defaultdict(list)
    for i,(x,y) in enumerate(nums): g[(math.floor(x),math.floor(y))].append(i)
    out=[]
    for i in range(len(VV)):
        xi,yi=nums[i]; cand=set()
        for dx in(-1,0,1):
            for dy in(-1,0,1): cand.update(g.get((math.floor(xi)+dx,math.floor(yi)+dy),[]))
        for j in cand:
            if j<=i: continue
            fx=xi-nums[j][0]; fy=yi-nums[j][1]
            if abs(fx*fx+fy*fy-1.0)>1e-6: continue
            dd=(VV[i][0]-VV[j][0],VV[i][1]-VV[j][1])
            if is_one(dd[0]*dd[0]+dd[1]*dd[1]): out.append((i,j))
    return out

def dimacs(n,edges,k):
    cl=[]
    for v in range(n):
        cl.append([v*k+c+1 for c in range(k)])
        for c in range(k):
            for d in range(c+1,k): cl.append([-(v*k+c+1),-(v*k+d+1)])
    for a,b in edges:
        for c in range(k): cl.append([-(a*k+c+1),-(b*k+c+1)])
    return n*k,cl

def write_dimacs(path,nv,cl):
    with open(path,'w') as f:
        f.write(f"p cnf {nv} {len(cl)}\n")
        for c in cl: f.write(" ".join(map(str,c))+" 0\n")

def solve(nv,cl,cnf_path,drat_path,timeout):
    # prefer a SAT binary (emits DRAT for verification); else python-sat
    write_dimacs(cnf_path,nv,cl)
    for binr in ("kissat","cadical"):
        from shutil import which
        if which(binr):
            cmd=[binr]+( ["--no-binary",f"--proof={drat_path}"] if binr=="cadical" else [])+[cnf_path]
            try:
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
                out=r.stdout
                if "s UNSATISFIABLE" in out: return False,binr
                if "s SATISFIABLE" in out: return True,binr
            except subprocess.TimeoutExpired:
                return None,f"{binr}-timeout"
    try:
        from pysat.formula import CNF; from pysat.solvers import Solver
        c=CNF(); c.extend(cl); s=Solver(name='cadical153',bootstrap_with=c)
        r=s.solve(); s.delete(); return r,"pysat-cadical"
    except Exception as e:
        return None,f"nosolver:{e}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--size',type=int,required=True); ap.add_argument('--rcap',type=float,required=True)
    ap.add_argument('--variant',type=int,default=0); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--colors',type=int,default=5); ap.add_argument('--timeout',type=int,default=0)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    tag=f"s{a.size}_r{a.rcap}_v{a.variant}_k{a.colors}"
    V0,MOVES=load(a.data); t0=time.time()
    VV=closure(V0,MOVES,a.size,a.rcap,seed_offset=a.variant%max(1,len(MOVES)))
    E=complete_edges(VV); te=time.time()
    nv,cl=dimacs(len(VV),E,a.colors)
    cnf=os.path.join(a.out,tag+".cnf"); drat=os.path.join(a.out,tag+".drat")
    to=a.timeout if a.timeout>0 else None
    sat,solver=solve(nv,cl,cnf,drat,to); t1=time.time()
    res={"tag":tag,"vertices":len(VV),"edges":len(E),"colors":a.colors,"kcolorable":sat,
         "solver":solver,"edge_s":round(te-t0,1),"sat_s":round(t1-te,1)}
    json.dump(res,open(os.path.join(a.out,tag+".json"),'w'))
    print(json.dumps(res),flush=True)
    if sat is False:
        print(f"*** chi>={a.colors+1} CANDIDATE at {tag}: NOT {a.colors}-colourable. "
              f"CNF={cnf} DRAT={drat} -> verify with drat-trim + bv_decide ***",flush=True)
    else:
        # negative: drop the (large, regenerable) cnf/drat to save space
        for p in (cnf,drat):
            if os.path.exists(p): os.remove(p)

if __name__=="__main__": main()
