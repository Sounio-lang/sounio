#!/usr/bin/env python3
"""Analyze process-isolated β-ablation (beta levels 0 and 100). z3-gate every instance,
pair β=0 vs β>0 conflicts, report mean Δ + bootstrap CI + paired sign test.
Each instance was solved in its OWN process (no cross-instance state leak)."""
import re, sys, subprocess, tempfile, os, random
PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/iso_all.out"
random.seed(20260629)
lines = open(PATH).read().splitlines()
inst = {}; cur=None; in_d=False
for ln in lines:
    m=re.match(r'^INSTANCE seed=(\d+)', ln)
    if m: cur=int(m.group(1)); inst[cur]={"dimacs":[],"cells":{}}; in_d=False; continue
    if ln.startswith('p cnf'): inst[cur]["dimacs"]=[ln]; in_d=True; continue
    r=re.match(r'^RESULT seed=(\d+) beta=(\d+) result=(-?\d+) decisions=(\d+) conflicts=(\d+)', ln)
    if r:
        in_d=False; s,b,res,dec,conf=map(int,r.groups()); inst[s]["cells"][b]=(res,dec,conf); continue
    if in_d and re.match(r'^-?\d', ln): inst[cur]["dimacs"].append(ln)
def z3v(dl):
    with tempfile.NamedTemporaryFile('w',suffix='.cnf',delete=False) as f:
        f.write("\n".join(dl)+"\n"); fn=f.name
    try: out=subprocess.run(["z3","-dimacs",fn],capture_output=True,text=True,timeout=60).stdout
    finally: os.unlink(fn)
    return "UNSAT" if "unsat" in out.lower() else ("SAT" if "sat" in out.lower() else "?")
kept=[]; excl=[]
for seed in sorted(inst):
    c=inst[seed]["cells"]
    if 0 not in c or 100 not in c: excl.append((seed,"missing cell")); continue
    if c[0][0]!=0 or c[100][0]!=0: excl.append((seed,f"not UNSAT r0={c[0][0]} r100={c[100][0]}")); continue
    if z3v(inst[seed]["dimacs"])!="UNSAT": excl.append((seed,"z3 disagrees")); continue
    kept.append((seed,c[0][2],c[100][2]))
n=len(kept)
print(f"kept(UNSAT,z3-agreed)={n} excluded={len(excl)}")
for s,w in excl: print(f"  EXCLUDED {s}: {w}")
if n==0: sys.exit(0)
deltas=[c0-c1 for (_,c0,c1) in kept]   # >0 => β>0 fewer conflicts (helps)
m0=sum(c0 for (_,c0,_) in kept)/n; m1=sum(c1 for (_,_,c1) in kept)/n; md=sum(deltas)/n
B=20000; ms=[]
for _ in range(B):
    samp=[deltas[random.randrange(n)] for _ in range(n)]; ms.append(sum(samp)/n)
ms.sort(); lo=ms[int(.025*B)]; hi=ms[int(.975*B)]
pos=sum(1 for d in deltas if d>0); neg=sum(1 for d in deltas if d<0); tie=n-pos-neg
print(f"\n[PROCESS-ISOLATED] mean conflicts  β=0: {m0:.1f}   β>0: {m1:.1f}")
print(f"mean Δ (β0-βpos) = {md:+.2f}   95% bootstrap CI [{lo:+.2f},{hi:+.2f}]  (>0 => β>0 helps)")
print(f"paired sign: β>0 better {pos} / worse {neg} / tie {tie}  (of {n})")
print("VERDICT:", "CI excludes 0 -> significant" if (lo>0 or hi<0) else "CI includes 0 -> null")
print("\nper-instance (seed, β0, βpos, Δ):")
for (s,c0,c1) in kept: print(f"  {s}: {c0:5d} {c1:5d}  Δ={c0-c1:+d}")
