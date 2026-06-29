#!/usr/bin/env python3
"""Analyze beta_sweep_v1 output: z3-gate instances, then for each disc_beta_scale level
report mean conflicts-to-UNSAT and the paired effect vs β=0 (damage curve).
DESCRIPTIVE characterisation — NOT a confirmation test (per pre-registration)."""
import re, sys, subprocess, tempfile, os, random

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/beta_sweep.out"
random.seed(20260629)
lines = open(PATH).read().splitlines()

instances = {}
cur = None; in_d = False
for ln in lines:
    m = re.match(r'^INSTANCE seed=(\d+)', ln)
    if m: cur=int(m.group(1)); instances[cur]={"dimacs":[],"cells":{}}; in_d=False; continue
    if ln.startswith('p cnf'): instances[cur]["dimacs"]=[ln]; in_d=True; continue
    r = re.match(r'^RESULT seed=(\d+) beta=(\d+) result=(-?\d+) decisions=(\d+) conflicts=(\d+)', ln)
    if r:
        in_d=False; s,b,res,dec,conf=map(int,r.groups())
        instances[s]["cells"][b]=(res,dec,conf); continue
    if in_d and re.match(r'^-?\d', ln): instances[cur]["dimacs"].append(ln)

def z3v(dl):
    with tempfile.NamedTemporaryFile('w',suffix='.cnf',delete=False) as f:
        f.write("\n".join(dl)+"\n"); fn=f.name
    try: out=subprocess.run(["z3","-dimacs",fn],capture_output=True,text=True,timeout=60).stdout
    finally: os.unlink(fn)
    return "UNSAT" if "unsat" in out.lower() else ("SAT" if "sat" in out.lower() else "?")

LEVELS=[0,25,50,100]
kept=[]; excluded=[]
for seed in sorted(instances):
    cells=instances[seed]["cells"]
    if any(b not in cells for b in LEVELS): excluded.append((seed,"missing cell")); continue
    if any(cells[b][0]!=0 for b in LEVELS): excluded.append((seed,"not all UNSAT")); continue
    if z3v(instances[seed]["dimacs"])!="UNSAT": excluded.append((seed,"z3 disagrees")); continue
    kept.append(seed)

n=len(kept)
print(f"kept(UNSAT,z3-agreed)={n}  excluded={len(excluded)}")
for s,w in excluded: print(f"  EXCLUDED {s}: {w}")
if n==0: sys.exit(0)

def conf(seed,b): return instances[seed]["cells"][b][2]
base=[conf(s,0) for s in kept]
mean0=sum(base)/n
print(f"\ndisc_beta_scale damage curve (n={n}, baseline β=0 mean conflicts={mean0:.1f}):")
print(f"{'scale':>6} {'mean_conf':>10} {'mean_Δ_vs_β0':>13} {'95%CI':>20} {'worse/better/tie':>18}")
for b in LEVELS:
    vals=[conf(s,b) for s in kept]; mean=sum(vals)/n
    deltas=[conf(s,0)-conf(s,b) for s in kept]   # >0 => this level better than β0
    md=sum(deltas)/n
    B=20000; ms=[]
    for _ in range(B):
        samp=[deltas[random.randrange(n)] for _ in range(n)]; ms.append(sum(samp)/n)
    ms.sort(); lo=ms[int(0.025*B)]; hi=ms[int(0.975*B)]
    worse=sum(1 for d in deltas if d<0); better=sum(1 for d in deltas if d>0); tie=n-worse-better
    sc=b/100
    print(f"{sc:>6.2f} {mean:>10.1f} {md:>+13.1f} {('['+format(lo,'+.1f')+','+format(hi,'+.1f')+']'):>20} {f'{worse}/{better}/{tie}':>18}")
print("\n(mean_Δ_vs_β0 < 0 means that scale is WORSE than β=0; monotone-negative => dose-response damage)")
