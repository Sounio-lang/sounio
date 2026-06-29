#!/usr/bin/env python3
"""Analyze the 2x2 conjunction ablation. z3-gate every instance; pair each config
(BANDIT 3/0, POL 1/1, CONJ 3/1) against BASE (1/0) by conflicts-to-UNSAT."""
import re, sys, subprocess, tempfile, os, random
PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/iso_conj.out"
random.seed(20260629)
lines = open(PATH).read().splitlines()
inst={}; cur=None; in_d=False
for ln in lines:
    m=re.match(r'^INSTANCE seed=(\d+)', ln)
    if m: cur=int(m.group(1)); inst[cur]={"dimacs":[],"cells":{}}; in_d=False; continue
    if ln.startswith('p cnf'): inst[cur]["dimacs"]=[ln]; in_d=True; continue
    r=re.match(r'^RESULT seed=(\d+) sm=(\d+) pm=(\d+) result=(-?\d+) decisions=(\d+) conflicts=(\d+)', ln)
    if r:
        in_d=False; s,sm,pm,res,dec,conf=map(int,r.groups()); inst[s]["cells"][(sm,pm)]=(res,conf); continue
    if in_d and re.match(r'^-?\d', ln): inst[cur]["dimacs"].append(ln)
def z3v(dl):
    with tempfile.NamedTemporaryFile('w',suffix='.cnf',delete=False) as f:
        f.write("\n".join(dl)+"\n"); fn=f.name
    try: out=subprocess.run(["z3","-dimacs",fn],capture_output=True,text=True,timeout=60).stdout
    finally: os.unlink(fn)
    return "UNSAT" if "unsat" in out.lower() else ("SAT" if "sat" in out.lower() else "?")
CFGS=[(1,0,"BASE"),(3,0,"BANDIT"),(1,1,"POL"),(3,1,"CONJ")]
kept=[]; excl=[]
for seed in sorted(inst):
    c=inst[seed]["cells"]
    if any((sm,pm) not in c for sm,pm,_ in CFGS): excl.append((seed,"missing")); continue
    if any(c[(sm,pm)][0]!=0 for sm,pm,_ in CFGS): excl.append((seed,"not all UNSAT")); continue
    if z3v(inst[seed]["dimacs"])!="UNSAT": excl.append((seed,"z3 disagrees")); continue
    kept.append(seed)
n=len(kept)
print(f"kept(UNSAT,z3-agreed)={n} excluded={len(excl)}")
for s,w in excl: print(f"  EXCLUDED {s}: {w}")
if n==0: sys.exit(0)
def conf(seed,sm,pm): return inst[seed]["cells"][(sm,pm)][1]
basevals=[conf(s,1,0) for s in kept]; mbase=sum(basevals)/n
print(f"\n[ISOLATED 2x2] BASE(mean,saved) mean conflicts = {mbase:.1f}  (n={n})")
print(f"{'config':>8} {'mean_conf':>10} {'meanΔ_vs_BASE':>14} {'95%CI':>20} {'better/worse/tie':>18}")
for sm,pm,lab in CFGS:
    vals=[conf(s,sm,pm) for s in kept]; mean=sum(vals)/n
    d=[conf(s,1,0)-conf(s,sm,pm) for s in kept]; md=sum(d)/n   # >0 => config better than BASE
    B=20000; ms=[]
    for _ in range(B):
        samp=[d[random.randrange(n)] for _ in range(n)]; ms.append(sum(samp)/n)
    ms.sort(); lo=ms[int(.025*B)]; hi=ms[int(.975*B)]
    better=sum(1 for x in d if x>0); worse=sum(1 for x in d if x<0); tie=n-better-worse
    print(f"{lab:>8} {mean:>10.1f} {md:>+14.1f} {('['+format(lo,'+.1f')+','+format(hi,'+.1f')+']'):>20} {f'{better}/{worse}/{tie}':>18}")
print("\n(meanΔ_vs_BASE > 0 => that config uses FEWER conflicts than mean-only BASE => helps)")
