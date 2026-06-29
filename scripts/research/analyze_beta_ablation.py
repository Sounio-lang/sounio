#!/usr/bin/env python3
"""Analyze beta_ablation_v1 output: z3-gate every instance, pair beta=0 vs beta>0
conflicts-to-UNSAT, report mean delta + bootstrap CI + paired sign test.

Single source of truth: the DIMACS solved is the DIMACS emitted by the .sio harness.
Null result is a valid, reportable outcome (see the pre-registration)."""
import re, sys, subprocess, tempfile, os, random

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/beta_abl.out"
random.seed(20260629)  # fixed — analysis reproducibility, not instance generation

lines = open(PATH).read().splitlines()

# Parse instances: INSTANCE header -> following 'p cnf' + clause lines; RESULT lines.
instances = {}   # seed -> {"dimacs": [...], "cells": {beta: (result, dec, conf)}}
cur = None
in_dimacs = False
for ln in lines:
    m = re.match(r'^INSTANCE seed=(\d+)', ln)
    if m:
        cur = int(m.group(1)); instances[cur] = {"dimacs": [], "cells": {}}; in_dimacs = False; continue
    if ln.startswith('p cnf'):
        instances[cur]["dimacs"] = [ln]; in_dimacs = True; continue
    r = re.match(r'^RESULT seed=(\d+) beta=(\d+) result=(-?\d+) decisions=(\d+) conflicts=(\d+)', ln)
    if r:
        in_dimacs = False
        s, b, res, dec, conf = map(int, r.groups())
        instances[s]["cells"][b] = (res, dec, conf)
        continue
    if in_dimacs and re.match(r'^-?\d', ln):
        instances[cur]["dimacs"].append(ln)

def z3_verdict(dimacs_lines):
    with tempfile.NamedTemporaryFile('w', suffix='.cnf', delete=False) as f:
        f.write("\n".join(dimacs_lines) + "\n"); fn = f.name
    try:
        out = subprocess.run(["z3","-dimacs",fn], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.unlink(fn)
    if "unsat" in out.lower(): return "UNSAT"
    if "sat" in out.lower(): return "SAT"
    return "?"

# Gate + pair
kept, excluded = [], []
for seed in sorted(instances):
    inst = instances[seed]
    cells = inst["cells"]
    if 0 not in cells or 10 not in cells:
        excluded.append((seed,"missing cell")); continue
    r0, d0, c0 = cells[0]; r1, d1, c1 = cells[10]
    if r0 != 0 or r1 != 0:
        excluded.append((seed,f"not UNSAT (r0={r0},r1={r1}) — possibly censored")); continue
    z = z3_verdict(inst["dimacs"])
    if z != "UNSAT":
        excluded.append((seed,f"z3={z} disagrees")); continue
    kept.append((seed, c0, c1, d0, d1))

n = len(kept)
print(f"instances parsed={len(instances)} kept(UNSAT,z3-agreed,uncensored)={n} excluded={len(excluded)}")
for s,why in excluded: print(f"  EXCLUDED seed={s}: {why}")
if n == 0:
    print("no usable instances"); sys.exit(0)

deltas = [c0 - c1 for (_, c0, c1, _, _) in kept]   # >0 means beta>0 used FEWER conflicts (helps)
mean_c0 = sum(c0 for (_,c0,_,_,_) in kept)/n
mean_c1 = sum(c1 for (_,_,c1,_,_) in kept)/n
mean_d = sum(deltas)/n
# bootstrap 95% CI of mean delta
B=20000; means=[]
for _ in range(B):
    samp=[deltas[random.randrange(n)] for _ in range(n)]
    means.append(sum(samp)/n)
means.sort(); lo=means[int(0.025*B)]; hi=means[int(0.975*B)]
# paired sign test (two-sided), exact-ish via normal approx + raw counts
pos=sum(1 for d in deltas if d>0); neg=sum(1 for d in deltas if d<0); ties=sum(1 for d in deltas if d==0)

print(f"\nmean conflicts  beta=0: {mean_c0:.1f}   beta>0: {mean_c1:.1f}")
print(f"mean delta (beta0 - betaPos) = {mean_d:+.2f}   95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]")
print(f"  (positive delta => beta>0 reduces conflicts)")
print(f"paired sign: beta>0 better on {pos} / worse on {neg} / tie on {ties}  (of {n})")
sig = "CI excludes 0 -> significant" if (lo>0 or hi<0) else "CI includes 0 -> NOT significant (null)"
print(f"VERDICT: {sig}")
print("\nper-instance (seed, conf_b0, conf_bpos, delta):")
for (s,c0,c1,_,_) in kept: print(f"  {s}: {c0:5d} {c1:5d}  d={c0-c1:+d}")
