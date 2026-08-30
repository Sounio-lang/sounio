#!/usr/bin/env python3
"""Conflict-section workload stats + distinct-epm counting cross-check."""
import sys, time
sys.path.insert(0, ".")
from gen_elplus_data import load_go
from gen_go_full_data import topo_order

H, NR, sub, exsub, disj, rsub, rcomp = load_go("../go_full_elplus_tbox.txt")
order, parents = topo_order(H, sub)
anc = [(1 << c) | (1 << H) for c in range(H)]
for c in order:
    m = anc[c]
    for p in parents[c]:
        m |= anc[p]
    anc[c] = m

eps = sorted({x for pr in disj for x in pr})
ep_idx = {e: k for k, e in enumerate(eps)}
pb = [0] * len(eps)
for a, b in disj:
    pb[ep_idx[a]] |= 1 << ep_idx[b]
    pb[ep_idx[b]] |= 1 << ep_idx[a]
t0 = time.time()
epm = [0] * H
for e in eps:
    bit = 1 << e
    be = 1 << ep_idx[e]
    for c in range(H):
        if anc[c] & bit:
            epm[c] |= be
pm = [0] * H
actors = []
for c in range(H):
    m, p, k = epm[c], 0, 0
    while m:
        if m & 1:
            p |= pb[k]
        m >>= 1
        k += 1
    pm[c] = p
    if p or epm[c]:
        actors.append(c)
print(f"eps={len(eps)} nact={len(actors)} build={time.time()-t0:.1f}s")
n_pm = sum(1 for c in actors if pm[c])
n_ep = sum(1 for c in actors if epm[c])
print(f"actors with pm!=0: {n_pm}; with epm!=0: {n_ep}")

t0 = time.time()
n_conf = 0
for c1 in actors:
    for c2 in actors:
        if c1 != c2 and (pm[c1] & epm[c2]):
            n_conf += 1
print(f"naive conf={n_conf} ({time.time()-t0:.1f}s)")

# distinct-epm grouping
t0 = time.time()
from collections import Counter
mult = Counter(epm[c] for c in actors if epm[c])
vals = list(mult.items())
print(f"distinct epm values among actors: {len(vals)}")
conf2 = 0
for c1 in actors:
    p = pm[c1]
    if not p:
        continue
    s = 0
    for v, mlt in vals:
        if v & p:
            s += mlt
    if epm[c1] & p:
        s -= 1   # exclude c2 == c1
    conf2 += s
print(f"grouped conf={conf2} ({time.time()-t0:.1f}s)")
print("MATCH" if conf2 == n_conf == 792814846 else "MISMATCH")
