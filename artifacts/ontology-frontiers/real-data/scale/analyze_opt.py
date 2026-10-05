#!/usr/bin/env python3
"""Round 13 analysis: workload statistics of the round-12 bitmask reduction
on the full GO go-plus TBox, and a python prototype of the optimized
fixpoint (worklist + version-skipped roleComp) cross-checked against the
round-12 mirror semantics (dict-based naive fixpoint)."""
import sys, time
sys.path.insert(0, ".")
from gen_elplus_data import load_go
from gen_go_full_data import topo_order, bitmask_reduce

H, NR, sub, exsub, disj, rsub, rcomp = load_go("../go_full_elplus_tbox.txt")
print(f"H={H} NR={NR} sub={len(sub)} exsub={len(exsub)} disj={len(disj)} "
      f"rsub={len(rsub)} rcomp={len(rcomp)}")

# chains per r1 role
from collections import Counter
c1 = Counter(k[0] for k in rcomp)
print("chains per r1: max", max(c1.values()), "dist", dict(c1))
# roles actually used in exsub
used = Counter(r for _, r, _ in exsub)
print("roles used in exsub:", len(used))

t0 = time.time()
order, parents = topo_order(H, sub)
anc = [(1 << c) | (1 << H) for c in range(H)]
for c in order:
    m = anc[c]
    for p in parents[c]:
        m |= anc[p]
    anc[c] = m
print(f"anc build: {time.time()-t0:.1f}s")

# role hierarchy closure
rclos = [[r == s for s in range(NR)] for r in range(NR)]
for r, s in rsub:
    rclos[r][s] = True
changed = True
while changed:
    changed = False
    for a in range(NR):
        for b in range(NR):
            if rclos[a][b]:
                for d in range(NR):
                    if rclos[b][d] and not rclos[a][d]:
                        rclos[a][d] = True; changed = True
rclos_pairs = [(r, s) for r in range(NR) for s in range(NR)
               if rclos[r][s] and r != s]
print("proper roleSub pairs:", len(rclos_pairs))
# supers per role
supers = {}
for r, s in rclos_pairs:
    supers.setdefault(r, []).append(s)
print("max supers per role:", max(len(v) for v in supers.values()))

t0 = time.time()
F = [dict() for _ in range(NR)]
for c, r, f in exsub:
    F[r][c] = F[r].get(c, 0) | anc[f]
for c in order:
    for p in parents[c]:
        for r in range(NR):
            fr = F[r]
            if p in fr:
                v = fr.get(c, 0) | fr[p]
                if v != fr.get(c, 0):
                    fr[c] = v
ne_cells = sum(len(fr) for fr in F)
bits = sum(bin(m).count("1") for fr in F for m in fr.values())
print(f"after seed+expand: non-empty cells={ne_cells} bits={bits} "
      f"({time.time()-t0:.1f}s)")

# ── optimized fixpoint prototype ──────────────────────────────────────
# worklist roleSub + roleComp with per-(chain,cell) version skipping.
t0 = time.time()
ver = [dict() for _ in range(NR)]   # ver[r][c] monotone version
gver = [0]
def bump(r, c):
    gver[0] += 1
    ver[r][c] = gver[0]

dirty = [set(fr.keys()) for fr in F]
for r in range(NR):
    for c in F[r]:
        bump(r, c)

# per-role chain lists, with slot index per chain
chains_of = {}
for k, (r1, r2, r3) in enumerate(rcomp):
    chains_of.setdefault(r1, []).append(k)
# last processed input-version per (chain, cell) — sparse dict
lastproc = {}   # (k, c) -> version
work_or = 0     # instrumented: number of row-OR operations
work_scan = 0   # bit scans

rounds = 0
while True:
    rounds += 1
    any_change = False
    # roleSub worklist drain
    while any(dirty):
        cur = dirty
        dirty = [set() for _ in range(NR)]
        for r, slist in supers.items():
            cr = cur[r]
            if not cr:
                continue
            for s in slist:
                Fs = F[s]
                for c in cr:
                    v = Fs.get(c, 0)
                    nv = v | F[r][c]
                    work_or += 1
                    if nv != v:
                        Fs[c] = nv
                        dirty[s].add(c)
                        bump(s, c)
                        any_change = True
    # roleComp scan with version skip
    for k, (r1, r2, r3) in enumerate(rcomp):
        F1, F2, F3 = F[r1], F[r2], F[r3]
        v2 = ver[r2]
        for c, m in F1.items():
            v1 = ver[r1].get(c, 0)
            lp = lastproc.get((k, c), -1)
            if v1 <= lp:
                # check whether any f in m has ver[r2][f] > lp
                mm = m
                new = False
                while mm:
                    b = mm & -mm
                    f = b.bit_length() - 1
                    mm ^= b
                    work_scan += 1
                    if v2.get(f, 0) > lp:
                        new = True
                        break
                if not new:
                    continue
            acc = 0
            mm = m
            while mm:
                b = mm & -mm
                f = b.bit_length() - 1
                mm ^= b
                fv = F2.get(f, 0)
                if fv:
                    acc |= fv
                    work_or += 1
            if acc:
                v3 = F3.get(c, 0)
                nv = v3 | acc
                if nv != v3:
                    F3[c] = nv
                    dirty[r3].add(c)
                    bump(r3, c)
                    any_change = True
            lastproc[(k, c)] = gver[0]
    if not any_change:
        break

ne2 = sum(len(fr) for fr in F)
bits2 = sum(bin(m).count("1") for fr in F for m in fr.values())
print(f"optimized fixpoint: rounds={rounds} cells={ne2} role_edges={bits2} "
      f"row-ORs={work_or} bit-scans={work_scan} ({time.time()-t0:.1f}s)")

# cross-check against the reference naive implementation
ref = bitmask_reduce(H, NR, sub, exsub, disj, rsub, rcomp)
print("reference: role_edges_atom =", ref["role_edges_atom"],
      "rounds =", ref["rounds"])
print("MATCH" if bits2 == ref["role_edges_atom"] else "MISMATCH!")
