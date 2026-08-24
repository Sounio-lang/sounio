#!/usr/bin/env python3
"""Round 13 prototype: EXACT simulation of the planned optimized Sounio
algorithm (sparse sorted-list rows, single-queue roleSub worklist,
per-(chain,cell) version-skipped roleComp, grouped conflict counting).
Must reproduce the round-12 mirror numbers in all three configurations."""
import sys, time
sys.path.insert(0, ".")
from gen_elplus_data import load_go
from gen_go_full_data import topo_order, bitmask_reduce

H, NR, sub, exsub, disj, rsub, rcomp = load_go("../go_full_elplus_tbox.txt")
order, parents = topo_order(H, sub)

t0 = time.time()
anc = [(1 << c) | (1 << H) for c in range(H)]
for c in order:
    m = anc[c]
    for p in parents[c]:
        m |= anc[p]
    anc[c] = m
atomic_edges = sum(bin(anc[c] & ((1 << H) - 1)).count("1") for c in range(H))

# role hierarchy closure -> supers CSR
rclos = [[r == s for s in range(NR)] for r in range(NR)]
for r, s in rsub:
    rclos[r][s] = True
ch = True
while ch:
    ch = False
    for a in range(NR):
        for b in range(NR):
            if rclos[a][b]:
                for d in range(NR):
                    if rclos[b][d] and not rclos[a][d]:
                        rclos[a][d] = True; ch = True
supers = {r: [] for r in range(NR)}
for r in range(NR):
    for s in range(NR):
        if r != s and rclos[r][s]:
            supers[r].append(s)
chains_of = {r: [] for r in range(NR)}
for k, (r1, r2, r3) in enumerate(rcomp):
    chains_of[r1].append(k)

def run(with_rsub, with_rcomp):
    # rows: dict cell -> sorted list; cell = r*HC + c simulated as (r, c)
    F = [dict() for _ in range(NR)]     # r -> {c: sorted list}
    ver = [dict() for _ in range(NR)]   # r -> {c: version}
    lpv = {}                            # (k, c) -> version
    rmax = [0] * NR
    gver = [0]
    from collections import deque
    dq = deque()
    inq = [dict() for _ in range(NR)]
    stats = [0, 0]  # merges, merge_ops

    def bump(r, c):
        gver[0] += 1
        ver[r][c] = gver[0]
        rmax[r] = gver[0]
        if not inq[r].get(c):
            inq[r][c] = True
            dq.append((r, c))

    def merge(dst, src):
        # sorted union; returns (newlist, changed)
        i = j = 0
        out = []
        nd, ns = len(dst), len(src)
        while i < nd and j < ns:
            dv, sv = dst[i], src[j]
            stats[1] += 1
            if dv < sv:
                out.append(dv); i += 1
            elif sv < dv:
                out.append(sv); j += 1
            else:
                out.append(dv); i += 1; j += 1
        out.extend(dst[i:]); out.extend(src[j:])
        return out, len(out) != nd

    def merge_into(r, c, src):
        stats[0] += 1
        dst = F[r].get(c)
        if dst is None:
            F[r][c] = list(src)
            bump(r, c)
            return True
        out, changed = merge(dst, src)
        if changed:
            F[r][c] = out
            bump(r, c)
        return changed

    # seed
    for c, r, f in exsub:
        m = anc[f]
        lst = []
        while m:
            b = m & -m
            lst.append(b.bit_length() - 1)
            m ^= b
        merge_into(r, c, lst)
    # expand (topo order, skip empty parent rows)
    for c in order:
        for p in parents[c]:
            for r in range(NR):
                row = F[r].get(p)
                if row:
                    merge_into(r, c, row)
    ne_after_seed = sum(len(fr) for fr in F)

    rounds = 0
    changed = True
    while changed:
        changed = False
        rounds += 1
        # roleSub drain
        while dq:
            r, c = dq.popleft()
            inq[r][c] = False
            if with_rsub:
                row = F[r].get(c)
                if row:
                    for s in supers[r]:
                        if merge_into(s, c, row):
                            changed = True
        # roleComp scan with version skip
        if with_rcomp:
            for r1 in range(NR):
                ks = chains_of[r1]
                if not ks:
                    continue
                F1 = F[r1]
                for c in list(F1.keys()):
                    for k in ks:
                        row = F1[c]   # fresh per chain: a chain with r3 == r1
                        v1 = ver[r1].get(c, 0)   # may grow the row mid-loop
                        r2, r3 = rcomp[k][1], rcomp[k][2]
                        lp = lpv.get((k, c), 0)
                        if v1 > lp:
                            proc = True
                        else:
                            proc = False
                            v2 = ver[r2]
                            for f in row:
                                if v2.get(f, 0) > lp:
                                    proc = True
                                    break
                        if not proc:
                            continue
                        # acc = union of F[r2][f] over f in row
                        accm = 0
                        F2 = F[r2]
                        for f in row:
                            m2 = F2.get(f)
                            if m2:
                                for g in m2:
                                    accm |= 1 << g
                                    stats[1] += 1
                        if accm:
                            lst = []
                            while accm:
                                b = accm & -accm
                                lst.append(b.bit_length() - 1)
                                accm ^= b
                            if merge_into(r3, c, lst):
                                changed = True
                        lpv[(k, c)] = gver[0]
    total = sum(len(v) for fr in F for v in fr.values())
    return total, rounds, ne_after_seed, stats[0], stats[1]

for name, ws, wc in [("full", True, True), ("no-roleComp", True, False),
                     ("no-roleSub", False, True)]:
    t0 = time.time()
    total, rounds, ne0, nm, nops = run(ws, wc)
    print(f"{name}: role_edges={total} rounds={rounds} "
          f"cells_after_seed={ne0} merges={nm} merge_ops={nops} "
          f"({time.time()-t0:.1f}s)")

# conflicts with grouping
t0 = time.time()
eps = sorted({x for pr in disj for x in pr})
ep_idx = {e: k for k, e in enumerate(eps)}
pb = [0] * len(eps)
for a, b in disj:
    pb[ep_idx[a]] |= 1 << ep_idx[b]
    pb[ep_idx[b]] |= 1 << ep_idx[a]
epm = [0] * H
for e in eps:
    bit = 1 << e
    be = 1 << ep_idx[e]
    for c in range(H):
        if anc[c] & bit:
            epm[c] |= be
pm = [0] * H
for c in range(H):
    m, p, k = epm[c], 0, 0
    while m:
        if m & 1:
            p |= pb[k]
        m >>= 1
        k += 1
    pm[c] = p
from collections import Counter
mult = Counter(v for v in epm if v)
vals = list(mult.items())
conf = 0
for c in range(H):
    p = pm[c]
    if not p:
        continue
    s = 0
    for v, mlt in vals:
        if v & p:
            s += mlt
    if epm[c] & p:
        s -= 1
    conf += s
print(f"conflicts (grouped): {conf} ({time.time()-t0:.1f}s)")
print("expected: full=2135207 norc=1883813 nors=597305 conf=792814846 "
      f"atomic={atomic_edges}")
