#!/usr/bin/env python3
"""Apply rules H1, H2, H3, H3' and H4 of examples/kmc/hot_atom_diffraction.sio
to its array results. Rules are copied from that header (with the H3
amendment made before any production data). Written before any M7 data was
read. Errors come from seed-to-seed scatter only.

Usage: hot_atom_aggregate.py result-task0.txt ... result-task19.txt
"""
import math
import re
import sys
from collections import defaultdict

KV = re.compile(r"(\w+)=([-\d.]+)")
SHELLS = (1, 2, 4, 5, 8, 9, 10, 13, 16)
CPS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
LAM = {0: 0.05, 1: 0.1, 2: 0.25, 3: 0.5}
NAME = {0: "(1,0) lam=.05", 1: "(1,0) lam=.1", 2: "(1,0) lam=.25", 3: "(1,0) lam=.5",
        4: "(1,1) lam=.2", 5: "histogram", 6: "(1,0) lam=.1 excl", 7: "histogram excl",
        8: "(1,1) lam=.2 excl"}
FIELD = {0: "F0 (1/2,1/2)", 1: "F1 (0,1/2)", 2: "F2 (1/2,0)"}


def cp_of(theta):
    return min((c for c in CPS if theta >= c - 1e-9), key=lambda c: abs(c - theta))


def mean(x):
    return sum(x) / len(x)


def sem(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1) / len(x))


def jack_ratio(nums, dens):
    n = len(nums)
    full = mean(nums) / mean(dens)
    loo = [(sum(nums) - nums[i]) / (sum(dens) - dens[i]) for i in range(n)]
    lm = mean(loo)
    return full, math.sqrt((n - 1) / n * sum((v - lm) ** 2 for v in loo))


def poisson_allow(n_tests, rate=0.0047, level=0.99):
    lam = rate * n_tests
    cum, k = 0.0, 0
    while True:
        cum += math.exp(-lam) * lam ** k / math.factorial(k)
        if cum >= level:
            return k, lam
        k += 1


def parse(paths):
    spot = defaultdict(dict)    # (kind, scen, field, cp) -> rep -> {shell: S}, plus pairs
    parity = defaultdict(dict)  # (scen, field, cp) -> rep -> (pairs, same, same_book, qlat, qpair)
    ends, sentinels = [], {}
    for p in paths:
        for line in open(p):
            head = line.split(" ", 1)[0]
            if head in ("SPOT", "PRED"):
                d = dict(KV.findall(line))
                f = int(line.split("field=F")[1][0])
                key = (head, int(d["scen"]), f, cp_of(float(d["theta"])))
                spot[key][int(d["rep"])] = ({k: float(d[f"S{k}u"]) * 1e-6 for k in SHELLS}, int(d["pairs"]))
            elif head == "PARITY":
                d = dict(KV.findall(line))
                f = int(line.split("field=F")[1][0])
                parity[(int(d["scen"]), f, cp_of(float(d["theta"])))][int(d["rep"])] = (
                    int(d["pairs"]), int(d["same"]), int(d["same_book"]), int(d["qlat"]), int(d["qpair"]))
            elif head == "HOT_END":
                ends.append(line.strip())
            elif head.startswith("M7_TASK"):
                sentinels[p] = line.strip()
    return spot, parity, ends, sentinels


def main(paths):
    spot, parity, ends, sentinels = parse(paths)
    print(f"== sentinels: {sum(1 for s in sentinels.values() if s.startswith('M7_TASK_OK'))}/{len(paths)} tasks OK;"
          f" runs with identities=ok: {sum('identities=ok' in e for e in ends)}/{len(ends)} ==")
    v = {}

    q_bad = sum(1 for reps in parity.values() for r in reps.values() if r[3] != r[4])
    b_bad = sum(1 for reps in parity.values() for r in reps.values() if r[1] != r[2])
    n_chk = sum(len(reps) for reps in parity.values())
    print(f"\n== H1 exact charge identity: {n_chk} checks, {q_bad} failures ==")
    print(f"== H3' exact successful-hop parity: {n_chk} checks, {b_bad} failures ==")
    v["H1"] = "HOLDS" if q_bad == 0 and n_chk else "FAILS"
    v["H3'"] = "HOLDS" if b_bad == 0 and n_chk else "FAILS"

    print("\n== H2: coverage 0.01, no exclusion, SPOT - PRED paired per seed, every shell ==")
    n_cmp, n_out, worst = 0, 0, 0.0
    for scen in range(6):
        row = []
        for f in (0, 1, 2):
            S = spot[("SPOT", scen, f, 0.01)]
            P = spot[("PRED", scen, f, 0.01)]
            reps = sorted(set(S) & set(P))
            for k in SHELLS:
                diffs = [S[r][0][k] - P[r][0][k] for r in reps]
                e = sem(diffs)
                z = abs(mean(diffs)) / e if e > 0 else 0.0
                n_cmp += 1
                worst = max(worst, z)
                n_out += z > 3
            s1 = mean([S[r][0][1] for r in reps])
            p1 = mean([P[r][0][1] for r in reps])
            row.append(f"{FIELD[f]} centre SPOT/PRED = {s1 / p1:.3f}")
        print(f"  {NAME[scen]:<16} " + "   ".join(row))
    allow, lam = poisson_allow(n_cmp)
    print(f"  {n_cmp} comparisons, {n_out} beyond 3 SE (worst {worst:.1f}); expectation {lam:.2f}, 99% allowance {allow}")
    v["H2"] = "HOLDS" if n_out <= allow else "FAILS"

    print("\n== H3 (expected to fail, partner blocking): F0 same-sublattice fraction at coverage 0.01 ==")
    h3 = True
    for scen, lam_s in LAM.items():
        fr = [same / pairs for (pairs, same, *_rest) in parity[(scen, 0, 0.01)].values()]
        m, e = mean(fr), sem(fr)
        pred = (1 - math.exp(-4 * lam_s)) / 2
        z = (m - pred) / e
        h3 = h3 and abs(z) < 3
        print(f"  lam={lam_s:<5} realised {m:.4f} +- {e:.4f}   unobstructed prediction {pred:.4f}   {z:+.1f} SE")
    v["H3"] = "HOLDS" if h3 else "FAILS"

    print("\n== H4 (exploratory): centre-shell SPOT/PRED vs coverage under NN exclusion (jackknife over seeds) ==")
    for scen, fields in ((6, (0,)), (7, (0, 2)), (8, (1, 2))):
        for f in fields:
            cells = []
            for cp in CPS:
                S = spot[("SPOT", scen, f, cp)]
                P = spot[("PRED", scen, f, cp)]
                reps = sorted(set(S) & set(P))
                if len(reps) < 5:
                    continue
                r, e = jack_ratio([S[x][0][1] for x in reps], [P[x][0][1] for x in reps])
                flag = "*" if abs(r - 1) > 0.10 and abs(r - 1) > 3 * e else ""
                cells.append(f"{cp:.2f}:{r:.2f}+-{e:.2f}{flag}")
            print(f"  {NAME[scen]:<18} {FIELD[f]:<13} " + "  ".join(cells))
    print("  (* = deviates from the exact reading by more than 10% and more than 3 SE)")

    print("\n== verdicts ==")
    for k in ("H1", "H2", "H3", "H3'"):
        print(f"  {k}: {v[k]}")


if __name__ == "__main__":
    main(sys.argv[1:])
