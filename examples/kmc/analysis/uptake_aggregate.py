#!/usr/bin/env python3
"""Apply rules P1-P5 of examples/kmc/uptake_diffraction.sio to its array
results. Rules are copied from that header, fixed before the run.

Errors are from independent seeds only (SD/sqrt(n) of per-seed values, or a
jackknife for ratios). Written before any M6 production data was looked at.

Usage: uptake_aggregate.py result-task0.txt ... result-task19.txt
"""
import math
import re
import sys
from collections import defaultdict

KV = re.compile(r"(\w+)=([-\d.()]+)")
SHELLS = (1, 2, 4, 5, 8, 9, 10, 13, 16)
CHECKPOINTS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
SCEN = {0: "(1,0)", 1: "(1,1)", 2: "(2,0)", 3: "(1,1)+10%(2,0)", 4: "(1,1)+30%(2,0)",
        5: "(1,1) NN-excl", 6: "(2,0) NN-excl", 7: "(4,0) NN-excl"}
# which fields each scenario makes pure dipole (conserved)
DIPOLE_FIELDS = {0: {0}, 1: {1, 2}, 2: set(), 3: set(), 4: set(), 5: {1, 2}, 6: set(), 7: set()}
N = 256 * 256


def cp_of(theta):
    """Map an actual coverage to the checkpoint it was recorded at."""
    return min(CHECKPOINTS, key=lambda c: abs(c - theta) if theta >= c - 1e-9 else 9)


def parse(paths):
    spots = defaultdict(lambda: defaultdict(dict))   # (scen, field, cp) -> rep -> (theta, pairs, {shell: S})
    charges = defaultdict(list)                       # scen -> [(cp, rep, Q0, Q1, Q2)]
    analytic = {}                                     # (dx, dy, field) -> {shell: A}
    sentinels = {}
    for p in paths:
        for line in open(p):
            if line.startswith("SPOT "):
                d = dict(KV.findall(line))
                scen, rep, fi = int(d["scen"]), int(d["rep"]), int(line.split("field=F")[1][0])
                theta = float(d["theta"])
                S = {k: float(d[f"S{k}u"]) * 1e-6 for k in SHELLS}
                spots[(scen, fi, cp_of(theta))][rep] = (theta, int(d["pairs"]), S)
            elif line.startswith("CHARGE "):
                d = dict(KV.findall(line))
                charges[int(d["scen"])].append((cp_of(float(d["theta"])), int(d["rep"]),
                                                int(d["Q0"]), int(d["Q1"]), int(d["Q2"])))
            elif line.startswith("ANALYTIC "):
                m = re.search(r"d=\((\d+),(\d+)\) field=F(\d)", line)
                d = dict(KV.findall(line))
                analytic[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = \
                    {k: float(d[f"A{k}"]) for k in SHELLS}
            elif line.startswith("M6_TASK"):
                sentinels[p] = line.strip()
    return spots, charges, analytic, sentinels


def mean(x):
    return sum(x) / len(x)


def sem(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1) / len(x))


def jack(items, stat):
    n = len(items)
    full = stat(items)
    loo = [stat(items[:i] + items[i + 1:]) for i in range(n)]
    lm = mean(loo)
    return full, math.sqrt((n - 1) / n * sum((v - lm) ** 2 for v in loo))


def main(paths):
    spots, charges, analytic, sentinels = parse(paths)
    ok = sum(1 for s in sentinels.values() if s.startswith("M6_TASK_OK"))
    print(f"== sentinels: {ok}/{len(paths)} tasks OK ==")
    verdict = {}

    print("\n== P1: exact zero total charge in every dipole field, at every checkpoint ==")
    bad = 0
    total = 0
    for scen, rows in charges.items():
        for cp, rep, q0, q1, q2 in rows:
            for f, q in ((0, q0), (1, q1), (2, q2)):
                if f in DIPOLE_FIELDS[scen]:
                    total += 1
                    if q != 0:
                        bad += 1
    print(f"  {total} checks, {bad} nonzero")
    verdict["P1"] = "HOLDS" if bad == 0 and total > 0 else "FAILS"

    print("\n== P2: coverage 0.01, no exclusion, simulation vs (pairs/N)*A(q) in every shell ==")
    p2_fail = 0
    p2_n = 0
    worst = 0.0
    for scen, (dx, dy) in ((0, (1, 0)), (1, (1, 1)), (2, (2, 0))):
        for f in (0, 1, 2):
            reps = spots[(scen, f, 0.01)]
            if len(reps) < 3:
                print(f"  scen {scen} F{f}: only {len(reps)} seeds")
                p2_fail += 1
                continue
            A = analytic[(dx, dy, f)]
            cells = []
            for k in SHELLS:
                ratios = [S[k] / (pairs / N) for (_, pairs, S) in reps.values()]
                m, e = mean(ratios), sem(ratios)
                z = abs(m - A[k]) / e if e > 0 else (0.0 if abs(m - A[k]) < 1e-12 else float("inf"))
                worst = max(worst, z)
                p2_n += 1
                if z > 3:
                    p2_fail += 1
                cells.append(f"n2={k}: {m:.4g}/{A[k]:.4g} ({z:.1f})")
            print(f"  {SCEN[scen]:<6} F{f}  " + "  ".join(cells[:4]) + "  ...")
    # The header states P2 as "within 3 SE in every shell". With ~81 comparisons
    # and 40 seeds (t tail P(|t|>3) ~ 0.0047), that strict reading fails by chance
    # about a third of the time -- a mis-specified rule, found by calibrating this
    # script on synthetic data BEFORE any real M6 data was read. Both readings are
    # reported; the multiplicity-aware one allows as many exceedances as a
    # Poisson(0.0047 n) count reaches with 99% probability.
    lam = 0.0047 * p2_n
    allow, cum, k = 0, 0.0, 0
    while True:
        cum += math.exp(-lam) * lam ** k / math.factorial(k)
        if cum >= 0.99:
            allow = k
            break
        k += 1
    print(f"  {p2_n} shell comparisons, {p2_fail} beyond 3 SE, worst {worst:.1f} SE "
          f"(Poisson expectation {lam:.2f}; 99% allowance {allow})")
    verdict["P2"] = ("HOLDS" if p2_fail <= allow else "FAILS") + \
        f" (multiplicity-aware; strict header reading: {'HOLDS' if p2_fail == 0 else 'FAILS'})"

    print("\n== P3: small-q slope s = log2(S4/S1) at coverage 0.05 ==")
    p3 = True
    for scen in (0, 1, 2):
        for f in (0, 1, 2):
            reps = list(spots[(scen, f, 0.05)].values())
            s, e = jack(reps, lambda rs: math.log2(mean([r[2][4] for r in rs]) / mean([r[2][1] for r in rs])))
            dip = f in DIPOLE_FIELDS[scen]
            if dip:
                passed = s - 1 > 3 * e
            else:
                passed = abs(s) < 1
            p3 = p3 and passed
            print(f"  {SCEN[scen]:<6} F{f} {'dipole  ' if dip else 'monopole'}  s = {s:+.3f} +- {e:.3f}  "
                  f"{'ok' if passed else 'FAIL'}")
    verdict["P3"] = "HOLDS" if p3 else "FAILS"

    print("\n== P4: p(2x2) plateau S(q_min) at coverage 0.01 linear in the (2,0) fraction f ==")
    p4 = True
    for scen, fr in ((3, 0.1), (4, 0.3)):
        for f in (1, 2):
            reps = spots[(scen, f, 0.01)]
            ratios = [S[1] / (pairs / N) for (_, pairs, S) in reps.values()]
            m, e = mean(ratios), sem(ratios)
            pred = (1 - fr) * analytic[(1, 1, f)][1] + fr * analytic[(2, 0, f)][1]
            z = abs(m - pred) / e
            p4 = p4 and z < 3
            print(f"  f={fr} F{f}: S1/(pairs/N) = {m:.4f} +- {e:.4f}   predicted {pred:.4f}   {z:.1f} SE")
    verdict["P4"] = "HOLDS" if p4 else "FAILS"

    print("\n== P5 (exploratory): NN exclusion, spot profiles vs coverage, seed means ==")
    for scen in (5, 6, 7):
        print(f"  {SCEN[scen]}")
        for cp in CHECKPOINTS:
            line = []
            for f, name in ((0, "(1/2,1/2)"), (1, "(0,1/2)"), (2, "(1/2,0)")):
                reps = list(spots[(scen, f, cp)].values())
                if len(reps) < 3:
                    continue
                s1 = mean([r[2][1] for r in reps])
                s4 = mean([r[2][4] for r in reps])
                line.append(f"{name} S1={s1:.3g} s={math.log2(s4 / s1):+.2f}")
            if line:
                print(f"    theta={cp:.2f}  " + "   ".join(line))

    print("\n== verdicts ==")
    for k in ("P1", "P2", "P3", "P4"):
        print(f"  {k}: {verdict[k]}")


if __name__ == "__main__":
    main(sys.argv[1:])
