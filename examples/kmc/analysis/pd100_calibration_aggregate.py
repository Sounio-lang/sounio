#!/usr/bin/env python3
"""Apply rules C1-C4 of examples/kmc/uptake_pd100_calibration.sio. Written
while its array ran, before any of its data was read. Errors come only from
seed-to-seed scatter.

AMENDMENT TO C4, made while writing this script and before any data: the
line-profile "shoulder" (mean of S at m = 48..80 along [10] and [01]) is a bad
normaliser. For an axis-aligned pair such as (4,0), the component
perpendicular to the scan has dy = 0 and contributes cos(0) = 1 at every q, so
no band along a line averages it away. Analytically at low coverage R_line is
1.140 for Lin-Jiang and 1.147 for (4,0): it cannot tell them apart, although
their true centre ratios are 1.185 and 2.000. The correct normaliser is the
diffuse intensity averaged over the whole Brillouin zone (excluding Bragg
points), which by Parseval equals the coverage theta exactly; then
R = S(centre) / theta = 2 P_same, the prediction card's ratio, and C1/C3's
"centre per pair" divided by 2 is that quantity. C4 is still printed, as an
explicit demonstration of the wrong normalisation, not as a discriminator.

Usage: pd100_calibration_aggregate.py result-task0.txt ... result-task19.txt
"""
import math
import re
import sys
from collections import defaultdict

KV = re.compile(r"(\w+)=([-\d.]+)")
CPS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MS = (1, 2, 4, 8, 16, 32, 40, 48, 56, 64, 72, 80)
BAND = (48, 56, 64, 72, 80)
L = 256
N = L * L
SCEN = {0: "Lin-Jiang, NN exclusion", 1: "(4,0) Bukas-Reuter, NN exclusion",
        2: "Lin-Jiang, no exclusion", 3: "(4,0), no exclusion"}

LJ = {(1, 0): 0.0404, (1, 1): 0.0378, (2, 0): 0.4318, (2, 1): 0.2426, (2, 2): 0.0439, (3, 0): 0.0955,
      (3, 1): 0.0662, (3, 2): 0.0138, (4, 0): 0.0095, (4, 1): 0.0112, (5, 0): 0.0017}
BR = {(4, 0): 1.0}
HIST = {0: LJ, 1: BR, 2: LJ, 3: BR}


def images(dx, dy):
    return [(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy), (dy, dx), (-dy, dx), (dy, -dx), (-dy, -dx)]


def g(field, x, y):
    return -1 if ({0: x + y, 1: y, 2: x}[field] % 2) else 1


def analytic_line(hist, field, m):
    """Low-coverage S/rho along the line profile, averaged over [10] and [01]."""
    tot = sum(hist.values())
    q = 2 * math.pi * m / L
    acc = 0.0
    for (dx, dy), w in hist.items():
        ims = images(dx, dy)
        acc += (w / tot) * sum(2 + g(field, ex, ey) * (math.cos(q * ex) + math.cos(q * ey)) for ex, ey in ims) / 8
    return acc


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


def parse(paths):
    spot = defaultdict(dict)
    line = defaultdict(dict)
    ok = 0
    for p in paths:
        for text in open(p):
            head = text.split(" ", 1)[0]
            if head.startswith("M8_TASK_OK"):
                ok += 1
            if head not in ("SPOT", "LINE"):
                continue
            d = dict(KV.findall(text))
            f = int(text.split("field=F")[1][0])
            key = (int(d["scen"]), f, cp_of(float(d["theta"])))
            rep = int(d["rep"])
            rho = int(d["pairs"]) / N
            if head == "SPOT":
                spot[key][rep] = float(d["S1u"]) * 1e-6 / rho
            else:
                line[key][rep] = {m: float(d[f"m{m}u"]) * 1e-6 for m in MS}
    return spot, line, ok


def main(paths):
    spot, line, ok = parse(paths)
    print(f"== {ok}/{len(paths)} tasks OK ==\n")

    print("== C1: no exclusion, coverage <= 0.02, centre per pair vs 4 P_same ==")
    c1 = True
    for scen in (2, 3):
        for f in (0, 1, 2):
            target = analytic_line(HIST[scen], f, 0)   # q = 0 limit = 4 P_same
            for cp in (0.005, 0.01, 0.02):
                vals = list(spot[(scen, f, cp)].values())
                if len(vals) < 5:
                    continue
                m, e = mean(vals), sem(vals)
                z = (m - target) / e
                passed = abs(z) < 3
                if f == 0:
                    c1 = c1 and passed
                print(f"  {SCEN[scen]:<26} F{f} theta={cp:<5} {m:.3f} +- {e:.3f}   exact {target:.3f}   "
                      f"{z:+.1f} SE{'' if f == 0 else '   (F1/F2 reported, rule is on F0)'}")
    print(f"  C1: {'HOLDS' if c1 else 'FAILS'}")

    print("\n== C3: with NN exclusion, (1/2,1/2) centre per pair, Lin-Jiang vs (4,0) ==")
    last_ok = None
    for cp in CPS:
        a = list(spot[(0, 0, cp)].values())
        b = list(spot[(1, 0, cp)].values())
        if len(a) < 5 or len(b) < 5:
            continue
        ma, ea, mb, eb = mean(a), sem(a), mean(b), sem(b)
        z = (mb - ma) / math.hypot(ea, eb)
        if z > 5:
            last_ok = cp
        print(f"  theta={cp:<5}  Lin-Jiang {ma:.3f} +- {ea:.3f}   (4,0) {mb:.3f} +- {eb:.3f}   "
              f"separation {z:.1f} SE {'(>5)' if z > 5 else ''}")
    print(f"  C3: largest coverage with separation > 5 SE: {last_ok}")

    print("\n== C4: R_line = S(m=1) / mean S(m=48..80), jackknife over seeds; analytic at low coverage ==")
    for scen in (0, 1, 2, 3):
        for f in (0, 2):
            exact = analytic_line(HIST[scen], f, 1) / mean([analytic_line(HIST[scen], f, m) for m in BAND])
            cells = []
            for cp in CPS:
                reps = line[(scen, f, cp)]
                if len(reps) < 5:
                    continue
                nums = [r[1] for r in reps.values()]
                dens = [mean([r[m] for m in BAND]) for r in reps.values()]
                rr, ee = jack_ratio(nums, dens)
                cells.append(f"{cp}:{rr:.2f}+-{ee:.2f}")
            print(f"  {SCEN[scen]:<32} F{f}  exact(low cov) {exact:.3f}  |  " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
