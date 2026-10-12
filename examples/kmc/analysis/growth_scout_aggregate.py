#!/usr/bin/env python3
"""Aggregate the M5d scout (examples/kmc/kmc_c2x2_growth.sio, one result file
per Slurm array task).

Two questions, both answered only from independent runs:

1. Is the batch-means SE honest at q=8, L=32 without diffusion? The observed
   standard deviation of the per-run means is set against the mean claimed
   SE at each blocking level. A claimed SE that is too small shows up as a
   ratio well above 1 at small block sizes, falling toward 1 if blocking
   reaches past the correlation time.

2. How does the c(2x2) domain size grow with and without hops? Every error
   bar here comes from seed-to-seed scatter: leave-one-seed-out jackknife on
   the seed-averaged curve, never from within-run statistics.

Usage: growth_scout_aggregate.py result-task0.txt result-task1.txt ...
"""
import math
import re
import sys
from collections import defaultdict

P_INF = 0.025521  # Onsager pair density at q=8 (c2x2_exact_anchors.sio)
KV = re.compile(r"(\w+)=([-\d.]+)")


def parse(paths):
    eq, growth, sentinels = [], defaultdict(dict), {}
    for p in paths:
        for line in open(p):
            if line.startswith("EQ "):
                d = dict(KV.findall(line))
                eq.append((float(d["pairs"]), [float(d[f"se_b{k}"]) for k in range(7)],
                           "checks=ok" in line))
            elif line.startswith("GROWTH "):
                d = dict(KV.findall(line))
                dyn = "hops" if "dyn=hops" in line else "dimer"
                key = (int(d["L"]), dyn)
                growth[key].setdefault(int(d["seed"]), []).append(
                    (float(d["t"]), float(d["pairs"]), float(d["blockM"]),
                     float(d["ellC"]), "SATURATED" in line))
            elif line.startswith("M5D_TASK"):
                sentinels[p] = line.strip()
    return eq, growth, sentinels


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def jackknife(values_by_seed, stat):
    seeds = list(values_by_seed)
    full = stat([values_by_seed[s] for s in seeds])
    loo = [stat([values_by_seed[s] for s in seeds if s != d]) for d in seeds]
    n = len(seeds)
    lm = sum(loo) / n
    return full, math.sqrt((n - 1) / n * sum((x - lm) ** 2 for x in loo))


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def main(paths):
    eq, growth, sentinels = parse(paths)

    print("== task sentinels ==")
    for p in paths:
        print(f"  {p}: {sentinels.get(p, 'MISSING -- task did not finish')}")
    ok_tasks = sum(1 for s in sentinels.values() if s.startswith("M5D_TASK_OK"))
    print(f"  {ok_tasks}/{len(paths)} tasks OK")

    print("\n== 1. batch-means error bar, q=8 L=32 dimer only ==")
    good = [r for r in eq if r[2]]
    if len(good) >= 3:
        means = [r[0] for r in good]
        m, sd = mean_sd(means)
        print(f"  {len(good)} independent runs, mean pairs {m:.6f}, observed SD of run means {sd:.6f}")
        print(f"  (SD itself uncertain by ~{100 / math.sqrt(2 * (len(good) - 1)):.0f}%)")
        print("  merged batches  blocks  mean claimed SE   observed SD / claimed")
        for k in range(7):
            claimed = sum(r[1][k] for r in good) / len(good)
            print(f"  {2 ** k:>13}  {256 // 2 ** k:>6}   {claimed:.6f}          {sd / claimed:.2f}")

    print("\n== 2. domain growth, seed-averaged, jackknife errors over seeds ==")
    for key in sorted(growth):
        L, dyn = key
        runs = growth[key]
        nseed = len(runs)
        times = sorted({t for r in runs.values() for (t, *_rest) in r})
        print(f"\n  L={L} {dyn}: {nseed} seeds")
        print("        t        ellC            excess pairs        1/excess     blockM   seeds")
        usable = []
        for t in times:
            pts = {s: [x for x in r if x[0] == t][0] for s, r in runs.items() if any(x[0] == t for x in r)}
            if len(pts) < max(3, nseed // 2):
                continue
            sat = any(v[4] for v in pts.values())
            ell, ell_e = jackknife({s: v[3] for s, v in pts.items()}, lambda v: sum(v) / len(v))
            exc, exc_e = jackknife({s: v[1] - P_INF for s, v in pts.items()}, lambda v: sum(v) / len(v))
            bm = sum(v[2] for v in pts.values()) / len(pts)
            flag = " SAT" if sat else ""
            inv = 1.0 / exc if exc > 0 else float("nan")
            print(f"  {t:>11.3f}  {ell:7.3f} +- {ell_e:5.3f}   {exc:.5f} +- {exc_e:.5f}   {inv:8.2f}   {bm:.3f}   {len(pts)}{flag}")
            if not sat and len(pts) == nseed:
                usable.append((t, pts))

        win = [(t, pts) for t, pts in usable
               if 3.0 < sum(v[3] for v in pts.values()) / len(pts) < L / 8.0]
        if len(win) >= 3:
            seeds = list(win[0][1])

            def alpha(ss, which):
                xs, ys = [], []
                for t, pts in win:
                    if which == "ellC":
                        y = sum(pts[s][3] for s in ss) / len(ss)
                    else:
                        y = 1.0 / (sum(pts[s][1] - P_INF for s in ss) / len(ss))
                    xs.append(math.log(t))
                    ys.append(math.log(y))
                return slope(xs, ys)

            for which in ("ellC", "1/excess"):
                full = alpha(seeds, which)
                loo = [alpha([s for s in seeds if s != d], which) for d in seeds]
                lm = sum(loo) / len(loo)
                err = math.sqrt((len(seeds) - 1) / len(seeds) * sum((x - lm) ** 2 for x in loo))
                print(f"  effective exponent from {which:8s} over 3 < ellC < L/8 "
                      f"({len(win)} points, t {win[0][0]:.1f}..{win[-1][0]:.1f}): {full:.3f} +- {err:.3f}")
        else:
            print(f"  scaling window 3 < ellC < L/8 has only {len(win)} usable points -- no exponent")


if __name__ == "__main__":
    main(sys.argv[1:])
