#!/usr/bin/env python3
"""Apply decision rules R1-R4 of examples/kmc/kmc_c2x2_production.sio to the
result files of its Slurm array. The rules are copied from that file's
header, which was fixed before the run; this script does not tune them.

Every error bar is from independent seeds (jackknife, or SD/sqrt(n) for plain
means). No within-run error estimate is used anywhere.

Usage: production_aggregate.py result-task0.txt ... result-task15.txt
"""
import math
import re
import sys
from collections import defaultdict

KV = re.compile(r"(\w+)=([-\d.]+)")
SQRT2 = math.sqrt(2.0)
CFG = {
    0: "L=128 D=0", 1: "L=128 D=1e-3", 2: "L=128 D=1e-2", 3: "L=128 D=1",
    4: "L=64 D=0", 5: "L=64 D=1",
    6: "L=32 D=0 clean", 7: "L=32 D=0 slab", 8: "L=16 D=0 clean", 9: "L=16 D=0 slab",
    10: "L=32 D=1 ref", 11: "L=16 D=1 ref",
}
CFG_L = {0: 128, 1: 128, 2: 128, 3: 128, 4: 64, 5: 64, 6: 32, 7: 32, 8: 16, 9: 16, 10: 32, 11: 16}


def parse(paths):
    seed_cfg, runs, sentinels, run_ok = {}, defaultdict(dict), {}, {}
    for p in paths:
        for line in open(p):
            if line.startswith("RUN_BEGIN"):
                d = dict(KV.findall(line))
                seed_cfg[int(d["seed"])] = (int(d["cfg"]), int(d["rep"]))
            elif line.startswith("RUN_END"):
                d = dict(KV.findall(line))
                run_ok[(int(d["cfg"]), int(d["rep"]))] = "checks=ok" in line
            elif line.startswith("GROWTH "):
                d = dict(KV.findall(line))
                cfg, rep = seed_cfg[int(d["seed"])]
                runs[cfg].setdefault(rep, []).append({
                    "t": float(d["t"]), "pairs": float(d["pairs"]), "win": float(d["winm"]) / 1000.0,
                    "ell": float(d["ellC"]), "S1": float(d["S1m"]), "S4": float(d["S4m"]),
                    "sat": "SATURATED" in line})
            elif line.startswith("M5E_TASK"):
                sentinels[p] = line.strip()
    return runs, sentinels, run_ok


def mean(xs):
    return sum(xs) / len(xs)


def se_mean(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def jackknife(items, stat):
    full = stat(items)
    n = len(items)
    loo = [stat(items[:i] + items[i + 1:]) for i in range(n)]
    lm = mean(loo)
    return full, math.sqrt((n - 1) / n * sum((x - lm) ** 2 for x in loo))


def at_time(run, t):
    for s in run:
        if abs(s["t"] - t) < 1e-9 * max(1.0, t):
            return s
    return None


def common_times(cfg_runs):
    sets = [set(round(s["t"], 6) for s in r) for r in cfg_runs.values()]
    return sorted(set.intersection(*sets)) if sets else []


def fmt(x, e):
    return f"{x:.4f} +- {e:.4f}"


def r1(runs, cfg, L):
    reps = sorted(runs[cfg])
    rs = [runs[cfg][k] for k in reps]
    usable = []
    for t in common_times(runs[cfg]):
        pts = [at_time(r, t) for r in rs]
        if any(p["sat"] for p in pts):
            continue
        if 4.0 < mean([p["ell"] for p in pts]) < L / 8.0:
            usable.append(t)
    if len(usable) < 2:
        return None, None, usable
    per_seed = []
    for r in rs:
        vals = [math.log2(at_time(r, t)["S4"] / at_time(r, t)["S1"]) for t in usable
                if at_time(r, t)["S1"] > 0 and at_time(r, t)["S4"] > 0]
        per_seed.append(mean(vals))
    s, e = jackknife(per_seed, mean)
    return s, e, usable


def alpha(runs, cfg, L):
    reps = sorted(runs[cfg])
    rs = [runs[cfg][k] for k in reps]
    win = []
    for t in common_times(runs[cfg]):
        pts = [at_time(r, t) for r in rs]
        if any(p["sat"] for p in pts):
            continue
        if 3.0 < mean([p["ell"] for p in pts]) < L / 8.0:
            win.append(t)
    if len(win) < 3:
        return None, None, win

    def fit(subset):
        xs = [math.log(t) for t in win]
        ys = [math.log(mean([at_time(r, t)["ell"] for r in subset])) for t in win]
        mx, my = mean(xs), mean(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

    a, e = jackknife(rs, fit)
    return a, e, win


def windows(run):
    """Time-weighted pair density over (T/4, T/2] and (T/2, T], T = last sample."""
    T = run[-1]["t"]
    acc = {"early": [0.0, 0.0], "late": [0.0, 0.0]}
    for s in run:
        dt = s["t"] * (1.0 - 1.0 / SQRT2)
        if T / 4.0 < s["t"] <= T / 2.0 * (1 + 1e-9):
            acc["early"][0] += s["win"] * dt
            acc["early"][1] += dt
        elif T / 2.0 < s["t"] <= T * (1 + 1e-9):
            acc["late"][0] += s["win"] * dt
            acc["late"][1] += dt
    return acc["early"][0] / acc["early"][1], acc["late"][0] / acc["late"][1], T


def main(paths):
    runs, sentinels, run_ok = parse(paths)
    print("== sentinels ==")
    ok = sum(1 for s in sentinels.values() if s.startswith("M5E_TASK_OK"))
    print(f"  {ok}/{len(paths)} tasks OK; runs with engine checks ok: "
          f"{sum(run_ok.values())}/{len(run_ok)}")
    for p in paths:
        if not sentinels.get(p, "").startswith("M5E_TASK_OK"):
            print(f"  NOT OK: {p}: {sentinels.get(p, 'missing')}")
    for cfg in sorted(runs):
        print(f"  cfg {cfg:>2} {CFG[cfg]:<16} seeds={len(runs[cfg])}")

    verdicts = {}

    print("\n== R1: small-k slope s = log2(S[shell 4]/S[shell 1]), L=128, samples with 4 < ellC < L/8 ==")
    s0, e0, u0 = r1(runs, 0, 128)
    s3, e3, u3 = r1(runs, 3, 128)
    if s0 is None or s3 is None:
        print("  not enough samples in the window -- R1 UNDECIDED")
        verdicts["R1"] = "UNDECIDED"
    else:
        print(f"  D=0: s = {fmt(s0, e0)} over {len(u0)} samples (t {u0[0]:.0f}..{u0[-1]:.0f}); need s-1 > 3SE: "
              f"{(s0 - 1) / e0:.1f} SE")
        print(f"  D=1: s = {fmt(s3, e3)} over {len(u3)} samples (t {u3[0]:.1f}..{u3[-1]:.1f}); need -s > 3SE: "
              f"{-s3 / e3:.1f} SE")
        verdicts["R1"] = "SUPPORTED" if (s0 - 1 > 3 * e0 and -s3 > 3 * e3) else "NOT SUPPORTED"
    print(f"  R1: {verdicts['R1']}")

    print("\n== R2: growth exponents over 3 < ellC < L/8 ==")
    passes = []
    for L, c0, c1 in ((128, 0, 3), (64, 4, 5)):
        a0, ea0, w0 = alpha(runs, c0, L)
        a1, ea1, w1 = alpha(runs, c1, L)
        if a0 is None or a1 is None:
            print(f"  L={L}: window too short (D=0: {len(w0)} points, D=1: {len(w1)}) -- undecided")
            passes.append(None)
            continue
        z = abs(a1 - a0) / math.sqrt(ea0 ** 2 + ea1 ** 2)
        print(f"  L={L}: D=0 alpha = {fmt(a0, ea0)} ({len(w0)} pts)   D=1 alpha = {fmt(a1, ea1)} ({len(w1)} pts)   "
              f"difference {z:.1f} SE")
        passes.append(z > 3)
    verdicts["R2"] = "UNDECIDED" if None in passes else ("SUPPORTED" if all(passes) else "NOT SUPPORTED")
    print(f"  R2: {verdicts['R2']}")

    print("\n== R3: crossover ordering at the latest common time, L=128 ==")
    tc = sorted(set(common_times(runs[0])) & set(common_times(runs[1])) & set(common_times(runs[2])))
    if not tc:
        verdicts["R3"] = "UNDECIDED"
    else:
        t = tc[-1]
        ells = {}
        for c in (0, 1, 2):
            vals = [at_time(r, t)["ell"] for r in runs[c].values()]
            ells[c] = (mean(vals), se_mean(vals), any(at_time(r, t)["sat"] for r in runs[c].values()))
        for c in (0, 1, 2):
            print(f"  t={t:.0f}  {CFG[c]:<14} ellC = {fmt(ells[c][0], ells[c][1])}{'  (some saturated)' if ells[c][2] else ''}")
        g21 = (ells[2][0] - ells[1][0]) / math.hypot(ells[2][1], ells[1][1])
        g10 = (ells[1][0] - ells[0][0]) / math.hypot(ells[1][1], ells[0][1])
        print(f"  gap D=1e-2 over D=1e-3: {g21:.1f} SE    gap D=1e-3 over D=0: {g10:.1f} SE")
        verdicts["R3"] = "SUPPORTED" if (g21 > 3 and g10 > 3) else "NOT SUPPORTED"
    print(f"  R3: {verdicts['R3']}")

    print("\n== R4: equilibrium at L=16, 32 (D=0), windows (T/4,T/2] and (T/2,T] ==")
    r4 = []
    excess = {}
    for L, cc, cs, cr in ((32, 6, 7, 10), (16, 8, 9, 11)):
        res = {}
        for name, c in (("clean", cc), ("slab", cs), ("ref", cr)):
            ws = [windows(r) for r in runs[c].values()]
            res[name] = ws
        drift = [e - l for e, l, _ in res["clean"]] + [e - l for e, l, _ in res["slab"]]
        dm, ds = mean(drift), se_mean(drift)
        lc = [l for _, l, _ in res["clean"]]
        ls = [l for _, l, _ in res["slab"]]
        lr = [l for _, l, _ in res["ref"]]
        diff = mean(lc) - mean(ls)
        dse = math.hypot(se_mean(lc), se_mean(ls))
        T = res["clean"][0][2]
        informative = ds < 0.001 and dse < 0.001
        no_drift = abs(dm) < 3 * ds
        agree = abs(diff) < 3 * dse
        print(f"  L={L} (T={T:.0f}, {len(lc)}+{len(ls)} seeds): drift early-late = {fmt(dm, ds)} ({abs(dm) / ds:.1f} SE)   "
              f"clean-slab (late) = {fmt(diff, dse)} ({abs(diff) / dse:.1f} SE)")
        print(f"        late pairs: clean {fmt(mean(lc), se_mean(lc))}  slab {fmt(mean(ls), se_mean(ls))}  "
              f"hops ref {fmt(mean(lr), se_mean(lr))}")
        if not informative:
            v = "UNINFORMATIVE (an SE >= 0.001)"
        elif no_drift and agree:
            v = "equilibrium NOT EXCLUDED"
        else:
            v = "NOT EQUILIBRATED"
        print(f"        {v}")
        r4.append(v)
        pooled = lc + ls
        excess[L] = (mean(pooled) - mean(lr), math.hypot(se_mean(pooled), se_mean(lr)))
    verdicts["R4"] = "; ".join(f"L={L}: {v}" for L, v in zip((32, 16), r4))
    e32, e16 = excess[32], excess[16]
    ratio = e32[0] / e16[0]
    rse = abs(ratio) * math.hypot(e32[1] / e32[0], e16[1] / e16[0])
    print(f"  excess over hops ref: L=32 {fmt(*e32)}  L=16 {fmt(*e16)}  ratio {fmt(ratio, rse)} (1/L would be 0.5)")
    print(f"  R4: {verdicts['R4']}")

    print("\n== verdicts ==")
    for k in ("R1", "R2", "R3", "R4"):
        print(f"  {k}: {verdicts[k]}")

    exploratory_ring(runs)


def exploratory_ring(runs):
    """EXPLORATORY, NOT A DECISION RULE. Written while array 12050 was running,
    after its validation lines but before any GROWTH data had been looked at.

    Once verify_kawasaki_mapping.py showed the D=0 dynamics is Kawasaki Ising
    coarsening, the part that might be new is its diffraction face: a
    centre-suppressed ("ring") c(2x2) half-order spot, s = log2(S4/S1) > 0, and
    how long a hop rate D lets it survive. Two scaling arguments, recorded
    before the data:
      gamma = 3: wall velocities dl/dt = G_B/l^2 + G_A(D)/l with G_A ~ D give
                 l_x ~ 1/D and t_x ~ l_x^3 ~ D^-3;
      gamma = 1: the spot centre fills directly by non-conserving moves at a
                 rate ~ D, t_x ~ 1/D.
    Definitions: t_ring(D) = first common sample time with s_D < 0 by more than
    2 SE while s_0 > 1 at the same time; t_ell(D) = first common time with
    ellC_D - ellC_0 > 3 combined SE. gamma_est = log10(t(1e-3) / t(1e-2)).
    """
    print("\n== EXPLORATORY: ring survival vs hop rate (not a pre-registered rule) ==")
    need = (0, 1, 2)
    if any(c not in runs for c in need):
        print("  missing configurations")
        return
    times = sorted(set.intersection(*[set(common_times(runs[c])) for c in need]))
    series = {}
    for c in (0, 1, 2):
        rs = list(runs[c].values())
        series[c] = []
        for t in times:
            pts = [at_time(r, t) for r in rs]
            sv = [math.log2(p["S4"] / p["S1"]) for p in pts if p["S1"] > 0 and p["S4"] > 0]
            ev = [p["ell"] for p in pts]
            series[c].append((t, mean(sv), se_mean(sv), mean(ev), se_mean(ev)))
    print("        t      s(D=0)            s(D=1e-3)          s(D=1e-2)          ell D=0   ell 1e-3  ell 1e-2")
    for k, t in enumerate(times):
        if t < 1.0:
            continue
        a, b, c = series[0][k], series[1][k], series[2][k]
        print(f"  {t:>9.1f}  {a[1]:+.3f} +- {a[2]:.3f}   {b[1]:+.3f} +- {b[2]:.3f}   {c[1]:+.3f} +- {c[2]:.3f}"
              f"   {a[3]:7.3f}  {b[3]:7.3f}  {c[3]:7.3f}")
    tx = {}
    for c, dname in ((1, "1e-3"), (2, "1e-2")):
        t_ring = t_ell = None
        for k, t in enumerate(times):
            s0, sd = series[0][k], series[c][k]
            if t_ring is None and s0[1] > 1 and sd[1] < 0 and -sd[1] > 2 * sd[2]:
                t_ring = t
            if t_ell is None and sd[3] - s0[3] > 3 * math.hypot(sd[4], s0[4]):
                t_ell = t
        tx[c] = (t_ring, t_ell)
        print(f"  D={dname}: t_ring = {t_ring}   t_ell = {t_ell}")
    for idx, name in ((0, "t_ring"), (1, "t_ell")):
        a, b = tx[1][idx], tx[2][idx]
        if a and b:
            print(f"  gamma from {name}: {math.log10(a / b):.2f}   (candidates 1 and 3; the sqrt(2) time grid "
                  f"limits resolution to about +-0.15)")
        else:
            print(f"  gamma from {name}: not measurable (crossover not reached for both D in the window)")


if __name__ == "__main__":
    main(sys.argv[1:])
