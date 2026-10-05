#!/usr/bin/env python3
"""Evaluate the blind excluded-volume predictions against hot_atom_blind.sio.

Predictions are READ from blind_predictions_excluded_volume.txt as committed
in d419730f9c (md5 checked below), never recomputed here, so nothing about
the simulation can leak into them. Written before the blind array's data
existed.

BJ1: for every (scenario, field) marked in_BJ1=True, the predicted
     SPOT/PRED of shell n^2=1 at coverage 0.05 and 0.10 agrees with the
     simulated jackknife ratio within 3 SE; multiplicity-aware (Poisson 99%
     allowance at the t-tail rate 0.0047).
BJ2: for the same comparisons, whether |prediction - 1| > 3 SE: a prediction
     that the no-correction null could not be told apart from is reported as
     uninformative, not as a success.
Also reported: coverage 0.02, 0.15, 0.20, and the null's own z-scores.

Usage: blind_evaluate.py PREDICTIONS_FILE result-task0.txt ...
"""
import hashlib
import math
import re
import sys
from collections import defaultdict

EXPECTED_MD5 = "95e8276d3b10a6b2677626f851a6e691"
KV = re.compile(r"(\w+)=([-\d.]+)")
CPS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
SCEN = {"B0 (3,0)": 10, "B1 (2,1)": 11, "B2 (2,2)": 12, "B3 mix (1,1).5 (2,0).3 (3,1).2": 13}


def cp_of(theta):
    return min((c for c in CPS if theta >= c - 1e-9), key=lambda c: abs(c - theta))


def mean(x):
    return sum(x) / len(x)


def load_predictions(path):
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if md5 != EXPECTED_MD5:
        raise SystemExit(f"prediction file md5 {md5} != committed {EXPECTED_MD5}: refusing")
    preds = {}
    for line in open(path):
        if not line.startswith("PREDICT"):
            continue
        parts = [p.strip() for p in line.split("|")]
        name = parts[0][len("PREDICT "):]
        field = int(parts[1][1])
        use = parts[3].endswith("True")
        vals = {float(a): float(b) for a, b in re.findall(r"theta=([\d.]+): ([\d.]+)", parts[4])}
        preds[(SCEN[name], field)] = (name, use, vals)
    return preds


def main(argv):
    preds = load_predictions(argv[0])
    spot = defaultdict(dict)
    sentinels = 0
    for p in argv[1:]:
        for line in open(p):
            head = line.split(" ", 1)[0]
            if head.startswith("M7B_TASK_OK"):
                sentinels += 1
            if head not in ("SPOT", "PRED"):
                continue
            d = dict(KV.findall(line))
            f = int(line.split("field=F")[1][0])
            spot[(head, int(d["scen"]), f, cp_of(float(d["theta"])))][int(d["rep"])] = float(d["S1u"])
    print(f"prediction file md5 verified; {sentinels}/{len(argv) - 1} tasks OK\n")

    def sim_ratio(scen, f, cp):
        S, P = spot[("SPOT", scen, f, cp)], spot[("PRED", scen, f, cp)]
        reps = sorted(set(S) & set(P))
        if len(reps) < 5:
            return None
        nums, dens = [S[r] for r in reps], [P[r] for r in reps]
        n = len(reps)
        full = mean(nums) / mean(dens)
        loo = [(sum(nums) - nums[i]) / (sum(dens) - dens[i]) for i in range(n)]
        lm = mean(loo)
        return full, math.sqrt((n - 1) / n * sum((v - lm) ** 2 for v in loo))

    n_cmp = n_out = n_sharp = 0
    for (scen, f), (name, use, vals) in sorted(preds.items()):
        cells = []
        for cp in (0.02, 0.05, 0.10, 0.15, 0.20):
            r = sim_ratio(scen, f, cp)
            if r is None:
                continue
            rs, es = r
            pr = vals[cp]
            z = (rs - pr) / es
            znull = (rs - 1.0) / es
            mark = ""
            if use and cp in (0.05, 0.10):
                n_cmp += 1
                n_out += abs(z) > 3
                sharp = abs(pr - 1.0) > 3 * es
                n_sharp += sharp
                mark = "[BJ1" + (",sharp]" if sharp else ",not sharp]")
            cells.append(f"{cp:.2f}: sim {rs:.3f}+-{es:.3f} pred {pr:.3f} ({z:+.1f}; null {znull:+.1f}){mark}")
        print(f"{name} F{f} {'(in BJ1)' if use else '(dipole centre, excluded)'}")
        for c in cells:
            print("    " + c)
    lam = 0.0047 * n_cmp
    cum, allow = 0.0, 0
    while True:
        cum += math.exp(-lam) * lam ** allow / math.factorial(allow)
        if cum >= 0.99:
            break
        allow += 1
    print(f"\nBJ1: {n_cmp} comparisons, {n_out} beyond 3 SE, allowance {allow} -> "
          f"{'HOLDS' if n_cmp and n_out <= allow else 'FAILS'}")
    print(f"BJ2: {n_sharp} of {n_cmp} comparisons sharp against the no-correction null")


if __name__ == "__main__":
    main(sys.argv[1:])
