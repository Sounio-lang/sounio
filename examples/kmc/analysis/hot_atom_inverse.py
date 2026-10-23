#!/usr/bin/env python3
"""The inverse problem for hot_atom_diffraction.sio: recover pair-separation
statistics from half-order spot profiles alone, and check them against the
truth the simulation knows (PARITY lines).

PRE-REGISTERED, written while array 12090 was running and before any of its
data was read.

Estimator. The exact low-coverage form, image-averaged over lattice
symmetry, expands at small q as
    S_F(q) / rho = 2 + 2 <g_F> - (q^2 / 2) <g_F |D|^2> + O(q^4),
rho = pairs / N (coverage / 2, measurable independently). A least-squares
line in q^2 through the seed-mean S_F/rho of shells n^2 = 1, 2, 4
(q = 2 pi n / L) gives intercept a and slope b:
    P_same_hat = a / 4            (fraction of pairs on one sublattice of F)
    M2_hat     = -2 b             (<g_F |D|^2>, parity-signed second moment)
Errors: jackknife over seeds (the fit is redone leaving each seed out).

Rules.
  I1  at coverage 0.01, no exclusion (scenarios 0-5), in every field, P_same_hat
      agrees with the realised same-sublattice fraction (seed-mean of PARITY
      same/pairs) within 3 combined SE. 18 comparisons; multiplicity-aware
      allowance as elsewhere.
  I2  exploratory: the same comparison at coverage 0.05 and 0.10, and under
      nearest-neighbour exclusion (scenarios 6-8) up to 0.30, reporting the
      bias P_same_hat - P_same_true -- the coverage at which the inverse
      reading stops being trustworthy.
M2 is reported without a rule: its truth would need the full D distribution,
which the PARITY lines do not carry.
"""
import math
import re
import sys
from collections import defaultdict

KV = re.compile(r"(\w+)=([-\d.]+)")
CPS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
L = 256
N = L * L
NAME = {0: "(1,0) lam=.05", 1: "(1,0) lam=.1", 2: "(1,0) lam=.25", 3: "(1,0) lam=.5", 4: "(1,1) lam=.2",
        5: "histogram", 6: "(1,0) lam=.1 excl", 7: "histogram excl", 8: "(1,1) lam=.2 excl"}


def cp_of(theta):
    return min((c for c in CPS if theta >= c - 1e-9), key=lambda c: abs(c - theta))


def mean(x):
    return sum(x) / len(x)


def parse(paths):
    spot = defaultdict(dict)
    parity = defaultdict(dict)
    for p in paths:
        for line in open(p):
            head = line.split(" ", 1)[0]
            if head not in ("SPOT", "PARITY"):
                continue
            d = dict(KV.findall(line))
            f = int(line.split("field=F")[1][0])
            key = (int(d["scen"]), f, cp_of(float(d["theta"])))
            rep = int(d["rep"])
            if head == "SPOT":
                pairs = int(d["pairs"])
                spot[key][rep] = {k: float(d[f"S{k}u"]) * 1e-6 / (pairs / N) for k in (1, 2, 4)}
            else:
                parity[key][rep] = int(d["same"]) / int(d["pairs"])
    return spot, parity


def fit(reps):
    xs = [(2 * math.pi / L) ** 2 * k for k in (1, 2, 4)]
    ys = [mean([r[k] for r in reps]) for k in (1, 2, 4)]
    mx, my = mean(xs), mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return a / 4.0, -2.0 * b


def jack(reps):
    n = len(reps)
    full = fit(reps)
    loo = [fit(reps[:i] + reps[i + 1:]) for i in range(n)]
    out = []
    for j in range(2):
        lm = mean([x[j] for x in loo])
        out.append((full[j], math.sqrt((n - 1) / n * sum((x[j] - lm) ** 2 for x in loo))))
    return out


def compare(spot, parity, scen, f, cp):
    key = (scen, f, cp)
    reps = sorted(set(spot[key]) & set(parity[key]))
    if len(reps) < 5:
        return None
    (p_hat, pe), (m2, m2e) = jack([spot[key][r] for r in reps])
    truth = [parity[key][r] for r in reps]
    t = mean(truth)
    te = math.sqrt(sum((v - t) ** 2 for v in truth) / (len(truth) - 1) / len(truth))
    z = (p_hat - t) / math.hypot(pe, te)
    return p_hat, pe, t, te, z, m2, m2e


def main(paths):
    spot, parity = parse(paths)
    print("== I1: inverse P_same from spot profiles vs realised truth, coverage 0.01, no exclusion ==")
    n, out, worst = 0, 0, 0.0
    for scen in range(6):
        for f in (0, 1, 2):
            c = compare(spot, parity, scen, f, 0.01)
            if c is None:
                continue
            p_hat, pe, t, te, z, m2, m2e = c
            n += 1
            out += abs(z) > 3
            worst = max(worst, abs(z))
            print(f"  {NAME[scen]:<14} F{f}: P_hat {p_hat:.4f} +- {pe:.4f}   truth {t:.4f} +- {te:.4f}   "
                  f"{z:+.1f} SE   M2_hat {m2:+.2f} +- {m2e:.2f}")
    lam = 0.0047 * n
    cum, allow = 0.0, 0
    while True:
        cum += math.exp(-lam) * lam ** allow / math.factorial(allow)
        if cum >= 0.99:
            break
        allow += 1
    print(f"  {n} comparisons, {out} beyond 3 SE (worst {worst:.1f}); 99% allowance {allow}")
    print(f"  I1: {'HOLDS' if n and out <= allow else 'FAILS'}")

    print("\n== I2 (exploratory): bias P_hat - truth vs coverage ==")
    for scen, fields, cps in ((1, (0,), (0.01, 0.05, 0.10)), (5, (0, 2), (0.01, 0.05, 0.10)),
                              (6, (0,), CPS), (7, (0, 2), CPS), (8, (1, 2), CPS)):
        for f in fields:
            cells = []
            for cp in cps:
                c = compare(spot, parity, scen, f, cp)
                if c is None:
                    continue
                p_hat, pe, t, te, z, *_ = c
                cells.append(f"{cp:.2f}:{p_hat - t:+.3f}({z:+.1f})")
            print(f"  {NAME[scen]:<18} F{f}: " + "  ".join(cells))
    print("  (entries: coverage: bias (bias in combined SE))")


if __name__ == "__main__":
    main(sys.argv[1:])
