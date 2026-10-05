#!/usr/bin/env python3
"""C4 aggregation + predeclared gate evaluation (task_freeze.md, 2026-08-09).

Reads results/seedXX_L*.json (+ sha256 sidecars, verified before use),
aggregates over the 20 paired seeds, and applies the six promotion criteria
literally:

  1. OctTree-8 − CliffTree-8 > 0, paired 95% CI excl. 0   (Task B test)
  2. OctTree-8 − RealTree-8 > 0, paired 95% CI excl. 0    (Task B test)
  3. OctTree-8 >= LearnedBilinTree − 2 pp                  (Task B test)
  4. direction replicates on Task A test (no CI requirement)
  5. CountBaseline <= 55% on Task B (task validity)
  6. NEG arm: all models within chance CI (pipeline soundness)

Cross-L aggregation (amendment 2026-08-09, pre-results): L=128 is the primary
endpoint (exploratory precedent); other L are robustness. Promotion requires
gates 1–3 at L=128 and sign consistency of (1)-(2) across the L grid.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

MODELS = ["CountBaseline", "RealTree-8", "CliffTree-8", "LearnedBilinTree",
          "OctTree-8", "GRU-8"]
ARMS = ["A", "B", "NEG"]
LS = [64, 128, 256, 512]
N_SEEDS = 20
CHANCE = 0.5
EPS_LEARNED = 0.02
COUNT_GATE = 0.55
T95_19 = 2.093  # two-sided 95% CI, df=19


def ci_mean(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, T95_19 * math.sqrt(var / n)


def load_results(resdir):
    cells = {}  # (seed_idx, L) -> dict
    for p in sorted(resdir.glob("seed*_L*.json")):
        side = p.with_suffix(".sha256")
        if not side.exists():
            print(f"SKIP {p.name}: missing sha256 sidecar", file=sys.stderr)
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        recorded = side.read_text().split()[0].strip()
        if digest != recorded:
            print(f"SKIP {p.name}: sha256 MISMATCH", file=sys.stderr)
            continue
        d = json.loads(p.read_text())
        cells[(d["seed_idx"], d["L"])] = d
    return cells


def paired_diff(cells, L, arm, m1, m2, key="test_acc_final"):
    diffs = []
    for i in range(N_SEEDS):
        c = cells.get((i, L))
        if c is None:
            continue
        a = c["arms"][arm][m1][key]
        b = c["arms"][arm][m2][key]
        diffs.append(a - b)
    return diffs


def series(cells, L, arm, model, key="test_acc_final"):
    return [cells[(i, L)]["arms"][arm][model][key]
            for i in range(N_SEEDS) if (i, L) in cells]


def main(resdir):
    resdir = Path(resdir)
    cells = load_results(resdir)
    have = sorted(cells)
    print(f"cells loaded: {len(have)} / {N_SEEDS * len(LS)}")
    missing = [(i, L) for i in range(N_SEEDS) for L in LS if (i, L) not in cells]
    if missing:
        print(f"missing: {missing}")

    # ---- descriptive tables --------------------------------------------
    for arm in ARMS:
        for L in LS:
            row = []
            for m in MODELS:
                xs = series(cells, L, arm, m)
                if not xs:
                    row.append(f"{m:16s}  —")
                    continue
                mean, half = ci_mean(xs)
                row.append(f"{m:16s} {mean:.3f} ± {half:.3f}")
            print(f"[{arm:3s} L={L:4d}] " + " | ".join(row))

    # ---- gates, per L ----------------------------------------------------
    verdicts = {}
    for L in LS:
        n = len(series(cells, L, "B", "OctTree-8"))
        if n == 0:
            continue
        g = {}
        d1 = paired_diff(cells, L, "B", "OctTree-8", "CliffTree-8")
        m1, h1 = ci_mean(d1)
        g["G1_oct_minus_cliff"] = (m1, h1, m1 - h1 > 0)
        d2 = paired_diff(cells, L, "B", "OctTree-8", "RealTree-8")
        m2, h2 = ci_mean(d2)
        g["G2_oct_minus_real"] = (m2, h2, m2 - h2 > 0)
        d3 = paired_diff(cells, L, "B", "OctTree-8", "LearnedBilinTree")
        m3, h3 = ci_mean(d3)
        g["G3_oct_minus_learned"] = (m3, h3, m3 >= -EPS_LEARNED)
        d4 = paired_diff(cells, L, "A", "OctTree-8", "CliffTree-8")
        m4, _ = ci_mean(d4)
        d4r = paired_diff(cells, L, "A", "OctTree-8", "RealTree-8")
        m4r, _ = ci_mean(d4r)
        g["G4_taskA_direction"] = (m4, m4r, m4 > 0 and m4r > 0)
        cb = series(cells, L, "B", "CountBaseline")
        mcb, _ = ci_mean(cb)
        g["G5_countbaseline"] = (mcb, mcb <= COUNT_GATE)
        chance_half = 1.96 * math.sqrt(0.25 / 4096)  # frozen test n
        negok = True
        neg_detail = {}
        for m in MODELS:
            xs = series(cells, L, "NEG", m)
            if not xs:
                continue
            mm, _ = ci_mean(xs)
            neg_detail[m] = mm
            if abs(mm - CHANCE) > chance_half:
                negok = False
        g["G6_neg_chance"] = (neg_detail, negok)
        verdicts[L] = g
        print(f"\n== gates L={L} (n={n} seeds) ==")
        print(f"  G1 Oct−Cliff  {m1:+.4f} ± {h1:.4f}  pass={g['G1_oct_minus_cliff'][2]}")
        print(f"  G2 Oct−Real   {m2:+.4f} ± {h2:.4f}  pass={g['G2_oct_minus_real'][2]}")
        print(f"  G3 Oct−Learn  {m3:+.4f} ± {h3:.4f}  pass={g['G3_oct_minus_learned'][2]}  (>= −2pp)")
        print(f"  G4 A-direct.  cliff {m4:+.4f}, real {m4r:+.4f}  pass={g['G4_taskA_direction'][2]}")
        print(f"  G5 CountBase  {mcb:.4f}  pass={g['G5_countbaseline'][1]}")
        print(f"  G6 NEG chance pass={negok}  detail="
              + ", ".join(f"{k}={v:.3f}" for k, v in neg_detail.items()))

    # ---- promotion verdict (primary L=128) -------------------------------
    out = {"cells": len(have), "verdicts": {
        str(L): {k: (v if not isinstance(v, tuple) else
                     [x if not isinstance(x, dict) else x for x in v])
                 for k, v in g.items()} for L, g in verdicts.items()}}
    if 128 in verdicts:
        g = verdicts[128]
        promo = (g["G1_oct_minus_cliff"][2] and g["G2_oct_minus_real"][2]
                 and g["G3_oct_minus_learned"][2] and g["G4_taskA_direction"][2]
                 and g["G5_countbaseline"][1] and g["G6_neg_chance"][1])
        sign_ok = all(
            verdicts[L]["G1_oct_minus_cliff"][0] > 0
            and verdicts[L]["G2_oct_minus_real"][0] > 0
            for L in verdicts)
        out["promotion"] = {
            "primary_L128_gates_pass": promo,
            "sign_consistency_all_L": sign_ok,
            "claim_promoted": promo and sign_ok,
        }
        print(f"\nPROMOTION (primary L=128): gates={promo} "
              f"sign-consistency={sign_ok} "
              f"=> claim_promoted={promo and sign_ok}")
    (resdir / "aggregate_c4.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {resdir / 'aggregate_c4.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
