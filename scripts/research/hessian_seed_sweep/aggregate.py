#!/usr/bin/env python3
"""
Parse the stdout files produced by run_sweep.py and compute:

- Seed robustness (P1.3): distribution of cohort-level Ictal/Interictal/Baseline
  mean ||H_offdiag|| across seeds (median, IQR, 95% percentile CI,
  sign-direction consistency for Ictal vs Baseline).
- LOO generalization (P1.4): per-fold per-seed test_acc, pooled across seeds
  as median + IQR per held-out patient.

Outputs:
  artifacts/research/hessian_seed_sweep/seed_cohort.tsv      cohort stats per seed
  artifacts/research/hessian_seed_sweep/seed_per_fold.tsv    per-fold per-seed values
  artifacts/research/hessian_seed_sweep/summary.json         aggregated summary
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics


COHORT_RE = re.compile(
    r"seed_offset=(?P<seed>-?\d+).*?"
    r"Ictal:\s+(?P<ictal>\d+).*?"
    r"Interictal:\s+(?P<inter>\d+).*?"
    r"Baseline:\s+(?P<base>\d+).*?"
    r"Ictal / Baseline:\s+(?P<r_ib>\d+).*?"
    r"Interictal / Baseline:\s+(?P<r_nb>\d+).*?"
    r"Ictal / Interictal:\s+(?P<r_in>\d+)",
    re.DOTALL,
)

FOLD_RE = re.compile(
    r"Fold\s+(?P<fold>\d+)\s*/\s*(?P<n>\d+)\s*\.\.\..*?"
    r"train_acc=\s*(?P<tr>\d+)\s*%.*?"
    r"test_acc=\s*(?P<te>\d+)\s*%.*?"
    r"hessians=\s*(?P<h>\d+)",
    re.DOTALL,
)


def parse_seed_file(text: str) -> dict | None:
    m = COHORT_RE.search(text)
    if not m:
        return None
    folds = [
        {
            "fold": int(f["fold"]),
            "train_acc": int(f["tr"]),
            "test_acc": int(f["te"]),
            "hessians": int(f["h"]),
        }
        for f in FOLD_RE.finditer(text)
    ]
    return {
        "seed": int(m["seed"]),
        "ictal_od": int(m["ictal"]),
        "inter_od": int(m["inter"]),
        "base_od": int(m["base"]),
        "r_ictal_base_x1000": int(m["r_ib"]),
        "r_inter_base_x1000": int(m["r_nb"]),
        "r_ictal_inter_x1000": int(m["r_in"]),
        "folds": folds,
    }


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    idx = max(0, min(len(xs_sorted) - 1, int(round(q * (len(xs_sorted) - 1)))))
    return xs_sorted[idx]


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "iqr_lo": pct(vals, 0.25),
        "iqr_hi": pct(vals, 0.75),
        "ci95_lo": pct(vals, 0.025),
        "ci95_hi": pct(vals, 0.975),
        "min": min(vals),
        "max": max(vals),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default="artifacts/research/hessian_seed_sweep")
    args = p.parse_args()
    root = pathlib.Path(args.sweep_dir)
    stdout_files = sorted(root.glob("hess_seed*.stdout"))
    if not stdout_files:
        print(f"no stdout files under {root}")
        return 1

    parsed = []
    for f in stdout_files:
        rec = parse_seed_file(f.read_text())
        if rec is None:
            print(f"WARN unparseable: {f}")
            continue
        parsed.append(rec)
    parsed.sort(key=lambda r: r["seed"])

    cohort_tsv = root / "seed_cohort.tsv"
    with cohort_tsv.open("w") as f:
        f.write("seed\tictal_od\tinter_od\tbase_od\tr_IB_x1000\tr_NB_x1000\tr_IN_x1000\n")
        for r in parsed:
            f.write(
                f"{r['seed']}\t{r['ictal_od']}\t{r['inter_od']}\t{r['base_od']}"
                f"\t{r['r_ictal_base_x1000']}\t{r['r_inter_base_x1000']}"
                f"\t{r['r_ictal_inter_x1000']}\n"
            )

    fold_tsv = root / "seed_per_fold.tsv"
    with fold_tsv.open("w") as f:
        f.write("seed\tfold\ttrain_acc\ttest_acc\thessians\n")
        for r in parsed:
            for fo in r["folds"]:
                f.write(
                    f"{r['seed']}\t{fo['fold']}\t{fo['train_acc']}"
                    f"\t{fo['test_acc']}\t{fo['hessians']}\n"
                )

    ratios = [r["r_ictal_base_x1000"] / 1000.0 for r in parsed]
    flips = sum(1 for x in ratios if x < 1.0)
    total = len(ratios)
    by_fold: dict[int, list[int]] = {}
    for r in parsed:
        for fo in r["folds"]:
            by_fold.setdefault(fo["fold"], []).append(fo["test_acc"])

    # Patient-name mapping inferred from seizure_hessian_manifest.tsv load order
    # (discover_sites() enumerates in order of first appearance).
    PATIENT_BY_FOLD = {
        0: "chb01", 1: "chb02", 2: "chb03", 3: "chb04", 4: "chb05",
        5: "chb06", 6: "chb09", 7: "chb10", 8: "chb11",
    }
    # chb04 and chb09 are baseline-only in the manifest (no ictal windows);
    # binary "ictal vs non-ictal" LOO test_acc on those folds is uninformative
    # (always predict non-ictal → acc depends on model bias).
    BASELINE_ONLY_FOLDS = {3, 6}

    pooled_loo = []
    pooled_loo_informative = []
    for r in parsed:
        accs = [fo["test_acc"] for fo in r["folds"]]
        if accs:
            pooled_loo.append(statistics.fmean(accs))
        accs_inf = [
            fo["test_acc"]
            for fo in r["folds"]
            if fo["fold"] not in BASELINE_ONLY_FOLDS
        ]
        if accs_inf:
            pooled_loo_informative.append(statistics.fmean(accs_inf))

    summary = {
        "n_seeds": total,
        "ratio_ictal_over_baseline": summarize(ratios),
        "direction_ictal_gt_baseline_count": total - flips,
        "direction_ictal_lt_baseline_count": flips,
        "ictal_od": summarize([r["ictal_od"] for r in parsed]),
        "inter_od": summarize([r["inter_od"] for r in parsed]),
        "base_od": summarize([r["base_od"] for r in parsed]),
        "per_fold_test_acc": {
            PATIENT_BY_FOLD.get(k, f"fold{k}"): summarize([float(v) for v in vs])
            for k, vs in sorted(by_fold.items())
        },
        "pooled_loo_acc_all_folds": summarize(pooled_loo),
        "pooled_loo_acc_ictal_patients_only": summarize(pooled_loo_informative),
        "baseline_only_folds_excluded_from_informative": sorted(
            PATIENT_BY_FOLD[f] for f in BASELINE_ONLY_FOLDS
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"parsed {total} seed runs")
    print(f"direction Ictal > Baseline: {total - flips}/{total} seeds")
    r = summary["ratio_ictal_over_baseline"]
    print(
        f"ratio_IB: median={r['median']:.3f}  IQR=[{r['iqr_lo']:.3f}, {r['iqr_hi']:.3f}]  "
        f"min={r['min']:.3f}  max={r['max']:.3f}"
    )
    p = summary["pooled_loo_acc_ictal_patients_only"]
    print(
        f"pooled_LOO (7 ictal patients): median={p['median']:.1f}%  "
        f"IQR=[{p['iqr_lo']:.1f}, {p['iqr_hi']:.1f}]  "
        f"CI95=[{p['ci95_lo']:.1f}, {p['ci95_hi']:.1f}]"
    )
    print(f"wrote {cohort_tsv.name}, {fold_tsv.name}, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
