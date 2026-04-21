#!/usr/bin/env python3
# P5.5 / PROTOCOL §2 T3 — Channel-subset robustness aggregator.
#
# Inputs :
#   --draw-manifest  draw_manifest.tsv produced by plan_channel_draws.py
#   --stdout-root    dir with <patient>/draw_<NNN>.stdout from the sbatch run
#
# Statistics (PROTOCOL §1):
#   per (patient p, draw d):
#     dip_{p,d}   = (FAR - min(PRE30, PRE10, PRE5)) / FAR
#     spike_{p,d} = (IC - PRE5) / PRE5
#   per draw d (cohort across 24 patients):
#     dip_median_d   = median_p dip_{p,d}
#     spike_median_d = median_p spike_{p,d}
#
# Decision rule (PROTOCOL §2 T3): robust iff ≥ 95 % of the n_draws draws
# yield a cohort-median with the canonical positive sign, SEPARATELY for
# dip and for spike. Both must pass.
#
# Outputs :
#   <out-dir>/t3_per_draw.tsv          patient, draw, dip, spike, assoc_{FAR,...}
#   <out-dir>/t3_cohort_by_draw.tsv    draw, dip_median, spike_median, ...
#   <out-dir>/t3_robust.json           headline numbers + pass/fail
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

WINDOWS = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]
LINE_RE = re.compile(
    r"^\s+(FAR|PRE30|PRE10|PRE5|IC|POST)\s+[+-]?\d+s\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_stdout(path):
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                m = LINE_RE.match(ln)
                if m:
                    out[m.group(1)] = float(m.group(4))   # assoc column
    except FileNotFoundError:
        return None
    return out if len(out) == 6 else None


def dip_spike(assoc):
    far   = assoc["FAR"]
    pre5  = assoc["PRE5"]
    p_min = min(assoc["PRE30"], assoc["PRE10"], assoc["PRE5"])
    dip_denom = far  if far  > 1e-12 else 1e-12
    sp_denom  = pre5 if pre5 > 1e-12 else 1e-12
    return (far - p_min) / dip_denom, (assoc["IC"] - pre5) / sp_denom


def load_draws(draw_manifest, stdout_root):
    entries = []
    with open(draw_manifest) as f:
        header = f.readline().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            entries.append({
                "patient": c[ix["patient"]],
                "draw_id": int(c[ix["draw_id"]]),
                "channels": c[ix["channels"]],
            })
    per = []
    missing = 0
    for e in entries:
        path = os.path.join(stdout_root, e["patient"],
                            f"draw_{e['draw_id']:03d}.stdout")
        assoc = parse_stdout(path)
        if assoc is None:
            missing += 1
            continue
        d, s = dip_spike(assoc)
        per.append({**e, **{f"assoc_{w}": assoc[w] for w in WINDOWS},
                    "dip": d, "spike": s})
    return per, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw-manifest", required=True)
    ap.add_argument("--stdout-root",   required=True)
    ap.add_argument("--out-dir",       required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    per, missing = load_draws(args.draw_manifest, args.stdout_root)
    if not per:
        sys.exit("no parseable draws found")

    per_path = os.path.join(args.out_dir, "t3_per_draw.tsv")
    with open(per_path, "w") as f:
        f.write("patient\tdraw\tdip\tspike\t"
                + "\t".join(f"assoc_{w}" for w in WINDOWS)
                + "\tchannels\n")
        for e in per:
            f.write(f"{e['patient']}\t{e['draw_id']}"
                    f"\t{e['dip']:.6f}\t{e['spike']:.6f}\t"
                    + "\t".join(f"{e[f'assoc_{w}']:.6f}" for w in WINDOWS)
                    + f"\t{e['channels']}\n")

    # Cohort-median per draw index
    by_draw = defaultdict(list)
    for e in per:
        by_draw[e["draw_id"]].append((e["dip"], e["spike"]))
    by_draw_path = os.path.join(args.out_dir, "t3_cohort_by_draw.tsv")
    dip_medians, spike_medians = [], []
    with open(by_draw_path, "w") as f:
        f.write("draw\tN_patients\tdip_median\tspike_median\n")
        for d in sorted(by_draw):
            arr = np.array(by_draw[d])
            dm, sm = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
            f.write(f"{d}\t{len(arr)}\t{dm:.6f}\t{sm:.6f}\n")
            dip_medians.append(dm)
            spike_medians.append(sm)

    dm_arr = np.asarray(dip_medians)
    sm_arr = np.asarray(spike_medians)
    frac_dip_pos   = float(np.mean(dm_arr   > 0.0))
    frac_spike_pos = float(np.mean(sm_arr > 0.0))
    dip_median   = float(np.median(dm_arr))
    spike_median = float(np.median(sm_arr))
    dip_ci   = [float(np.percentile(dm_arr, 2.5)),
                float(np.percentile(dm_arr, 97.5))]
    spike_ci = [float(np.percentile(sm_arr, 2.5)),
                float(np.percentile(sm_arr, 97.5))]

    summary = {
        "n_draws": len(dm_arr),
        "missing_stdouts": missing,
        "dip_median_across_draws":   dip_median,
        "dip_ci95_across_draws":     dip_ci,
        "dip_sign_preserved_frac":   frac_dip_pos,
        "spike_median_across_draws": spike_median,
        "spike_ci95_across_draws":   spike_ci,
        "spike_sign_preserved_frac": frac_spike_pos,
        "threshold": 0.95,
        "t3_dip_passes":   bool(frac_dip_pos   >= 0.95),
        "t3_spike_passes": bool(frac_spike_pos >= 0.95),
        "t3_overall_pass": bool(frac_dip_pos   >= 0.95 and frac_spike_pos >= 0.95),
    }
    summary_path = os.path.join(args.out_dir, "t3_robust.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
