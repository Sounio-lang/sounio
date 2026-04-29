#!/usr/bin/env python3
# Compute per-patient control-biomarker trajectories (576 epochs per patient)
# for the Phase A peer-review response. Three controls plus a re-verification
# of the sedenion associator:
#   - sedenion  : Sounio-parity sedenion associator (sanity re-run)
#   - permuted  : sedenion associator with a random channel->basis permutation
#   - rand_tri  : fixed random 16x16x16 trilinear tensor replacing sed_mul
#   - bispec    : ||<x_t x_t^T x_t>_t||_F, SSM-free, algebra-free
#
# Outputs one trajectory TSV per (biomarker, patient) plus a cohort TSV, in
# the exact format consumed by permutation_test.py:
#     columns: epoch  t_rel_s  assoc   (per-patient)
#     columns: patient  epoch  t_rel_s  assoc   (cohort)
#
# Usage:
#   python3 scripts/research/door_f_cohort/run_controls.py \
#       --manifest scripts/research/door_f_cohort/chbmit_manifest.tsv \
#       --edf-dir  data/chbmit \
#       --out-root artifacts/research/door_f_cohort_N24/controls \
#       [--seed 20260421] [--only chb02,chb03] [--biomarkers sedenion,bispec]
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate import load_windows                                  # noqa: E402
from sedenion_numpy import (                                       # noqa: E402
    make_a,
    assoc_trajectory,
    assoc_trajectory_permuted,
    random_trilinear_trajectory,
    bispectrum_norm,
)

FS, WINDOW_LEN, RADIUS_S = 256, 80, 90
TOTAL_EPOCHS = int(RADIUS_S * 2 * FS / WINDOW_LEN)

# Match permutation_test.py PROBE_OFFSETS_S and generate.py WINDOWS
PROBE_OFFSETS_S = [("FAR", -60), ("PRE30", -30), ("PRE10", -10),
                   ("PRE5", -5), ("IC", 0), ("POST", 5)]


def parse_manifest(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#") or ln.startswith("patient\t"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], int(parts[2])))
    return rows


def epoch_offsets(onset_s):
    base = int(round(onset_s * FS)) - RADIUS_S * FS
    return [base + k * WINDOW_LEN for k in range(TOTAL_EPOCHS)]


def t_rel(epoch):
    return -RADIUS_S + (epoch * WINDOW_LEN + WINDOW_LEN / 2) / FS


def compute_patient_trajectories(null_pool, a, perm, tri_tensor, bios):
    """null_pool: (N_ep, 16, 80). Returns dict {biomarker: (N_ep,)}."""
    out = {}
    N = null_pool.shape[0]
    if "sedenion" in bios:
        out["sedenion"] = np.array(
            [assoc_trajectory(a, null_pool[i].T) for i in range(N)])
    if "permuted" in bios:
        out["permuted"] = np.array(
            [assoc_trajectory_permuted(a, null_pool[i].T, perm)
             for i in range(N)])
    if "rand_tri" in bios:
        out["rand_tri"] = np.array(
            [random_trilinear_trajectory(a, null_pool[i].T, tri_tensor)
             for i in range(N)])
    if "bispec" in bios:
        out["bispec"] = np.array(
            [bispectrum_norm(null_pool[i].T) for i in range(N)])
    return out


def write_patient(trajectories_dir, biomarker, patient, values):
    pdir = os.path.join(trajectories_dir, biomarker)
    os.makedirs(pdir, exist_ok=True)
    path = os.path.join(pdir, f"{patient}.tsv")
    with open(path, "w") as f:
        f.write("epoch\tt_rel_s\tassoc\n")
        for ep, v in enumerate(values):
            f.write(f"{ep}\t{t_rel(ep):.6f}\t{v:.10f}\n")
    return path


def write_cohort(trajectories_dir, biomarker, per_patient_values):
    pdir = os.path.join(trajectories_dir, biomarker)
    path = os.path.join(pdir, "cohort_trajectory.tsv")
    with open(path, "w") as f:
        f.write("patient\tepoch\tt_rel_s\tassoc\n")
        for p in sorted(per_patient_values):
            for ep, v in enumerate(per_patient_values[p]):
                f.write(f"{p}\t{ep}\t{t_rel(ep):.6f}\t{v:.10f}\n")
    return path


def write_cohort_windows(trajectories_dir, biomarker, per_patient_values):
    """Emit a 6-window TSV matching door_f_cohort.tsv columns for LOO use."""
    pdir = os.path.join(trajectories_dir, biomarker)
    path = os.path.join(pdir, "cohort.tsv")
    cols = ["patient"] + [f"assoc_{w}" for w, _ in PROBE_OFFSETS_S]
    ts = np.array([t_rel(ep) for ep in range(TOTAL_EPOCHS)])
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for p in sorted(per_patient_values):
            vals = per_patient_values[p]
            row = [p]
            for _, off in PROBE_OFFSETS_S:
                ep = int(np.argmin(np.abs(ts - off)))
                row.append(f"{vals[ep]:.6f}")
            f.write("\t".join(row) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--edf-dir",  required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed",     type=int, default=20260421)
    ap.add_argument("--only",     help="comma-separated patient filter")
    ap.add_argument("--biomarkers",
                    default="sedenion,permuted,rand_tri,bispec",
                    help="subset of {sedenion,permuted,rand_tri,bispec}")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    bios = set(args.biomarkers.split(","))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(16)
    tri_tensor = rng.normal(size=(16, 16, 16))
    tri_tensor /= np.sqrt((tri_tensor * tri_tensor).sum())
    a = make_a(0.2)

    rows = parse_manifest(args.manifest)
    per_patient = {b: {} for b in bios}
    skipped = []

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "PROVENANCE.md"), "w") as pf:
        pf.write("# Phase A control-biomarker run\n\n")
        pf.write(f"seed: {args.seed}\n")
        pf.write(f"permutation: {list(map(int, perm))}\n")
        pf.write(f"trilinear tensor: Gaussian, unit Frobenius\n")
        pf.write(f"biomarkers: {sorted(bios)}\n")

    for patient, edf, onset in rows:
        if only and patient not in only:
            continue
        edf_path = os.path.join(args.edf_dir, edf)
        if not os.path.exists(edf_path):
            skipped.append((patient, "missing EDF"))
            continue
        try:
            _, _, null_pool = load_windows(
                edf_path, onset, null_offsets=epoch_offsets(onset))
        except Exception as e:
            skipped.append((patient, str(e)))
            continue
        t0 = time.time()
        tr = compute_patient_trajectories(null_pool, a, perm, tri_tensor, bios)
        for b, vals in tr.items():
            per_patient[b][patient] = vals
            write_patient(args.out_root, b, patient, vals)
        sys.stderr.write(f"{patient}: {time.time()-t0:.1f}s\n")

    for b in bios:
        if per_patient[b]:
            path = write_cohort(args.out_root, b, per_patient[b])
            wpath = write_cohort_windows(args.out_root, b, per_patient[b])
            sys.stderr.write(f"{b}: cohort -> {path}, windows -> {wpath}\n")

    if skipped:
        sys.stderr.write(f"\nSKIPPED {len(skipped)}: {skipped}\n")


if __name__ == "__main__":
    main()
