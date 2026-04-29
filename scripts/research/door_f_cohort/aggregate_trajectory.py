#!/usr/bin/env python3
# Aggregator for P5.3 Door F trajectory-chunk stdouts.
#
# Reads the chunk manifest emitted by plan_trajectory_chunks.py plus the
# per-chunk stdout captured by the cluster runner, stitches NULL_ASSOC
# <slot> <value> pairs into a per-patient trajectory TSV keyed by global
# epoch index (0 .. N_EPOCHS-1).
#
# Stdout format from souc's print: each NULL_ASSOC entry spans two lines,
#   NULL_ASSOC <slot_local>
#    <f64>
# (the leading space on line 2 is how `print(f64)` renders). slot_local is
# 0..n_epochs-1 within the chunk; global epoch = epoch_start + slot_local.
#
# Per-patient output:
#   <run_root>/trajectories/<patient>.tsv
#     columns: epoch  t_rel_s  assoc
# plus a cohort-level <run_root>/trajectories/cohort_trajectory.tsv with
# columns: patient epoch t_rel_s assoc
#
# Usage:
#   python3 aggregate_trajectory.py \
#       --chunk-manifest <run_root>/chunk_manifest.tsv \
#       --stdout-dir     <run_root>/stdout \
#       --master-manifest scripts/research/door_f_cohort/chbmit_manifest.tsv \
#       --out-dir <run_root>/trajectories \
#       [--chunk-epochs 20] [--window-len 80] [--fs 256]
import argparse
import os
import re
import sys

NULL_LINE_RE = re.compile(r"^NULL_ASSOC\s+(\d+)\s*$")


def parse_chunk_stdout(path):
    slots = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m = NULL_LINE_RE.match(lines[i])
        if m and i + 1 < len(lines):
            try:
                val = float(lines[i + 1].strip())
            except ValueError:
                i += 1
                continue
            slots[int(m.group(1))] = val
            i += 2
        else:
            i += 1
    return slots


def parse_master_manifest(path):
    onsets = {}
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#") or ln.startswith("patient\t"):
                continue
            p = ln.split("\t")
            if len(p) >= 3:
                onsets[p[0]] = int(p[2])
    return onsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-manifest",  required=True)
    ap.add_argument("--stdout-dir",      required=True,
                    help="directory holding <patient>/chunk_<NNN>.stdout")
    ap.add_argument("--master-manifest", required=True)
    ap.add_argument("--out-dir",         required=True)
    ap.add_argument("--chunk-epochs",    type=int, default=20)
    ap.add_argument("--window-len",      type=int, default=80)
    ap.add_argument("--fs",              type=int, default=256)
    ap.add_argument("--radius-s",        type=int, default=90,
                    help="must match plan_trajectory_chunks.py --radius-s")
    args = ap.parse_args()

    onsets = parse_master_manifest(args.master_manifest)
    os.makedirs(args.out_dir, exist_ok=True)

    by_patient = {}
    missing = []
    with open(args.chunk_manifest) as f:
        header = f.readline()
        for ln in f:
            p, cid, estart, nep, sio = ln.rstrip("\n").split("\t")
            cid, estart, nep = int(cid), int(estart), int(nep)
            stdout_path = os.path.join(args.stdout_dir, p, f"chunk_{cid:03d}.stdout")
            if not os.path.exists(stdout_path):
                missing.append((p, cid))
                continue
            slots = parse_chunk_stdout(stdout_path)
            got = len(slots)
            if got != nep:
                sys.stderr.write(f"WARN {p} chunk {cid}: got {got}/{nep} slots\n")
            patient_rows = by_patient.setdefault(p, {})
            for local, v in slots.items():
                patient_rows[estart + local] = v

    cohort_path = os.path.join(args.out_dir, "cohort_trajectory.tsv")
    with open(cohort_path, "w") as cf:
        cf.write("patient\tepoch\tt_rel_s\tassoc\n")
        for p in sorted(by_patient):
            onset = onsets.get(p)
            if onset is None:
                sys.stderr.write(f"skip {p}: not in master manifest\n")
                continue
            rows = by_patient[p]
            per_patient = os.path.join(args.out_dir, f"{p}.tsv")
            with open(per_patient, "w") as pf:
                pf.write("epoch\tt_rel_s\tassoc\n")
                for ep in sorted(rows):
                    # t_rel matches the offset layout from
                    # plan_trajectory_chunks.py: base=onset-radius, stride=WIN
                    t_rel = (-args.radius_s
                             + (ep * args.window_len + args.window_len / 2)
                               / args.fs)
                    pf.write(f"{ep}\t{t_rel:.6f}\t{rows[ep]:.10f}\n")
                    cf.write(f"{p}\t{ep}\t{t_rel:.6f}\t{rows[ep]:.10f}\n")
            sys.stderr.write(f"{p}: {len(rows)} epochs -> {per_patient}\n")
    if missing:
        sys.stderr.write(f"\nMISSING {len(missing)} chunk stdouts: "
                         f"{missing[:5]}{'...' if len(missing) > 5 else ''}\n")
    sys.stderr.write(f"\ncohort trajectory: {cohort_path}\n")


if __name__ == "__main__":
    main()
