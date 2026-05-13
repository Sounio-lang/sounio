#!/usr/bin/env python3
# Trajectory-chunk planner for the P5.3 Door F sedenion-associator sweep.
#
# Per patient in the canonical manifest, emits one .sio file per chunk of the
# +/- RADIUS_S second window around seizure onset. Each chunk has CHUNK_EPOCHS
# non-overlapping 80-sample epochs, each corresponding to one associator
# measurement in template_with_null.sio.part. Emits a chunk-manifest TSV that
# the Slurm submit script consumes.
#
# Chunk-start sample (relative to recording start):
#   base = round(onset_s * FS) - RADIUS_S * FS
#   chunk_i_start = base + i * CHUNK_EPOCHS * WINDOW_LEN
# Global epoch index (0 .. N_EPOCHS-1) for slot k of chunk i:
#   epoch = i * CHUNK_EPOCHS + k
# Time relative to onset (seconds) for slot k of chunk i (epoch centre):
#   t_rel = (chunk_i_start + k * WINDOW_LEN + WINDOW_LEN // 2) / FS - onset_s
#
# Output layout:
#   <out_root>/sio/chb03/chunk_000.sio            (one per chunk)
#   <out_root>/chunk_manifest.tsv                 (master index)
#     columns: patient  chunk_id  epoch_start  n_epochs  sio_path
# Usage:
#   python3 scripts/research/door_f_cohort/plan_trajectory_chunks.py \
#       --manifest scripts/research/door_f_cohort/chbmit_manifest.tsv \
#       --edf-dir /orangefs/training/sounio/chbmit \
#       --out-root /orangefs/training/sounio/door-f-runs/traj-<id> \
#       [--chunk-epochs 20] [--radius-s 90] [--only chb03,chb07]
import argparse
import os
import subprocess
import sys

FS           = 256
WINDOW_LEN   = 80
GENERATE_PY  = os.path.join(os.path.dirname(__file__), "generate.py")
TAIL_PART    = os.path.join(os.path.dirname(__file__),
                            "template_with_null.sio.part")


def parse_manifest(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#") or ln.startswith("patient\t"):
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], int(parts[2])))
    return rows


def plan_chunks(onset_s, radius_s, chunk_epochs):
    base = int(round(onset_s * FS)) - int(radius_s * FS)
    total_epochs = int(radius_s * 2 * FS / WINDOW_LEN)
    n_chunks = (total_epochs + chunk_epochs - 1) // chunk_epochs
    plan = []
    for i in range(n_chunks):
        start = i * chunk_epochs
        end   = min(start + chunk_epochs, total_epochs)
        offsets = [base + (start + k) * WINDOW_LEN for k in range(end - start)]
        plan.append((i, start, end - start, offsets))
    return plan, total_epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",     required=True)
    ap.add_argument("--edf-dir",      required=True)
    ap.add_argument("--out-root",     required=True)
    ap.add_argument("--chunk-epochs", type=int, default=20)
    ap.add_argument("--radius-s",     type=int, default=90)
    ap.add_argument("--only",         help="comma-separated patient filter")
    ap.add_argument("--tail",         default=TAIL_PART)
    ap.add_argument("--python",       default=sys.executable)
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    manifest_rows = parse_manifest(args.manifest)
    sio_root = os.path.join(args.out_root, "sio")
    os.makedirs(sio_root, exist_ok=True)
    manifest_out = os.path.join(args.out_root, "chunk_manifest.tsv")
    total = 0
    with open(manifest_out, "w") as mf:
        mf.write("patient\tchunk_id\tepoch_start\tn_epochs\tsio_path\n")
        for patient, edf, onset in manifest_rows:
            if only and patient not in only:
                continue
            edf_path = os.path.join(args.edf_dir, edf)
            patient_dir = os.path.join(sio_root, patient)
            os.makedirs(patient_dir, exist_ok=True)
            plan, total_epochs = plan_chunks(onset, args.radius_s,
                                             args.chunk_epochs)
            sys.stderr.write(f"{patient}: {len(plan)} chunks "
                             f"({total_epochs} epochs total)\n")
            for i, start, n, offsets in plan:
                sio_path = os.path.join(patient_dir, f"chunk_{i:03d}.sio")
                cmd = [args.python, GENERATE_PY,
                       "--patient", patient,
                       "--edf",     edf_path,
                       "--onset",   str(onset),
                       "--out",     sio_path,
                       "--tail",    args.tail,
                       "--null-offsets", ",".join(str(x) for x in offsets)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    sys.stderr.write(f"  FAIL chunk {i}: {r.stderr}\n")
                    raise SystemExit(2)
                mf.write(f"{patient}\t{i}\t{start}\t{n}\t{sio_path}\n")
                total += 1
    sys.stderr.write(f"\nwrote {total} chunks; manifest: {manifest_out}\n")


if __name__ == "__main__":
    main()
