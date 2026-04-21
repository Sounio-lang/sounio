#!/usr/bin/env python3
# P5.5 / PROTOCOL §2 T3 — Channel-subset robustness planner.
#
# For each patient in the N=24 manifest, draws n_draws 16-channel subsets
# uniformly at random (without replacement within each draw) from the set
# of available EDF channels (≥23 per CHB-MIT), and generates one Sounio
# file per (patient, draw_id). The same draw_id d is matched across
# patients, so the aggregator can compute a cohort-median dip/spike at
# each draw index d (PROTOCOL §2 T3 decision rule).
#
# RNG seed is the PROTOCOL-frozen value (20260420); re-runs are bitwise
# reproducible.
#
# Output layout (under --out-root):
#   sio/<patient>/draw_<NNN>.sio
#   draw_manifest.tsv   patient<TAB>draw_id<TAB>channels<TAB>sio<TAB>edf
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Import generate.py as a module (avoid sys.path pollution).
_SPEC = importlib.util.spec_from_file_location(
    "door_f_generate", os.path.join(HERE, "generate.py"))
GEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GEN)


def read_manifest(path):
    rows = []
    with open(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if parts[0] == "patient":
                continue
            rows.append((parts[0], parts[1], int(parts[2])))
    return rows


def count_channels(edf_path):
    import pyedflib
    r = pyedflib.EdfReader(edf_path)
    n = r.signals_in_file
    r.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="N=24 cohort TSV (scripts/research/door_f_cohort/chbmit_manifest.tsv)")
    ap.add_argument("--edf-dir",  required=True,
                    help="directory containing the EDF files listed in the manifest")
    ap.add_argument("--out-root", required=True,
                    help="root directory for sio/<patient>/draw_<NNN>.sio + draw_manifest.tsv")
    ap.add_argument("--n-draws",  type=int, default=100,
                    help="number of random 16-channel draws per patient (default 100)")
    ap.add_argument("--seed",     type=int, default=20260420,
                    help="PROTOCOL-frozen RNG seed")
    ap.add_argument("--only",
                    help="comma-separated patient whitelist")
    ap.add_argument("--tail",     default=os.path.join(HERE, "template.sio.part"),
                    help="alternate tail template (default template.sio.part)")
    ap.add_argument("--n-ch",     type=int, default=16,
                    help="channels per draw (must match header.sio.part; default 16)")
    ap.add_argument("--manifest-only", action="store_true",
                    help="re-derive the RNG draws and re-emit the draw_manifest.tsv "
                         "without rewriting the .sio files (intended to recover "
                         "from a truncated manifest when the .sio files already "
                         "exist from a previous byte-identical run)")
    args = ap.parse_args()

    if args.n_ch != GEN.N_CH:
        sys.exit(f"--n-ch must equal generate.N_CH ({GEN.N_CH})")

    rows_all = read_manifest(args.manifest)
    canonical_idx = {p: i for i, (p, _, _) in enumerate(rows_all)}
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows_all if r[0] in keep]
    else:
        rows = rows_all
    if not rows:
        sys.exit("no patients selected")

    tail_src = open(args.tail).read()
    os.makedirs(args.out_root, exist_ok=True)
    manifest_path = os.path.join(args.out_root, "draw_manifest.tsv")

    # Per-patient deterministic seed derived from (base_seed, canonical_index)
    # so that splitting the run with --only yields bit-identical .sio files
    # as the unfiltered run for the same patients.
    # The manifest is assembled in-memory and flushed in one write() at the
    # end; OrangeFS/pvfs2 truncates long series of small appends on an open
    # handle to a parallel filesystem, so a single write is essential.
    manifest_lines = ["patient\tdraw_id\tchannels\tsio\tedf"]
    for patient, edf_name, onset in rows:
        edf_path = os.path.join(args.edf_dir, edf_name)
        if not os.path.exists(edf_path):
            sys.exit(f"missing EDF: {edf_path}")
        n_ch_avail = count_channels(edf_path)
        if n_ch_avail < args.n_ch:
            sys.exit(f"{patient}: only {n_ch_avail} channels available, need ≥{args.n_ch}")
        sio_dir = os.path.join(args.out_root, "sio", patient)
        os.makedirs(sio_dir, exist_ok=True)
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, canonical_idx[patient]]))
        for d in range(args.n_draws):
            ch = sorted(rng.choice(n_ch_avail, size=args.n_ch, replace=False).tolist())
            sio_path = os.path.join(sio_dir, f"draw_{d:03d}.sio")
            if not args.manifest_only:
                GEN.generate(patient, edf_path, onset, sio_path,
                             ch_map=ch,
                             null_pool_size=0,
                             tail_override=tail_src)
            manifest_lines.append(
                f"{patient}\t{d}\t{','.join(str(c) for c in ch)}"
                f"\t{sio_path}\t{edf_path}")
        sys.stderr.write(f"  {patient}: {args.n_draws} draws "
                         f"({n_ch_avail} chans available)\n")

    with open(manifest_path, "w") as mf:
        mf.write("\n".join(manifest_lines) + "\n")
    sys.stderr.write(f"manifest: {manifest_path}  ({len(manifest_lines) - 1} entries)\n")


if __name__ == "__main__":
    main()
