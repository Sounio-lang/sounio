#!/usr/bin/env python3
"""High-gamma (20-40 Hz) power baseline, matched window schedule to Door F.

For each patient in the manifest, read the EDF, extract the same 7 windows
Door F uses (II @+10s, FAR @onset-60, PRE30 @-30, PRE10 @-10, PRE5 @-5,
IC @onset, POST @onset+5; each 80 samples at 256 Hz, 16 channels), compute
channel-averaged bandpower in 20-40 Hz via Welch's method, and emit per-patient
TSV plus a cohort summary.

This is the standard literature baseline against which our non-associative
observable (Door F sedenion associator norm) must be compared head-to-head.

Usage:
    python3 high_gamma_baseline.py \\
        --manifest scripts/research/door_f_cohort/chbmit_manifest.tsv \\
        --edf-dir  /orangefs/training/sounio/chbmit \\
        --out-dir  /orangefs/.../results/baseline \\
        --cohort-tsv /orangefs/.../results/baseline_cohort.tsv
"""
import os
import sys
import argparse
import numpy as np
import pyedflib
from scipy.signal import welch

FS          = 256
WINDOW_LEN  = 80
N_CH        = 16
II_S        = 10.0
BAND_LO     = 20.0
BAND_HI     = 40.0
WINDOWS     = [("FAR", -60), ("PRE30", -30), ("PRE10", -10),
               ("PRE5", -5), ("IC", 0), ("POST", +5)]


def load_window(f, ch_map, t0_s, length=WINDOW_LEN):
    t0 = int(round(t0_s * FS))
    sigs = np.stack([f.readSignal(i, t0, length) for i in ch_map])
    return sigs  # (N_CH, length)


def bandpower(sigs, fs=FS, lo=BAND_LO, hi=BAND_HI):
    """Channel-averaged bandpower in [lo, hi] via Welch."""
    # nperseg must be <= signal length
    nperseg = min(sigs.shape[1], 64)
    freqs, pxx = welch(sigs, fs=fs, nperseg=nperseg, axis=1)
    band_mask = (freqs >= lo) & (freqs <= hi)
    if not band_mask.any():
        return float("nan")
    band_power = pxx[:, band_mask].mean(axis=1)  # per channel
    return float(band_power.mean())


def process(patient, edf_path, onset_s, out_tsv):
    f = pyedflib.EdfReader(edf_path)
    fs = int(f.getSampleFrequency(0))
    if fs != FS:
        raise RuntimeError(f"fs mismatch: expected {FS}, got {fs}")
    ch_map = list(range(min(N_CH, f.signals_in_file)))

    ii_raw = load_window(f, ch_map, II_S)
    probes = {name: load_window(f, ch_map, onset_s + off) for name, off in WINDOWS}
    f.close()

    # Normalization: z-score by II window's own per-channel mean/std so
    # bandpower is comparable across patients despite absolute-µV variance.
    mu = ii_raw.mean(axis=1, keepdims=True)
    sd = ii_raw.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)

    def norm(x):
        return (x - mu) / sd

    ii_bp     = bandpower(norm(ii_raw))
    probe_bp  = {name: bandpower(norm(probes[name])) for name, _ in WINDOWS}

    with open(out_tsv, "w") as g:
        g.write("# high-gamma (20-40 Hz) power, Welch, z-normalized on II\n")
        g.write(f"# patient={patient}  edf={edf_path}  onset_s={onset_s}\n")
        g.write("window\tt_rel_s\tpower\n")
        g.write(f"II\t+10\t{ii_bp:.8e}\n")
        for name, off in WINDOWS:
            g.write(f"{name}\t{off:+d}\t{probe_bp[name]:.8e}\n")

    return {"II": ii_bp, **probe_bp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--edf-dir",    required=True)
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--cohort-tsv", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "patient":
                continue
            pid, edf_name, onset_s = parts[0], parts[1], int(parts[2])
            edf = os.path.join(args.edf_dir, edf_name)
            out_tsv = os.path.join(args.out_dir, f"baseline_{pid}.tsv")
            try:
                bp = process(pid, edf, onset_s, out_tsv)
                rows.append((pid, bp))
                print(f"  {pid:8s}  II={bp['II']:.3e}  "
                      f"IC={bp['IC']:.3e}  (spike%={(bp['IC']/bp['PRE5']-1)*100:+.1f}%)")
            except Exception as e:
                print(f"  FAIL {pid}: {e}", file=sys.stderr)

    with open(args.cohort_tsv, "w") as g:
        g.write("# high-gamma (20-40 Hz) cohort — channel-avg Welch bandpower\n")
        g.write("patient\tII\tFAR\tPRE30\tPRE10\tPRE5\tIC\tPOST\n")
        for pid, bp in rows:
            g.write(f"{pid}\t{bp['II']:.6e}\t{bp['FAR']:.6e}\t{bp['PRE30']:.6e}"
                    f"\t{bp['PRE10']:.6e}\t{bp['PRE5']:.6e}\t{bp['IC']:.6e}\t{bp['POST']:.6e}\n")
    print(f"wrote {args.cohort_tsv}  (N={len(rows)})")


if __name__ == "__main__":
    main()
