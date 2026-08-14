#!/usr/bin/env python3
"""
scripts/research/lemon_ffi_bridge.py

Pure I/O bridge for the Sounio-side LEMON G2 re-analysis
(examples/cayley_dickson_lemon_g2_ffi.sio). Invoked by Sounio via FFI
(extern "C" system()) -- per CLAUDE.md's "science stays in Sounio"
principle, this script does NOT compute any octonion/G2/correlation
math. Its only job is the part Sounio genuinely cannot do itself:
reading the binary .npy EEG format and the endpoints CSV, matching
subjects across both, downsampling so the output is small enough for
Sounio to parse with plain text I/O, and emitting a flat CSV.

Everything downstream (O-SSM forward pass, G2 derivation features,
Spearman correlation) is computed in Sounio.

Output columns: subject_id,cerq_rumination,neo_neuroticism,t,ch0..ch6
"""
import csv
import os
import sys
import numpy as np

CACHE_DIR = "/workspace/data/lemon/preprocessed"
ENDPOINTS_PATH = "/workspace/data/lemon/endpoints.csv"
N_SUBJECTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_TIMESTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
OUT_PATH = sys.argv[3] if len(sys.argv) > 3 else "/dev/stdout"


def main():
    endpoints = {}
    with open(ENDPOINTS_PATH) as f:
        for row in csv.DictReader(f):
            rum = row.get("cerq_rumination", "").strip()
            neu = row.get("neo_neuroticism", "").strip()
            if rum and neu:
                try:
                    endpoints[row["subject_id"]] = (float(rum), float(neu))
                except ValueError:
                    continue

    rows = []
    n_written = 0
    for sid, (rum, neu) in endpoints.items():
        if n_written >= N_SUBJECTS:
            break
        epoch_path = os.path.join(CACHE_DIR, f"{sid}_epochs.npy")
        if not os.path.exists(epoch_path):
            continue
        epochs = np.load(epoch_path)  # (n_epochs, 7, 1000)
        if epochs.shape[0] < 1:
            continue
        epoch = epochs[0]  # (7, 1000)
        n_t = epoch.shape[1]
        stride = max(1, n_t // N_TIMESTEPS)
        idxs = list(range(0, n_t, stride))[:N_TIMESTEPS]
        for t_out, t_in in enumerate(idxs):
            ch = epoch[:, t_in]  # (7,)
            rows.append([sid, rum, neu, t_out] + list(ch))
        n_written += 1

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "cerq_rumination", "neo_neuroticism", "t",
                    "ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6"])
        w.writerows(rows)

    print(f"[bridge] wrote {len(rows)} rows for {n_written} subjects -> {OUT_PATH}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
