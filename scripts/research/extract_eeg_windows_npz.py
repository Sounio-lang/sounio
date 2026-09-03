#!/usr/bin/env python3
"""Extract all windows from EEGMMIDB EDF files and save as npz.
Run locally (needs pyedflib). Output is numpy-only, no pyedflib needed on cluster.

Output: windows_normed.npz with keys:
  windows_normed: (N, 16, 80) float32  — per-channel normalized inputs
  targets:        (N, 80)     float32  — CH0 one-step-ahead targets
"""
import os, sys, argparse
import numpy as np

WIN_LEN = 81
TRAIN_N = 64
CH_COUNT = 16

def load_edf(path):
    import pyedflib
    try:
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        sigs = np.stack([f.readSignal(i) for i in range(min(n, CH_COUNT))])
        f.close()
        return sigs
    except Exception as e:
        sys.stderr.write(f"SKIP {path}: {e}\n")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", default="data/eegmmidb")
    ap.add_argument("--stride",  type=int, default=40)
    ap.add_argument("--output",  default="/tmp/windows_normed.npz")
    args = ap.parse_args()

    edf_files = sorted([os.path.join(args.edf_dir, f)
                        for f in os.listdir(args.edf_dir) if f.endswith(".edf")])
    print(f"Processing {len(edf_files)} EDF files...", flush=True)

    all_normed = []
    all_targets = []

    for path in edf_files:
        sigs = load_edf(path)
        if sigs is None or sigs.shape[0] < CH_COUNT:
            continue
        _, T = sigs.shape
        start = 0
        while start + WIN_LEN <= T:
            w = sigs[:CH_COUNT, start:start+WIN_LEN]
            data = w[:, :80]
            target_raw = w[0, 1:]  # (80,)
            scales = np.abs(data[:, :TRAIN_N]).max(axis=1) + 1e-6
            normed = (data / scales[:, None]).astype(np.float32)
            tscale = np.abs(target_raw[:TRAIN_N]).max() + 1e-6
            target = (target_raw / tscale).astype(np.float32)
            if np.abs(target[:TRAIN_N]).max() < 1e-6:
                start += args.stride; continue
            all_normed.append(normed)
            all_targets.append(target)
            start += args.stride

    W = np.stack(all_normed)   # (N, 16, 80)
    T = np.stack(all_targets)  # (N, 80)
    print(f"Extracted {len(W)} windows  shape: {W.shape}", flush=True)
    np.savez_compressed(args.output, windows_normed=W, targets=T)
    print(f"Saved → {args.output}  "
          f"({os.path.getsize(args.output)/1e6:.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
