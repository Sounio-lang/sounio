#!/usr/bin/env python3
"""Process a single LEMON subject — standalone script for parallel execution."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.research.ossm_168_dryrun import (
    _features_fast as feat_fast,
    features as feat_mod,
    lemon_preprocess as lp,
    ossm as ossm_mod,
)
import statistics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--subject-index", type=int, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--subsample-seed", type=int, default=20260421)
    args = ap.parse_args()

    sid = args.subject
    si = args.subject_index
    t0_total = time.time()

    result = {
        "subject": sid,
        "subject_index": si,
        "status": "unknown",
        "wall_time_s": 0,
    }

    # Check cache
    npy = args.cache_dir / f"{sid}_epochs.npy"
    meta = args.cache_dir / f"{sid}_meta.json"

    if npy.is_file():
        try:
            epochs = np.load(npy)
            result["status"] = "cached"
            result["n_epochs_cache"] = int(epochs.shape[0])
        except Exception as e:
            result["status"] = f"cache_error: {e}"
            print(json.dumps(result))
            return 1
    else:
        # Preprocess
        try:
            t0 = time.time()
            res = lp.preprocess_subject(str(args.raw_root), sid)
            epochs = np.asarray(res.epochs, dtype=np.float64)

            prov = {
                "subject": sid,
                "sfreq": res.sfreq,
                "n_total_epochs": res.n_total_epochs,
                "n_retained_epochs": res.n_retained_epochs,
                "retention_rate": res.retention_rate(),
                "n_ica_rejected": res.n_ica_rejected,
                "n_ica_total": res.n_ica_total,
            }

            np.save(npy, epochs)
            meta.write_text(json.dumps(prov, indent=2))

            result["status"] = "processed"
            result["preprocess_time_s"] = time.time() - t0
            result["n_total_epochs"] = res.n_total_epochs
            result["n_retained_epochs"] = res.n_retained_epochs
        except Exception as e:
            result["status"] = f"preprocess_error: {type(e).__name__}: {e}"
            result["wall_time_s"] = time.time() - t0_total
            print(json.dumps(result))
            return 1

    if epochs.shape[0] == 0:
        result["status"] = "zero_epochs"
        result["wall_time_s"] = time.time() - t0_total
        print(json.dumps(result))
        return 0

    # Extract features
    try:
        n = epochs.shape[0]
        max_ep = args.max_epochs
        if max_ep is not None and n > max_ep:
            rng = np.random.default_rng(args.subsample_seed + si)
            idx = np.sort(rng.choice(n, size=max_ep, replace=False))
            epochs = epochs[idx]
            n_eff = max_ep
        else:
            n_eff = n

        f1s: list[float] = []
        f2s: list[float] = []
        f3s: list[float] = []

        for epoch in epochs:
            traj = ossm_mod.forward_pass(np.ascontiguousarray(epoch), si)
            f1s.append(feat_mod.feature_f1_from_trajectory(traj))
            f2s.append(feat_mod.feature_f2_from_trajectory(traj))
            f3s.append(feat_fast.feature_f3_fast(traj))

        result["F1"] = statistics.median(f1s)
        result["F2"] = statistics.median(f2s)
        result["F3"] = statistics.median(f3s)
        result["n_epochs"] = int(n_eff)
        result["status"] = "success"
    except Exception as e:
        result["status"] = f"feature_error: {type(e).__name__}: {e}"

    result["wall_time_s"] = time.time() - t0_total
    print(json.dumps(result))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
