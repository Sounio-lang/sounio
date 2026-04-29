#!/usr/bin/env python3
"""Preprocess LEMON subjects and save to cache — fast parallel version.

This script ONLY preprocesses (no feature extraction), allowing efficient
parallelization with separate subprocesses.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.research.ossm_168_dryrun import lemon_preprocess as lp


def preprocess_single(sid: str, raw_root: Path, cache_dir: Path) -> dict:
    """Preprocess one subject and save to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{sid}_epochs.npy"
    meta = cache_dir / f"{sid}_meta.json"

    if npy.is_file() and meta.is_file():
        try:
            prov = json.loads(meta.read_text())
            return {"subject": sid, "status": "cached", "n_epochs": prov.get("n_retained_epochs", 0)}
        except Exception:
            pass  # Reprocess if cache is corrupted

    t0 = time.time()
    try:
        res = lp.preprocess_subject(str(raw_root), sid)
        epochs = np.asarray(res.epochs, dtype=np.float64)

        prov = {
            "subject": sid,
            "sfreq": res.sfreq,
            "n_total_epochs": res.n_total_epochs,
            "n_retained_epochs": res.n_retained_epochs,
            "retention_rate": res.retention_rate(),
            "n_ica_rejected": res.n_ica_rejected,
            "n_ica_total": res.n_ica_total,
            "channels_used_n": len(res.channel_names_used),
            "wall_time_s": time.time() - t0,
        }

        np.save(npy, epochs)
        meta.write_text(json.dumps(prov, indent=2))

        return {
            "subject": sid,
            "status": "processed",
            "n_epochs": res.n_retained_epochs,
            "wall_time_s": time.time() - t0,
        }
    except Exception as e:
        return {
            "subject": sid,
            "status": f"error: {type(e).__name__}: {e}",
            "wall_time_s": time.time() - t0,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--subjects-list", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects_list.read_text().splitlines() if s.strip()]
    print(f"[preprocess] {len(subjects)} subjects to process")

    # Check which subjects need processing
    to_process = []
    for sid in subjects:
        npy = args.cache_dir / f"{sid}_epochs.npy"
        meta = args.cache_dir / f"{sid}_meta.json"
        if not (npy.exists() and meta.exists()):
            to_process.append(sid)

    print(f"[preprocess] {len(to_process)} subjects need preprocessing")

    if not to_process:
        print("[preprocess] All subjects already cached.")
        return 0

    # Process in parallel using multiprocessing (no MNE pickling issues as we spawn fresh processes)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    t0_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(preprocess_single, sid, args.raw_root, args.cache_dir): sid for sid in to_process}

        for i, future in enumerate(as_completed(futures), 1):
            sid = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result["status"]
                n_ep = result.get("n_epochs", 0)
                wall = result.get("wall_time_s", 0)
                print(f"[{i}/{len(to_process)}] {sid}: {status} n_ep={n_ep} wall={wall:.1f}s")
            except Exception as e:
                print(f"[{i}/{len(to_process)}] {sid}: FAILED {e}")
                results.append({"subject": sid, "status": f"exception: {e}"})

    elapsed = time.time() - t0_start
    successful = sum(1 for r in results if r["status"] in ("processed", "cached"))
    print(f"\n[preprocess] Completed {len(results)} subjects in {elapsed:.1f}s")
    print(f"[preprocess] Successful: {successful}/{len(results)}")

    # Save results
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
