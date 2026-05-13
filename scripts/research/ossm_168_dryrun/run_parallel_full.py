#!/usr/bin/env python3
"""Parallel execution of OSSM-168 v3 confirmatory analysis on all 220 LEMON subjects.

Uses multiprocessing to utilize all CPU cores for preprocessing and feature extraction.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
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
from scripts.research.ossm_168_dryrun.run_lemon_confirmatory import (
    SubjectFeatures,
    _load_endpoints,
    _iter_subjects,
    _merge_with_endpoints,
    _run_hypothesis_tests,
    _write_features,
    _log_endpoint_coverage,
    HYPOTHESES,
    SECONDARY,
    BOOTSTRAP_SEED,
)


def preprocess_or_load(
    subject: str,
    raw_root: Path,
    cache_dir: Path,
) -> tuple[np.ndarray | None, dict, str]:
    """Return (epochs_tensor, provenance, status). Caches .npy + .json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{subject}_epochs.npy"
    meta = cache_dir / f"{subject}_meta.json"

    if npy.is_file():
        try:
            epochs = np.load(npy)
            if meta.is_file():
                prov = json.loads(meta.read_text())
            else:
                prov = {
                    "subject": subject,
                    "n_retained_epochs": int(epochs.shape[0]),
                    "n_total_epochs": int(epochs.shape[0]),
                    "source": "cached_npy_no_meta",
                }
                meta.write_text(json.dumps(prov, indent=2))
            return epochs, prov, "cached"
        except Exception as e:
            return None, {}, f"cache_load_error: {e}"

    t0 = time.time()
    try:
        res = lp.preprocess_subject(str(raw_root), subject)
        epochs = np.asarray(res.epochs, dtype=np.float64)
        prov = {
            "subject": subject,
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
        return epochs, prov, "processed"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return None, {"error": error_msg, "wall_time_s": time.time() - t0}, f"error: {error_msg}"


def features_for_subject(
    subject_index: int,
    epochs: np.ndarray,
    max_epochs: int | None = None,
    subsample_seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return (F1_median, F2_median, F3_median) across retained epochs."""
    import statistics

    n = epochs.shape[0]
    if max_epochs is not None and n > max_epochs:
        rng = np.random.default_rng(subsample_seed + subject_index)
        idx = np.sort(rng.choice(n, size=max_epochs, replace=False))
        epochs = epochs[idx]

    f1s: list[float] = []
    f2s: list[float] = []
    f3s: list[float] = []

    for epoch in epochs:
        traj = ossm_mod.forward_pass(np.ascontiguousarray(epoch), subject_index)
        f1s.append(feat_mod.feature_f1_from_trajectory(traj))
        f2s.append(feat_mod.feature_f2_from_trajectory(traj))
        f3s.append(feat_fast.feature_f3_fast(traj))

    return (statistics.median(f1s), statistics.median(f2s), statistics.median(f3s))


def process_single_subject(
    args_tuple: tuple,
) -> tuple[str, SubjectFeatures | None, dict, str]:
    """Process one subject: preprocess + features. Returns (subject_id, features, prov, status)."""
    sid, si, raw_root, cache_dir, max_epochs = args_tuple

    t0 = time.time()

    # Preprocess or load from cache
    epochs, prov, status = preprocess_or_load(sid, raw_root, cache_dir)

    if epochs is None:
        return sid, None, prov, status

    if epochs.shape[0] == 0:
        return sid, None, prov, "zero_epochs"

    try:
        # Extract features
        F1, F2, F3 = features_for_subject(si, epochs, max_epochs=max_epochs)
        n_eff = min(epochs.shape[0], max_epochs) if max_epochs else epochs.shape[0]

        rec = SubjectFeatures(
            subject_id=sid,
            n_epochs=int(n_eff),
            F1=F1,
            F2=F2,
            F3=F3,
        )

        wall_time = time.time() - t0
        prov["total_wall_time_s"] = wall_time

        return sid, rec, prov, f"success ({status})"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        prov["feature_error"] = error_msg
        return sid, None, prov, f"feature_error: {error_msg}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subjects-list", type=Path, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: CPU count - 1)")
    ap.add_argument("--skip-existing-features", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load endpoints and subjects
    endpoints = _load_endpoints(args.endpoints)
    subjects = _iter_subjects(args.raw_root, args.subjects_list)

    print(f"[parallel] {len(subjects)} subjects total")
    print(f"[parallel] {len(endpoints)} subjects with endpoints")

    # Check for existing features
    feat_csv = args.out_dir / "features.csv"
    already: dict[str, SubjectFeatures] = {}
    if args.skip_existing_features and feat_csv.is_file():
        with feat_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                already[row["subject_id"]] = SubjectFeatures(
                    subject_id=row["subject_id"],
                    n_epochs=int(row["n_epochs"]),
                    F1=float(row["F1"]),
                    F2=float(row["F2"]),
                    F3=float(row["F3"]),
                )
        print(f"[parallel] {len(already)} subjects already in features.csv")

    # Filter to subjects needing processing
    subjects_to_process = [s for s in subjects if s not in already]
    print(f"[parallel] {len(subjects_to_process)} subjects to process")

    if not subjects_to_process:
        print("[parallel] All subjects already processed.")
        feat_rows = list(already.values())
    else:
        # Build subject index map for deterministic seeds
        sub_index_map = {sid: i for i, sid in enumerate(sorted(endpoints.keys()))}

        # Prepare work items
        work_items = [
            (sid, sub_index_map.get(sid, 0), args.raw_root, args.cache_dir, args.max_epochs)
            for sid in subjects_to_process
            if sid in endpoints  # Only process subjects with endpoint data
        ]

        print(f"[parallel] Processing {len(work_items)} subjects with endpoint data")

        # Determine worker count
        n_workers = args.workers or max(1, os.cpu_count() - 1)
        print(f"[parallel] Using {n_workers} workers")

        # Process in parallel
        feat_rows: list[SubjectFeatures] = list(already.values())
        results_log: list[dict] = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_subject, item): item[0] for item in work_items}

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                sid = futures[future]
                completed += 1

                try:
                    sid_out, rec, prov, status = future.result()

                    log_entry = {
                        "subject": sid_out,
                        "status": status,
                        "wall_time_s": prov.get("total_wall_time_s", prov.get("wall_time_s", 0)),
                    }
                    if "error" in prov:
                        log_entry["error"] = prov["error"]
                    if "n_retained_epochs" in prov:
                        log_entry["n_epochs"] = prov["n_retained_epochs"]

                    results_log.append(log_entry)

                    if rec is not None:
                        feat_rows.append(rec)
                        print(f"[{completed}/{len(work_items)}] {sid_out}: "
                              f"n_ep={rec.n_epochs} F1={rec.F1:.4f} F2={rec.F2:.4f} F3={rec.F3:.4f} "
                              f"({status})")
                    else:
                        print(f"[{completed}/{len(work_items)}] {sid_out}: FAILED ({status})")

                except Exception as e:
                    print(f"[{completed}/{len(work_items)}] {sid}: EXCEPTION {type(e).__name__}: {e}")
                    results_log.append({"subject": sid, "status": f"exception: {e}"})

                # Save incremental progress
                if completed % 10 == 0:
                    _write_features(feat_rows, feat_csv)
                    (args.out_dir / "processing_log.json").write_text(
                        json.dumps(results_log, indent=2)
                    )

        # Final save
        _write_features(feat_rows, feat_csv)
        (args.out_dir / "processing_log.json").write_text(
            json.dumps(results_log, indent=2)
        )

    # Merge with endpoints and run hypothesis tests
    if not feat_rows:
        print("[parallel] no successful subjects — aborting.", flush=True)
        return 1

    merged, merged_path = _merge_with_endpoints(feat_rows, endpoints, args.out_dir)
    print(f"[parallel] merged {len(merged)} subjects -> {merged_path}")
    _log_endpoint_coverage(merged)

    results = _run_hypothesis_tests(merged)
    results_path = args.out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[parallel] wrote {results_path}")
    print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
