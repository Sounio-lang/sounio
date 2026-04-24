#!/usr/bin/env python3
"""Sequential execution of OSSM-168 v3 confirmatory analysis on all 220 LEMON subjects.

Processes subjects one by one with incremental saves (safe for long runs).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    BOOTSTRAP_SEED,
)
import statistics


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subjects-list", type=Path, default=None)
    ap.add_argument("--max-epochs", type=int, default=50,
                    help="subsample epochs per subject (default: 50, None for all)")
    ap.add_argument("--skip-existing-features", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load endpoints and subjects
    endpoints = _load_endpoints(args.endpoints)
    subjects = _iter_subjects(args.raw_root, args.subjects_list)

    print(f"[sequential] {len(subjects)} subjects total")
    print(f"[sequential] {len(endpoints)} subjects with endpoints")

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
        print(f"[sequential] {len(already)} subjects already in features.csv")

    # Build subject index map for deterministic seeds
    sub_index_map = {sid: i for i, sid in enumerate(sorted(endpoints.keys()))}

    # Filter to subjects needing processing with endpoint data
    subjects_to_process = [
        s for s in subjects
        if s not in already and s in endpoints
    ]
    print(f"[sequential] {len(subjects_to_process)} subjects to process")

    feat_rows: list[SubjectFeatures] = list(already.values())
    results_log: list[dict] = []

    for pos, sid in enumerate(subjects_to_process, 1):
        si = sub_index_map.get(sid, 0)
        t0_total = time.time()

        # Preprocess or load
        epochs, prov, status = preprocess_or_load(sid, args.raw_root, args.cache_dir)

        if epochs is None:
            results_log.append({
                "subject": sid,
                "status": status,
                "wall_time_s": time.time() - t0_total,
            })
            print(f"[{pos}/{len(subjects_to_process)}] {sid}: FAILED ({status})")
            continue

        if epochs.shape[0] == 0:
            results_log.append({
                "subject": sid,
                "status": "zero_epochs",
                "wall_time_s": time.time() - t0_total,
            })
            print(f"[{pos}/{len(subjects_to_process)}] {sid}: 0 epochs retained")
            continue

        try:
            # Extract features
            F1, F2, F3 = features_for_subject(si, epochs, max_epochs=args.max_epochs)
            n_eff = min(epochs.shape[0], args.max_epochs) if args.max_epochs else epochs.shape[0]

            rec = SubjectFeatures(
                subject_id=sid,
                n_epochs=int(n_eff),
                F1=F1,
                F2=F2,
                F3=F3,
            )
            feat_rows.append(rec)

            wall_time = time.time() - t0_total
            results_log.append({
                "subject": sid,
                "status": f"success ({status})",
                "n_epochs": int(n_eff),
                "wall_time_s": wall_time,
            })

            print(f"[{pos}/{len(subjects_to_process)}] {sid}: "
                  f"n_ep={rec.n_epochs} F1={rec.F1:.4f} F2={rec.F2:.4f} F3={rec.F3:.4f} "
                  f"wall={wall_time:.1f}s ({status})")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            wall_time = time.time() - t0_total
            results_log.append({
                "subject": sid,
                "status": f"feature_error: {error_msg}",
                "wall_time_s": wall_time,
            })
            print(f"[{pos}/{len(subjects_to_process)}] {sid}: FEATURE ERROR {error_msg}")

        # Save incremental progress every 5 subjects
        if pos % 5 == 0:
            _write_features(feat_rows, feat_csv)
            (args.out_dir / "processing_log.json").write_text(
                json.dumps(results_log, indent=2)
            )

    # Final save
    _write_features(feat_rows, feat_csv)
    (args.out_dir / "processing_log.json").write_text(
        json.dumps(results_log, indent=2)
    )

    print(f"\n[sequential] Processed {len(feat_rows)} subjects successfully")

    # Merge with endpoints and run hypothesis tests
    if not feat_rows:
        print("[sequential] no successful subjects — aborting.", flush=True)
        return 1

    merged, merged_path = _merge_with_endpoints(feat_rows, endpoints, args.out_dir)
    print(f"[sequential] merged {len(merged)} subjects -> {merged_path}")
    _log_endpoint_coverage(merged)

    results = _run_hypothesis_tests(merged)
    results_path = args.out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[sequential] wrote {results_path}")
    print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
