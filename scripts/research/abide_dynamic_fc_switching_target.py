#!/usr/bin/env python3
"""Build ABIDE dynamic-FC state-switching targets.

This is a target-builder and audit surface, not a model evaluation. It turns
cached ABIDE CC200 ROI time series into fold-local dynamic functional
connectivity states and switching events. PCA and k-means are fit on training
windows only inside each split, then applied to held-out windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "neurodyn.abide_dynamic_fc_switching_target.v1"
CLAIM_BOUNDARY = (
    "Target-builder and readiness audit only; no diagnostic, biomarker, "
    "mechanism, treatment, ASD-detection, or O-SSM superiority claim."
)

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-999", "-9999"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotypic-csv",
        type=Path,
        default=None,
        help="ABIDE phenotypic CSV. Defaults to <cache-dir>/phenotypic.csv.",
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=None,
        help="Directory containing *_rois_cc200.1D files. Defaults to <cache-dir>.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/research/abide"),
        help="Fallback directory containing phenotypic.csv and ROI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/research/abide_dynamic_fc_switching"),
        help="Output artifact directory.",
    )
    parser.add_argument("--window-tr", type=int, default=30)
    parser.add_argument("--step-tr", type=int, default=3)
    parser.add_argument("--min-timepoints", type=int, default=90)
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sensitivity-k", default="3,5,6,7")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument(
        "--split-policy",
        choices=["leave_one_site_out", "stratified_smoke"],
        default="leave_one_site_out",
    )
    parser.add_argument(
        "--smoke-test-frac",
        type=float,
        default=0.25,
        help="Held-out subject fraction for --split-policy=stratified_smoke.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Optional deterministic cap for local smoke runs.",
    )
    parser.add_argument(
        "--roi-limit",
        type=int,
        default=0,
        help="Optional leading-ROI cap for smoke tests; 0 uses all columns.",
    )
    parser.add_argument("--min-subjects", type=int, default=50)
    parser.add_argument("--min-sites", type=int, default=5)
    parser.add_argument("--min-state-occupancy-frac", type=float, default=0.05)
    parser.add_argument("--min-switch-event-frac", type=float, default=0.02)
    parser.add_argument("--max-switch-event-frac", type=float, default=0.80)
    parser.add_argument("--max-zero-switch-subject-frac", type=float, default=0.25)
    parser.add_argument("--max-low-window-subject-frac", type=float, default=0.20)
    parser.add_argument("--max-window-switch-abs-corr", type=float, default=0.50)
    return parser.parse_args()


def is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in MISSING_TOKENS


def label_from_dx(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if value == "1":
        return "ASD"
    if value == "2":
        return "CONTROL"
    return None


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def read_pheno(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_timeseries(path: Path, roi_limit: int = 0) -> np.ndarray | None:
    rows: list[list[float]] = []
    width_counts: Counter[int] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [safe_float(value) for value in line.split()]
            if any(value is None for value in values):
                continue
            row = [float(value) for value in values if value is not None]
            if roi_limit > 0:
                row = row[:roi_limit]
            if row:
                width_counts[len(row)] += 1
                rows.append(row)
    if not rows:
        return None
    target_width = width_counts.most_common(1)[0][0]
    clean: list[list[float]] = []
    for row in rows:
        if len(row) >= target_width:
            clean.append(row[:target_width])
        else:
            clean.append(row + [0.0] * (target_width - len(row)))
    arr = np.asarray(clean, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def window_slices(n_timepoints: int, window_tr: int, step_tr: int) -> list[tuple[int, int]]:
    if window_tr <= 1 or step_tr <= 0:
        raise ValueError("window_tr must be >1 and step_tr must be positive")
    if n_timepoints < window_tr:
        return []
    return [(start, start + window_tr) for start in range(0, n_timepoints - window_tr + 1, step_tr)]


def window_fc_vectors(ts: np.ndarray, slices: list[tuple[int, int]]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    tri = np.triu_indices(ts.shape[1], k=1)
    for start, end in slices:
        window = ts[start:end, :]
        mu = window.mean(axis=0, keepdims=True)
        sigma = window.std(axis=0, keepdims=True, ddof=1)
        sigma = np.where(sigma > 1.0e-8, sigma, 1.0)
        z = (window - mu) / sigma
        corr = (z.T @ z) / max(1, z.shape[0] - 1)
        corr = np.clip(corr, -0.999999, 0.999999)
        fish = np.arctanh(corr)
        vectors.append(fish[tri].astype(np.float64))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(vectors)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) * (x - mx) for x in xs)
    vy = sum((y - my) * (y - my) for y in ys)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(vx * vy)


def fit_pca(train_x: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    mean = train_x.mean(axis=0)
    centered = train_x - mean
    max_components = max(1, min(components, centered.shape[0] - 1, centered.shape[1]))
    if centered.shape[0] < centered.shape[1]:
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        nonzero = eigvals > 1.0e-12
        eigvals = eigvals[nonzero]
        eigvecs = eigvecs[:, nonzero]
        take = min(max_components, eigvals.shape[0])
        if take <= 0:
            basis = np.zeros((centered.shape[1], 1), dtype=np.float64)
            variances = np.zeros(1, dtype=np.float64)
        else:
            svals = np.sqrt(eigvals[:take])
            basis = centered.T @ (eigvecs[:, :take] / svals[None, :])
            variances = eigvals / max(1, centered.shape[0] - 1)
    else:
        cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        basis = eigvecs[:, :max_components]
        variances = eigvals
    total = float(variances.sum())
    explained = [float(value / total) if total > 0 else 0.0 for value in variances[: basis.shape[1]]]
    return mean, basis, explained


def project_pca(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mean) @ basis


def fit_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int = 100) -> np.ndarray:
    if k <= 1:
        raise ValueError("k must be greater than one")
    if x.shape[0] < k:
        raise ValueError(f"cannot fit k={k} with only {x.shape[0]} rows")
    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, x.shape[0]))
    centers = [x[first_idx]]
    min_dist = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(min_dist.sum())
        if total <= 1.0e-12:
            idx = len(centers) % x.shape[0]
        else:
            idx = int(rng.choice(x.shape[0], p=min_dist / total))
        centers.append(x[idx])
        min_dist = np.minimum(min_dist, np.sum((x - centers[-1]) ** 2, axis=1))
    centroids = np.vstack(centers)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        new_labels = assign_kmeans(x, centroids)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            mask = labels == cluster
            if mask.any():
                centroids[cluster] = x[mask].mean(axis=0)
            else:
                centroids[cluster] = x[int(rng.integers(0, x.shape[0]))]
    return centroids


def assign_kmeans(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist, axis=1).astype(np.int64)


def build_subjects(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pheno_path = args.phenotypic_csv or (args.cache_dir / "phenotypic.csv")
    roi_dir = args.roi_dir or args.cache_dir
    if not pheno_path.exists():
        raise SystemExit(f"missing phenotypic CSV: {pheno_path}")
    if not roi_dir.exists():
        raise SystemExit(f"missing ROI directory: {roi_dir}")

    specs: list[dict[str, str]] = []
    skipped = Counter()
    for row in read_pheno(pheno_path):
        subject_id = (row.get("SUB_ID") or row.get("subject_id") or "").strip()
        file_id = (row.get("FILE_ID") or "").strip()
        site = (row.get("SITE_ID") or row.get("site") or "").strip()
        label = label_from_dx(row.get("DX_GROUP"))
        if is_missing(file_id) or file_id == "no_filename":
            skipped["missing_file_id"] += 1
            continue
        if not subject_id or not site or label is None:
            skipped["missing_metadata"] += 1
            continue
        ts_path = roi_dir / f"{file_id}_rois_cc200.1D"
        if not ts_path.exists():
            skipped["missing_roi"] += 1
            continue
        specs.append(
            {
                "subject_id": subject_id,
                "file_id": file_id,
                "site": site,
                "label": label,
                "ts_path": str(ts_path),
            }
        )

    if args.max_subjects > 0 and len(specs) > args.max_subjects:
        specs = select_balanced_specs(specs, args.max_subjects)

    subjects: list[dict[str, Any]] = []
    for spec in specs:
        ts_path = Path(spec["ts_path"])
        ts = load_timeseries(ts_path, roi_limit=args.roi_limit)
        if ts is None:
            skipped["bad_roi"] += 1
            continue
        if ts.shape[0] < args.min_timepoints:
            skipped["short_time_series"] += 1
            continue
        slices = window_slices(ts.shape[0], args.window_tr, args.step_tr)
        if len(slices) < args.min_windows:
            skipped["few_windows"] += 1
            continue
        fc = window_fc_vectors(ts, slices)
        if fc.shape[0] != len(slices) or fc.shape[1] <= 0:
            skipped["bad_fc"] += 1
            continue
        subjects.append(
            {
                "subject_id": spec["subject_id"],
                "file_id": spec["file_id"],
                "site": spec["site"],
                "label": spec["label"],
                "n_timepoints": int(ts.shape[0]),
                "roi_dim": int(ts.shape[1]),
                "slices": slices,
                "fc": fc,
            }
        )

    source = {
        "phenotypic_csv": str(pheno_path),
        "roi_dir": str(roi_dir),
        "skipped": dict(sorted(skipped.items())),
    }
    return subjects, source


def select_balanced_specs(specs: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Deterministically cap subjects without collapsing to one site/label."""
    by_bucket: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for spec in specs:
        by_bucket[(spec["site"], spec["label"])].append(spec)
    buckets = sorted(by_bucket)
    selected: list[dict[str, str]] = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for bucket in buckets:
            rows = by_bucket[bucket]
            if index < len(rows):
                selected.append(rows[index])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return selected


def make_folds(subjects: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.split_policy == "leave_one_site_out":
        return [
            {
                "fold_id": f"loso_{site}",
                "holdout_site": site,
                "test_subject_ids": {s["subject_id"] for s in subjects if s["site"] == site},
            }
            for site in sorted({s["site"] for s in subjects})
        ]

    rng = np.random.default_rng(args.seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subject in subjects:
        by_group[subject["label"]].append(subject)
    test_ids: set[str] = set()
    for group_subjects in by_group.values():
        shuffled = list(group_subjects)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * args.smoke_test_frac)))
        if n_test >= len(shuffled) and len(shuffled) > 1:
            n_test = len(shuffled) - 1
        test_ids.update(subject["subject_id"] for subject in shuffled[:n_test])
    return [{"fold_id": "stratified_smoke_0", "holdout_site": "STRATIFIED_SMOKE", "test_subject_ids": test_ids}]


def subject_window_rows(subjects: list[dict[str, Any]], fold: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    test_ids = fold["test_subject_ids"]
    for subject in subjects:
        split = "test" if subject["subject_id"] in test_ids else "train"
        for window_idx, (start, end) in enumerate(subject["slices"]):
            vectors.append(subject["fc"][window_idx])
            rows.append(
                {
                    "subject_id": subject["subject_id"],
                    "file_id": subject["file_id"],
                    "site": subject["site"],
                    "label": subject["label"],
                    "split": split,
                    "window_index": window_idx,
                    "window_start": start,
                    "window_end": end,
                    "n_timepoints": subject["n_timepoints"],
                    "roi_dim": subject["roi_dim"],
                    "window_count": len(subject["slices"]),
                }
            )
    return np.vstack(vectors), rows


def summarize_fold(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    train_state_counts = Counter(row["state"] for row in rows if row["split"] == "train")
    test_state_counts = Counter(row["state"] for row in rows if row["split"] == "test")
    test_switch_values = [
        int(row["switch_event"])
        for row in rows
        if row["split"] == "test" and str(row["switch_event"]) in {"0", "1"}
    ]
    train_total = sum(train_state_counts.values())
    min_train_occ = (
        min(train_state_counts.get(state, 0) / train_total for state in range(k))
        if train_total
        else 0.0
    )
    return {
        "split_window_counts": dict(sorted(split_counts.items())),
        "train_state_counts": dict(sorted(train_state_counts.items())),
        "test_state_counts": dict(sorted(test_state_counts.items())),
        "min_train_state_occupancy_frac": min_train_occ,
        "test_switch_event_count": sum(test_switch_values),
        "test_switch_event_denominator": len(test_switch_values),
        "test_switch_event_frac": sum(test_switch_values) / len(test_switch_values)
        if test_switch_values
        else 0.0,
    }


def assign_states_for_k(
    subjects: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    k: int,
    pca_components: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        x, rows = subject_window_rows(subjects, fold)
        train_mask = np.array([row["split"] == "train" for row in rows], dtype=bool)
        if int(train_mask.sum()) < max(k, 3):
            continue
        mean, basis, explained = fit_pca(x[train_mask], pca_components)
        z = project_pca(x, mean, basis)
        centroids = fit_kmeans(z[train_mask], k=k, seed=seed + 7919 * fold_idx + k)
        states = assign_kmeans(z, centroids)
        previous_by_subject: dict[str, int] = {}
        for row, state in zip(rows, states.tolist(), strict=True):
            subject_id = row["subject_id"]
            prev = previous_by_subject.get(subject_id)
            switch_event: str | int = "" if prev is None else int(state != prev)
            previous_by_subject[subject_id] = state
            all_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "holdout_site": fold["holdout_site"],
                    "k": k,
                    "pca_components": int(basis.shape[1]),
                    "pca_explained_variance": f"{sum(explained):.10f}",
                    **row,
                    "state": int(state),
                    "switch_event": switch_event,
                }
            )
        fold_rows = [row for row in all_rows if row["fold_id"] == fold["fold_id"] and row["k"] == k]
        fold_summaries.append(
            {
                "fold_id": fold["fold_id"],
                "holdout_site": fold["holdout_site"],
                "k": k,
                "train_subject_count": len({row["subject_id"] for row in fold_rows if row["split"] == "train"}),
                "test_subject_count": len({row["subject_id"] for row in fold_rows if row["split"] == "test"}),
                "pca_components": int(basis.shape[1]),
                "pca_explained_variance": sum(explained),
                **summarize_fold(fold_rows, k),
            }
        )
    return all_rows, fold_summaries


def summarize_subjects(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        if row["split"] == "test":
            grouped[(row["fold_id"], row["subject_id"], int(row["k"]), row["holdout_site"])].append(row)
    out: list[dict[str, Any]] = []
    for (fold_id, subject_id, k, holdout_site), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["window_index"]))
        switches = [int(row["switch_event"]) for row in rows if str(row["switch_event"]) in {"0", "1"}]
        state_counts = Counter(int(row["state"]) for row in rows)
        dwell: list[int] = []
        current_state = None
        current_len = 0
        for row in rows:
            state = int(row["state"])
            if current_state is None or state == current_state:
                current_state = state
                current_len += 1
            else:
                dwell.append(current_len)
                current_state = state
                current_len = 1
        if current_len:
            dwell.append(current_len)
        total = len(rows)
        occupancy = {f"state_{state}_occupancy": state_counts.get(state, 0) / total for state in range(k)}
        out.append(
            {
                "fold_id": fold_id,
                "holdout_site": holdout_site,
                "k": k,
                "subject_id": subject_id,
                "file_id": rows[0]["file_id"],
                "site": rows[0]["site"],
                "label": rows[0]["label"],
                "window_count": total,
                "n_timepoints": rows[0]["n_timepoints"],
                "roi_dim": rows[0]["roi_dim"],
                "switch_count": sum(switches),
                "switch_denominator": len(switches),
                "switching_rate": sum(switches) / len(switches) if switches else 0.0,
                "zero_switch": int(sum(switches) == 0),
                "mean_dwell": sum(dwell) / len(dwell) if dwell else 0.0,
                "max_dwell": max(dwell) if dwell else 0,
                **occupancy,
            }
        )
    return out


def audit_payload(
    args: argparse.Namespace,
    subjects: list[dict[str, Any]],
    source: dict[str, Any],
    window_rows: list[dict[str, Any]],
    subject_summary: list[dict[str, Any]],
    fold_summaries: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> dict[str, Any]:
    site_counts = Counter(subject["site"] for subject in subjects)
    label_counts = Counter(subject["label"] for subject in subjects)
    windows = [len(subject["slices"]) for subject in subjects]
    low_window_count = sum(1 for value in windows if value < args.min_windows)
    primary_subject_rows = [row for row in subject_summary if int(row["k"]) == args.k]
    switch_rates = [float(row["switching_rate"]) for row in primary_subject_rows]
    window_counts = [float(row["window_count"]) for row in primary_subject_rows]
    zero_switch_frac = (
        sum(int(row["zero_switch"]) for row in primary_subject_rows) / len(primary_subject_rows)
        if primary_subject_rows
        else 1.0
    )
    switch_values = [
        int(row["switch_event"])
        for row in window_rows
        if int(row["k"]) == args.k and row["split"] == "test" and str(row["switch_event"]) in {"0", "1"}
    ]
    switch_frac = sum(switch_values) / len(switch_values) if switch_values else 0.0
    window_switch_corr = pearson(window_counts, switch_rates)
    min_train_occ = min(
        (float(summary["min_train_state_occupancy_frac"]) for summary in fold_summaries if int(summary["k"]) == args.k),
        default=0.0,
    )
    failures: list[str] = []
    if len(subjects) < args.min_subjects:
        failures.append("subject_count_below_minimum")
    if len(site_counts) < args.min_sites:
        failures.append("site_count_below_minimum")
    if windows and low_window_count / len(windows) > args.max_low_window_subject_frac:
        failures.append("low_window_subject_fraction_above_threshold")
    if min_train_occ < args.min_state_occupancy_frac:
        failures.append("state_occupancy_below_minimum")
    if switch_frac < args.min_switch_event_frac:
        failures.append("switch_event_prevalence_below_minimum")
    if switch_frac > args.max_switch_event_frac:
        failures.append("switch_event_prevalence_above_maximum")
    if zero_switch_frac > args.max_zero_switch_subject_frac:
        failures.append("zero_switch_subject_fraction_above_threshold")
    if window_switch_corr is not None and abs(window_switch_corr) > args.max_window_switch_abs_corr:
        failures.append("switching_rate_window_count_correlation_above_threshold")

    return {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "fail" if failures else "pass",
        "failures": failures,
        "source": source,
        "parameters": {
            "window_tr": args.window_tr,
            "step_tr": args.step_tr,
            "min_timepoints": args.min_timepoints,
            "min_windows": args.min_windows,
            "pca_components": args.pca_components,
            "k": args.k,
            "sensitivity_k": args.sensitivity_k,
            "split_policy": args.split_policy,
            "seed": args.seed,
            "roi_limit": args.roi_limit,
        },
        "thresholds": {
            "min_subjects": args.min_subjects,
            "min_sites": args.min_sites,
            "min_state_occupancy_frac": args.min_state_occupancy_frac,
            "min_switch_event_frac": args.min_switch_event_frac,
            "max_switch_event_frac": args.max_switch_event_frac,
            "max_zero_switch_subject_frac": args.max_zero_switch_subject_frac,
            "max_low_window_subject_frac": args.max_low_window_subject_frac,
            "max_window_switch_abs_corr": args.max_window_switch_abs_corr,
        },
        "subject_count": len(subjects),
        "site_count": len(site_counts),
        "site_counts": dict(sorted(site_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "window_count_min": min(windows) if windows else 0,
        "window_count_max": max(windows) if windows else 0,
        "window_count_mean": sum(windows) / len(windows) if windows else 0.0,
        "low_window_subject_count": low_window_count,
        "primary_k": args.k,
        "primary_test_switch_event_frac": switch_frac,
        "primary_test_switch_event_count": sum(switch_values),
        "primary_test_switch_event_denominator": len(switch_values),
        "primary_zero_switch_subject_frac": zero_switch_frac,
        "primary_min_train_state_occupancy_frac": min_train_occ,
        "primary_switching_rate_window_count_corr": window_switch_corr,
        "fold_summaries": fold_summaries,
        "sensitivity": sensitivity,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# ABIDE Dynamic-FC Switching Target Audit",
        "",
        f"Schema: `{audit['schema']}`",
        "",
        f"Status: `{audit['status']}`",
        "",
        f"Claim boundary: {audit['claim_boundary']}",
        "",
        "## Summary",
        "",
        f"- Subjects: {audit['subject_count']}",
        f"- Sites: {audit['site_count']}",
        f"- Primary k: {audit['primary_k']}",
        f"- Test switch-event fraction: {audit['primary_test_switch_event_frac']:.6f}",
        f"- Zero-switch subject fraction: {audit['primary_zero_switch_subject_frac']:.6f}",
        f"- Min train state occupancy fraction: {audit['primary_min_train_state_occupancy_frac']:.6f}",
        f"- Switching-rate/window-count correlation: {audit['primary_switching_rate_window_count_corr']}",
        "",
        "## Failures",
        "",
    ]
    if audit["failures"]:
        lines.extend(f"- `{failure}`" for failure in audit["failures"])
    else:
        lines.append("- none")
    lines.extend(["", "## Site Counts", ""])
    for site, count in audit["site_counts"].items():
        lines.append(f"- `{site}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    subjects, source = build_subjects(args)
    if not subjects:
        raise SystemExit("no usable subjects for dynamic-FC switching target")
    folds = make_folds(subjects, args)
    sensitivity_ks = [int(value) for value in args.sensitivity_k.split(",") if value.strip()]
    all_ks = [args.k] + [value for value in sensitivity_ks if value != args.k]

    rows_by_k: dict[int, list[dict[str, Any]]] = {}
    fold_summaries_all: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for k in all_ks:
        rows, fold_summaries = assign_states_for_k(subjects, folds, k, args.pca_components, args.seed)
        rows_by_k[k] = rows
        fold_summaries_all.extend(fold_summaries)
        switch_values = [
            int(row["switch_event"])
            for row in rows
            if row["split"] == "test" and str(row["switch_event"]) in {"0", "1"}
        ]
        min_occ = min((float(item["min_train_state_occupancy_frac"]) for item in fold_summaries), default=0.0)
        sensitivity.append(
            {
                "k": k,
                "fold_count": len(fold_summaries),
                "test_switch_event_frac": sum(switch_values) / len(switch_values) if switch_values else 0.0,
                "test_switch_event_denominator": len(switch_values),
                "min_train_state_occupancy_frac": min_occ,
            }
        )

    primary_rows = rows_by_k[args.k]
    subject_summary = summarize_subjects(primary_rows)
    audit = audit_payload(args, subjects, source, primary_rows, subject_summary, fold_summaries_all, sensitivity)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    window_fields = [
        "fold_id",
        "holdout_site",
        "k",
        "subject_id",
        "file_id",
        "site",
        "label",
        "split",
        "window_index",
        "window_start",
        "window_end",
        "state",
        "switch_event",
        "window_count",
        "n_timepoints",
        "roi_dim",
        "pca_components",
        "pca_explained_variance",
    ]
    summary_fields = [
        "fold_id",
        "holdout_site",
        "k",
        "subject_id",
        "file_id",
        "site",
        "label",
        "window_count",
        "n_timepoints",
        "roi_dim",
        "switch_count",
        "switch_denominator",
        "switching_rate",
        "zero_switch",
        "mean_dwell",
        "max_dwell",
    ] + [f"state_{state}_occupancy" for state in range(args.k)]
    write_tsv(out / "dynamic_fc_window_table.tsv", primary_rows, window_fields)
    write_tsv(out / "dynamic_fc_subject_summary.tsv", subject_summary, summary_fields)
    with (out / "dynamic_fc_target_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(out / "dynamic_fc_target_audit.md", audit)

    print(json.dumps({"status": audit["status"], "failures": audit["failures"], "output_dir": str(out)}, sort_keys=True))
    if audit["failures"]:
        raise SystemExit("ABIDE dynamic-FC target audit failed: " + ",".join(audit["failures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
