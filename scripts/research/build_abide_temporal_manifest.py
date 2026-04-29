#!/usr/bin/env python3
"""Build a richer temporal ABIDE manifest from cached CC200 ROI time series.

The current Sounio ABIDE benchmark consumes flat subject rows, but the Python
campaign stack already supports arbitrary sequence length. This builder keeps
the benchmark-compatible `input_dim=8` contract while increasing temporal
resolution by:

- z-scoring each ROI time series per subject
- partitioning the time axis into `seq_len` windows
- pooling the 200 ROI channels into `input_dim` contiguous groups
- flattening the resulting `seq_len x input_dim` subject sequence back into
  `f0..fN` columns

That gives us a richer temporal manifest without needing to redesign the O-SSM
step algebra in the same change.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

CANONICAL_EXCLUDED_SUBJECT_IDS = {
    "51134",
    "51135",
    "51136",
    "51137",
    "51138",
    "51139",
    "51140",
    "51141",
    "51142",
    "51146",
    "51147",
    "51148",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal ABIDE manifest from cached .1D files")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/research/abide",
        help="Directory containing phenotypic.csv and cached *_rois_cc200.1D files",
    )
    parser.add_argument(
        "--output",
        default="artifacts/research/abide/abide_roi_temporal_manifest_v3.tsv",
        help="Output manifest path",
    )
    parser.add_argument("--seq-len", type=int, default=32, help="Number of temporal windows per subject")
    parser.add_argument("--input-dim", type=int, default=8, help="Number of pooled ROI groups per step")
    parser.add_argument(
        "--feature-recipe",
        choices=[
            "contiguous-window-mean",
            "global-pca-window-mean",
            "global-pca-whiten-window-delta",
        ],
        default="contiguous-window-mean",
        help="How to turn CC200 ROI time series into seq_len x input_dim subject sequences",
    )
    parser.add_argument(
        "--min-timepoints",
        type=int,
        default=32,
        help="Minimum required timepoints after loading a .1D file",
    )
    parser.add_argument(
        "--allow-extra-controls",
        action="store_true",
        help="Disable the canonical 12-control trim and emit every locally eligible subject",
    )
    parser.add_argument(
        "--basis-output",
        default="",
        help="Optional .npy output path for the learned global PCA basis",
    )
    return parser.parse_args()


def _label_from_dx_group(raw: str) -> str | None:
    value = raw.strip()
    if value == "1":
        return "ASD"
    if value == "2":
        return "CONTROL"
    return None


def _time_bins(total: int, count: int) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    for idx in range(count):
        start = (idx * total) // count
        end = ((idx + 1) * total) // count
        if end <= start:
            end = min(total, start + 1)
        bins.append((start, end))
    return bins


def _load_timeseries(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(value) for value in line.split()])
    return rows


def _sanitize_rows(rows: list[list[float]], roi_dim_hint: int | None = None) -> list[list[float]]:
    if not rows:
        return []
    counts: dict[int, int] = {}
    for row in rows:
        counts[len(row)] = counts.get(len(row), 0) + 1
    if roi_dim_hint is None:
        target_roi_dim = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    else:
        target_roi_dim = roi_dim_hint
    cleaned: list[list[float]] = []
    for row in rows:
        if len(row) >= target_roi_dim:
            cleaned.append(row[:target_roi_dim])
        else:
            cleaned.append(row + [0.0] * (target_roi_dim - len(row)))
    return cleaned


def _zscore_columns(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return []
    timepoints = len(rows)
    rois = len(rows[0])
    means = [0.0] * rois
    for row in rows:
        for roi_idx, value in enumerate(row):
            means[roi_idx] += value
    means = [value / timepoints for value in means]

    variances = [0.0] * rois
    for row in rows:
        for roi_idx, value in enumerate(row):
            delta = value - means[roi_idx]
            variances[roi_idx] += delta * delta
    stds = [math.sqrt(value / max(1, timepoints)) for value in variances]

    normalized: list[list[float]] = []
    for row in rows:
        normalized_row: list[float] = []
        for roi_idx, value in enumerate(row):
            std = stds[roi_idx]
            if std <= 1e-8 or math.isnan(std):
                normalized_row.append(0.0)
            else:
                normalized_row.append((value - means[roi_idx]) / std)
        normalized.append(normalized_row)
    return normalized


def _window_pool_subject(rows: list[list[float]], seq_len: int, input_dim: int) -> list[list[float]]:
    normalized = _zscore_columns(rows)
    time_bins = _time_bins(len(normalized), seq_len)
    roi_bins = _time_bins(len(normalized[0]), input_dim)
    pooled: list[list[float]] = []
    for time_start, time_end in time_bins:
        step_rows = normalized[time_start:time_end]
        if not step_rows:
            step_rows = [normalized[min(len(normalized) - 1, time_start)]]
        step: list[float] = []
        for roi_start, roi_end in roi_bins:
            total = 0.0
            count = 0
            for row in step_rows:
                for value in row[roi_start:roi_end]:
                    total += value
                    count += 1
            step.append(total / max(1, count))
        pooled.append(step)
    return pooled


def _zscore_numpy(rows: list[list[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    means = arr.mean(axis=0, keepdims=True)
    stds = arr.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    norm = (arr - means) / stds
    norm[np.isnan(norm)] = 0.0
    return norm


def _fit_global_pca(subject_rows: list[list[list[float]]], input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not subject_rows:
        raise ValueError("cannot fit PCA without subject rows")
    roi_dim = len(subject_rows[0][0])
    cov = np.zeros((roi_dim, roi_dim), dtype=np.float64)
    total_steps = 0
    for rows in subject_rows:
        norm = _zscore_numpy(rows)
        cov += norm.T @ norm
        total_steps += norm.shape[0]
    cov /= max(1, total_steps)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:input_dim]
    basis = eigvecs[:, order]
    values = eigvals[order]
    for idx in range(basis.shape[1]):
        vec = basis[:, idx]
        pivot = int(np.argmax(np.abs(vec)))
        if vec[pivot] < 0:
            basis[:, idx] = -vec
    return basis, values


def _window_mean_projected(norm: np.ndarray, basis: np.ndarray, seq_len: int) -> list[list[float]]:
    projected = norm @ basis
    pooled: list[list[float]] = []
    for time_start, time_end in _time_bins(projected.shape[0], seq_len):
        window = projected[time_start:time_end]
        if window.shape[0] == 0:
            window = projected[min(projected.shape[0] - 1, time_start) : min(projected.shape[0], time_start + 1)]
        pooled.append([float(value) for value in window.mean(axis=0)])
    return pooled


def _window_delta_projected(
    norm: np.ndarray,
    basis: np.ndarray,
    eigvals: np.ndarray,
    seq_len: int,
    output_dim: int,
) -> list[list[float]]:
    if output_dim % 2 != 0:
        raise ValueError("window-delta recipe requires an even output dimension")
    projected = norm @ basis
    scales = np.sqrt(np.maximum(eigvals, 1e-8))
    whitened = projected / scales
    whitened = np.clip(whitened, -3.0, 3.0)

    half = output_dim // 2
    mean_steps: list[np.ndarray] = []
    for time_start, time_end in _time_bins(whitened.shape[0], seq_len):
        window = whitened[time_start:time_end]
        if window.shape[0] == 0:
            window = whitened[min(whitened.shape[0] - 1, time_start) : min(whitened.shape[0], time_start + 1)]
        mean_steps.append(window.mean(axis=0))

    pooled: list[list[float]] = []
    prev = np.zeros(half, dtype=np.float64)
    for idx, current in enumerate(mean_steps):
        delta = current - prev if idx > 0 else np.zeros_like(current)
        step = np.concatenate([current, np.clip(delta, -3.0, 3.0)])
        pooled.append([float(value) for value in step])
        prev = current
    return pooled


def main() -> int:
    args = parse_args()
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be positive")
    if args.input_dim != 8:
        raise SystemExit("current Sounio ABIDE benchmark requires --input-dim=8")

    cache_dir = Path(args.cache_dir)
    pheno_path = cache_dir / "phenotypic.csv"
    output_path = Path(args.output)
    if not pheno_path.exists():
        raise SystemExit(f"missing phenotypic file: {pheno_path}")

    subject_specs: list[dict[str, str]] = []
    subject_timeseries: list[list[list[float]]] = []
    rows_out: list[dict[str, str]] = []
    site_names: set[str] = set()
    skipped_missing = 0
    skipped_short = 0
    skipped_bad_label = 0
    skipped_canonical_trim = 0
    use_canonical_subset = not args.allow_extra_controls
    roi_dim_hint: int | None = None

    with pheno_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = (row.get("SUB_ID") or row.get("subject") or "").strip()
            file_id = (row.get("FILE_ID") or "").strip()
            site = (row.get("SITE_ID") or "").strip()
            label = _label_from_dx_group((row.get("DX_GROUP") or "").strip())
            if not file_id or file_id == "no_filename":
                skipped_missing += 1
                continue
            if not site or label is None:
                skipped_bad_label += 1
                continue
            if use_canonical_subset and subject_id in CANONICAL_EXCLUDED_SUBJECT_IDS:
                skipped_canonical_trim += 1
                continue

            ts_path = cache_dir / f"{file_id}_rois_cc200.1D"
            if not ts_path.exists():
                skipped_missing += 1
                continue

            matrix = _load_timeseries(ts_path)
            if matrix:
                if roi_dim_hint is None:
                    roi_dim_hint = max(len(row) for row in matrix)
                matrix = _sanitize_rows(matrix, roi_dim_hint=roi_dim_hint)
            if not matrix or len(matrix) < args.min_timepoints or len(matrix[0]) < args.input_dim:
                skipped_short += 1
                continue

            subject_specs.append(
                {
                    "subject_id": subject_id or file_id,
                    "file_id": file_id,
                    "label": label,
                    "site": site,
                }
            )
            subject_timeseries.append(matrix)
            site_names.add(site)

    basis_path = Path(args.basis_output) if args.basis_output else None
    explained_variance_bp = 0
    basis: np.ndarray | None = None
    if args.feature_recipe == "global-pca-window-mean":
        basis, eigvals = _fit_global_pca(subject_timeseries, args.input_dim)
        total = float(sum(max(0.0, value) for value in eigvals))
        explained_variance_bp = int(total * 1_000_000.0)
        if basis_path is not None:
            basis_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(basis_path, basis)
    elif args.feature_recipe == "global-pca-whiten-window-delta":
        basis, eigvals = _fit_global_pca(subject_timeseries, args.input_dim // 2)
        total = float(sum(max(0.0, value) for value in eigvals))
        explained_variance_bp = int(total * 1_000_000.0)
        if basis_path is not None:
            basis_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(basis_path, basis)

    for spec, matrix in zip(subject_specs, subject_timeseries, strict=True):
        if args.feature_recipe == "global-pca-window-mean":
            assert basis is not None
            pooled = _window_mean_projected(_zscore_numpy(matrix), basis, args.seq_len)
        elif args.feature_recipe == "global-pca-whiten-window-delta":
            assert basis is not None
            pooled = _window_delta_projected(_zscore_numpy(matrix), basis, eigvals, args.seq_len, args.input_dim)
        else:
            pooled = _window_pool_subject(matrix, args.seq_len, args.input_dim)
        flat = [value for step in pooled for value in step]
        record: dict[str, str] = {
            "subject_id": spec["subject_id"],
            "label": spec["label"],
            "site": spec["site"],
        }
        for index, value in enumerate(flat):
            record[f"f{index}"] = f"{value:.10f}"
        rows_out.append(record)

    fieldnames = ["subject_id", "label", "site"] + [f"f{idx}" for idx in range(args.seq_len * args.input_dim)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        if args.feature_recipe == "global-pca-window-mean":
            schema = "brain_ossm.abide.v4"
        elif args.feature_recipe == "global-pca-whiten-window-delta":
            schema = "brain_ossm.abide.v5"
        else:
            schema = "brain_ossm.abide.v3"
        handle.write(f"# schema={schema}\n")
        handle.write(f"# seq_len={args.seq_len}\n")
        handle.write(f"# input_dim={args.input_dim}\n")
        handle.write("# feature_layout=flat\n")
        handle.write("# label_space=asd_vs_control\n")
        handle.write("# split_policy=leave_one_site_out\n")
        handle.write("# atlas=CC200\n")
        handle.write(f"# feature_recipe={args.feature_recipe}\n")
        if args.feature_recipe == "global-pca-window-mean":
            handle.write("# roi_pooling=global_pca_mean\n")
        elif args.feature_recipe == "global-pca-whiten-window-delta":
            handle.write("# roi_pooling=global_pca_whitened_mean_plus_delta\n")
        else:
            handle.write("# roi_pooling=contiguous_mean\n")
        handle.write("# time_pooling=window_mean\n")
        handle.write(f"# source_cache_dir={cache_dir}\n")
        handle.write(f"# source_phenotypic={pheno_path}\n")
        handle.write(f"# canonical_subset={'yes' if use_canonical_subset else 'no'}\n")
        if basis_path is not None and args.feature_recipe in {"global-pca-window-mean", "global-pca-whiten-window-delta"}:
            handle.write(f"# pca_basis_path={basis_path}\n")
            handle.write(f"# pca_topk_variance_bp={explained_variance_bp}\n")
        handle.write(f"# subject_count={len(rows_out)}\n")
        handle.write(f"# site_count={len(site_names)}\n")
        handle.write(f"# skipped_missing={skipped_missing}\n")
        handle.write(f"# skipped_short={skipped_short}\n")
        handle.write(f"# skipped_bad_label={skipped_bad_label}\n")
        handle.write(f"# skipped_canonical_trim={skipped_canonical_trim}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"wrote temporal manifest: {output_path}")
    print(f"subjects={len(rows_out)} sites={len(site_names)} seq_len={args.seq_len} input_dim={args.input_dim}")
    print(f"feature_recipe={args.feature_recipe}")
    if basis_path is not None and args.feature_recipe in {"global-pca-window-mean", "global-pca-whiten-window-delta"}:
        print(f"basis_path={basis_path}")
    print(
        "skips:"
        f" missing={skipped_missing}"
        f" short={skipped_short}"
        f" bad_label={skipped_bad_label}"
        f" canonical_trim={skipped_canonical_trim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
