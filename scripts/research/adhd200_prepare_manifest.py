#!/usr/bin/env python3
"""Prepare an ADHD-200 dimensional O-SSM manifest from phenotypic and ROI data.

The rich manifest is the phenotype/readiness source of truth. The optional
O-SSM compatibility view intentionally drops dimensional phenotype columns so
current Sounio model readers can consume the same feature rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg


SCHEMA = "neurodyn.adhd200_dimensional_manifest.v1"
OSSM_VIEW_SCHEMA = "neurodyn.adhd200_ossm_compat_manifest.v1"
CLAIM_BOUNDARY = (
    "Dataset preparation only; no diagnostic, biomarker, mechanism, treatment, "
    "or O-SSM superiority claim."
)

N_EIGVECS = 7
N_ROIS_TARGET = 200
N_STEPS = 8
N_DIMS = 8
N_FEATURES = 64
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-999", "-9999"}


COLUMN_ALIASES = {
    "subject_id": [
        "subject_id",
        "sub_id",
        "subject",
        "participant_id",
        "scan_dir_id",
        "scandirid",
        "scan_dir id",
        "ScanDir ID",
        "scanid",
        "id",
    ],
    "file_id": [
        "file_id",
        "fileid",
        "filename",
        "scan_dir_id",
        "scandirid",
        "scan_dir id",
        "ScanDir ID",
        "scanid",
        "subject_id",
        "participant_id",
    ],
    "site": [
        "site",
        "site_id",
        "siteid",
        "scan_site",
        "data_set",
        "dataset",
    ],
    "diagnosis": [
        "dx",
        "dx_group",
        "diagnosis",
        "adhd",
        "adhd_diagnosis",
        "diagnostic_status",
    ],
    "inattention": [
        "inattention",
        "inattentive",
        "adhd_inattentive",
        "adhd_inattention",
        "adhd_rs_inattention",
        "adhd_rs_iv_inattention",
        "adhd_inattentive_score",
        "conners_inattention",
    ],
    "hyperactivity_impulsivity": [
        "hyperactivity_impulsivity",
        "hyper_impulsive",
        "hyperactive_impulsive",
        "hyperactivity",
        "impulsivity",
        "adhd_hyperactivity",
        "adhd_hyper_impulsive",
        "adhd_rs_hyperactivity_impulsivity",
        "adhd_rs_iv_hyperactivity_impulsivity",
        "conners_hyperactivity",
    ],
    "adhd_total": [
        "adhd_total",
        "adhd_index",
        "adhd index",
        "adhd_index_score",
        "adhd_rs_total",
        "adhd_rs_iv_total",
        "conners_adhd_index",
        "total",
    ],
    "age": ["age", "age_at_scan", "age_years"],
    "sex": ["sex", "gender"],
    "iq": ["iq", "fiq", "full_scale_iq", "fullscale_iq", "full2_iq", "full4_iq", "Full2 IQ", "Full4 IQ"],
    "medication_status": [
        "medication_status",
        "med_status",
        "current_med_status",
        "medicated",
        "lifetime_med_status",
    ],
    "qc_status": [
        "qc_status",
        "qc",
        "quality_control",
        "qc_rater",
        "func_quality",
    ],
    "mean_fd": [
        "mean_fd",
        "func_mean_fd",
        "motion_mean_fd",
        "fd",
    ],
}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum())


def is_missing(value: Any) -> bool:
    return str(value).strip().lower() in MISSING_TOKENS


def parse_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        result = float(str(value).strip())
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def discover_columns(fieldnames: list[str]) -> dict[str, str | None]:
    by_norm = {normalize_name(name): name for name in fieldnames}
    result: dict[str, str | None] = {}
    for logical, aliases in COLUMN_ALIASES.items():
        result[logical] = None
        for alias in aliases:
            found = by_norm.get(normalize_name(alias))
            if found:
                result[logical] = found
                break
    return result


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise SystemExit(f"phenotypic CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def diagnosis_label(value: Any) -> str | None:
    text = str(value).strip()
    low = text.lower()
    if is_missing(text):
        return None
    if low in {"td", "tdc", "control", "typically developing", "healthy", "0", "2"}:
        return "TD"
    if low in {"adhd", "adhd-c", "adhd-i", "adhd-h", "combined", "inattentive", "hyperactive", "1", "3"}:
        return "ADHD"
    numeric = parse_float(text)
    if numeric is not None:
        if int(numeric) == 0 or int(numeric) == 2:
            return "TD"
        if int(numeric) in {1, 3}:
            return "ADHD"
    return None


def ossm_label(label: str) -> str:
    if label == "ADHD":
        return "1"
    if label == "TD":
        return "0"
    raise SystemExit(f"unsupported O-SSM compatibility label: {label}")


def download_file(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as exc:  # pragma: no cover - network failure path
        print(f"download failed: {url}: {exc}", file=sys.stderr)
        return False


def candidate_roi_paths(roi_dir: Path, file_id: str, subject_id: str) -> list[Path]:
    candidates = []
    names = []
    for stem in [file_id, subject_id, f"sub-{subject_id}", f"sub_{subject_id}"]:
        if stem and stem not in names:
            names.append(stem)
    suffixes = [
        "",
        "_rois_cc200.1D",
        "_rois_aal.1D",
        ".1D",
        "_timeseries.tsv",
        "_timeseries.txt",
        ".tsv",
        ".txt",
        ".csv",
    ]
    for stem in names:
        for suffix in suffixes:
            path = roi_dir / f"{stem}{suffix}"
            if path.exists() and path.is_file():
                candidates.append(path)
    return candidates


def load_timeseries(path: Path) -> np.ndarray | None:
    try:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        data = np.loadtxt(path, delimiter=delimiter, comments="#")
    except Exception:
        return None
    if data.ndim != 2 or data.shape[0] < 20 or data.shape[1] < 8:
        return None
    return data


def extract_eigenvectors(ts: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    try:
        n_rois = ts.shape[1]
        n_timepoints = ts.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(ts.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        adj = np.maximum(corr, 0.0)
        deg = adj.sum(axis=1)
        laplacian = np.diag(deg) - adj
        eigenvalues, eigenvectors = linalg.eigh(laplacian)
        if len(eigenvalues) < N_EIGVECS + 1:
            return None
        return eigenvectors[:, 1 : N_EIGVECS + 1].T, n_rois, n_timepoints
    except Exception:
        return None


def eigvecs_to_features(evecs: np.ndarray, n_rois: int) -> np.ndarray:
    if evecs.shape[0] < N_STEPS:
        padded = np.zeros((N_STEPS, evecs.shape[1]))
        padded[: evecs.shape[0], :] = evecs
        evecs = padded
    if n_rois < N_ROIS_TARGET:
        padded = np.zeros((N_STEPS, N_ROIS_TARGET))
        padded[:, :n_rois] = evecs[:, :n_rois]
        evecs = padded
    elif n_rois > N_ROIS_TARGET:
        evecs = evecs[:, :N_ROIS_TARGET]
    block_size = N_ROIS_TARGET // N_DIMS
    features = np.zeros((N_STEPS, N_DIMS))
    for dim in range(N_DIMS):
        start = dim * block_size
        end = start + block_size
        features[:, dim] = evecs[:, start:end].mean(axis=1)
    max_abs = np.abs(features).max()
    if max_abs > 1.0e-12:
        features = features / max_abs
    return features.flatten()


def timeseries_to_temporal_block_features(ts: np.ndarray) -> np.ndarray:
    n_timepoints, n_rois = ts.shape
    if n_rois < N_ROIS_TARGET:
        padded = np.zeros((n_timepoints, N_ROIS_TARGET))
        padded[:, :n_rois] = ts[:, :n_rois]
        ts = padded
    elif n_rois > N_ROIS_TARGET:
        ts = ts[:, :N_ROIS_TARGET]

    means = ts.mean(axis=0, keepdims=True)
    stds = ts.std(axis=0, keepdims=True)
    stds = np.where(stds > 1.0e-12, stds, 1.0)
    zts = (ts - means) / stds

    block_size = N_ROIS_TARGET // N_DIMS
    features = np.zeros((N_STEPS, N_DIMS))
    edges = np.linspace(0, n_timepoints, N_STEPS + 1, dtype=int)
    for step in range(N_STEPS):
        start = int(edges[step])
        end = int(edges[step + 1])
        if end <= start:
            end = min(n_timepoints, start + 1)
        window = zts[start:end, :]
        for dim in range(N_DIMS):
            roi_start = dim * block_size
            roi_end = roi_start + block_size
            features[step, dim] = float(window[:, roi_start:roi_end].mean())

    max_abs = np.abs(features).max()
    if max_abs > 1.0e-12:
        features = features / max_abs
    return features.flatten()


def canonical_value(row: dict[str, str], columns: dict[str, str | None], logical: str) -> str:
    column = columns.get(logical)
    if not column:
        return ""
    value = row.get(column, "")
    return "" if is_missing(value) else str(value).strip()


def numeric_text(row: dict[str, str], columns: dict[str, str | None], logical: str) -> str:
    value = parse_float(canonical_value(row, columns, logical))
    return "" if value is None else f"{value:.8g}"


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        for key, value in meta.items():
            handle.write(f"# {key}={value}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def subject_universe_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash ordered subject_id/label/site rows; keep in sync with the probe."""
    h = hashlib.sha256()
    for row in rows:
        h.update(str(row["subject_id"]).encode("utf-8"))
        h.update(b"\t")
        h.update(str(row["label"]).encode("utf-8"))
        h.update(b"\t")
        h.update(str(row["site"]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phenotypic-csv", type=Path, default=None)
    parser.add_argument("--phenotypic-url", default="")
    parser.add_argument("--roi-dir", type=Path, default=None)
    parser.add_argument("--download-derivatives", action="store_true")
    parser.add_argument(
        "--derivative-url-template",
        default="",
        help="Template containing {file_id} and/or {subject_id}; required with --download-derivatives.",
    )
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--write-ossm-view", action="store_true")
    parser.add_argument("--allow-missing-primary", action="store_true")
    parser.add_argument("--sampling-mode", choices=["ordered", "site_balanced"], default="site_balanced")
    parser.add_argument(
        "--feature-mode",
        choices=["temporal_roi_block", "laplacian_eigenblock"],
        default="temporal_roi_block",
        help="Default preserves an 8-step temporal sequence; laplacian mode is a static-summary ablation.",
    )
    parser.add_argument("--primary-phenotypes", default="inattention,hyperactivity_impulsivity,adhd_total")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.phenotypic_csv:
        pheno_path = args.phenotypic_csv
        if not pheno_path.exists():
            raise SystemExit(f"phenotypic CSV not found: {pheno_path}")
        phenotypic_source = str(pheno_path)
    else:
        pheno_path = cache_dir / "phenotypic.csv"
        phenotypic_source = args.phenotypic_url or "manual_required"
        if args.skip_download:
            raise SystemExit("no --phenotypic-csv supplied and --skip-download is set")
        if not args.phenotypic_url:
            raise SystemExit("no public ADHD-200 phenotypic URL configured; pass --phenotypic-csv")
        if not download_file(args.phenotypic_url, pheno_path):
            raise SystemExit("failed to download phenotypic CSV; pass --phenotypic-csv after manual access")

    roi_dir = args.roi_dir if args.roi_dir else cache_dir
    if args.download_derivatives and not args.derivative_url_template:
        raise SystemExit("--download-derivatives requires --derivative-url-template")

    fieldnames, pheno_rows = read_csv_rows(pheno_path)
    columns = discover_columns(fieldnames)
    for logical in ["subject_id", "site", "diagnosis"]:
        if not columns.get(logical):
            raise SystemExit(f"could not identify required phenotypic column: {logical}")

    primary = [value.strip() for value in args.primary_phenotypes.split(",") if value.strip()]
    missing_primary = [logical for logical in primary if not columns.get(logical)]
    if missing_primary and not args.allow_missing_primary:
        raise SystemExit(
            "missing primary phenotype columns: "
            + ",".join(missing_primary)
            + " (use --allow-missing-primary for readiness-only manifests)"
        )

    prepared: list[dict[str, Any]] = []
    skipped = Counter()
    for row in pheno_rows:
        subject_id = canonical_value(row, columns, "subject_id")
        file_id = canonical_value(row, columns, "file_id") or subject_id
        site = canonical_value(row, columns, "site") or "UNKNOWN_SITE"
        label = diagnosis_label(canonical_value(row, columns, "diagnosis"))
        if not subject_id:
            skipped["missing_subject_id"] += 1
            continue
        if label not in {"ADHD", "TD"}:
            skipped["missing_or_unknown_label"] += 1
            continue

        paths = candidate_roi_paths(roi_dir, file_id, subject_id)
        if not paths and args.download_derivatives:
            target = roi_dir / f"{file_id}_rois_cc200.1D"
            url = args.derivative_url_template.format(file_id=file_id, subject_id=subject_id)
            if download_file(url, target):
                paths = [target]
        if not paths:
            skipped["missing_roi"] += 1
            continue

        ts = None
        ts_path = None
        for candidate in paths:
            ts = load_timeseries(candidate)
            if ts is not None:
                ts_path = candidate
                break
        if ts is None or ts_path is None:
            skipped["unreadable_roi"] += 1
            continue
        n_rois = int(ts.shape[1])
        n_timepoints = int(ts.shape[0])
        if args.feature_mode == "laplacian_eigenblock":
            extracted = extract_eigenvectors(ts)
            if extracted is None:
                skipped["feature_extraction_failed"] += 1
                continue
            evecs, n_rois, n_timepoints = extracted
            features = eigvecs_to_features(evecs, n_rois)
            feature_layout = "8x8_laplacian_eigenblock"
        else:
            features = timeseries_to_temporal_block_features(ts)
            feature_layout = "8x8_temporal_roi_block"
        record: dict[str, Any] = {
            "subject_id": subject_id,
            "label": label,
            "site": site,
            "inattention": numeric_text(row, columns, "inattention"),
            "hyperactivity_impulsivity": numeric_text(row, columns, "hyperactivity_impulsivity"),
            "adhd_total": numeric_text(row, columns, "adhd_total"),
            "age": numeric_text(row, columns, "age"),
            "sex": canonical_value(row, columns, "sex"),
            "iq": numeric_text(row, columns, "iq"),
            "medication_status": canonical_value(row, columns, "medication_status"),
            "qc_status": canonical_value(row, columns, "qc_status"),
            "mean_fd": numeric_text(row, columns, "mean_fd"),
            "source_file_id": file_id,
            "roi_path": str(ts_path),
            "n_rois": n_rois,
            "n_timepoints": n_timepoints,
        }
        for idx, value in enumerate(features):
            record[f"f{idx}"] = f"{value:.8f}"
        prepared.append(record)
        if args.max_subjects > 0 and len(prepared) >= args.max_subjects:
            break

    if args.sampling_mode == "site_balanced":
        by_site_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in prepared:
            by_site_label[(str(record["site"]), str(record["label"]))].append(record)
        selected: list[dict[str, Any]] = []
        for site in sorted({str(record["site"]) for record in prepared}):
            n = min(len(by_site_label[(site, "ADHD")]), len(by_site_label[(site, "TD")]))
            selected.extend(by_site_label[(site, "ADHD")][:n])
            selected.extend(by_site_label[(site, "TD")][:n])
        prepared = selected

    if not prepared:
        raise SystemExit("no subjects processed; check phenotypic fields, ROI cache, and access settings")
    universe_hash = subject_universe_sha256(prepared)

    rich_fields = [
        "subject_id",
        "label",
        "site",
        "inattention",
        "hyperactivity_impulsivity",
        "adhd_total",
        "age",
        "sex",
        "iq",
        "medication_status",
        "qc_status",
        "mean_fd",
        "source_file_id",
        "roi_path",
        "n_rois",
        "n_timepoints",
    ] + [f"f{i}" for i in range(N_FEATURES)]
    meta = {
        "schema": SCHEMA,
        "seq_len": str(N_STEPS),
        "input_dim": str(N_DIMS),
        "feature_layout": feature_layout,
        "feature_mode": args.feature_mode,
        "label_space": "adhd_vs_td_for_stratification_only",
        "primary_phenotypes": ",".join(primary),
        "split_policy": "leave_one_site_out",
        "sampling_mode": args.sampling_mode,
        "phenotypic_source": phenotypic_source,
        "subject_universe_sha256": universe_hash,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest_path = args.output_dir / "adhd200_roi_manifest.tsv"
    write_tsv(manifest_path, prepared, rich_fields, meta)

    ossm_view_path = None
    if args.write_ossm_view:
        ossm_fields = ["subject_id", "label", "site"] + [f"f{i}" for i in range(N_FEATURES)]
        ossm_rows = []
        for row in prepared:
            view_row = dict(row)
            view_row["label"] = ossm_label(str(row["label"]))
            ossm_rows.append(view_row)
        ossm_meta = {
            "schema": OSSM_VIEW_SCHEMA,
            "seq_len": str(N_STEPS),
            "input_dim": str(N_DIMS),
            "feature_layout": feature_layout,
            "feature_mode": args.feature_mode,
            "label_space": "1_ADHD_0_TD_for_model_smoke_only",
            "label_encoding": "1=ADHD;0=TD",
            "split_policy": "leave_one_site_out",
            "source_manifest": str(manifest_path),
            "subject_universe_sha256": universe_hash,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        ossm_view_path = args.output_dir / "adhd200_ossm_manifest.tsv"
        write_tsv(ossm_view_path, ossm_rows, ossm_fields, ossm_meta)

    summary = {
        "schema": "neurodyn.adhd200_prepare_manifest.summary.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest": str(manifest_path),
        "ossm_view": str(ossm_view_path) if ossm_view_path else None,
        "phenotypic_source": phenotypic_source,
        "roi_dir": str(roi_dir),
        "column_mapping": columns,
        "missing_primary_columns": missing_primary,
        "processed_subjects": len(prepared),
        "subject_universe_sha256": universe_hash,
        "site_counts": dict(sorted(Counter(str(row["site"]) for row in prepared).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in prepared).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    summary_path = args.output_dir / "adhd200_prepare_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
