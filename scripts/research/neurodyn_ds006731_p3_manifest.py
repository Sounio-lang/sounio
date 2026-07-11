#!/usr/bin/env python3
"""Build a durable ds006731 NeuroDyn P3 balanced manifest package.

This is a plumbing helper for the P3 MDD ROI lane. It takes two already
materialized ROI stores:

* a survivor directory containing ``roi/sub-*.1D`` files
* a rerun directory containing ``roi/sub-*.1D`` plus rerun receipts

and emits the ``balanced_manifest`` artifact class consumed by
``neurodyn_evidence_bundle.py`` and ``neurodyn_orangefs_not_required_gate.py``.

It intentionally does not run fMRIPrep, train O-SSM, or make any clinical claim.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import shutil
from pathlib import Path


DEFAULT_SUBJECTS: tuple[tuple[str, int, str], ...] = (
    ("HVS1", 0, "survivor"),
    ("HVS10", 0, "survivor"),
    ("HVS11", 0, "survivor"),
    ("HVS12", 0, "survivor"),
    ("HVS13", 0, "rerun"),
    ("HVS14", 0, "rerun"),
    ("HVS15", 0, "rerun"),
    ("HVS17", 0, "rerun"),
    ("T006", 1, "survivor"),
    ("T007", 1, "survivor"),
    ("T008", 1, "survivor"),
    ("T009", 1, "survivor"),
    ("T010", 1, "rerun"),
    ("T012", 1, "rerun"),
    ("T013", 1, "rerun"),
    ("T014", 1, "rerun"),
)

CLAIM_BOUNDARY = (
    "ROI/manifest persistence only. No positive, clinical, mechanistic, "
    "biomarker, or replicated O-SSM claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivor-root", default="", help="Directory containing survivor roi/sub-*.1D files")
    parser.add_argument("--rerun-root", default="", help="Directory containing rerun roi/sub-*.1D files")
    parser.add_argument(
        "--roster-tsv",
        default="",
        help="Optional readiness roster with subject_id, label, and primary_roi_path columns.",
    )
    parser.add_argument("--out-root", required=True, help="Durable output root for the balanced_manifest package")
    parser.add_argument("--tag", default="", help="Optional output tag; defaults to UTC timestamp")
    parser.add_argument("--manifest-prefix", default="", help="Optional output package prefix; defaults to 8x8 or 14x14.")
    parser.add_argument(
        "--roi-buckets",
        type=int,
        default=200,
        help="Number of contiguous ROI buckets to emit per time window; use 8 for current O-SSM runner.",
    )
    parser.add_argument(
        "--site-policy",
        choices=["openneuro", "pseudo_fold4", "pseudo_balanced"],
        default="openneuro",
        help="Site labels for O-SSM grouped holdout; pseudo_balanced creates balanced internal folds.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the target directory if it exists")
    return parser.parse_args()


def read_roi(path: Path) -> list[list[float]]:
    data: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append([float(item) for item in line.split()])
    if len(data) < 32:
        raise SystemExit(f"short ROI {path}: {len(data)} rows")
    width = len(data[0])
    if width != 200:
        raise SystemExit(f"unexpected ROI width {path}: {width}")
    if any(len(row) != width for row in data):
        raise SystemExit(f"ragged ROI {path}")
    return data


def window_features(data: list[list[float]], roi_buckets: int, mode: str = "none") -> list[float]:
    n_rows = len(data)
    n_roi = len(data[0])
    if roi_buckets <= 0 or roi_buckets > n_roi:
        raise SystemExit(f"invalid roi_buckets={roi_buckets}; ROI width is {n_roi}")
    columns: list[list[float]] = []
    for col_idx in range(n_roi):
        col = [data[row_idx][col_idx] for row_idx in range(n_rows)]
        mean = sum(col) / n_rows
        var = sum((value - mean) * (value - mean) for value in col) / max(1, n_rows - 1)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        columns.append([(value - mean) / sd for value in col])

    z_rows = [[columns[col_idx][row_idx] for col_idx in range(n_roi)] for row_idx in range(n_rows)]
    order = list(range(n_rows))
    if mode == "reverse":
        order = list(reversed(order))
    elif mode == "shuffle":
        rnd = random.Random(60731 + n_rows + n_roi)
        rnd.shuffle(order)

    values: list[float] = []
    for window in range(32):
        start = (window * n_rows) // 32
        end = ((window + 1) * n_rows) // 32
        idx = order[start:end] or [order[min(start, n_rows - 1)]]
        for bucket in range(roi_buckets):
            roi_start = (bucket * n_roi) // roi_buckets
            roi_end = ((bucket + 1) * n_roi) // roi_buckets
            denom = len(idx) * max(1, roi_end - roi_start)
            acc = 0.0
            for row_idx in idx:
                for col_idx in range(roi_start, roi_end):
                    acc += z_rows[row_idx][col_idx]
            values.append(acc / denom)
    return values


def site_for_row(row: dict[str, str], site_policy: str) -> str:
    if site_policy == "openneuro":
        return "openneuro"
    return row["SITE_ID"]


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    out_dir: Path,
    site_policy: str,
    roi_buckets: int,
    temporal: str = "none",
    label_shuffle: bool = False,
) -> None:
    labels = {row["subject_id"]: int(row["diagnosis"]) for row in rows}
    if label_shuffle:
        ordered = [row["subject_id"] for row in rows]
        labels = dict(zip(ordered, list(reversed([labels[subject] for subject in ordered]))))
    label_control = "within_site_shuffle" if label_shuffle else "none"

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# schema=brain_ossm.neurodyn.v1\n"
            "# dataset_id=ds006731\n"
            "# seq_len=32\n"
            f"# input_dim={roi_buckets}\n"
            "# feature_layout=flat\n"
            "# label_space=ds006731_mdd_vs_control\n"
            "# label_positive_name=MDD\n"
            "# label_negative_name=CONTROL\n"
            "# split_policy=leave_one_site_out\n"
            "# atlas=Schaefer2018_200Parcels_7Networks\n"
            f"# feature_recipe=zscore_window_contiguous_roi_mean_bucket{roi_buckets}\n"
            f"# site_policy={site_policy}\n"
            f"# temporal_control={temporal}\n"
            f"# label_control={label_control}\n"
            f"# source_phenotypic={out_dir / 'mdd_phenotypic.csv'}\n"
            f"# source_roi_dir={out_dir / 'rois'}\n"
            f"# subject_count={len(rows)}\n"
            f"# site_count={len(set(site_for_row(row, site_policy) for row in rows))}\n"
        )
        header = ["subject_id", "label", "site"] + [f"f{i}" for i in range(32 * roi_buckets)]
        handle.write("\t".join(header) + "\n")
        for row in rows:
            roi_path = out_dir / "rois" / (row["subject_id"] + ".1D")
            values = window_features(read_roi(roi_path), roi_buckets, temporal)
            fields = [row["subject_id"], str(labels[row["subject_id"]]), site_for_row(row, site_policy)]
            fields.extend(f"{value:.10f}" for value in values)
            handle.write("\t".join(fields) + "\n")


def pseudo_fold(subject: str) -> str:
    # Four folds, each with two controls and two MDD subjects in DEFAULT_SUBJECTS order.
    fold_map = {
        "HVS1": 0,
        "HVS13": 0,
        "T006": 0,
        "T010": 0,
        "HVS10": 1,
        "HVS14": 1,
        "T007": 1,
        "T012": 1,
        "HVS11": 2,
        "HVS15": 2,
        "T008": 2,
        "T013": 2,
        "HVS12": 3,
        "HVS17": 3,
        "T009": 3,
        "T014": 3,
    }
    return f"pseudo_fold_{fold_map[subject]:02d}"


def apply_balanced_pseudo_folds(rows: list[dict[str, str]], per_label_per_fold: int = 2) -> None:
    by_label: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for row in rows:
        by_label.setdefault(row["diagnosis"], []).append(row)
    if len(by_label.get("0", [])) != len(by_label.get("1", [])):
        raise SystemExit("pseudo_balanced requires equal class counts")
    if not by_label.get("0"):
        raise SystemExit("pseudo_balanced requires at least one subject per class")
    for label_rows in by_label.values():
        for idx, row in enumerate(label_rows):
            row["SITE_ID"] = f"pseudo_fold_{idx // per_label_per_fold:02d}"


def read_roster_rows(roster_tsv: Path) -> list[dict[str, str]]:
    with roster_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"subject_id", "label", "primary_roi_path"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"roster missing columns: {','.join(sorted(missing))}")
    return rows


def copy_rois_from_roster(roster_tsv: Path, out_dir: Path, site_policy: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    (out_dir / "rois").mkdir(parents=True, exist_ok=True)
    for item in read_roster_rows(roster_tsv):
        subject_id = item["subject_id"]
        label = str(item["label"])
        source = Path(item["primary_roi_path"])
        if label not in {"0", "1"}:
            raise SystemExit(f"bad label for {subject_id}: {label}")
        if not source.exists() or source.stat().st_size == 0:
            raise SystemExit(f"missing ROI {subject_id}: {source}")
        roi = read_roi(source)
        target = out_dir / "rois" / f"{subject_id}.1D"
        shutil.copy2(source, target)
        rows.append(
            {
                "subject_id": subject_id,
                "FILE_ID": subject_id,
                "SITE_ID": "openneuro",
                "diagnosis": label,
                "source_roi_path": str(source),
                "timepoints": str(len(roi)),
                "roi_dim": str(len(roi[0])),
                "provenance": item.get("primary_roi_path", ""),
            }
        )
    if site_policy == "pseudo_balanced":
        apply_balanced_pseudo_folds(rows)
    return rows


def copy_rois(survivor_root: Path, rerun_root: Path, out_dir: Path, site_policy: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    (out_dir / "rois").mkdir(parents=True, exist_ok=True)
    for subject, label, provenance in DEFAULT_SUBJECTS:
        source_root = survivor_root if provenance == "survivor" else rerun_root
        roi_path = source_root / "roi" / f"sub-{subject}.1D"
        if not roi_path.exists() or roi_path.stat().st_size == 0:
            raise SystemExit(f"missing ROI {subject}: {roi_path}")
        target = out_dir / "rois" / f"sub-{subject}.1D"
        shutil.copy2(roi_path, target)
        site = "openneuro" if site_policy == "openneuro" else pseudo_fold(subject)
        rows.append(
            {
                "subject_id": f"sub-{subject}",
                "FILE_ID": f"sub-{subject}",
                "SITE_ID": site,
                "diagnosis": str(label),
                "source_roi_path": str(target),
                "timepoints": "200",
                "roi_dim": "200",
                "provenance": provenance,
            }
        )
    if site_policy == "pseudo_balanced":
        apply_balanced_pseudo_folds(rows)
    return rows


def write_summary(
    out_dir: Path,
    rows: list[dict[str, str]],
    rerun_root: Path,
    site_policy: str,
    roi_buckets: int,
    source_descriptor: str,
) -> None:
    site_label_counts: dict[str, dict[str, int]] = {}
    label_counts = {"0": 0, "1": 0}
    for row in rows:
        site_counts = site_label_counts.setdefault(row["SITE_ID"], {"0": 0, "1": 0})
        site_counts[row["diagnosis"]] += 1
        label_counts[row["diagnosis"]] = label_counts.get(row["diagnosis"], 0) + 1
    summary = {
        "schema": "brain_ossm.mdd_neurodyn_prepare.v1",
        "dataset_id": "ds006731",
        "atlas": "Schaefer2018_200Parcels_7Networks",
        "seq_len": 32,
        "input_dim": roi_buckets,
        "selected_subject_count": len(rows),
        "selected_negative_count": label_counts.get("0", 0),
        "selected_positive_count": label_counts.get("1", 0),
        "selected_site_label_counts": site_label_counts,
        "eligible_subject_count": len(rows),
        "eligible_negative_count": label_counts.get("0", 0),
        "eligible_positive_count": label_counts.get("1", 0),
        "site_policy": site_policy,
        "source_descriptor": source_descriptor,
        "source_phenotypic": str(out_dir / "mdd_phenotypic.csv"),
        "phenotypic_output": str(out_dir / "mdd_phenotypic.csv"),
        "source_roi_dir": str(out_dir / "rois"),
        "roi_cache_dir": str(out_dir / "rois"),
        "manifests": {
            "original": str(out_dir / "mdd_neurodyn_manifest.tsv"),
            "temporal_shuffle": str(out_dir / "mdd_neurodyn_temporal_shuffle_manifest.tsv"),
            "temporal_reverse": str(out_dir / "mdd_neurodyn_temporal_reverse_manifest.tsv"),
            "label_within_site_shuffle": str(out_dir / "mdd_neurodyn_label_within_site_shuffle_manifest.tsv"),
        },
        "candidate_stats": {
            "rows_seen": len(rows),
            "skipped_bad_label": 0,
            "skipped_missing_path": 0,
            "skipped_short": 0,
            "skipped_site_limit": 0,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (out_dir / "mdd_neurodyn_prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "mdd_neurodyn_inventory.tsv").write_text(
        "subject_id\tlabel\tsite\tprovenance\n"
        + "".join(
            f"{row['subject_id']}\t{row['diagnosis']}\t{row['SITE_ID']}\t{row['provenance']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    if str(rerun_root) != ".":
        receipt = rerun_root / "rerun_receipt.env"
        if receipt.exists():
            shutil.copy2(receipt, out_dir / "rerun_receipt.env")
        (out_dir / "RERUN_SOURCE_DIR.txt").write_text(str(rerun_root) + "\n", encoding="utf-8")
    (out_dir / "claim_boundary.txt").write_text(CLAIM_BOUNDARY + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    survivor_root = Path(args.survivor_root).resolve() if args.survivor_root else Path()
    rerun_root = Path(args.rerun_root).resolve() if args.rerun_root else Path()
    roster_tsv = Path(args.roster_tsv).resolve() if args.roster_tsv else Path()
    out_root = Path(args.out_root).resolve()
    tag = args.tag or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if roster_tsv:
        package_prefix = args.manifest_prefix or "14x14"
    else:
        package_prefix = args.manifest_prefix or "8x8"
    out_dir = out_root / f"manifest_{package_prefix}_rerun_{args.site_policy}_bucket{args.roi_buckets}_{tag}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if roster_tsv:
        rows = copy_rois_from_roster(roster_tsv, out_dir, args.site_policy)
        source_descriptor = str(roster_tsv)
    else:
        if not args.survivor_root or not args.rerun_root:
            raise SystemExit("either --roster-tsv or both --survivor-root/--rerun-root are required")
        rows = copy_rois(survivor_root, rerun_root, out_dir, args.site_policy)
        source_descriptor = f"survivor_root={survivor_root};rerun_root={rerun_root}"

    with (out_dir / "mdd_phenotypic.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_manifest(out_dir / "mdd_neurodyn_manifest.tsv", rows, out_dir, args.site_policy, args.roi_buckets)
    write_manifest(
        out_dir / "mdd_neurodyn_temporal_shuffle_manifest.tsv",
        rows,
        out_dir,
        args.site_policy,
        args.roi_buckets,
        temporal="shuffle",
    )
    write_manifest(
        out_dir / "mdd_neurodyn_temporal_reverse_manifest.tsv",
        rows,
        out_dir,
        args.site_policy,
        args.roi_buckets,
        temporal="reverse",
    )
    write_manifest(
        out_dir / "mdd_neurodyn_label_within_site_shuffle_manifest.tsv",
        rows,
        out_dir,
        args.site_policy,
        args.roi_buckets,
        label_shuffle=True,
    )
    write_summary(out_dir, rows, rerun_root, args.site_policy, args.roi_buckets, source_descriptor)
    with (out_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in out_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            handle.write(f"{digest}  {path}\n")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
