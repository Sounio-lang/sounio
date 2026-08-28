#!/usr/bin/env python3
"""Build a paired temporal-arrow manifest from durable NeuroDyn ROI files.

This is a clinical-claim-free task.  Each subject contributes two records:

* forward: the ROI sequence in its observed order, label 1
* reverse: the same ROI sequence reversed in time, label 0

Both records inherit the same site/fold, preventing subject leakage across
leave-one-site-out folds.  The goal is to test whether the state-space model
uses temporal direction structure rather than disease labels.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "brain_ossm.neurodyn_temporal_arrow_prepare.v1"
CLAIM_BOUNDARY = (
    "Temporal-arrow diagnostic only. No clinical, mechanistic, biomarker, "
    "or replicated claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="TSV with subject_id, site, and provenance/path columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-buckets", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--swap-labels",
        action="store_true",
        help="Emit the paired control with forward=0 and reverse=1 for polarity auditing.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def zscore_columns(data: list[list[float]]) -> list[list[float]]:
    n_rows = len(data)
    n_roi = len(data[0])
    columns: list[list[float]] = []
    for col_idx in range(n_roi):
        col = [data[row_idx][col_idx] for row_idx in range(n_rows)]
        mean = sum(col) / n_rows
        var = sum((value - mean) * (value - mean) for value in col) / max(1, n_rows - 1)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        columns.append([(value - mean) / sd for value in col])
    return [[columns[col_idx][row_idx] for col_idx in range(n_roi)] for row_idx in range(n_rows)]


def window_features(data: list[list[float]], roi_buckets: int, reverse: bool) -> list[float]:
    z_rows = zscore_columns(data)
    n_rows = len(z_rows)
    n_roi = len(z_rows[0])
    if roi_buckets <= 0 or roi_buckets > n_roi:
        raise SystemExit(f"invalid roi_buckets={roi_buckets}; ROI width is {n_roi}")
    order = list(range(n_rows))
    if reverse:
        order = list(reversed(order))
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


def roi_path_for(row: dict[str, str]) -> Path:
    raw = row.get("provenance") or row.get("primary_roi_path") or row.get("source_roi_path") or ""
    if not raw:
        raise SystemExit(f"missing ROI path for {row.get('subject_id')}")
    return Path(raw)


def write_manifest(path: Path, records: list[dict[str, Any]], roi_buckets: int, swap_labels: bool) -> None:
    label_space = "ds006731_temporal_arrow_reverse_vs_forward" if swap_labels else "ds006731_temporal_arrow_forward_vs_reverse"
    positive_name = "REVERSE" if swap_labels else "FORWARD"
    negative_name = "FORWARD" if swap_labels else "REVERSE"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# schema=brain_ossm.neurodyn.v1\n"
            "# dataset_id=ds006731\n"
            "# seq_len=32\n"
            f"# input_dim={roi_buckets}\n"
            "# feature_layout=flat\n"
            f"# label_space={label_space}\n"
            f"# label_positive_name={positive_name}\n"
            f"# label_negative_name={negative_name}\n"
            "# split_policy=leave_one_site_out_subject_paired\n"
            "# atlas=Schaefer2018_200Parcels_7Networks\n"
            f"# feature_recipe=zscore_window_contiguous_roi_mean_bucket{roi_buckets}\n"
            "# task=temporal_arrow\n"
            f"# subject_count={len({record['source_subject_id'] for record in records})}\n"
            f"# record_count={len(records)}\n"
            f"# site_count={len({record['site'] for record in records})}\n"
        )
        header = ["subject_id", "label", "site"] + [f"f{i}" for i in range(32 * roi_buckets)]
        handle.write("\t".join(header) + "\n")
        for record in records:
            fields = [record["record_id"], str(record["label"]), record["site"]]
            fields.extend(f"{value:.10f}" for value in record["features"])
            handle.write("\t".join(fields) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    inventory_path = Path(args.inventory).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = read_tsv(inventory_path)
    records: list[dict[str, Any]] = []
    phenotypic: list[dict[str, Any]] = []
    for row in inventory:
        subject_id = row["subject_id"]
        site = row["site"]
        source_label = row.get("label", "")
        roi_path = roi_path_for(row)
        data = read_roi(roi_path)
        for direction, base_label, reverse in (("forward", 1, False), ("reverse", 0, True)):
            label = 1 - base_label if args.swap_labels else base_label
            record_id = f"{subject_id}__{direction}"
            features = window_features(data, args.roi_buckets, reverse)
            records.append(
                {
                    "record_id": record_id,
                    "source_subject_id": subject_id,
                    "direction": direction,
                    "label": label,
                    "site": site,
                    "source_clinical_label": source_label,
                    "source_roi_path": str(roi_path),
                    "timepoints": len(data),
                    "roi_dim": len(data[0]),
                    "features": features,
                }
            )
            phenotypic.append(
                {
                    "subject_id": record_id,
                    "source_subject_id": subject_id,
                    "direction": direction,
                    "diagnosis": label,
                    "SITE_ID": site,
                    "source_clinical_label": source_label,
                    "source_roi_path": str(roi_path),
                    "timepoints": len(data),
                    "roi_dim": len(data[0]),
                }
            )

    manifest = output_dir / "temporal_arrow_manifest.tsv"
    write_manifest(manifest, records, args.roi_buckets, args.swap_labels)
    write_csv(output_dir / "temporal_arrow_phenotypic.csv", phenotypic)
    write_json(
        output_dir / "temporal_arrow_prepare_summary.json",
        {
            "schema": SCHEMA,
            "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "dataset_id": "ds006731",
            "source_inventory": str(inventory_path),
            "manifest": str(manifest),
            "seq_len": 32,
            "input_dim": args.roi_buckets,
            "source_subject_count": len(inventory),
            "record_count": len(records),
            "swap_labels": bool(args.swap_labels),
            "label_positive_name": "REVERSE" if args.swap_labels else "FORWARD",
            "label_negative_name": "FORWARD" if args.swap_labels else "REVERSE",
            "label_counts": {
                "0": sum(1 for record in records if record["label"] == 0),
                "1": sum(1 for record in records if record["label"] == 1),
            },
            "site_label_counts": {
                site: {
                    "0": sum(1 for record in records if record["site"] == site and record["label"] == 0),
                    "1": sum(1 for record in records if record["site"] == site and record["label"] == 1),
                }
                for site in sorted({record["site"] for record in records})
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path}\n")
    print(f"manifest={manifest}")
    print(f"records={len(records)}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
