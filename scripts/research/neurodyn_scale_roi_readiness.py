#!/usr/bin/env python3
"""Inventory durable ROI availability for a scaled NeuroDyn ds006731 cohort."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.scale_roi_readiness.v1"
CLAIM_BOUNDARY = "Operational ROI readiness only; no O-SSM, clinical, or biomarker claim."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-subjects", required=True)
    parser.add_argument("--roi-root", action="append", default=[], help="Root to search for roi/sub-*.1D files.")
    parser.add_argument(
        "--roi-index",
        action="append",
        default=[],
        help="Optional TSV with subject and path columns for ROIs discovered outside this filesystem.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-per-class", type=int, default=14)
    return parser.parse_args()


def read_selected(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_roi_index(path: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    if not path.exists():
        return found
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            subject = (row.get("subject") or row.get("subject_id") or "").removeprefix("sub-")
            roi_path = row.get("path") or row.get("roi_path") or row.get("primary_roi_path") or ""
            if subject and roi_path:
                found.setdefault(subject, []).append(roi_path)
    return found


def find_roi_files(roots: list[Path], roi_indexes: list[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("sub-*.1D")):
            if path.parent.name not in {"roi", "rois"}:
                continue
            subject = path.name.removeprefix("sub-").removesuffix(".1D")
            found.setdefault(subject, []).append(str(path.resolve()))
    for index_path in roi_indexes:
        for subject, paths in read_roi_index(index_path).items():
            found.setdefault(subject, []).extend(paths)
    return {subject: sorted(paths) for subject, paths in sorted(found.items())}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label_name(label: str) -> str:
    return "MDD" if str(label) == "1" else "CONTROL"


def main() -> int:
    args = parse_args()
    selected_path = Path(args.selected_subjects).resolve()
    output_dir = Path(args.output_dir).resolve()
    roots = [Path(raw).resolve() for raw in args.roi_root]
    roi_indexes = [Path(raw).resolve() for raw in args.roi_index]

    selected = read_selected(selected_path)
    roi_files = find_roi_files(roots, roi_indexes)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        subject_id = row["subject_id"]
        subject = subject_id.removeprefix("sub-")
        paths = roi_files.get(subject, [])
        rows.append(
            {
                "candidate_order": index,
                "subject": subject,
                "subject_id": subject_id,
                "label": row["label"],
                "label_name": label_name(row["label"]),
                "raw_label": row.get("raw_label", ""),
                "has_roi": bool(paths),
                "roi_path_count": len(paths),
                "primary_roi_path": paths[0] if paths else "",
                "bold_s3_uri": row.get("bold_s3_uri", ""),
                "anat_s3_uri": row.get("anat_s3_uri", ""),
            }
        )

    selected_for_target: list[dict[str, Any]] = []
    missing_for_target: list[dict[str, Any]] = []
    counts = {"0": 0, "1": 0}
    available_counts = {"0": 0, "1": 0}
    for row in rows:
        label = str(row["label"])
        if row["has_roi"]:
            available_counts[label] = available_counts.get(label, 0) + 1
        if counts.get(label, 0) >= args.target_per_class:
            continue
        counts[label] = counts.get(label, 0) + 1
        selected_for_target.append(row)
        if not row["has_roi"]:
            missing_for_target.append(row)

    summary = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "selected_subjects": str(selected_path),
        "roi_roots": [str(root) for root in roots],
        "roi_indexes": [str(path) for path in roi_indexes],
        "target_per_class": args.target_per_class,
        "candidate_count": len(rows),
        "candidate_label_counts": {
            "0": sum(1 for row in rows if str(row["label"]) == "0"),
            "1": sum(1 for row in rows if str(row["label"]) == "1"),
        },
        "available_roi_label_counts": available_counts,
        "target_selected_count": len(selected_for_target),
        "target_selected_label_counts": counts,
        "target_available_label_counts": {
            "0": sum(1 for row in selected_for_target if str(row["label"]) == "0" and row["has_roi"]),
            "1": sum(1 for row in selected_for_target if str(row["label"]) == "1" and row["has_roi"]),
        },
        "target_missing_count": len(missing_for_target),
        "target_missing_subjects": [row["subject"] for row in missing_for_target],
        "decision": "ready_for_scaled_manifest" if not missing_for_target else "blocked_missing_roi",
        "next_action": "build scaled manifest" if not missing_for_target else "run fMRIPrep/ROI extraction for target_missing_subjects",
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "scale_roi_readiness.json", summary)
    write_tsv(output_dir / "scale_roi_inventory.tsv", rows)
    write_tsv(output_dir / "scale_target_selected.tsv", selected_for_target)
    write_tsv(output_dir / "scale_target_missing.tsv", missing_for_target)
    print(f"decision={summary['decision']}")
    print(f"available_roi_label_counts={summary['available_roi_label_counts']}")
    print(f"target_available_label_counts={summary['target_available_label_counts']}")
    print(f"target_missing_subjects={','.join(summary['target_missing_subjects'])}")
    print(f"summary={output_dir / 'scale_roi_readiness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
