#!/usr/bin/env python3
"""Bootstrap a bounded ADHD-200 PCP/S3 pilot cache.

The public FCP-INDI bucket exposes ADHD-200 phenotypic CSVs and C-PAC ROI
timeseries. This helper downloads small phenotypic tables plus a deliberately
bounded CC200 ROI subset and rewrites the ROI filenames into the local
`<ScanDir ID>_rois_cc200.1D` convention consumed by adhd200_prepare_manifest.py.

It is not a full cohort downloader unless --allow-full-download is explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.adhd200_s3_bootstrap.v1"
CLAIM_BOUNDARY = (
    "Data acquisition bootstrap only; no diagnostic, biomarker, mechanism, "
    "treatment, or O-SSM superiority claim."
)
S3_BUCKET_URL = "https://s3.amazonaws.com/fcp-indi"
S3_XML_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PHENO_PREFIX = "data/Projects/ADHD200/RawData/"
DEFAULT_PIPELINE = "pipeline_adhd200-benchmark__freq-filter"
DEFAULT_REGRESSOR = "_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf0"
DEFAULT_BANDPASS = "_bandpass_freqs_0.01.0.1"


def s3_url(key: str) -> str:
    return f"{S3_BUCKET_URL}/{urllib.parse.quote(key)}"


def list_s3(prefix: str, max_keys: int = 1000) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)})
    request = urllib.request.Request(
        f"{S3_BUCKET_URL}?{query}",
        headers={"User-Agent": "sounio-adhd200-s3-bootstrap/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    rows = []
    for item in root.findall("s:Contents", S3_XML_NS):
        key = item.findtext("s:Key", default="", namespaces=S3_XML_NS)
        size = int(item.findtext("s:Size", default="0", namespaces=S3_XML_NS) or 0)
        if key:
            rows.append({"key": key, "size": size})
    return rows


def list_s3_prefixes(prefix: str, max_keys: int = 1000) -> list[str]:
    query = urllib.parse.urlencode(
        {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": str(max_keys)}
    )
    request = urllib.request.Request(
        f"{S3_BUCKET_URL}?{query}",
        headers={"User-Agent": "sounio-adhd200-s3-bootstrap/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    return [
        item.findtext("s:Prefix", default="", namespaces=S3_XML_NS)
        for item in root.findall("s:CommonPrefixes", S3_XML_NS)
        if item.findtext("s:Prefix", default="", namespaces=S3_XML_NS)
    ]


def download_file(key: str, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
        return {"key": key, "path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "cached": True}
    request = urllib.request.Request(s3_url(key), headers={"User-Agent": "sounio-adhd200-s3-bootstrap/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    path.write_bytes(data)
    return {"key": key, "path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "cached": False}


def is_missing(value: Any) -> bool:
    return str(value).strip().lower() in {"", "na", "n/a", "nan", "none", "null", "-999", "-9999"}


def diagnosis_label(value: Any) -> str:
    text = str(value).strip()
    if text in {"0", "2"}:
        return "TD"
    if text in {"1", "3"}:
        return "ADHD"
    return ""


def read_phenotypic(path: Path, source_key: str) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scan_id = str(row.get("ScanDir ID", "")).strip()
            label = diagnosis_label(row.get("DX", ""))
            if not scan_id or not label:
                continue
            row = dict(row)
            row["source_phenotypic_key"] = source_key
            row["source_site_name"] = Path(source_key).name.replace("_phenotypic.csv", "")
            rows.append(row)
    return rows


def has_primary(row: dict[str, str]) -> bool:
    return not any(is_missing(row.get(name, "")) for name in ["Inattentive", "Hyper/Impulsive", "ADHD Index"])


def select_rows(rows: list[dict[str, str]], max_subjects: int, available_scan_ids: set[str]) -> list[dict[str, str]]:
    by_site_label: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not has_primary(row):
            continue
        scan_id = str(row.get("ScanDir ID", "")).strip()
        if available_scan_ids and scan_id not in available_scan_ids:
            continue
        label = diagnosis_label(row.get("DX", ""))
        site = row.get("source_site_name", row.get("Site", "UNKNOWN_SITE"))
        by_site_label[(site, label)].append(row)
    selected = []
    target = max_subjects if max_subjects > 0 else sum(len(v) for v in by_site_label.values())
    while len(selected) < target:
        progressed = False
        for site in sorted({key[0] for key in by_site_label}):
            for label in ["ADHD", "TD"]:
                bucket = by_site_label[(site, label)]
                if bucket:
                    selected.append(bucket.pop(0))
                    progressed = True
                    if len(selected) >= target:
                        break
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def available_subjects_for_pipeline(pipeline: str) -> set[str]:
    prefix = f"data/Projects/ADHD200/Outputs/cpac/raw_outputs/{pipeline}/"
    subjects = set()
    for item in list_s3_prefixes(prefix, max_keys=1000):
        leaf = item.rstrip("/").split("/")[-1]
        if leaf.endswith("_session_1"):
            subjects.add(leaf[: -len("_session_1")])
    return subjects


def roi_key_for(scan_id: str, pipeline: str, regressor: str, bandpass: str) -> str:
    parts = [
        "data/Projects/ADHD200/Outputs/cpac/raw_outputs",
        pipeline,
        f"{scan_id}_session_1",
        "roi_timeseries",
        "_scan_rest_1",
        "_csf_threshold_0.96",
        "_gm_threshold_0.7",
        "_wm_threshold_0.96",
        regressor,
    ]
    if bandpass:
        parts.append(bandpass)
    parts.extend(["_mask_CC200", "roi_CC200.1D"])
    return "/".join(parts)


def write_merged_phenotypic(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "ScanDir ID",
        "Site",
        "Gender",
        "Age",
        "DX",
        "ADHD Index",
        "Inattentive",
        "Hyper/Impulsive",
        "Full2 IQ",
        "Full4 IQ",
        "Med Status",
        "QC_Rest_1",
        "source_site_name",
        "source_phenotypic_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ADHD-200 S3 Bootstrap",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "## Outputs",
        "",
        f"- Phenotypic CSV: `{summary['outputs']['phenotypic_csv']}`",
        f"- ROI dir: `{summary['outputs']['roi_dir']}`",
        f"- Downloaded ROI files: {summary['downloaded_roi_count']}",
        "",
        "## Next Command",
        "",
        "```bash",
        summary.get("next_command", ""),
        "```",
        "",
        "## Notes",
        "",
        "- Source bucket: `s3://fcp-indi/data/Projects/ADHD200/`.",
        "- Default derivative: C-PAC benchmark with frequency filter, CC200 ROI timeseries, no global signal regression.",
        "- This is a bounded cache bootstrap, not a clinical or mechanistic result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-subjects", type=int, default=24)
    parser.add_argument("--allow-full-download", action="store_true")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--regressor", default=DEFAULT_REGRESSOR)
    parser.add_argument("--bandpass", default=DEFAULT_BANDPASS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_subjects <= 0 and not args.allow_full_download:
        raise SystemExit("--max-subjects must be positive unless --allow-full-download is set")
    if args.max_subjects > 64 and not args.allow_full_download:
        raise SystemExit("--max-subjects >64 requires --allow-full-download")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pheno_dir = args.output_dir / "phenotypic_sources"
    roi_dir = args.output_dir / "rois"
    summary_path = args.output_dir / "adhd200_s3_bootstrap_summary.json"
    readme_path = args.output_dir / "ADHD200_S3_BOOTSTRAP.md"
    if not args.overwrite and (summary_path.exists() or readme_path.exists()):
        raise SystemExit(f"refusing to overwrite existing outputs in {args.output_dir}; pass --overwrite")

    pheno_keys = [
        item["key"]
        for item in list_s3(PHENO_PREFIX)
        if item["key"].endswith("_phenotypic.csv") and "TestRelease" not in item["key"]
    ]
    pheno_downloads = []
    all_rows: list[dict[str, str]] = []
    for key in sorted(pheno_keys):
        path = pheno_dir / Path(key).name
        receipt = download_file(key, path)
        pheno_downloads.append(receipt)
        all_rows.extend(read_phenotypic(path, key))

    available_scan_ids = available_subjects_for_pipeline(args.pipeline)
    selected = select_rows(all_rows, args.max_subjects, available_scan_ids)
    if not selected:
        raise SystemExit(
            "no ADHD-200 phenotypic rows with primary dimensional endpoints and available C-PAC ROI subjects were selected"
        )

    merged_pheno = args.output_dir / "adhd200_phenotypic.csv"
    write_merged_phenotypic(merged_pheno, selected)

    roi_receipts = []
    missing = []
    for row in selected:
        scan_id = str(row["ScanDir ID"]).strip()
        key = roi_key_for(scan_id, args.pipeline, args.regressor, args.bandpass)
        target = roi_dir / f"{scan_id}_rois_cc200.1D"
        if args.dry_run:
            roi_receipts.append({"key": key, "path": str(target), "dry_run": True})
            continue
        try:
            roi_receipts.append(download_file(key, target))
        except Exception as exc:
            missing.append({"scan_id": scan_id, "key": key, "error": str(exc)})

    status = "ready" if not missing and not args.dry_run else ("dry_run" if args.dry_run else "partial")
    next_command = (
        f"PHENOTYPIC_CSV='{merged_pheno}' ROI_DIR='{roi_dir}' "
        f"OUTPUT_DIR='{args.output_dir / 'pilot'}' SOUNIO_SOUC_ENGINE=lean_single "
        "scripts/research/adhd200_dimensional_pilot_smoke.sh"
    )
    counts = Counter(row.get("source_site_name", "UNKNOWN_SITE") for row in selected)
    label_counts = Counter(diagnosis_label(row.get("DX", "")) for row in selected)
    summary = {
        "schema": SCHEMA,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_bucket": "s3://fcp-indi/data/Projects/ADHD200/",
        "pipeline": args.pipeline,
        "available_cpac_subject_count": len(available_scan_ids),
        "regressor": args.regressor,
        "bandpass": args.bandpass,
        "selected_subject_count": len(selected),
        "site_counts": dict(sorted(counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "phenotypic_downloads": pheno_downloads,
        "downloaded_roi_count": sum(1 for receipt in roi_receipts if not receipt.get("dry_run")),
        "missing_roi": missing,
        "outputs": {"phenotypic_csv": str(merged_pheno), "roi_dir": str(roi_dir)},
        "next_command": "" if status == "partial" else next_command,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(readme_path, summary)
    print(json.dumps({"status": status, "summary": str(summary_path), "readme": str(readme_path)}, sort_keys=True))
    return 0 if status in {"ready", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
