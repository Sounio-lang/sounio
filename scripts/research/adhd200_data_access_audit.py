#!/usr/bin/env python3
"""Audit ADHD-200 data access before running the dimensional O-SSM pilot.

This script is intentionally conservative: it does not download large
derivatives, does not treat ABIDE files as ADHD-200, and does not convert a
missing dataset into a synthetic smoke. Its job is to make the next executable
step explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.adhd200_data_access_audit.v1"
CLAIM_BOUNDARY = (
    "Data-access/readiness audit only; no diagnostic, biomarker, mechanism, "
    "treatment, or O-SSM superiority claim."
)

SOURCE_URLS = [
    "https://fcon_1000.projects.nitrc.org/indi/adhd200/",
    "https://preprocessed-connectomes-project.org/adhd200/",
    "https://preprocessed-connectomes-project.org/adhd200/download.html",
    "https://raw.githubusercontent.com/preprocessed-connectomes-project/adhd200/gh-pages/download.html",
]

PHENO_NAME_RE = re.compile(r"(adhd|phenotypic|phenotype|participants).*\.(csv|tsv|txt)$", re.I)
ROI_NAME_RE = re.compile(r"(adhd|nyu|pitt|peking|washu|ohsu|neuro|kki|brown|kennedy|sub[-_]?|^[A-Za-z0-9_.-]+)(_rois_cc200)?\.(1D|tsv|txt|csv)$", re.I)
ABIDE_NAME_RE = re.compile(r"abide|autism|asd", re.I)
ADHD_CONTEXT_RE = re.compile(r"adhd[-_ ]?200|adhd", re.I)
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "target",
    "vendor",
}

REQUIRED_HINTS = {
    "subject": ["subject_id", "sub_id", "subject", "participant_id", "scan_dir_id", "scandirid", "scan_dir id", "ScanDir ID", "scanid", "id"],
    "site": ["site", "site_id", "siteid", "scan_site", "data_set", "dataset"],
    "diagnosis": ["dx", "dx_group", "diagnosis", "adhd", "adhd_diagnosis", "diagnostic_status"],
}
PRIMARY_HINTS = {
    "inattention": ["inattention", "inattentive", "adhd_rs_iv_inattention", "conners_inattention"],
    "hyperactivity_impulsivity": [
        "hyperactivity_impulsivity",
        "hyper_impulsive",
        "Hyper/Impulsive",
        "hyperactivity",
        "impulsivity",
        "adhd_rs_iv_hyperactivity_impulsivity",
        "conners_hyperactivity",
    ],
    "adhd_total": ["adhd_total", "adhd_index", "adhd_rs_iv_total", "conners_adhd_index"],
}


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def header_hits(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "readable": False,
        "required_hits": {},
        "primary_hits": {},
        "field_count": 0,
    }
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            except csv.Error:
                dialect = csv.excel_tab if path.suffix.lower() in {".tsv", ".txt"} else csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            fields = list(reader.fieldnames or [])
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["readable"] = bool(fields)
    result["field_count"] = len(fields)
    by_norm = {normalize_name(field): field for field in fields}
    for logical, aliases in REQUIRED_HINTS.items():
        result["required_hits"][logical] = next(
            (by_norm[normalize_name(alias)] for alias in aliases if normalize_name(alias) in by_norm),
            "",
        )
    for logical, aliases in PRIMARY_HINTS.items():
        result["primary_hits"][logical] = next(
            (by_norm[normalize_name(alias)] for alias in aliases if normalize_name(alias) in by_norm),
            "",
        )
    return result


def has_adhd_context(path: Path) -> bool:
    return bool(ADHD_CONTEXT_RE.search(str(path)))


def looks_like_roi(path: Path, require_adhd_context: bool) -> bool:
    if require_adhd_context and not has_adhd_context(path):
        return False
    name = path.name
    if ABIDE_NAME_RE.search(str(path)):
        return False
    if not ROI_NAME_RE.search(name):
        return False
    if path.stat().st_size < 256:
        return False
    return True


def scan_tree(root: Path, max_files: int, require_adhd_context: bool) -> tuple[list[Path], list[Path], int, bool]:
    phenotypic: list[Path] = []
    roi: list[Path] = []
    visited = 0
    truncated = False
    if not root.exists():
        return phenotypic, roi, visited, truncated
    if root.is_file():
        visited = 1
        if PHENO_NAME_RE.search(root.name) and (not require_adhd_context or has_adhd_context(root)):
            phenotypic.append(root)
        if looks_like_roi(root, require_adhd_context):
            roi.append(root)
        return phenotypic, roi, visited, truncated
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            visited += 1
            if visited > max_files:
                truncated = True
                return phenotypic, roi, visited, truncated
            if (
                PHENO_NAME_RE.search(filename)
                and not ABIDE_NAME_RE.search(str(path))
                and (not require_adhd_context or has_adhd_context(path))
            ):
                phenotypic.append(path)
            if looks_like_roi(path, require_adhd_context):
                roi.append(path)
    return phenotypic, roi, visited, truncated


def summarize_roi_dirs(paths: list[Path]) -> list[dict[str, Any]]:
    by_dir: dict[Path, int] = {}
    examples: dict[Path, list[str]] = {}
    for path in paths:
        by_dir[path.parent] = by_dir.get(path.parent, 0) + 1
        examples.setdefault(path.parent, [])
        if len(examples[path.parent]) < 5:
            examples[path.parent].append(path.name)
    rows = [
        {"roi_dir": str(directory), "candidate_file_count": count, "examples": examples.get(directory, [])}
        for directory, count in sorted(by_dir.items(), key=lambda item: (-item[1], str(item[0])))
    ]
    return rows


def probe_url(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "sounio-adhd200-access-audit/1.0"})
    result: dict[str, Any] = {"url": url, "ok": False}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(4096)
            result.update(
                {
                    "ok": 200 <= int(response.status) < 400,
                    "status": int(response.status),
                    "content_type": response.headers.get("content-type", ""),
                    "sample_bytes": len(body),
                }
            )
    except Exception as exc:  # pragma: no cover - network environment dependent
        result["error"] = str(exc)
    return result


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_command(pheno: str, roi_dir: str, output_dir: str) -> str:
    return (
        f"PHENOTYPIC_CSV={shell_quote(pheno)} "
        f"ROI_DIR={shell_quote(roi_dir)} "
        f"OUTPUT_DIR={shell_quote(output_dir)} "
        "SOUNIO_SOUC_ENGINE=lean_single "
        "scripts/research/adhd200_dimensional_pilot_smoke.sh"
    )


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# ADHD-200 Data Access Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "## Inputs",
        "",
        f"- Explicit phenotypic CSV: `{audit['explicit_inputs'].get('phenotypic_csv') or ''}`",
        f"- Explicit ROI dir: `{audit['explicit_inputs'].get('roi_dir') or ''}`",
        f"- Search roots: {', '.join('`' + root + '`' for root in audit['search_roots'])}",
        "",
        "## Local Findings",
        "",
        f"- Phenotypic candidates: {audit['local']['phenotypic_candidate_count']}",
        f"- ROI candidate files: {audit['local']['roi_candidate_file_count']}",
    ]
    if audit["local"]["phenotypic_candidates"]:
        lines.append("")
        lines.append("Phenotypic candidates:")
        for row in audit["local"]["phenotypic_candidates"][:10]:
            required = ",".join(k for k, v in row.get("required_hits", {}).items() if v) or "none"
            primary = ",".join(k for k, v in row.get("primary_hits", {}).items() if v) or "none"
            lines.append(f"- `{row['path']}` required={required} primary={primary}")
    if audit["local"]["roi_dirs"]:
        lines.append("")
        lines.append("ROI candidate directories:")
        for row in audit["local"]["roi_dirs"][:10]:
            lines.append(f"- `{row['roi_dir']}` files={row['candidate_file_count']}")
    lines.extend(["", "## Next Action", ""])
    if audit.get("next_command"):
        lines.append("Run:")
        lines.append("")
        lines.append("```bash")
        lines.append(audit["next_command"])
        lines.append("```")
    else:
        lines.append(audit["next_action"])
    if audit.get("source_probes"):
        lines.extend(["", "## Source Probes", ""])
        for row in audit["source_probes"]:
            state = "ok" if row.get("ok") else "blocked"
            detail = row.get("status", row.get("error", ""))
            lines.append(f"- `{state}` {row['url']} {detail}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phenotypic-csv", type=Path, default=None)
    parser.add_argument("--roi-dir", type=Path, default=None)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Local root to scan. Defaults to artifacts/research and data if present.",
    )
    parser.add_argument("--max-files", type=int, default=25000)
    parser.add_argument("--probe-network", action="store_true")
    parser.add_argument("--network-timeout", type=float, default=12.0)
    parser.add_argument("--pilot-output-dir", default="/tmp/adhd200_dimensional_pilot_real")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "adhd200_data_access_audit.json"
    md_path = args.output_dir / "adhd200_data_access_audit.md"
    if not args.overwrite and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"refusing to overwrite existing audit outputs in {args.output_dir}; pass --overwrite")

    default_roots = [Path("artifacts/research"), Path("data"), Path("datasets")]
    roots = args.search_root or [root for root in default_roots if root.exists()] or [Path(".")]
    roots = [root.resolve() for root in roots]

    phenotypic_paths: list[Path] = []
    roi_paths: list[Path] = []
    scans = []
    for root in roots:
        pheno, roi, visited, truncated = scan_tree(root, args.max_files, require_adhd_context=True)
        phenotypic_paths.extend(pheno)
        roi_paths.extend(roi)
        scans.append({"root": str(root), "visited_files": visited, "truncated": truncated})

    if args.phenotypic_csv:
        phenotypic_paths.insert(0, args.phenotypic_csv.resolve())
    if args.roi_dir and args.roi_dir.exists():
        _, explicit_roi, visited, truncated = scan_tree(args.roi_dir.resolve(), args.max_files, require_adhd_context=False)
        roi_paths = explicit_roi + roi_paths
        scans.append({"root": str(args.roi_dir.resolve()), "visited_files": visited, "truncated": truncated})

    seen_pheno: set[str] = set()
    pheno_summaries = []
    for path in phenotypic_paths:
        key = str(path)
        if key in seen_pheno or not path.exists() or not path.is_file():
            continue
        seen_pheno.add(key)
        pheno_summaries.append(header_hits(path))

    seen_roi: set[str] = set()
    unique_roi_paths: list[Path] = []
    for path in roi_paths:
        key = str(path)
        if key not in seen_roi:
            seen_roi.add(key)
            unique_roi_paths.append(path)
    roi_dirs = summarize_roi_dirs(unique_roi_paths)

    explicit_pheno_ok = bool(args.phenotypic_csv and args.phenotypic_csv.exists())
    explicit_roi_ok = bool(args.roi_dir and args.roi_dir.exists() and roi_dirs)
    usable_pheno = [
        row
        for row in pheno_summaries
        if row.get("readable")
        and all(row.get("required_hits", {}).get(key) for key in ["subject", "site", "diagnosis"])
    ]
    primary_complete = [
        row
        for row in usable_pheno
        if all(row.get("primary_hits", {}).get(key) for key in PRIMARY_HINTS)
    ]
    explicit_primary_complete = bool(
        args.phenotypic_csv
        and primary_complete
        and Path(primary_complete[0]["path"]).resolve() == args.phenotypic_csv.resolve()
    )

    next_command = ""
    if explicit_pheno_ok and explicit_roi_ok and explicit_primary_complete:
        status = "ready"
        next_command = build_command(str(args.phenotypic_csv), str(args.roi_dir), args.pilot_output_dir)
        next_action = "Explicit ADHD-200 phenotypic and ROI inputs are present; run the pilot command."
    elif explicit_pheno_ok and explicit_roi_ok:
        status = "partial"
        next_action = (
            "Explicit phenotypic and ROI paths exist, but the phenotypic table is not confirmatory-ready. "
            "Check required ADHD dimensional subscale columns before running the pilot."
        )
    elif primary_complete and roi_dirs:
        status = "partial"
        next_command = build_command(primary_complete[0]["path"], roi_dirs[0]["roi_dir"], args.pilot_output_dir)
        next_action = "Candidate ADHD-200 inputs were found locally; inspect them, then run the pilot command if they are the intended dataset."
    elif usable_pheno or roi_dirs:
        status = "partial"
        next_action = (
            "Only part of the ADHD-200 input pair was found. Provide both PHENOTYPIC_CSV and ROI_DIR, "
            "or pass explicit --phenotypic-csv/--roi-dir to this audit."
        )
    else:
        status = "blocked"
        next_action = (
            "No usable local ADHD-200 phenotypic/ROI input pair was found. Acquire ADHD-200 phenotypic metadata "
            "and PCP ROI time-series derivatives manually or via the official PCP/NITRC access path, then rerun this audit."
        )

    source_probes = [probe_url(url, args.network_timeout) for url in SOURCE_URLS] if args.probe_network else []

    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "explicit_inputs": {
            "phenotypic_csv": str(args.phenotypic_csv) if args.phenotypic_csv else "",
            "roi_dir": str(args.roi_dir) if args.roi_dir else "",
        },
        "search_roots": [str(root) for root in roots],
        "scans": scans,
        "local": {
            "phenotypic_candidate_count": len(pheno_summaries),
            "usable_phenotypic_candidate_count": len(usable_pheno),
            "primary_complete_phenotypic_candidate_count": len(primary_complete),
            "phenotypic_candidates": pheno_summaries[:50],
            "roi_candidate_file_count": len(unique_roi_paths),
            "roi_dirs": roi_dirs[:50],
        },
        "source_probes": source_probes,
        "next_action": next_action,
        "next_command": next_command,
    }
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, audit)
    print(json.dumps({"status": status, "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
    return 0 if status in {"ready", "partial", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
