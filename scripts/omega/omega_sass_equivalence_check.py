#!/usr/bin/env python3
"""Sounio Omega Sprint 1 - structural SASS/CUBIN equivalence proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def readelf_sections(path: Path) -> str:
    readelf = shutil.which("readelf")
    if not readelf:
        return ""
    result = subprocess.run(
        [readelf, "-W", "-S", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        result = subprocess.run(
            [readelf, "-S", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode != 0:
        return ""
    return result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omega structural equivalence check")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--section-name", default=".note.sounio_omega")
    parser.add_argument("--report", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = Path(args.baseline)
    candidate = Path(args.candidate)

    if not baseline.exists():
        print(f"equivalence error: baseline missing: {baseline}", file=sys.stderr)
        return 2
    if not candidate.exists():
        print(f"equivalence error: candidate missing: {candidate}", file=sys.stderr)
        return 2

    baseline_sha = sha256_path(baseline)
    candidate_sha = sha256_path(candidate)
    baseline_size = baseline.stat().st_size
    candidate_size = candidate.stat().st_size

    checks = {
        "kernel": args.kernel,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "baseline_size": baseline_size,
        "candidate_size": candidate_size,
        "hash_changed": candidate_sha != baseline_sha,
        "size_non_decreasing": candidate_size >= baseline_size,
        "section_present": False,
    }

    sections = readelf_sections(candidate)
    if sections:
        checks["section_present"] = args.section_name in sections

    passed = (
        checks["hash_changed"]
        and checks["size_non_decreasing"]
        and checks["section_present"]
    )
    checks["passed"] = passed

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(checks, indent=2))

    if not passed:
        print(
            "equivalence fail: "
            f"kernel={args.kernel} hash_changed={checks['hash_changed']} "
            f"size_non_decreasing={checks['size_non_decreasing']} "
            f"section_present={checks['section_present']}",
            file=sys.stderr,
        )
        return 1

    print(
        "equivalence pass: "
        f"kernel={args.kernel} baseline_sha={baseline_sha[:12]} "
        f"candidate_sha={candidate_sha[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
