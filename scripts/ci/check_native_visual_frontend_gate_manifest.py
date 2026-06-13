#!/usr/bin/env python3
"""Validate the native visual frontend gate manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


VALID_MODES = {"check", "run", "compile", "script"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="scripts/ci/native_visual_frontend_gate.manifest.tsv",
        help="Path to the TSV manifest, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest

    errors: list[str] = []
    labels: set[str] = set()
    counts = {"check": 0, "run": 0, "compile": 0, "script": 0}
    required_labels = {
        "viz_headless",
        "viz_workbench_roundtrip",
        "viz_workbench_replay_session",
        "viz_workbench_replay_html_archive",
        "viz_html_passive_export",
        "viz_workbench_demo_run",
    }

    if not manifest.exists():
        print(f"error: manifest missing: {manifest}", file=sys.stderr)
        return 1

    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            errors.append(f"{lineno}: expected 4 TSV columns, got {len(parts)}")
            continue
        mode, label, rel_path, marker = parts
        if mode not in VALID_MODES:
            errors.append(f"{lineno}: invalid mode {mode!r}")
            continue
        if not label:
            errors.append(f"{lineno}: empty label")
        if label in labels:
            errors.append(f"{lineno}: duplicate label {label!r}")
        labels.add(label)
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(f"{lineno}: path must be repo-relative and not escape root: {rel_path!r}")
        else:
            path = root / rel_path
            if not path.exists():
                errors.append(f"{lineno}: missing path {rel_path!r}")
            elif mode == "script" and not path.is_file():
                errors.append(f"{lineno}: script entry must point to a file: {rel_path!r}")
            elif mode == "script" and not path.stat().st_mode & 0o111:
                errors.append(f"{lineno}: script entry must be executable: {rel_path!r}")
        if mode == "run":
            if marker == "-" or not marker:
                errors.append(f"{lineno}: run entries require an output marker")
        elif marker != "-":
            errors.append(f"{lineno}: non-run entries must use '-' marker")
        counts[mode] += 1

    missing_required = sorted(required_labels - labels)
    if missing_required:
        errors.append("missing required labels: " + ", ".join(missing_required))

    if counts["run"] < 10:
        errors.append("manifest must contain at least 10 run entries")
    if counts["compile"] < 3:
        errors.append("manifest must contain at least 3 compile entries")
    if counts["check"] < 2:
        errors.append("manifest must contain at least 2 check entries")
    if counts["script"] < 1:
        errors.append("manifest must contain at least 1 script entry")

    if errors:
        for err in errors:
            print(f"error: native visual frontend manifest: {err}", file=sys.stderr)
        return 1

    print(
        "native_visual_frontend_gate_manifest: PASS "
        f"check={counts['check']} run={counts['run']} compile={counts['compile']} script={counts['script']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
