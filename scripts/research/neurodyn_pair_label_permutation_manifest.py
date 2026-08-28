#!/usr/bin/env python3
"""Build a pair-preserving label-permutation null manifest.

This is for synthetic paired manifests with subject ids like
`synthetic_pair_0000__oriented` and `synthetic_pair_0000__swapped`.
For each pair, a seeded coin flip either keeps or swaps the two labels.
Features, sites, row order, and one-positive/one-negative pair balance are
preserved. The global relationship between orientation and label is broken.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.pair_label_permutation_manifest.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_id(subject_id: str) -> str:
    marker = "__"
    if marker not in subject_id:
        raise ValueError(f"subject_id lacks pair suffix: {subject_id}")
    return subject_id.split(marker, 1)[0]


def read_manifest(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    comments: list[str] = []
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            if raw.startswith("#"):
                comments.append(raw.rstrip("\n"))
                continue
            if not raw.strip():
                continue
            header = raw.rstrip("\n").split("\t")
            break
        if header is None:
            raise SystemExit(f"manifest header not found: {path}")
        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")
        rows = [dict(row) for row in reader if row.get("subject_id")]
    required = {"subject_id", "label", "site"}
    missing = required.difference(header)
    if missing:
        raise SystemExit(f"manifest missing required columns: {sorted(missing)}")
    return comments, header, rows


def write_manifest(path: Path, comments: list[str], header: list[str], rows: list[dict[str, str]], seed: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for comment in comments:
            if comment.startswith("# label_space="):
                handle.write("# label_space=pair_label_permutation_null\n")
            elif comment.startswith("# label_positive_name="):
                handle.write("# label_positive_name=PERMUTED_POSITIVE\n")
            elif comment.startswith("# label_negative_name="):
                handle.write("# label_negative_name=PERMUTED_NEGATIVE\n")
            elif comment.startswith("# task="):
                handle.write("# task=pair_label_permutation_null_control\n")
            else:
                handle.write(comment + "\n")
        handle.write(f"# null_control=pair_label_permutation\n")
        handle.write(f"# permutation_seed={seed}\n")
        handle.write("# claim_boundary=Synthetic non-clinical null control only. No clinical, biomarker, mechanistic, or O-SSM superiority claim.\n")
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comments, header, rows = read_manifest(args.input)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[pair_id(row["subject_id"])].append(row)
    bad = {key: value for key, value in groups.items() if len(value) != 2}
    if bad:
        raise SystemExit(f"expected exactly two rows per pair; bad pairs={sorted(bad)[:5]}")

    rng = random.Random(args.seed)
    flip_by_pair: dict[str, int] = {}
    output_rows = [dict(row) for row in rows]
    by_subject = {row["subject_id"]: row for row in output_rows}
    for key in sorted(groups):
        pair_rows = groups[key]
        labels = sorted(int(row["label"]) for row in pair_rows)
        if labels != [0, 1]:
            raise SystemExit(f"expected one label 0 and one label 1 for {key}: {labels}")
        flip = rng.randrange(2)
        flip_by_pair[key] = flip
        if flip:
            for row in pair_rows:
                by_subject[row["subject_id"]]["label"] = "0" if row["label"] == "1" else "1"

    manifest = args.output_dir / "pair_label_permutation_manifest.tsv"
    write_manifest(manifest, comments, header, output_rows, args.seed)

    site_label_counts = {
        site: dict(Counter(row["label"] for row in output_rows if row["site"] == site))
        for site in sorted({row["site"] for row in output_rows})
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "input": str(args.input),
        "manifest": str(manifest),
        "seed": args.seed,
        "pair_count": len(groups),
        "row_count": len(output_rows),
        "flipped_pair_count": sum(flip_by_pair.values()),
        "kept_pair_count": len(groups) - sum(flip_by_pair.values()),
        "label_counts": dict(Counter(row["label"] for row in output_rows)),
        "site_label_counts": site_label_counts,
        "input_sha256": sha256_file(args.input),
        "manifest_sha256": sha256_file(manifest),
    }
    (args.output_dir / "pair_label_permutation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(args.output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
