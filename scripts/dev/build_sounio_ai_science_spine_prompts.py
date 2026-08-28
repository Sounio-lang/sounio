#!/usr/bin/env python3
"""Convert the native-v2 science-spine manifest TSV into eval prompt JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tests/native-v2/science_spine/manifest.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/sounio-ai-eval/prompts.science_spine_reference.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_expected(path_text: str) -> str | None:
    if not path_text:
        return None
    path = REPO_ROOT / path_text
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def main() -> None:
    args = parse_args()
    rows = []
    with args.manifest.open("r", encoding="utf-8") as fh:
        header = fh.readline().lstrip("# ").rstrip("\n").split("\t")
        reader = csv.DictReader(fh, fieldnames=header, delimiter="\t")
        for row in reader:
            if not row.get("program"):
                continue
            case_id = row["case_id"]
            claim_class = row.get("claim_class") or "science_spine"
            rows.append(
                {
                    "id": f"science_spine_{case_id}",
                    "category": "science_spine",
                    "difficulty": "reference",
                    "expected_stdout": read_expected(row.get("expected_stdout", "")),
                    "prompt": (
                        f"Write the native-v2 science-spine Sounio program `{case_id}`. "
                        f"Claim class: `{claim_class}`. Return raw Sounio code only."
                    ),
                    "run": True,
                    "source_path": row["program"],
                    "tags": ["science_spine", claim_class, "reference_smoke"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} science-spine prompts -> {args.output}")


if __name__ == "__main__":
    main()
