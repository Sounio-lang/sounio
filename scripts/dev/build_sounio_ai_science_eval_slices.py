#!/usr/bin/env python3
"""Build focused scientific eval slices from a Sounio AI eval prompt set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "datasets/sounio-ai-eval/prompts.v2.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets/sounio-ai-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="prompts.v2")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def haystack(row: dict) -> str:
    parts = [
        str(row.get("id", "")),
        str(row.get("category", "")),
        str(row.get("source_path", "")),
        " ".join(str(tag) for tag in row.get("tags", [])),
    ]
    return " ".join(parts).lower()


def is_scientific(row: dict) -> bool:
    return row.get("category") in {"stdlib_science", "ode_pbpk", "epistemic"} or any(
        key in haystack(row)
        for key in ("ossm", "octon", "algebra", "clifford", "cayley", "pbpk", "rk4", "ode", "knowledge")
    )


def is_ode_pbpk(row: dict) -> bool:
    h = haystack(row)
    return row.get("category") == "ode_pbpk" or any(key in h for key in ("ode", "rk4", "pbpk", "pharmacokinetic", "darwin_pbpk"))


def is_ossm(row: dict) -> bool:
    return "ossm" in haystack(row) or "o_ssm" in haystack(row)


def is_algebra_octonion(row: dict) -> bool:
    return any(key in haystack(row) for key in ("octon", "algebra", "clifford", "cayley", "hypercomplex", "g2"))


def is_knowledge_octonion(row: dict) -> bool:
    h = haystack(row)
    return "knowledge" in h and "octon" in h


SLICES = {
    "scientific": is_scientific,
    "ode_pbpk": is_ode_pbpk,
    "ossm": is_ossm,
    "algebra_octonion": is_algebra_octonion,
    "knowledge_octonion": is_knowledge_octonion,
}


SUPPLEMENTAL_KNOWLEDGE_OCTONION = [
    {
        "id": "knowledge_octonion_knowledge_octonion_inner",
        "category": "knowledge_octonion",
        "difficulty": "heldout_slice",
        "prompt": "Write a Sounio program for the knowledge/octonion invariant `knowledge_octonion_inner`. Return raw Sounio code only.",
        "run": True,
        "source_path": "tests/run-pass/knowledge_octonion_inner.sio",
        "tags": ["knowledge_octonion", "octonion", "compiler_in_loop"],
    },
    {
        "id": "knowledge_octonion_knowledge_octonion_structure",
        "category": "knowledge_octonion",
        "difficulty": "heldout_slice",
        "prompt": "Write a Sounio program for the knowledge/octonion invariant `knowledge_octonion_structure`. Return raw Sounio code only.",
        "run": True,
        "source_path": "tests/run-pass/knowledge_octonion_structure.sio",
        "tags": ["knowledge_octonion", "octonion", "compiler_in_loop"],
    },
    {
        "id": "knowledge_octonion_knowledge_octonion_variance",
        "category": "knowledge_octonion",
        "difficulty": "heldout_slice",
        "prompt": "Write a Sounio program for the knowledge/octonion invariant `knowledge_octonion_variance`. Return raw Sounio code only.",
        "run": True,
        "source_path": "tests/native-v2/science_spine/knowledge_octonion_variance.sio",
        "tags": ["knowledge_octonion", "octonion", "compiler_in_loop"],
    },
    {
        "id": "knowledge_octonion_octonion_fano_168",
        "category": "knowledge_octonion",
        "difficulty": "heldout_slice",
        "prompt": "Write a Sounio program for the knowledge/octonion invariant `octonion_fano_168`. Return raw Sounio code only.",
        "run": True,
        "source_path": "tests/native-v2/science_spine/octonion_fano_168.sio",
        "tags": ["knowledge_octonion", "octonion", "compiler_in_loop"],
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.prompts)
    manifest = {
        "source_prompts": str(args.prompts.relative_to(REPO_ROOT) if args.prompts.is_relative_to(REPO_ROOT) else args.prompts),
        "source_count": len(rows),
        "slices": {},
    }
    for name, predicate in SLICES.items():
        selected = [row for row in rows if predicate(row)]
        supplemental_count = 0
        if name == "knowledge_octonion" and not selected:
            selected = SUPPLEMENTAL_KNOWLEDGE_OCTONION[:]
            supplemental_count = len(selected)
        out = args.output_dir / f"{args.prefix}.{name}.jsonl"
        write_jsonl(out, selected)
        manifest["slices"][name] = {
            "output": str(out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out),
            "count": len(selected),
            "category_counts": Counter(row.get("category", "<missing>") for row in selected),
            "supplemental_count": supplemental_count,
        }
        print(f"{name}: {len(selected)} -> {out}")

    manifest_path = args.output_dir / f"{args.prefix}.science-slices.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
