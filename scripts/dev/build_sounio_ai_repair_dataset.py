#!/usr/bin/env python3
"""Build a small syntax-repair dataset from current Sounio sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "datasets/sounio-ai-repair/repair.generated.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def rustify(source: str) -> str:
    lines = ["Here is the implementation:", "", "```rust"]
    for line in source.splitlines():
        if line.strip().startswith("println("):
            line = line.replace("println(", "println!(")
        if line.rstrip() and not line.rstrip().endswith(("{", "}", ";")) and not line.lstrip().startswith("//"):
            line = line.rstrip() + ";"
        lines.append(line)
    lines.extend(["```", "", "This should compile and run."])
    return "\n".join(lines) + "\n"


def category_for(path: Path) -> str:
    text = str(path).lower()
    if "octonion" in text or "fano" in text or "associator" in text:
        return "stdlib_science"
    if "ossm" in text:
        return "epistemic"
    return "basic_syntax"


def main() -> None:
    args = parse_args()
    paths = sorted((REPO_ROOT / "tests/run-pass").glob("*.sio"))[: args.limit]
    rows = []
    for path in paths:
        rel = path.relative_to(REPO_ROOT)
        good = path.read_text(encoding="utf-8", errors="replace")
        category = category_for(path)
        bad = rustify(good)
        rows.append(
            {
                "id": f"repair_generated_{len(rows) + 1:04d}_{path.stem}",
                "instruction": (
                    "Return one valid Sounio source file only. Do not use Markdown. "
                    "Repair the generated attempt while preserving program behavior."
                ),
                "input": bad,
                "output": good,
                "good": good,
                "bad": bad,
                "category": category,
                "difficulty": "generated",
                "rule": "generated_rust_surface_repair",
                "repair_kind": "generated_rust_surface_repair",
                "source_path": str(rel),
                "source_prompt_id": "",
                "tags": ["sounio", "repair", category],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} repair records -> {args.output}")


if __name__ == "__main__":
    main()
