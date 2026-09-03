#!/usr/bin/env python3
"""Build compiler-in-the-loop Sounio AI eval prompts from repo sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "datasets/sounio-ai-eval/prompts.generated.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=120)
    return parser.parse_args()


def category_for(path: Path) -> str:
    text = str(path).lower()
    if any(key in text for key in ("octonion", "fano", "associator", "cayley")):
        return "stdlib_science"
    if any(key in text for key in ("ossm", "o_ssm")):
        return "epistemic"
    if any(key in text for key in ("pbpk", "ode", "gum")):
        return "ode_pbpk"
    return "basic_syntax"


def expected_stdout(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "//@ expect-stdout:"
        if line.strip().startswith(marker):
            return line.split(marker, 1)[1].strip()
    stdout = path.with_suffix(".stdout")
    if stdout.exists():
        return stdout.read_text(encoding="utf-8", errors="replace").strip()
    return None


def prompt_for(index: int, path: Path) -> dict:
    rel = path.relative_to(REPO_ROOT)
    category = category_for(path)
    stem = path.stem
    return {
        "id": f"{category}_{index:03d}_{stem}",
        "category": category,
        "difficulty": "advanced" if category != "basic_syntax" else "intermediate",
        "expected_stdout": expected_stdout(path),
        "prompt": (
            f"Write one complete Sounio source file equivalent to `{stem}`. "
            f"Category: `{category}`. Contract: return raw Sounio code only, no Markdown. "
            "Use `var` for mutable bindings, `&!T` for exclusive references, no Rust macros, "
            "fixed-size arrays where needed, and explicit effects on functions."
        ),
        "run": True,
        "source_path": str(rel),
        "tags": sorted({"sounio", category, "run-pass"}),
    }


def main() -> None:
    args = parse_args()
    roots = [REPO_ROOT / "tests/run-pass", REPO_ROOT / "tests/native-v2/science_spine"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("*.sio")))
    rows = [prompt_for(i + 1, path) for i, path in enumerate(paths[: args.limit])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
