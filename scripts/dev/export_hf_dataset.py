#!/usr/bin/env python3
"""Build and optionally upload a Hugging Face dataset from Sounio tests."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "datasets" / "sounio-code-examples"
TEST_DIRS = [
    ("run-pass", ROOT / "tests" / "run-pass"),
    ("compile-fail", ROOT / "tests" / "compile-fail"),
]


@dataclass
class Example:
    source_path: str
    suite: str
    stem: str
    instruction: str
    completion: str
    description: str
    comments: list[str]
    annotations: dict[str, list[str]]
    ignore: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to write the dataset into.",
    )
    parser.add_argument(
        "--repo-id",
        default="sounio-lang/sounio-code-examples",
        help="Hugging Face dataset repo id to use when uploading.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the generated dataset to Hugging Face after export.",
    )
    return parser.parse_args()


def clean_comment(line: str) -> str:
    text = re.sub(r"^\s*//\s?", "", line).strip()
    text = text.strip()
    if not text:
        return ""
    if re.fullmatch(r"[-=*_#. ]{3,}", text):
        return ""
    return text


def parse_annotations(lines: list[str]) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    for line in lines:
        if not line.startswith("//@"):
            continue
        raw = line[3:].strip()
        if ":" in raw:
            key, value = raw.split(":", 1)
            annotations.setdefault(key.strip(), []).append(value.strip())
        else:
            annotations.setdefault(raw.strip(), []).append("true")
    return annotations


def leading_comments(lines: list[str]) -> list[str]:
    out: list[str] = []
    started_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//@"):
            continue
        if stripped.startswith("//") and not started_code:
            cleaned = clean_comment(line)
            if cleaned:
                out.append(cleaned)
            continue
        if not stripped and not started_code:
            if out:
                break
            continue
        started_code = True
        break
    return out


def build_summary(stem: str, annotations: dict[str, list[str]], comments: list[str]) -> str:
    if annotations.get("description"):
        return annotations["description"][0]
    if comments:
        joined = " ".join(comments[:3]).strip()
        joined = re.sub(rf"^{re.escape(stem)}(?:\.sio)?\s*[-:]\s*", "", joined, flags=re.IGNORECASE)
        return joined
    return stem.replace("_", " ")


def build_instruction(
    suite: str,
    stem: str,
    summary: str,
    annotations: dict[str, list[str]],
    ignore: bool,
) -> str:
    parts = [f"Write a Sounio {suite} example named `{stem}`."]
    if summary:
        summary = summary.strip()
        if not summary.endswith("."):
            summary += "."
        parts.append(summary)
    if suite == "run-pass":
        parts.append("It should compile and run successfully.")
        if annotations.get("expect-stdout"):
            parts.append(f"Expected stdout: {annotations['expect-stdout'][0]}")
    else:
        parts.append("It should fail to compile.")
        if annotations.get("error-pattern"):
            patterns = "; ".join(annotations["error-pattern"])
            parts.append(f"The compiler output should include: {patterns}.")
    if ignore:
        parts.append("This example is currently marked ignored in the upstream test suite.")
    return " ".join(parts).strip()


def load_examples() -> list[Example]:
    examples: list[Example] = []
    for suite, directory in TEST_DIRS:
        for path in sorted(directory.glob("*.sio")):
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            annotations = parse_annotations(lines)
            comments = leading_comments(lines)
            ignore = "ignore" in annotations
            summary = build_summary(path.stem, annotations, comments)
            instruction = build_instruction(suite, path.stem, summary, annotations, ignore)
            examples.append(
                Example(
                    source_path=str(path.relative_to(ROOT)),
                    suite=suite,
                    stem=path.stem,
                    instruction=instruction,
                    completion=text,
                    description=summary,
                    comments=comments,
                    annotations=annotations,
                    ignore=ignore,
                )
            )
    return examples


def split_examples(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    validation: list[Example] = []
    for idx, example in enumerate(examples):
        if idx % 10 == 0:
            validation.append(example)
        else:
            train.append(example)
    return train, validation


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            record = {
                "source_path": example.source_path,
                "suite": example.suite,
                "stem": example.stem,
                "instruction": example.instruction,
                "completion": example.completion,
                "description": example.description,
                "comments": example.comments,
                "annotations": example.annotations,
                "ignore": example.ignore,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_card(path: Path, train_count: int, validation_count: int) -> None:
    card = f"""---
license: apache-2.0
language:
- en
pretty_name: Sounio Code Examples
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- code
- compiler
- programming-language
- scientific-computing
- formal-verification
- uncertainty-propagation
- algebraic-effects
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# sounio-code-examples

Instruction/completion dataset for **Sounio**, a self-hosted systems + scientific programming language for epistemic computing.

## Contents

- `train.jsonl`: {train_count} examples
- `validation.jsonl`: {validation_count} examples
- Total: {train_count + validation_count} examples extracted from `tests/run-pass` and `tests/compile-fail`

Each record contains:

- `instruction`: natural-language prompt derived from test annotations, descriptions, and file names
- `completion`: the full `.sio` source file
- `suite`: `run-pass` or `compile-fail`
- `source_path`: original repository path
- `annotations`: extracted `//@ ...` metadata
- `ignore`: whether the upstream suite currently ignores the example

## Why this dataset exists

This dataset is designed to make Sounio legible to code models quickly:

- run-pass examples teach valid syntax and idioms
- compile-fail examples teach effect discipline, refinement failures, and epistemic boundary checks
- source paths preserve provenance back to the repository test corpus

## Rebuild locally

```bash
python3 scripts/dev/export_hf_dataset.py
```

## Upload

```bash
python3 scripts/dev/export_hf_dataset.py --upload
```

Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` before uploading.
"""
    path.write_text(card, encoding="utf-8")


def upload_dataset(output_dir: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set; exported dataset locally but cannot upload."
        )
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run `pip install huggingface_hub` and retry `--upload`."
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output_dir))


def main() -> None:
    args = parse_args()
    examples = load_examples()
    train, validation = split_examples(examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_count = write_jsonl(args.output_dir / "train.jsonl", train)
    validation_count = write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_card(args.output_dir / "README.md", train_count, validation_count)

    manifest = {
        "repo_id": args.repo_id,
        "total_examples": len(examples),
        "train_examples": train_count,
        "validation_examples": validation_count,
        "suites": {
            "run-pass": sum(1 for ex in examples if ex.suite == "run-pass"),
            "compile-fail": sum(1 for ex in examples if ex.suite == "compile-fail"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.upload:
        upload_dataset(args.output_dir, args.repo_id)

    print(
        f"Exported {len(examples)} examples to {args.output_dir} "
        f"({train_count} train / {validation_count} validation)."
    )


if __name__ == "__main__":
    main()
