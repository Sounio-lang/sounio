#!/usr/bin/env python3
"""Validate Sounio LoRA corpus and dataset assets without model dependencies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "docs" / "training" / "finetune" / "sounio_corpus.txt"
DEFAULT_CODE_DATASET = ROOT / "datasets" / "sounio-code-examples"
DEFAULT_CONTRASTIVE_DATASET = ROOT / "datasets" / "sounio-contrastive"


def fail(message: str) -> None:
    raise SystemExit(f"lora_assets_gate: FAIL: {message}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"missing JSONL: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_no}: invalid JSON: {exc}")
            if not isinstance(obj, dict):
                fail(f"{path}:{line_no}: expected object")
            rows.append(obj)
    return rows


def validate_code_split(path: Path, split: str) -> int:
    rows = read_jsonl(path)
    required = {
        "source_path",
        "suite",
        "stem",
        "instruction",
        "completion",
        "description",
        "comments",
        "annotations",
        "ignore",
    }
    seen_sources: set[str] = set()
    for idx, row in enumerate(rows, 1):
        missing = required.difference(row)
        if missing:
            fail(f"{path}:{idx}: missing fields: {sorted(missing)}")
        if row["source_path"] in seen_sources:
            fail(f"{path}:{idx}: duplicate source_path {row['source_path']}")
        seen_sources.add(row["source_path"])
        if row["suite"] not in {"run-pass", "compile-fail"}:
            fail(f"{path}:{idx}: invalid suite {row['suite']!r}")
        src = ROOT / row["source_path"]
        if not src.is_file():
            fail(f"{path}:{idx}: source_path does not exist: {row['source_path']}")
        if not str(row["instruction"]).strip():
            fail(f"{path}:{idx}: empty instruction")
        completion = str(row["completion"])
        if not completion.strip():
            fail(f"{path}:{idx}: empty completion")
        if src.read_text(encoding="utf-8", errors="replace") != completion:
            fail(f"{path}:{idx}: completion does not match source_path")
    print(f"  {split}: {len(rows)} examples")
    return len(rows)


def validate_code_dataset(path: Path) -> None:
    print(f"--> code dataset: {path}")
    train = validate_code_split(path / "train.jsonl", "train")
    validation = validate_code_split(path / "validation.jsonl", "validation")
    if train < 100:
        fail(f"train split too small: {train}")
    if validation < 10:
        fail(f"validation split too small: {validation}")
    manifest = path / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_train = data.get("train_examples", data.get("train_count"))
        manifest_validation = data.get("validation_examples", data.get("validation_count"))
        if manifest_train not in {None, train}:
            fail(f"manifest train count mismatch: {manifest_train} != {train}")
        if manifest_validation not in {None, validation}:
            fail(
                "manifest validation count mismatch: "
                f"{manifest_validation} != {validation}"
            )


def validate_contrastive_dataset(path: Path) -> None:
    print(f"--> contrastive dataset: {path}")
    contrastive = read_jsonl(path / "contrastive.jsonl")
    synthetic = read_jsonl(path / "synthetic.jsonl")
    if len(contrastive) < 50:
        fail(f"contrastive split too small: {len(contrastive)}")
    if len(synthetic) < 100:
        fail(f"synthetic split too small: {len(synthetic)}")
    for file_name, rows in {
        "contrastive.jsonl": contrastive,
        "synthetic.jsonl": synthetic,
    }.items():
        for idx, row in enumerate(rows, 1):
            if "rule" not in row:
                fail(f"{file_name}:{idx}: missing rule")
            if "explanation" not in row and "completion" not in row:
                fail(f"{file_name}:{idx}: missing explanation/completion")
    print(f"  contrastive: {len(contrastive)} examples")
    print(f"  synthetic:   {len(synthetic)} examples")


def validate_corpus(path: Path) -> None:
    print(f"--> corpus: {path}")
    if not path.is_file():
        fail(f"missing corpus: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = re.findall(r"^// === FILE: (.+) ===$", text, flags=re.MULTILINE)
    if len(markers) < 100:
        fail(f"too few file markers: {len(markers)}")
    missing = [marker for marker in markers if not (ROOT / marker).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        fail(f"corpus has markers for missing files: {preview}")
    print(f"  files: {len(markers)}")
    print(f"  bytes: {len(text.encode('utf-8'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--code-dataset", type=Path, default=DEFAULT_CODE_DATASET)
    parser.add_argument("--contrastive-dataset", type=Path, default=DEFAULT_CONTRASTIVE_DATASET)
    args = parser.parse_args()

    validate_corpus(args.corpus)
    validate_code_dataset(args.code_dataset)
    validate_contrastive_dataset(args.contrastive_dataset)
    print("lora_assets_gate: PASS")


if __name__ == "__main__":
    main()
