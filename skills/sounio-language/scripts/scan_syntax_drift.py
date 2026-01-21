#!/usr/bin/env python3
"""
Scan the repo for Rust-like syntax drift in Sounio-facing files.

Focus:
- `.sio` / `.sounio` source files
- Markdown docs that describe or demonstrate Sounio

Default scope (curated, actionable):
- `examples/`
- `tests/`
- `README.md`, `DEVELOPER.md`, `tests/README.md`, `docs/LLM_PROGRAMMING_GUIDE.md`

Opt-in:
- `--include-stdlib` to scan `stdlib/`
- `--docs-all` to scan `docs/` broadly (in addition to the guide)

Usage:
  python3 skills/sounio-language/scripts/scan_syntax_drift.py
  python3 skills/sounio-language/scripts/scan_syntax_drift.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    path: Path
    line_no: int
    line: str


RULES: list[tuple[str, str, str]] = [
    ("error", "rust-macro", r"\b(?:println!|assert!)\b"),
    ("error", "rust-borrow", r"\b&mut\b"),
    ("error", "rust-let-mut", r"\blet\s+mut\b"),
    ("error", "rust-attr", r"#\["),
    ("warn", "doc-dot-d", r"\.d\b"),
    ("warn", "module-keyword", r"^\s*module\s+"),
]

SOUNIO_MD_LANGS = {"sio", "sounio", "d"}  # `d` appears in older docs as Sounio fence


def iter_candidate_files(root: Path, include_stdlib: bool, docs_all: bool) -> Iterator[Path]:
    curated = [
        root / "README.md",
        root / "DEVELOPER.md",
        root / "tests" / "README.md",
        root / "docs" / "LLM_PROGRAMMING_GUIDE.md",
    ]
    for path in curated:
        if path.is_file():
            yield path

    for folder in ("examples", "tests"):
        base = root / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".sio", ".sounio", ".md"}:
                yield path

    if include_stdlib:
        base = root / "stdlib"
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in {".sio", ".sounio", ".md"}:
                    yield path

    if docs_all:
        base = root / "docs"
        if base.exists():
            for path in base.rglob("*.md"):
                if path.is_file():
                    yield path


def is_comment_line_sounio(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def iter_markdown_sounio_code_lines(text: str) -> Iterator[tuple[int, str]]:
    """
    Yield (line_no, line) for lines inside fenced code blocks labeled as Sounio.
    Supports fences like ```sio, ```sounio, ```d (legacy).
    """
    in_fence = False
    fence_lang: str | None = None

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                fence_lang = stripped[3:].strip().lower() or None
                in_fence = True
            else:
                in_fence = False
                fence_lang = None
            continue

        if in_fence and fence_lang in SOUNIO_MD_LANGS:
            yield line_no, line


def scan_file(root: Path, path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    findings: list[Finding] = []
    suffix = path.suffix.lower()
    try:
        rel = path.relative_to(root)
        top = rel.parts[0] if rel.parts else ""
    except ValueError:
        top = ""
    is_tests = top == "tests"

    if suffix in {".sio", ".sounio"}:
        in_block_comment = False
        for line_no, line in enumerate(text.splitlines(), start=1):
            if in_block_comment:
                end_idx = line.find("*/")
                if end_idx == -1:
                    continue
                line = line[end_idx + 2 :]
                in_block_comment = False

            while True:
                start_idx = line.find("/*")
                if start_idx == -1:
                    break
                end_idx = line.find("*/", start_idx + 2)
                if end_idx == -1:
                    line = line[:start_idx]
                    in_block_comment = True
                    break
                line = line[:start_idx] + line[end_idx + 2 :]

            if is_comment_line_sounio(line) or not line.strip():
                continue
            for severity, rule, pattern in RULES:
                if rule == "doc-dot-d":
                    continue
                if re.search(pattern, line):
                    effective_severity = severity if is_tests else "warn"
                    findings.append(
                        Finding(
                            severity=effective_severity,
                            rule=rule,
                            path=path,
                            line_no=line_no,
                            line=line.strip(),
                        )
                    )
        return findings

    if suffix == ".md":
        # Always scan markdown prose for stale `.d` references.
        for line_no, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\.d\b", line):
                findings.append(
                    Finding(
                        severity="warn",
                        rule="doc-dot-d",
                        path=path,
                        line_no=line_no,
                        line=line.strip(),
                    )
                )

        # Scan only Sounio-labeled fenced code blocks for syntax drift.
        for line_no, line in iter_markdown_sounio_code_lines(text):
            for severity, rule, pattern in RULES:
                if rule in {"doc-dot-d"}:
                    continue
                # Docs frequently include Rust-like snippets (sometimes as counterexamples),
                # so keep these as warnings by default.
                effective_severity = "warn"
                if re.search(pattern, line):
                    findings.append(
                        Finding(
                            severity=effective_severity,
                            rule=rule,
                            path=path,
                            line_no=line_no,
                            line=line.strip(),
                        )
                    )
        return findings

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-stdlib", action="store_true")
    parser.add_argument("--docs-all", action="store_true")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    findings: list[Finding] = []

    for path in iter_candidate_files(root, include_stdlib=args.include_stdlib, docs_all=args.docs_all):
        findings.extend(scan_file(root, path))

    if not findings:
        print("No drift findings.")
        return 0

    findings.sort(key=lambda f: (str(f.path), f.line_no, f.rule))
    errors = 0
    for f in findings:
        if f.severity == "error":
            errors += 1
        rel = f.path.relative_to(root) if f.path.is_relative_to(root) else f.path
        print(f"[{f.severity}] {f.rule}: {rel}:{f.line_no}: {f.line}")

    print(f"\nFindings: {len(findings)} (errors: {errors})")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
