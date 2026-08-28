#!/usr/bin/env python3
"""Audit Lean proof status without counting comments.

This is intentionally small and dependency-free. It strips Lean line comments
and nested block comments well enough for repository status accounting, then
counts actual `sorry` and `axiom` tokens in `.lean` files.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEAN_ROOT = ROOT / "formal" / "lean4"
TOKEN_RE = re.compile(r"\b(sorry|axiom)\b")
THEOREM_RE = re.compile(r"\b(theorem|lemma|axiom)\b")


def strip_lean_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    depth = 0
    while i < len(text):
        two = text[i : i + 2]

        if depth == 0 and two == "--":
            while i < len(text) and text[i] != "\n":
                i += 1
            if i < len(text):
                out.append("\n")
                i += 1
            continue

        if two == "/-":
            depth += 1
            i += 2
            continue

        if depth > 0:
            if two == "-/":
                depth -= 1
                i += 2
            else:
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            continue

        out.append(text[i])
        i += 1

    return "".join(out)


def main() -> int:
    files = sorted(LEAN_ROOT.glob("*.lean"))
    total_sorry = 0
    total_axiom = 0
    modules_with_sorry = 0
    modules_with_axiom = 0
    findings: list[str] = []

    print(f"lean_root={LEAN_ROOT}")
    print(f"lean_files={len(files)}")
    print()
    print("## Per-module status")

    for path in files:
        original = path.read_text(encoding="utf-8")
        stripped = strip_lean_comments(original)
        sorry_count = 0
        axiom_count = 0
        theorem_like = len(THEOREM_RE.findall(stripped))
        stripped_lines = stripped.splitlines()
        original_lines = original.splitlines()

        for idx, line in enumerate(stripped_lines, start=1):
            for match in TOKEN_RE.finditer(line):
                kind = match.group(1)
                if kind == "sorry":
                    sorry_count += 1
                elif kind == "axiom":
                    axiom_count += 1
                original_line = original_lines[idx - 1].strip() if idx <= len(original_lines) else line.strip()
                findings.append(f"{path.name}:{idx}:{kind}: {original_line}")

        total_sorry += sorry_count
        total_axiom += axiom_count
        if sorry_count:
            modules_with_sorry += 1
        if axiom_count:
            modules_with_axiom += 1

        print(f"{path.name}: theorem_like={theorem_like} sorry={sorry_count} axiom={axiom_count}")

    print()
    print("## Totals")
    print(f"sorry_tokens={total_sorry}")
    print(f"axiom_tokens={total_axiom}")
    print(f"modules_with_sorry={modules_with_sorry}")
    print(f"modules_with_axiom={modules_with_axiom}")

    if findings:
        print()
        print("## Token findings")
        for finding in findings:
            print(finding)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
