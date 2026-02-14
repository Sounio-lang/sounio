#!/usr/bin/env python3
"""Audit stdlib/compiler/*.sio `use`/`import` module paths against Rust ModuleLoader rules.

This is a bootstrap helper for Project Poseidon.
It does NOT try to fully parse Sounio; it uses a conservative regex scan.

Output:
- total statements
- unique module paths
- unresolved module paths (with first few example source locations)

Resolution rules intentionally mirror crates/souc/src/module_loader.rs:
- direct/local: search current dir, then parent dir, then stdlib root
- candidates: <path>.sio, <path>/mod.sio, <path>/lib.sio
- stdlib-qualified: `std::foo` resolves in stdlib root as `foo`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

RE_STMT = re.compile(r"^\s*(use|import)\s+([^;]+);\s*$")


@dataclass(frozen=True)
class Hit:
    kind: str
    module_path: tuple[str, ...]
    file: Path
    line_no: int
    raw: str


def extract_module_path(raw_path: str) -> tuple[str, ...] | None:
    s = raw_path.strip()

    # Strip trailing comments.
    if "//" in s:
        s = s.split("//", 1)[0].strip()

    # Drop braces selector: core::{A,B} -> core
    if "{" in s:
        s = s.split("{", 1)[0].strip()

    # Drop glob selector: foo::* -> foo
    if s.endswith("::*"):
        s = s[: -len("::*")]

    # For `use a::b::C` we treat it as module `a::b`.
    parts = [p for p in s.split("::") if p]
    if not parts:
        return None

    # If last segment is likely an item (capitalized or starts with _),
    # treat it as item import and remove it.
    if len(parts) > 1:
        last = parts[-1]
        if last[:1].isupper() or last.startswith("_"):
            parts = parts[:-1]

    return tuple(parts)


def scan_use_import(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for file in sorted(root.rglob("*.sio")):
        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = RE_STMT.match(line)
            if not m:
                continue
            kind, raw_path = m.group(1), m.group(2)
            mod_path = extract_module_path(raw_path)
            if not mod_path:
                continue
            hits.append(Hit(kind=kind, module_path=mod_path, file=file, line_no=i, raw=line))
    return hits


def resolve_in_directory(base: Path, segments: tuple[str, ...]) -> Path | None:
    joined = "/".join(segments)
    candidates = [
        base / f"{joined}.sio",
        base / joined / "mod.sio",
        base / joined / "lib.sio",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_like_rust(current_file: Path, segments: tuple[str, ...], stdlib_root: Path) -> Path | None:
    if not segments:
        return None

    # std::foo resolves inside stdlib root.
    if segments[0] == "std":
        if len(segments) == 1:
            return None
        segments = segments[1:]
        return resolve_in_directory(stdlib_root, segments)

    local_dir = current_file.parent

    hit = resolve_in_directory(local_dir, segments)
    if hit:
        return hit

    parent = local_dir.parent
    if parent != local_dir:
        hit = resolve_in_directory(parent, segments)
        if hit:
            return hit

    return resolve_in_directory(stdlib_root, segments)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stdlib_root = repo / "stdlib"
    compiler_root = stdlib_root / "compiler"

    hits = scan_use_import(compiler_root)

    by_mod: dict[tuple[str, ...], list[Hit]] = {}
    for h in hits:
        by_mod.setdefault(h.module_path, []).append(h)

    unresolved: dict[tuple[str, ...], list[Hit]] = {}
    resolved: dict[tuple[str, ...], Path] = {}

    for mod, hs in by_mod.items():
        # Use the first occurrence for resolution context.
        h0 = hs[0]
        path = resolve_like_rust(h0.file, mod, stdlib_root)
        if path is None:
            unresolved[mod] = hs
        else:
            resolved[mod] = path

    print("=== Poseidon stdlib/compiler import audit ===")
    print(f"Statements: {len(hits)}")
    print(f"Unique module paths: {len(by_mod)}")
    print(f"Resolved: {len(resolved)}")
    print(f"Unresolved: {len(unresolved)}")

    if unresolved:
        print("\n--- Unresolved module paths (top 50) ---")
        for mod in sorted(unresolved.keys())[:50]:
            locs = unresolved[mod][:3]
            loc_str = ", ".join(f"{h.file.relative_to(repo)}:{h.line_no}" for h in locs)
            print(f"- {'::'.join(mod)}  ({len(unresolved[mod])} hit(s))  e.g. {loc_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
