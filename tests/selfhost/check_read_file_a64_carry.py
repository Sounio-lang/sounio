#!/usr/bin/env python3
"""Source-shape gate: emit_read_file_a64 must normalize Darwin carry.

The Apple Self-Host native typecheck proof runs souc-lean-check-arm64-macos
--check on a 30-byte fixture. Without carry checks after open/mmap/read,
#2832's loop treats +errno as bytes and prints the 4 MiB refuse.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "self-hosted" / "compiler" / "lean_single.sio"


def extract_fn(text: str, name: str) -> str:
    start = text.find(f"fn {name}(")
    if start < 0:
        raise SystemExit(f"missing fn {name}")
    # Next top-level fn after this one.
    nxt = text.find("\nfn ", start + 1)
    if nxt < 0:
        raise SystemExit(f"could not bound fn {name}")
    return text[start:nxt]


def main() -> int:
    text = SRC.read_text()
    wrapper = extract_fn(text, "emit_read_file_a64")
    body = extract_fn(text, "emit_read_file_ordinary_a64")
    if "emit_read_file_ordinary_a64()" not in wrapper:
        print("FAIL emit_read_file_a64 must dispatch to emit_read_file_ordinary_a64")
        return 1
    if "/proc/self/environ" not in wrapper:
        print("FAIL emit_read_file_a64 must keep Darwin envp synthesis")
        return 1
    required = [
        "bss_alloc_aligned(8)",
        "b.cc open_ok",
        "b.cc mmap_ok",
        "b.cc ok (patched)",
        "file larger than 4 MiB is not supported on this target",
    ]
    missing = [item for item in required if item not in body]
    if missing:
        print("FAIL emit_read_file_ordinary_a64 missing carry/mmap guards:")
        for item in missing:
            print(f"  {item}")
        return 1
    if "with Mut, Panic, Div" not in body.split("{", 1)[0]:
        print("FAIL emit_read_file_ordinary_a64 must declare Div for bss_alloc_aligned")
        return 1
    print("PASS emit_read_file_a64 Darwin envp + ordinary carry/mmap guards present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
