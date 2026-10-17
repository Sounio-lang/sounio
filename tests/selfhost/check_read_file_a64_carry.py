#!/usr/bin/env python3
"""Source-shape gate: emit_read_file_a64 must keep Darwin envp synthesis.

The Apple Self-Host native typecheck proof runs souc-lean-check-arm64-macos
--check on a 30-byte fixture. Darwin has no procfs; io::env still speaks
the /proc/self/environ contract. The wrapper must synthesize that buffer
from argv-adjacent envp and dispatch every other path to
emit_read_file_ordinary_a64.

Ordinary-file open/mmap/read carry checks change lean_single codegen and
force a seed refresh (canonical_compiler_gate). They are out of scope for
this FO-inline PR: main already refreshed the seed around the envp wrapper.
Do not re-introduce ordinary carry here without a founder seed resync.
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
    if "with Mut, Panic {" not in wrapper.split("\n", 1)[0]:
        print("FAIL emit_read_file_a64 must match main's effect set (no extra Div)")
        return 1
    if "SYS_open" not in body and "openat" not in body:
        print("FAIL emit_read_file_ordinary_a64 must still emit open/openat")
        return 1
    print("PASS emit_read_file_a64 Darwin envp wrapper + ordinary dispatch present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
