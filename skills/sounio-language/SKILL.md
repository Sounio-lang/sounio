---
name: sounio-language
description: "Write and edit Sounio `.sio` code, examples, and language docs; enforce Sounio-native syntax (`var`, `&!`, explicit `with` effects) and reduce Rust-like drift across `examples/`, `tests/`, and `docs/`."
---

# Sounio Language

## Overview

Write correct Sounio and keep language-facing materials aligned with what the compiler actually accepts.

## Workflow

### 1) Decide “implemented” vs “aspirational”

- Read `compiler/docs/KNOWN_LIMITATIONS.md` and `docs/MV_CORE_CHECKLIST.md`.
- If you touch docs/examples, prefer aligning them to the compiler over adding new surface syntax.

### 2) Anchor to canonical examples

- For real, compilable syntax patterns, mirror `tests/run-pass/` and `stdlib/` (not blog-like examples).
- When unsure about a construct, search for it in:
  - `docs/LLM_PROGRAMMING_GUIDE.md`
  - `compiler/src/parser/tests/`

### 3) Enforce Sounio-native syntax (no Rust drift)

- Use `var` (not `let mut`)
- Use `&!T` (not `&mut T`)
- Avoid Rust macros in `.sio` (`println!`, `assert!`, attributes like `#[...]`)
- If drift appears, run the scan script and then update the file(s) to match Sounio conventions.

### 4) Validate quickly

- Prefer `cd compiler && cargo run -- check <file.sio>` for a single-file sanity check.
- Add/adjust a focused test in `tests/` when fixing language-facing behavior.

## References

- Repo navigation: `references/repo-navigation.md`
- Drift patterns + quick `rg` queries: `references/drift-scan-patterns.md`

## Scripts

- `scripts/scan_syntax_drift.py` scans `docs/`, `examples/`, `tests/`, `stdlib/` for Rust-like constructs and doc drift.
