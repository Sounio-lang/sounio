---
name: sounio-typeck-effects
description: "Work on Sounio type checking and effect system (bidirectional inference, effect inference/enforcement, and type rules); use when editing `compiler/src/typeck/`, `compiler/src/types/`, `compiler/src/check/`, or `compiler/src/effects/`."
---

# Sounio Typeck + Effects

## Overview

Make changes to the type system and effect checker without drifting toward Rust or weakening Sounio’s explicit-effects contract.

## Workflow

### 1) Anchor to language identity and current reality

- Read `CLAUDE.md` for Sounio-native syntax constraints.
- Check `compiler/docs/KNOWN_LIMITATIONS.md` before implementing surface syntax the compiler doesn’t support yet.
- Treat `spec/LANGUAGE_SPECIFICATION.md` as aspirational when it disagrees with the compiler.

### 2) Identify what layer you’re changing

- Parser/AST shape: `compiler/src/parser/`, `compiler/src/ast/`
- Type rules / inference: `compiler/src/check/`, `compiler/src/typeck/`, `compiler/src/types/`
- Effects enforcement + inference: `compiler/src/effects/` and `compiler/src/types/effects.rs`

### 3) Protect effect soundness

- If an operation performs IO/mutation/allocation/etc, ensure the effect set includes it.
- If an operation changes epistemic confidence/provenance, decide whether it must be effectful and enforce it consistently.

### 4) Add high-signal tests

- Rust integration: `compiler/tests/` (e.g. semantic types and effects)
- Diagnostics expectations: `tests/ui/`
- Language harness: `tests/run-pass/`, `tests/compile-fail/`

## References

- Typeck map: `references/typeck-navigation.md`
- Effects map: `references/effects-navigation.md`
- Claude command reference: `.claude/commands/sounio-effects.md`

