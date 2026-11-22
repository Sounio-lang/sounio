---
name: sounio-typeck-effects
description: "Work on Sounio type checking and effect system (bidirectional inference, effect inference/enforcement, and type rules); use when editing `self-hosted/check/` or `self-hosted/effects/`."
---

# Sounio Typeck + Effects

## Overview

Make changes to the type system and effect checker without drifting toward Rust or weakening Sounio’s explicit-effects contract.

## Workflow

### 1) Anchor to language identity and current reality

- Read `CLAUDE.md` for Sounio-native syntax constraints.
- Check `docs/compiler/KNOWN_LIMITATIONS.md` before implementing surface syntax the compiler doesn’t support yet.
- Treat `docs/spec/LANGUAGE_SPECIFICATION.md` as aspirational when it disagrees with the compiler.

### 2) Identify what layer you’re changing

- Parser shape: `self-hosted/parser/`
- Type rules / inference: `self-hosted/check/`
- Effects enforcement + inference: `self-hosted/effects/`

### 3) Protect effect soundness

- If an operation performs IO/mutation/allocation/etc, ensure the effect set includes it.
- If an operation changes epistemic confidence/provenance, decide whether it must be effectful and enforce it consistently.

### 4) Add high-signal tests

- Diagnostics expectations: `tests/ui/`
- Language harness: `tests/run-pass/`, `tests/compile-fail/`

## References

- Typeck map: `references/typeck-navigation.md`
- Effects map: `references/effects-navigation.md`
