---
name: sounio-examples-hygiene
description: "Keep `examples/` compiling and aligned with the compiler; use when editing examples or enforcing example hygiene."
---

# Sounio Examples Hygiene

## Workflow

1) Run the example gate
- Use `scripts/dev/gen_examples_index.sh` (runs every `examples/**/*.sio` that has `fn main` and writes `examples/INDEX.md`).

2) Fix first, stub only when a feature is missing
- Prefer minimal, targeted fixes (effects, syntax, imports) that keep the example intact.
- If a feature is unimplemented, preserve the original content inside a block comment and add a minimal compiling stub so the gate stays green.

3) Keep syntax Sounio-native
- No Rust macros or attributes in `.sio`.
- Use `var` and `&!` and explicit `with` effects.
- Drift scan: `skills/sounio-language/scripts/scan_syntax_drift.py` (also invoked by `scripts/dev/fast_gate.sh`).

4) Keep docs in sync if you changed visible semantics
- Update `docs/implementation/MV_CORE_CHECKLIST.md` and `docs/guide/LLM_PROGRAMMING_GUIDE.md` when behavior shifts.

## References
- `docs/implementation/MV_CORE_CHECKLIST.md`
- `docs/guide/LLM_PROGRAMMING_GUIDE.md`

## Scripts
- `scripts/dev/gen_examples_index.sh`
- `skills/sounio-language/scripts/scan_syntax_drift.py`
