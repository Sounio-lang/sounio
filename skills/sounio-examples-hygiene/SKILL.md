---
name: sounio-examples-hygiene
description: "Keep `examples/` compiling and aligned with the compiler; use when editing examples or enforcing example hygiene."
---

# Sounio Examples Hygiene

## Workflow

1) Run the example gate
- Use `scripts/check_all_examples.sh`.
- Failures are logged to `/tmp/sounio_examples_failures.txt`.

2) Fix first, stub only when a feature is missing
- Prefer minimal, targeted fixes (effects, syntax, imports) that keep the example intact.
- If a feature is unimplemented, preserve the original content inside a block comment and add a minimal compiling stub so the gate stays green.

3) Keep syntax Sounio-native
- No Rust macros or attributes in `.sio`.
- Use `var` and `&!` and explicit `with` effects.

4) Keep docs in sync if you changed visible semantics
- Update `docs/MV_CORE_CHECKLIST.md` and `docs/LLM_PROGRAMMING_GUIDE.md` when behavior shifts.

## References
- `docs/MV_CORE_CHECKLIST.md`
- `docs/LLM_PROGRAMMING_GUIDE.md`
- `.claude/commands/sounio-check.md`
- `.claude/commands/sounio-syntax.md`

## Scripts
- `scripts/check_all_examples.sh`
- `scripts/scan_syntax_drift.py`
