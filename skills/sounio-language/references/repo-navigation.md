# Sounio language navigation (repo-local)

## Canonical docs (start here)

- `CLAUDE.md` (repo “non-negotiables”, Sounio vs Rust)
- `docs/guide/LLM_PROGRAMMING_GUIDE.md` (examples and idioms)
- `docs/compiler/KNOWN_LIMITATIONS.md` (what is not implemented)
- `docs/implementation/MV_CORE_CHECKLIST.md` (definition of “real” core)
- `docs/spec/LANGUAGE_SPECIFICATION.md` (draft spec; may be ahead of implementation)

## Canonical “truth” for syntax

- `tests/run-pass/` (what should work)
- `tests/compile-fail/` and `tests/ui/` (what should fail + diagnostics)
- `stdlib/` (real-world Sounio usage patterns)

## Common entry points when changing syntax

- Parser: `self-hosted/parser/`
- Lexer: `self-hosted/lexer/` and `stdlib/compiler/lexer/` (Sounio lexer tests)
