# Sounio language navigation (repo-local)

## Canonical docs (start here)

- `CLAUDE.md` (repo “non-negotiables”, Sounio vs Rust)
- `docs/LLM_PROGRAMMING_GUIDE.md` (examples and idioms)
- `compiler/docs/KNOWN_LIMITATIONS.md` (what is not implemented)
- `docs/MV_CORE_CHECKLIST.md` (definition of “real” core)
- `spec/LANGUAGE_SPECIFICATION.md` (draft spec; may be ahead of implementation)

## Canonical “truth” for syntax

- `tests/run-pass/` (what should work)
- `tests/compile-fail/` and `tests/ui/` (what should fail + diagnostics)
- `stdlib/` (real-world Sounio usage patterns)
- `compiler/src/parser/tests/` (parser-level truth for new syntax)

## Common entry points when changing syntax

- Parser: `compiler/src/parser/mod.rs`
- AST: `compiler/src/ast/mod.rs`
- Lexer: `compiler/src/lexer/` (Rust lexer) and `stdlib/compiler/lexer/` (Sounio lexer tests)
