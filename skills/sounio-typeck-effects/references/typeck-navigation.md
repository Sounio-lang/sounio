# Type checking navigation (repo-local)

## Key directories

- Types and type constructors: `compiler/src/types/`
- Type-checking utilities: `compiler/src/typeck/`
- Main checker (large): `compiler/src/check/`

## Typical edit flows

- Add/adjust a type rule → update `compiler/src/check/` and any shared helpers in `compiler/src/typeck/`
- Add a new type form → update `compiler/src/types/` and make sure it roundtrips through diagnostics
- Tighten a rule → add a `tests/ui/` case + a small Rust integration test if possible

## Quick searches

- Find where a type is represented: `rg -n \"enum\\s+Type\\b|struct\\s+Type\\b\" compiler/src/types`
- Find error sites: `rg -n \"TypeError|type error\" compiler/src/check compiler/src/typeck`

