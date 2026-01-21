# Stdlib navigation

## Indexes and status

- `STDLIB_REFERENCE.md` (module-by-module status and intended coverage)

## Common module roots

- `stdlib/core/` (Option/Result)
- `stdlib/epistemic/` (knowledge + uncertainty semantics; see `SEMANTICS.md`)
- `stdlib/async/`, `stdlib/io/`, `stdlib/ffi/`
- `stdlib/ode/`, `stdlib/linalg/`, `stdlib/stats/`, `stdlib/random/`

## Finding patterns

- Find a symbol definition: `rg -n \"fn <name>\\b|struct <name>\\b\" stdlib`
- Find a module entrypoint: `rg -n \"^import|^module|mod\\.sio\" stdlib`
