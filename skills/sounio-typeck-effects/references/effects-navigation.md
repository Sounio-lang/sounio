# Effects navigation (repo-local)

## Key directories

- Runtime + handler infrastructure: `compiler/src/effects/`
- Effect types: `compiler/src/types/effects.rs`
- Effect checking integration: `compiler/src/effects/inference.rs` and `compiler/src/check/`

## Quick searches

- Effect checker entry points: `rg -n \"EffectChecker|EffectError\" compiler/src/effects`
- Where `with ...` is parsed/printed: `rg -n \"\\bwith\\b\" compiler/src/parser compiler/src/fmt`

