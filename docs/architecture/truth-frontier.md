<!-- docs:meta
topic_id: repo.docs.architecture.truth-frontier
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.truth-frontier
-->

# Rebuilt Checker Truth Frontier

## Summary

This note records the current tiny truth frontier for the rebuilt ontology
checker path.

It does not change ontology semantics. It records where the rebuilt direct
driver is currently trusted, where wrapper truth is still mixed with fallback
compile evidence, and where direct negative-truth restoration should focus
next.

Related docs:

- [semantic-contracts.md](./semantic-contracts.md)
- [compiler-maturity-blueprint.md](./compiler-maturity-blueprint.md)
- [truth-layers.md](./truth-layers.md)

## Current Truth Policy

- `rebuilt_direct` can be treated as provisionally trustworthy only where the
  direct driver verdict agrees with the wrapper and fallback compile verdict.
- `mixed` means the wrapper had to combine rebuilt direct output with fallback
  compile evidence.
- `unknown` does not count as semantic success.
- semantic closure stays paused until the direct rebuilt driver can reject a
  minimal bad frontier without fallback help.

## Tiny Truth Frontier

### Matrix

| File | Intended class | Direct rebuilt driver | Stage bits | Wrapper verdict | Provenance | Fallback compile |
|---|---|---|---|---|---|---|
| `self-hosted/ci/ontology_min_input.sio` | good | `witness=0`, `verdict=ok`, `rc=0` | `load=0 parse=0 collect=0 check=0` | `ok`, `rc=0` | `rebuilt_direct` | `ok` |
| `tests/run-pass/algebra_decl_basic.sio` | good | `witness=0`, `verdict=ok`, `rc=0` | `load=0 parse=0 collect=0 check=0` | `ok`, `rc=0` | `rebuilt_direct` | `ok` |
| `tests/compile-fail/ontology_subclass_reject.sio` | bad | `witness=0`, `verdict=ok`, `rc=0` | `load=0 parse=0 collect=0 check=0` | `unknown`, `rc=3` | `mixed` | `reject` |
| `tests/compile-fail/ontology_type_mismatch.sio` | bad | `witness=0`, `verdict=ok`, `rc=0` | `load=0 parse=0 collect=0 check=0` | `unknown`, `rc=3` | `mixed` | `reject` |
| `tests/compile-fail/acquisition_reason_requires_plan.sio` | bad | `witness=0`, `verdict=ok`, `rc=0` | `load=0 parse=0 collect=0 check=0` | `unknown`, `rc=3` | `mixed` | `reject` |

### Frontier Reading

- The direct rebuilt driver currently collapses all five files to the same
  coarse result: `witness=0`, `verdict=ok`, all stage bits unset.
- The wrapper is still operationally useful because it preserves provenance and
  converts rebuilt/fallback disagreement into `unknown`.
- No current tiny-frontier case is decided by `fallback_compile` alone.

## Trust Boundary

### Provisionally trusted today

- `self-hosted/ci/ontology_min_input.sio`
- `tests/run-pass/algebra_decl_basic.sio`

These are only provisionally trusted because the direct driver still emits the
collapsed `witness=0` shape. They are acceptable operationally because rebuilt
direct and fallback compile agree on `ok`.

### Still mixed / unknown today

- `tests/compile-fail/ontology_subclass_reject.sio`
- `tests/compile-fail/ontology_type_mismatch.sio`
- `tests/compile-fail/acquisition_reason_requires_plan.sio`

These remain outside the direct-driver trust frontier. Their rejection is still
coming from wrapper-level disagreement against fallback compile.

## Next Use

This frontier is sufficient to start direct-driver negative-truth restoration
work.

Why:

- the bad frontier is small
- the collapse pattern is stable across all three bad fixtures
- the wrapper now records that the current authority is mixed, not rebuilt
  direct truth

This frontier is not sufficient to reopen semantic closure work yet.
