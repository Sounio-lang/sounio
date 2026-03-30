<!-- docs:meta
topic_id: repo.docs.decisions.adr-005-algebra-keyword-compiler-infrastructure
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-005-algebra-keyword-compiler-infrastructure
-->

# ADR-005: `algebra` Keyword Is Compiler Infrastructure

**Status**: accepted
**Date**: 2026-03-30

## Context

Sounio's `algebra` keyword declares algebraic structure at the type level:

```sio
algebra Octonion over f64 {
    add: commutative, associative
    mul: alternative, non_commutative
    reassociate: fano_selective
}
```

This is not stdlib sugar. It affects:
- **Parsing**: `algebra` is a keyword, not a function call
- **Type checking**: algebraic properties (commutativity, associativity,
  alternativity) constrain which rewrites the optimizer may apply
- **IR optimization**: the e-graph rewriter uses Fano-selective reassociation
  rules for octonions — standard associativity rewrites would produce wrong
  results for non-associative algebras
- **Codegen**: algebra declarations inform SIMD/GPU lowering decisions

Sprint 247 (commit `8a672fcd`) proved this empirically: the Fano-selective
e-graph rewriter produces correct octonion reassociation that a generic
associativity pass would break. The algebraic structure is not a hint — it is a
semantic constraint that the compiler must enforce.

## Decision

The `algebra` keyword and its property declarations are **compiler
infrastructure**, not stdlib convenience.

Compiler-side responsibilities:
- Type family declaration and property validation
- Layout and ABI decisions informed by algebra structure
- IR rewrite rules constrained by declared properties
- Lowering strategy selection (scalar, SIMD, GPU) based on algebra dimension

Stdlib-side responsibilities:
- Concrete implementations of operations (multiply, conjugate, norm)
- Domain algorithms (rotation, interpolation, integration)
- Application-specific pipelines

The boundary: **if it changes what rewrites are legal or what codegen is
emitted, it belongs in the compiler. If it's an algorithm over those types,
it belongs in stdlib.**

## Consequences

- New algebra declarations must have compiler-side validation (property
  consistency, dimension checks).
- The e-graph rewriter must consult algebra properties before applying
  reassociation — never assume associativity.
- Future promotions (quaternion SIMD intrinsics, sedenion layout) follow the
  same gate: "does stdlib hit a wall that only compiler knowledge can solve?"
- This is the precedent for the scientific-core architecture lane.

## Grounded in

- Sprint 247: `8a672fcd` — Fano-selective e-graph, T61-T70 + T009-T018 PASS
- `algebra` keyword: `self-hosted/parser/parser.sio`, `self-hosted/check/`
- Architecture doc: `docs/architecture/scientific-core.md`
