<!-- docs:meta
topic_id: repo.docs.decisions.adr-007-second-order-compilation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-007-second-order-compilation
-->

# ADR-007: Madaros Second-Order Compilation

**Status**: experimental
**Date**: 2026-07-12

## Context

Sounio already treats precision, uncertainty, provenance, order, effects, and
truth layers as semantic information. A conventional artifact-only compiler
contract cannot show when its own representation, optimisation, lowering,
instrumentation, or fallback choices alter that information.

For scientific work, the compiler participates in the experiment. Reporting
only the final value can make a compiler-caused erasure look like a physical
zero, an equivalence, or an absence of effect.

## Decision

Madaros will implement second-order compilation as a first-class semantic
architecture, called C2.

A C2 compilation preserves the ordinary artifact and adds a bounded receipt of
compiler intervention. Controlled paired compilations may produce a
counterfactual comparison that identifies the first aligned observed
divergence, its compiler stage, realised semantic paths, fallbacks, and blind
spots.

The normative v0 contract is
`docs/architecture/second-order-compilation.md`.

The following rules are part of the decision:

- requested semantics and realised semantics are different receipt fields;
- comparison requires pinned or explicitly declared intervention dimensions;
- a first divergence requires aligned operation identities;
- absent observed divergence supports only a bounded observation claim;
- instrumentation is declared as an intervention unless non-interference is
  established for the stated surface;
- precision, order, correction, provenance, and fallback loss cannot be
  silent;
- compiler evidence does not by itself establish physical or clinical meaning.

## Consequences

- Compiler stages that participate in a C2 witness need stable semantic
  identities or a proven reconstruction mechanism.
- Optimisation and lowering must expose protected transformation decisions.
- Backends must distinguish unsupported semantics from explicit fallback.
- Receipts require positive, negative, tamper, and blind-spot tests.
- Hardware-specific implementations remain free to change while preserving the
  same language semantics and receipt obligations.
- C2 initially remains experimental and cannot be used as a broad novelty or
  scientific-discovery claim.

## First Acceptance Boundary

The first executable boundary is one controlled source compiled and executed
under EISA v1/dd64 and EISA v2/qd128, with a receipt that classifies the first
aligned observed divergence or honestly returns an unaligned, incomparable, or
blocked status.

This boundary tests the C2 architecture. It does not relabel expansion
arithmetic as IEEE high-precision formats and does not establish native
`f128`, `f256`, or `f512` support.

## Grounded In

- ADR-002 independent truth layers;
- ADR-003 wrapper provenance preservation;
- `SOUNIO-ZERO-PROVENANCE`;
- `SOUNIO-EPISTEMIC-NUMERIC-VALUE`;
- `SOUNIO-NONASSOCIATIVE-ORDER`;
- `SOUNIO-EXPLICIT-DISCHARGE`;
- `SOUNIO-PRECISION-PRESERVATION`;
- executed EISA W1-W5 runtime and bridge-conformance evidence merged in PR
  `#832`, without extending that evidence beyond its existing claim boundary.
