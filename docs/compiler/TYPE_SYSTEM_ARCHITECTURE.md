<!-- docs:meta
topic_id: website.docs.compiler.type-system
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler.type-system
-->

# Sounio Type System Architecture

The type system in the current tree is broad. It is where ordinary typing meets effects, units, epistemics, ownership, traits, refinements, and pattern reasoning. Contributor docs should reflect that breadth instead of collapsing the checker into a small Hindley-Milner story.

## Current checker map

Core typing and inference:

- `self-hosted/check/types.sio`
- `self-hosted/check/infer.sio`
- `self-hosted/check/check.sio`
- `self-hosted/check/env.sio`
- `self-hosted/check/defs.sio`

Subsystem-focused checking:

- `self-hosted/check/effects.sio`
- `self-hosted/check/units.sio`
- `self-hosted/check/epistemic.sio`
- `self-hosted/check/ownership.sio`
- `self-hosted/check/traits.sio`
- `self-hosted/check/refinement.sio`
- `self-hosted/check/patterns.sio`
- `self-hosted/check/pat_decision.sio`
- `self-hosted/check/exhaustiveness.sio`

## What the checked artifact proves

The current checked artifact advertises:

- algebraic effects with handlers
- units of measure
- refinement types
- epistemic types
- linear and affine types

That is feature breadth, not a blanket proof that every advanced code path is equally mature. For enforcement claims, prefer current run-pass and compile-fail fixtures.

Representative refusal evidence:

- `tests/compile-fail/vancomycin_low_conf.sio` demonstrates enforced epistemic refusal based on insufficient confidence

Representative success evidence:

- `tests/run-pass/vancomycin_propagation.sio`

## Documentation rules

- Use compile-fail and run-pass fixtures for enforcement claims.
- Use the checker source map for architecture claims.
- Do not imply that all advanced type subsystems are equally exercised by the default checked artifact.
