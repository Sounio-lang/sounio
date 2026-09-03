<!-- docs:meta
topic_id: repo.docs.research.m0-baseline-inventory
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.m0-baseline-inventory
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# M0 Baseline Inventory — What Already Runs

**Purpose**: snapshot of what Sounio infrastructure exists *before* the vancomycin-Knightian thrust starts. Used to scope M1-M3 and to detect breakage during the project.

**Date**: 2026-04-30

## Existing vancomycin pipeline

- `tests/run-pass/vancomycin_propagation.sio` — v1 with conservative `KnowledgeF64` struct, GUM mul/div, confidence gate at 0.82. **Status: passes baseline.**
- `tests/run-pass/vancomycin_propagation.sio` uses inline struct, not `stdlib/epistemic/knowledge.sio` `Epistemic` struct (different field names: `confidence: f64` vs `confidence: i64`). M3 v2 will reconcile.

## Existing epistemic stdlib (relevant subset)

- `stdlib/epistemic/knowledge.sio` — `Epistemic` struct (val/variance/confidence:i64), GUM ops (add/sub/mul/div), `is_credible(min_conf)`.
- `stdlib/epistemic/causal.sio` — `CausalDAG` (16 nodes, 64 edges), `EcBeta` distribution, backdoor adjustment, Pearl do-calculus.
- `stdlib/epistemic/observe.sio` — Bayesian normal-normal conjugate update, information gain in nats.
- `stdlib/epistemic/gum.sio` — GUM uncertainty propagation primitives.
- `stdlib/epistemic/multivariate.sio` — covariance-aware propagation.
- `stdlib/epistemic/sobol.sio` — Sobol sensitivity indices.
- `stdlib/epistemic/montecarlo.sio` — MC sampling.
- `stdlib/epistemic/aleatoric.sio` — aleatoric vs epistemic distinction.
- `stdlib/epistemic/confidence_gate.sio` — gating predicates.
- `stdlib/epistemic/budget.sio`, `budget64.sio` — uncertainty budgets (JCGM 100:2008 format).
- `stdlib/epistemic/correlation.sio`, `covariance.sio` — joint distributional pieces.

71 files total in `stdlib/epistemic/`.

## Effects already implemented

From compiler (`crates/souc/src/types/core.rs:973–1076` per `formal/lean4/SounioEffects.lean` references):
- `IO`, `Mut`, `Alloc`, `Prob`, `GPU`, `Epistemic`, `Div`, `Exn`, `Async`, `FFI`, `NonAssoc`
- **`Approx`** (commit `9c5fb768`)
- **`Causal`** (commit `2112074d`)
- **`Observe`** (used in `stdlib/epistemic/observe.sio`)

Composition between `Approx`, `Causal`, `Knowledge`, `Observe` is the **gap M1 fills**.

## Existing Lean 4 corpus (relevant subset)

60 Lean files in `formal/`. Directly relevant to vancomycin thrust:

- `formal/lean4/SounioEffects.lean` — algebraic effect rows, EffectRow as Effect → Bool.
- `formal/lean4/SounioCausality.lean` — causal semantics.
- `formal/lean4/SounioEpistemic.lean` — epistemic types.
- `formal/SecondOrderGUM.lean` — second-order GUM derivation (variance of variance).
- `formal/lean4/SounioMeasConf.lean` — measurement confidence.
- `formal/GUM.lean` — first-order GUM in Lean.
- `formal/Epistemic.lean` — top-level epistemic.
- `formal/KnowledgeArithmeticSoundness.lean` — soundness of Knowledge<T> arithmetic.

Gap: no `SounioApproxCausalKnowledge.lean` (composition); no `SounioKnightian.lean` (M2); no `proof_obligations/vancomycin_dosing_safety.lean` (M4).

## Key existing infrastructure for tests

- Test harness: `scripts/run_sio_test_suite.sh` (annotations: `//@ run-pass`, `//@ expect-stdout: X`, `//@ ignore`, `//@ check-only`, `//@ compile-fail`, `//@ error-pattern: X`).
- Stdlib gates: `scripts/stdlib_hyper_execution_gate.sh`, `scripts/stdlib_science_pipeline_gate.sh`, `scripts/stdlib_reliability_gate.sh`.
- Compiler: `bin/souc check`, `bin/souc run`, `bin/souc compile`.

## Known limitations affecting this thrust

From `CLAUDE.md`:
- `Knowledge<T>` is monomorphic (f64 only). `Knowledge<Knowledge<f64>>` will need a struct wrapper, not nested generics.
- No closure literals — use named fn refs.
- No unary minus — use `0 - x`.
- `&![T; N]` bare array mutation broken in JIT — use struct wrapper.
- GPU codegen exists but no end-to-end CLI path.

Implication: M2 Knightian operator is implemented as a **named struct** (`PBox` per `knightian_operator_choice.md`), not as a nested type literal.

## Baseline test pass count (pre-thrust)

```
[recorded at thrust start; will be rerun at each milestone gate]
```

Run via:
```bash
bash scripts/run_sio_test_suite.sh 2>&1 | tail -20
```

## Status

Inventory complete. M1 can begin.
