<!-- docs:meta
topic_id: website.docs.compiler.confidence-effect
authority: draft
audience: contributors
status: design-only / not implemented
last_validated: 2026-04-23
-->

# Compile-time Confidence effect — design

**Status:** design proposal. Not implemented. Runtime parallel exists at
`stdlib/epistemic/confidence_gate.sio` and, specialised for PD priors, at
`stdlib/darwin_pbpk/pd/pd_gate.sio`. This document scopes what the
*compile-time* half of the dissertation's second novel contribution
would look like, so the next session can implement against a fixed target.

## Why this exists

The file header of `stdlib/epistemic/confidence_gate.sio` already flags
the goal:

> *Future work: compile-time enforcement via the effect system
> (e.g., `fn dosing(c: Epistemic) with Confidence(0.95)` rejects at type-check)*

The runtime gate (`pd_gate_check`) catches prior-quality violations when
a simulation runs. A **compile-time** gate catches them when the code is
checked — before a run is even possible — which is what makes the
contribution "prevent low-confidence data from contaminating
high-confidence computations" genuinely novel in a systems language:
confidence becomes an element of the type, not of the value.

## Surface proposal

A new parameterised effect label, treated by the checker like `IO` but
carrying a numeric floor:

```sio
fn dosing_decision(k: Knowledge<mg>) -> Dose with Confidence(0.95) { ... }
```

Read as: "this function may only be called in contexts where every
`Knowledge<_>` argument it receives has a declared confidence at least
0.95." The checker rejects callers that cannot prove the floor.

Composability: if `f` has `Confidence(c_f)` and `g` calls `f`, `g`
must carry `Confidence(c_g)` with `c_g ≥ c_f` (monotone, exactly as
the runtime gate would enforce at the boundary). Callers can *weaken*
the floor at module boundaries with an explicit `with Confidence(c)`
restriction, never silently.

## Typecheck rule (informal)

For a call `f(args)` where `f : τ → σ with Confidence(c_f) ∪ E`:
- Each `arg_i` of Knowledge type must carry a *static* confidence
  attribute `conf_i`. The checker fails if any `conf_i < c_f`.
- The caller's effect set must contain `Confidence(c_caller)` with
  `c_caller ≥ c_f`, OR the call site must be inside a handler scope
  that weakens the effect (see "Handlers" below).

Existing e-graph / refinement infrastructure contains the ordering
primitives needed (compare f64 at typecheck time — `Knowledge<T>`
already stores `confidence: i16` in the 1000-scaled form per
`stdlib/epistemic/knowledge.sio`).

## Where confidence comes from

Three sources, in order of typecheck tractability:

1. **Literal construction**: `Knowledge::new(value, variance, 0.85)`
   — confidence is a compile-time constant, read directly off the
   constructor. Trivial.
2. **Struct literal with known priors**: e.g. `rapamycin_hill_priors()`
   returns `HillPriors { ... confidence: 0.85 }` — checker requires the
   confidence field to be a literal (not a runtime expression) in any
   function annotated `with Confidence(_)`. Inlined through pure
   constructors. Also tractable.
3. **Dynamic confidence (uncertainty-of-uncertainty)**: confidence
   computed at runtime by e.g. Bayesian update over observations.
   Checker cannot prove a floor; the call site must run through a
   runtime-gated handler (lifting the effect out).

Phase 1 covers cases (1) and (2) — which is what the dissertation PD
pipeline actually needs. Case (3) waits for Wave G or later.

## Handlers

Borrowing the effect-handler pattern already sketched in
`docs/architecture/EFFECT_HANDLERS_IMPLEMENTATION.md`:

```sio
fn run_with_clinical_gate<T>(body: fn() -> T with Confidence(0.60))
    -> (T, GateResult)
    with IO, Mut
{
    // runtime gate_check at the boundary
    // body runs with static floor lifted to 0.60
}
```

The handler is where the compile-time discipline meets the runtime
`pd_gate_check` — they are the same policy expressed at the two times.

## Implementation path (for the next session)

Phased, each phase independently mergeable:

**Phase A — Parser + AST** (~300 lines)
1. Extend `self-hosted/parser/parser.sio` to parse `Confidence(<float>)`
   as an effect label with one numeric argument.
2. Extend `self-hosted/check/ast.sio` to add the `EffConfidence(f64)`
   variant of the effect enum.

**Phase B — Checker** (~500 lines, the core work)
1. In `self-hosted/check/effects.sio`: add subsumption rule
   `Confidence(a) ≤ Confidence(b) ⇔ a ≤ b` (covariant on the floor).
2. In `self-hosted/check/types.sio`: when checking a call, inspect
   argument types for Knowledge-shaped structs; resolve static
   confidence fields; emit diagnostics on violation.
3. Extend `self-hosted/check/effects_row.sio` to treat `Confidence`
   as a *parameterised* label (it is the only one so far; design
   leaves room for more parameterised labels like `Dimension<_>`).

**Phase C — Stdlib migration** (~100 lines, mostly annotation)
1. Annotate `dose_from_plan`, `pd_endpoint_inhibition_auc`, and the
   other PD public entry points with `with Confidence(0.60)`.
2. Remove the runtime-gate wrapper in the hot path; keep it only at
   module boundaries as the Phase-1-to-Phase-3 bridge.

**Phase D — Test fixtures** (~200 lines)
1. `tests/compile-fail/confidence_too_low.sio` — Knowledge literal at
   0.40 passed into a `Confidence(0.60)` function. Must fail with a
   specific error message (`//@ error-pattern: confidence 0.400 < 0.600`).
2. `tests/run-pass/confidence_handler.sio` — weakened via handler.

## Interaction with existing systems

- `Knowledge<T>`: confidence field is already `i16` (1000-scaled). The
  checker reads this directly at type-check time; no new metadata
  needed.
- `graded_effects.sio`: the grading lattice already exists for effect
  strength. `Confidence(_)` is naturally a graded effect with the
  usual f64 ordering.
- `gum.sio`: expanded uncertainty ↔ confidence mapping
  (confidence = 1 − coverage-breach probability) is already computed;
  phase B checker can reuse that formula for structs with explicit
  variance + dof but no confidence field.

## What this does NOT try to do

- Not a dependent-type system. Confidence is a fixed numeric attribute,
  not a proof term. We don't prove *why* the confidence is what it is;
  we just propagate the declared value.
- Not a Bayesian update system. Runtime updates live on the runtime
  gate side (case 3 above) and don't cross the compile-time boundary.
- Not a replacement for `confidence_gate.sio` / `pd_gate.sio`. Runtime
  gates remain the mechanism for case 3 and for prior-set audits
  (where we *want* to see the worst-case confidence at run time even
  if the floor formally passes).

## Open questions (for the implementation session)

1. How strict is literal folding for confidence fields? Must the
   `HillEpParam { ... confidence: 0.85 }` literal be visible through
   one level of constructor inlining, or arbitrary inlining? (Phase B.2)
2. Does the user-facing diagnostic report the *chain* of arguments /
   confidence floors, or just the leaf violation? (UX choice; both
   are implementable.)
3. Do we need a `Confidence(auto)` shorthand that reads the tightest
   floor from the function body, or is explicit declaration required?
   (Affects ergonomics of Phase C migration.)

## References

- `stdlib/epistemic/confidence_gate.sio` — runtime gate, header points
  to this design
- `stdlib/darwin_pbpk/pd/pd_gate.sio` — PD-specific runtime gate (this
  commit)
- `stdlib/epistemic/knowledge.sio` — Knowledge<T> with i16 confidence
- `docs/compiler/EFFECT_SYSTEM_ARCHITECTURE.md` — current effect system
- `docs/architecture/EFFECT_HANDLERS_IMPLEMENTATION.md` — handler
  design background
