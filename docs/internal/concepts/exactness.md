<!-- docs:meta
topic_id: repo.docs.internal.concepts.exactness
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.exactness
-->

# Exactness


Status: **hypothesis**

Concept-ID: `SOUNIO-EXACTNESS`

## Founder Intent

Some questions must be **decided**, not measured. Where a result is an exact
arithmetic fact, the language must not convert it into a floating-point
comparison whose answer is chosen by a tolerance.

This is not a stronger form of [precision preservation]
(`SOUNIO-PRECISION-PRESERVATION`). Precision is quantitative: how many digits
survive a representation. Exactness is qualitative: whether rounding occurred
at all, and therefore whether equality is decidable.

The canonical case is Cayley-Dickson zero divisors. Over exact integers,
`a*b == 0` is a decidable equality — the pair either annihilates or it does
not. Over `f64`, the same test needs a tolerance, and the tolerance decides
the answer. The founder's ruling, stated in
`stdlib/algebra/cayley_dickson_exact_i64.sio`:

> `ab == 0` is a DECIDABLE i64 equality, not a tolerance-gated float
> measurement.

## The erasure this prevents

From `FOUNDER_INTENT.md`: *"a small or unresolved effect reported as zero"*.
A tolerance-gated zero test performs exactly this erasure in the direction the
tolerance chooses. Widening the float does not fix it; it moves the threshold.

## Current Surfaces

- `stdlib/algebra/cayley_dickson_exact_i64.sio` — concrete hand-monomorphized
  exact engine; ships the decidable ZD test today
- `stdlib/algebra/cayley_dickson_exact.sio` — generic `<F>` skeleton
- `stdlib/eisa/hypercomplex_zd.sio`
- `stdlib/algebra/sedenion_verdict.sio`

## Required Invariants

- An exact surface never falls back to floating point. A backend that cannot
  emit the exact path refuses; it does not approximate.
- Equality on an exact surface is decidable. A tolerance parameter on an exact
  comparison is a contract violation, not a tuning knob.
- Width is a **correctness precondition**, not a performance choice. If a
  product exceeds the representation, the operation must refuse detectably;
  silent wrap converts a decision into a fabrication.
- An exact result narrowed to float loses decidability and must be marked as
  having done so. The narrowed value is a measurement, not a fact.
- Exactness does not imply significance. A decided zero is a fact about the
  algebra, not evidence about the world.

## Current Frontier

The integer ladder stops at `i128`/`u128`. `i256` and `u256` are the widths
Cayley-Dickson exactness needs and do not exist in `TypeKind`; `i512` is a
declared Garden seed, not a gap.

`F = Rational` / `BigInt` is recorded as blocked by issue #651. That issue was
misdiagnosed: the cause was not a `[struct; N]` defect but the native
handle-table wrap at 2^20 plus a multimodule thin-link `rc=12`. Handle-table
capacity has since been raised to 2^22, so the blocker may no longer hold and
the repro (`docs/handoff/repros/d8_generic_struct_F_mul_segv.sio`) has not been
re-run against current Madaros.

The generic prerequisites named as blockers in the `exact_i64` header —
generic-struct-return, `impl Trait for Type`, trait-bounded dispatch — landed
on 2026-07-06 (`2adb8f061`, PR #650) and are in `main`. That header is stale.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
