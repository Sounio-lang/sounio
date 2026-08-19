<!-- docs:meta
topic_id: repo.docs.internal.concepts.precision-preservation
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.precision-preservation
-->

# Precision Preservation

Concept-ID: `SOUNIO-PRECISION-PRESERVATION`

## Founder Intent

Small, residual, or cancellation-sensitive effects must not disappear because
an implementation silently narrows the representation. `f128`, `f256`,
double-double, and quad-double paths are scientific surfaces.

## Current Surfaces

- `stdlib/math/dd64.sio`
- `stdlib/math/qd128.sio`
- EISA error lanes and qd bridges
- ENIR error kinds `dd64` and `qd128`
- native-v2 aggregate and arithmetic lowering

## Required Invariants

- **Narrowing is always an act.** Founder ruling, 2026-08-19: there is no
  exception and no heuristic. The earlier form of this invariant — *explicit
  **or** proven lossless* — was a disjunction, and the second branch is what a
  compiler reaches for when the first is inconvenient. Both halves now hold at
  once: the narrowing is written, and its justification carries the floor set
  by `SOUNIO-JUSTIFICATION` — a discharged proof obligation, not prose.
- **The domain is epistemic and exact values.** A loop counter, an index, or a
  scalar being narrowed for display carries no knowledge, so no degradation
  occurs and no act is required. The rule binds values that carry uncertainty
  (`Knowledge<T>`) and values that carry decidability (exact surfaces). The
  domain follows from `SOUNIO-NO-IMPLICIT-DEGRADATION` rather than being
  stipulated beside it.
- Equality after narrowing does not imply equal correction histories.
- Tests include adversarial cancellation and residual witnesses.
- Backend failure is never silent fallback to `f64`.
- Higher precision alone does not establish physical significance.
- Uncertainty is what makes narrowing survivable; exactness is what makes it
  fatal. A measurement can afford to lose digits — if the rounding sits below
  the declared uncertainty, nothing was lost that was not already unknown. A
  decided fact has no error bar in which a wrap can hide, which is why
  `SOUNIO-EXACTNESS` states width as a correctness precondition rather than a
  tuning choice. One rule covers both because on an exact value the absorbing
  uncertainty is zero.

## Current Frontier

`dd64` has a native passing control. The qd128/EISA graph lowers without the
former segmentation fault but native emission still fails closed on classified
paths. This is a compiler frontier, not permission to demote precision.

## Measured ladder (2026-08-19, `origin/main`)

Width kinds present in the checker:

| family | usable | Reserved (name taken, every use refused) | absent entirely |
|---|---|---|---|
| signed | `i8 i32 i64 i128` | — | **`i256` `i512`** |
| unsigned | `u8 u32 u64 u128` | — | **`u256` `u512`** |
| float | `f32 f64` | `f128` `f256` (`E218`) | — |

The asymmetry is not accidental: integers reach further than floats because
exactness needs wide **integers**, not wide floats. The Cayley-Dickson tower
squares component magnitudes as it doubles, and `i512` was named by the founder
as the **seed** of that tower.

`i256`, `i512`, `u256` and `u512` are named in the founder's specification and
exist in no enum — not even as `Reserved`. `Reserved` would be the honest
state for them: the name taken, every use refused with a named diagnostic,
which records the intent inside the compiler rather than only in a document.
That is a proposal, not a founder ruling.

## Claims Forbidden

- Do not describe the narrowing act or its proof obligation as implemented.
  Neither exists; this records a ruling of 2026-08-19.
- Do not read the absence of `i256`/`i512`/`u256`/`u512` as a decision against
  them. They were specified and never built.
- Do not cite `f128`/`f256` as available. They are `Reserved`: every use is
  refused with `E218`, and a refuse-fixture pair records that
  (`tests/typekind/index.tsv`).
