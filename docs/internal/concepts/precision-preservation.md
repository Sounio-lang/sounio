<!-- docs:meta
topic_id: repo.docs.internal.concepts.precision-preservation
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.precision-preservation
-->

# Precision Preservation


Status: **executable**

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

## Measured ladder (re-measured 2026-08-27 against `origin/main` @ `055825a3f9`)

`TypeKind` variants that name a width
(`git grep -ohE '\bTy(F|I|U)[0-9]+\b' -- self-hosted/`):

`TyI8 TyI32 TyI64 TyI128` · `TyU8 TyU32 TyU64 TyU128` · `TyF32 TyF64 TyF128 TyF256`

The enum is no longer the whole integer ladder. Since PR #2054 (merged
2026-08-20) the checker parses **arbitrary** `iN`/`uN` names — `name_wide_int_bits`
(`self-hosted/check/compat.sio:1401`) accepts `i`/`u` followed by decimal digits,
and `ty_wide_int` (`self-hosted/check/types.sio:386`) represents any width by
reusing the `TyI128`/`TyU128` kind and carrying the declared bit-width in the
otherwise-unused `clifford_p` field. A width therefore no longer needs an enum
variant to exist.

| family | usable, with a witness | accepted, no witness | Reserved (name taken, every use refused) | absent entirely |
|---|---|---|---|---|
| signed | `i8 i32 i64 i128` · **`i256` `i512`** | any other `iN` (`N > 64`) | — | — |
| unsigned | `u8 u32 u64 u128` | **`u256` `u512`**, and any other `uN` (`N > 64`) | — | — |
| float | `f32 f64` | — | `f128` `f256` (`E218`) | — |

What "with a witness" means for `i256`/`i512`: named binary-format descriptors
(`numeric_format_int256_id()` = `1256`, `numeric_format_int512_id()` = `1512`,
`self-hosted/check/numeric_format.sio`), limb lowering and x86-64 emission
(`IrWideMul`/`IrWideShr`), and two run-pass fixtures that compute
`217041893 · 2^65` and check the high limb —
`tests/run-pass/r1_i256_lorenz_peak.sio` and `tests/run-pass/r1_i512_lorenz_peak.sio`,
both `requires: madaros`, both carrying a `sabotage: wide-mul` positive control.
See `docs/audit/R1_I256_I512_LIMBS_2026-08-20.md`.

What "accepted, no witness" means for `u256`/`u512`: the same name parse and the
same width-generic representation admit them, and lowering carries signedness
(`lower_opt_type_wide_signed`, `self-hosted/ir/lower.sio`), but no format
descriptor names them, no fixture exercises them, and nothing here asserts that
they compute correctly. Accepted is not proven.

The asymmetry between the integer and float rows is not accidental: integers
reach further than floats because exactness needs wide **integers**, not wide
floats. The Cayley-Dickson tower squares component magnitudes as it doubles, and
`i512` was named by the founder as the **seed** of that tower — a seed that, as
of #2054, multiplies.

`Reserved` remains the honest state for a width that is named and refused; it is
what `f128`/`f256` are. It is no longer the right proposal for `i256`/`i512`,
which are past it, and it is a weaker proposal for `u256`/`u512`, which are
accepted rather than refused. What `u256`/`u512` lack is a witness pair, not a
reservation.

## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
- Do not describe the narrowing act or its proof obligation as implemented.
  Neither exists; this records a ruling of 2026-08-19.
- Do not read the state of `i256`/`i512`/`u256`/`u512` off this document's
  earlier drafts. As of 2026-08-20 `i256`/`i512` are implemented with run-pass
  witnesses; `u256`/`u512` are accepted by the width-generic path and have no
  witness. Neither is "specified and never built".
- Do not claim `u256`/`u512` compute correctly. The checker accepts them; no
  fixture measures them.
- Do not cite `f128`/`f256` as available. They are `Reserved`: every use is
  refused with `E218`, and a refuse-fixture pair records that
  (`tests/typekind/index.tsv`).
