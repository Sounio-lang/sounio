<!-- docs:meta
topic_id: website.docs.epistemic
authority: dual
audience: users
last_validated: 2026-09-21
validated_by: Claude
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.epistemic
-->

# Epistemic Value Reference

This document is the JOSS-facing API reference for Sounio's epistemic value model.

## Type Model

`Epistemic` represents a value with explicit uncertainty metadata.

```sio
pub struct Epistemic {
    pub val: f64,
    pub variance: f64,
    pub confidence: i64,   // 0..1000 (permille); higher = more trustworthy
}
```

There is no `provenance` field on `Epistemic`, and the constructors above
(`ep_measured`, `ep_certain`, `ep_new`) do not record a source — they only
build `{ val, variance, confidence }`. A W3C PROV-DM export module exists at
`stdlib/epistemic/prov.sio`, but its types and constructors are currently
private and are **not** part of the public/user-facing API, so provenance
cannot be attached or queried through it yet. Shared-source covariance is
tracked by the separate public `epistemic::affine` API (`stdlib/epistemic/affine.sio`),
a distinct surface not covered here.

## Two scalar value surfaces

Sounio exposes two distinct scalar epistemic value types; do not treat one as a
replacement for the other. Other representations exist for specialized needs —
notably the public `epistemic::affine` source-tracking API (see below).

- **Stdlib `Epistemic` / `ep_*` API (this document).** The user-facing value
  type `Epistemic { pub val, pub variance, pub confidence }` and the free
  functions `ep_measured` / `ep_certain` / `ep_new` / `ep_add` / … described
  below. Prefer this surface for new code.
- **Language-level `Knowledge<T>` / `measure` intrinsic.** A separate compiler
  intrinsic retained by the self-hosted checker and exercised by older
  run-pass tests. `measure(value: T, uncertainty: f64) -> Knowledge<T>`
  (`self-hosted/check/check.sio:1054`) builds a `Knowledge<T>`, whose nominal
  value is read via `.value` (see `tests/run-pass/variance_of_measure_sum.sio`).
  The standard-library index (`docs/reference/STDLIB_REFERENCE.md`) still routes
  `Knowledge<T>` lookups here. Treat `Knowledge<T>` as a distinct, lower-level
  surface rather than an alias for `Epistemic`.

These are the two scalar surfaces for a single value with uncertainty. For
shared-source covariance tracking, use the separate public `epistemic::affine`
API (`af_measured` / `af_add` / …, `stdlib/epistemic/affine.sio`, exercised by
`examples/vancomycin_auc_affine.sio`); it is a distinct, supported surface not
covered by this `Epistemic` reference.

## Constructors

Use the `ep_measured(val, std_dev)` / `ep_certain(val)` / `ep_new(val, variance, confidence)`
free fns (or the `Epistemic { val, variance, confidence }` struct literal) to create
epistemic values. `ep_measured` derives `variance = std_dev²` and seeds
confidence at 900/1000; `ep_certain` sets `variance = 0.0` with confidence 1000.

```sio
use epistemic::knowledge::{ep_measured, ep_val, ep_std}

let mass = ep_measured(70.0, 0.2)   // val=70.0, variance=0.04, confidence=900/1000
let dose = ep_measured(500.0, 2.5)  // builds { val, variance, confidence }; source provenance is not recorded
```

Guidelines:
- `val` is the nominal estimate.
- `variance` stores squared uncertainty (std = `ep_std(&e)`); accessors are `ep_val`, `ep_variance`, `ep_std`, `ep_confidence`.
- `confidence` (0..1000) should be interpreted consistently across a workflow.
- `Epistemic` carries no provenance field, and the constructors do not record a source. The `prov` module is not yet a public API (its types/ctors are private), so provenance cannot be attached through `epistemic::prov` today; shared-source covariance is
tracked by the separate public `epistemic::affine` API (`stdlib/epistemic/affine.sio`),
a distinct surface not covered here.

## Arithmetic and Propagation

Arithmetic on `Epistemic` propagates uncertainty automatically for common operations.

Typical first-order behavior:

- Addition/subtraction: combine independent variances (uncorrelated — `ep_add`/`ep_sub` drop the `2·Cov` term; use `ep_add_cov`/`ep_sub_cov` when covariance is known).
- Multiplication/division: GUM delta method via first-order sensitivity coefficients (`ep_mul`/`ep_div`; `ep_mul(&x, &x)` understates, so use `ep_square` for `X²`).
- Scalar transforms: `ep_scale`/`ep_shift` scale or shift `val` and adjust `variance`; `ep_square` squares.
- Confidence decay: arithmetic reduces confidence — `ep_add`/`ep_sub` ×99/100, `ep_mul`/`ep_square` ×98/100, `ep_div` ×97/100; `ep_scale`/`ep_shift` preserve it. `ep_merge` (inverse-variance weighted) averages confidence and may raise it.

Example:

```sio
use epistemic::knowledge::{ep_measured, ep_add, ep_val, ep_std}

let x = ep_measured(10.0, 0.5)
let y = ep_measured(20.0, 0.3)
// Canonical free-fn arithmetic; the `+` scalar operator is not part of the
// checked surface -- use ep_add() for `x + y` (uncorrelated quadrature).
let z = ep_add(&x, &y)
```

## Effect Annotations with Epistemic Values

Effect annotations make side effects explicit and composable with epistemic computation.

```sio
use epistemic::knowledge::{Epistemic, ep_val}

fn read_sensor() -> Epistemic with IO {
    // IO effect declared explicitly
}

fn main() with IO {
    let k = read_sensor()
    println(ep_val(&k))
}
```

Guidelines:
- If a function performs I/O, include `with IO`.
- If GPU kernels or device operations are used, include `with GPU`.
- The effect requirements differ per function and are exact: `ep_std` declares `with Mut, Div, Panic`; `ep_div` and `ep_merge` declare `with Div, Panic`. Include precisely the effects each function declares — do not add `Mut` to `ep_div`/`ep_merge`, and do not omit `Mut` from `ep_std`. Keep other uncertainty transforms effect-free when possible.

## Related References

- Standard library index: `docs/reference/STDLIB_REFERENCE.md`
- Full module inventory: `docs/stdlib/STDLIB_REFERENCE.md`
- Language specification: `spec/LANGUAGE_SPECIFICATION.md`
