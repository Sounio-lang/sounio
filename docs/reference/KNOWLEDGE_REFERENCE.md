<!-- docs:meta
topic_id: website.docs.epistemic
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.epistemic
-->

# Epistemic Value Reference

This document is the JOSS-facing API reference for Sounio's epistemic value model.

## Type Model

`Epistemic` represents a value with explicit uncertainty metadata.

```sio
pub struct Epistemic {
    val: f64,
    variance: f64,
    confidence: i64,   // 0..1000 (permille); higher = more trustworthy
}
```

There is no `provenance` field on `Epistemic`. Source and transformation
provenance are tracked separately via `epistemic::prov`
(`stdlib/epistemic/prov.sio`); shared-source covariance is tracked in
`stdlib/epistemic/affine.sio`.

## Constructors

Use the `ep_measured(val, std_dev)` / `ep_certain(val)` / `ep_new(val, variance, confidence)`
free fns (or the `Epistemic { val, variance, confidence }` struct literal) to create
epistemic values. `ep_measured` derives `variance = std_dev²` and seeds
confidence at 900/1000; `ep_certain` sets `variance = 0.0` with confidence 1000.

```sio
use epistemic::knowledge::{ep_measured, ep_val, ep_std}

let mass = ep_measured(70.0, 0.2)   // val=70.0, variance=0.04, confidence=900/1000
let dose = ep_measured(500.0, 2.5)  // provenance bookkeeping lives in stdlib/epistemic/prov.sio
```

Guidelines:
- `val` is the nominal estimate.
- `variance` stores squared uncertainty (std = `ep_std(&e)`); accessors are `ep_val`, `ep_variance`, `ep_std`, `ep_confidence`.
- `confidence` (0..1000) should be interpreted consistently across a workflow.
- Source/transformation provenance is tracked separately via `epistemic::prov`; shared-source covariance is tracked in `stdlib/epistemic/affine.sio`.

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
// Canonical free-fn arithmetic; the `*` scalar operator is not part of the
// checked surface -- use ep_add() (uncorrelated quadrature).
let z = ep_add(&x, &y)
```

## Effect Annotations with Epistemic Values

Effect annotations make side effects explicit and composable with epistemic computation.

```sio
use epistemic::knowledge::ep_val

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
- `ep_std`/`ep_div`/`ep_merge` may require `Div`/`Mut`/`Panic`; keep pure uncertainty transformations effect-free when possible.

## Related References

- Standard library index: `docs/reference/STDLIB_REFERENCE.md`
- Full module inventory: `docs/stdlib/STDLIB_REFERENCE.md`
- Language specification: `spec/LANGUAGE_SPECIFICATION.md`
