<!-- docs:meta
topic_id: website.docs.epistemic
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.epistemic
-->

# Knowledge<T> Reference

This document is the JOSS-facing API reference for Sounio's epistemic value model.

## Type Model

`Knowledge<T>` represents a value with explicit uncertainty metadata.

Conceptually:

```sio
struct Knowledge<T> {
    value: T,
    uncertainty: f64,
    confidence: f64,
    provenance: Source,
}
```

## Constructors

Use `Knowledge::new(...)`-style constructors (or domain helpers such as `measure(...)`) to create epistemic values.

```sio
let mass = Knowledge::new(70.0, uncertainty: 0.2)
let dose = measure(500.0, uncertainty: 2.5, source: "instrument")
```

Guidelines:
- `value` is the nominal estimate.
- `uncertainty` stores standard uncertainty in the same units as `value`.
- `confidence` (when present) should be interpreted consistently across a workflow.
- `provenance` records source and transformation context.

## Arithmetic and Propagation

Arithmetic on `Knowledge<T>` propagates uncertainty automatically for common operations.

Typical first-order behavior:

- Addition/subtraction: combine independent uncertainties in quadrature.
- Multiplication/division: propagate via first-order sensitivity coefficients.
- Mixed operations: preserve epistemic metadata unless explicitly extracted.

Example:

```sio
let x = Knowledge::new(10.0, uncertainty: 0.5)
let y = Knowledge::new(20.0, uncertainty: 0.3)
let z = x + y
```

## Effect Annotations with Epistemic Values

Effect annotations make side effects explicit and composable with epistemic computation.

```sio
fn read_sensor() -> Knowledge<f64> with IO {
    // IO effect declared explicitly
}

fn main() with IO {
    let k = read_sensor()
    println(k.value)
}
```

Guidelines:
- If a function performs I/O, include `with IO`.
- If GPU kernels or device operations are used, include `with GPU`.
- Keep pure uncertainty transformations effect-free when possible.

## Related References

- Standard library index: `docs/reference/STDLIB_REFERENCE.md`
- Full module inventory: `docs/stdlib/STDLIB_REFERENCE.md`
- Language specification: `spec/LANGUAGE_SPECIFICATION.md`
