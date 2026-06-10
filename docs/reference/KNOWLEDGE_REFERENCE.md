<!-- docs:meta
topic_id: website.docs.epistemic
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.epistemic
-->

# Knowledge<T> Reference

This document is the reference for Sounio's epistemic value model. It separates
the checked compiler builtin surface from package-local reference code and from
older design sketches.

## Claim Status

Status in this checkout:

- `Knowledge<T>` / `measure(...)`: **validated research**, limited to checked
  examples and named gates.
- Broad stdlib epistemic APIs: **prototype** unless a specific fixture or gate
  is cited.
- `Knowledge::new(...)`, named-argument constructors such as
  `Knowledge::new(70.0, uncertainty: 0.2)`, and self-contained `x + y`
  examples over `Knowledge<T>` are **design sketch**, not a current standalone
  API promise.

## Builtin `Knowledge<T>` Surface (Validated Fixtures)

The compiler recognizes `Knowledge<T>` and the builtin `measure(value,
uncertainty: ...)` form in checked fixtures. Accessing `.value` is an
epistemic boundary and requires the `Epistemic` effect.

Verified shape, from `tests/run-pass/gum_variance_shadow.sio`:

```sio
//@ run-pass

fn main() with IO, Mut, Epistemic {
    let k1: Knowledge<f64> = measure(100.0, uncertainty: 2.0)
    let k2: Knowledge<f64> = measure(50.0, uncertainty: 1.0)

    let a = k1.value
    let b = k2.value

    let sum = a + b
    let prod = a * b

    let va = variance_of(a)
    let vb = variance_of(b)

    print("a="); println(a)
    print("b="); println(b)
    print("var(a)="); println(va)
    print("var(b)="); println(vb)
    print("sum="); println(sum)
    print("prod="); println(prod)
}
```

This is a compiler builtin/fixture-backed surface, not the same thing as a
general package import or a documented `Knowledge::new` constructor.

## Package-Local `KCoreKnowledge`

`packages/epistemic-core/src/lib.sio` contains a package-local reference
implementation named `KCoreKnowledge`, with functions such as:

- `measure(value: f64, uncertainty: f64, source_label: string) -> KCoreKnowledge`
- `measure_with_confidence(...) -> KCoreKnowledge`
- `knowledge_add(&KCoreKnowledge, &KCoreKnowledge) -> KCoreKnowledge`
- `knowledge_div(&KCoreKnowledge, &KCoreKnowledge) -> KCoreKnowledge`
- `confidence_gate(...)`

This is useful reference code for GUM-style propagation, but it is not a
self-contained example unless the package/import wiring is part of the build
being tested.

```sio
// Package-local reference shape, not a standalone builtin example.
let dose = measure(500.0, 25.0, "HPLC_2025")
let volume = measure(10.0, 0.2, "pipette_class_A")
let concentration = knowledge_div(&dose, &volume)
```

## Design Sketches Not Yet Current API

The following older style is retained as a design sketch only:

```sio
let mass = Knowledge::new(70.0, uncertainty: 0.2)
let dose = measure(500.0, uncertainty: 2.5, source: "instrument")
let z = x + y
```

Do not use these forms as public current API examples unless a fixture or gate
in the same checkout proves them.

## Arithmetic and Propagation

The validated surface is narrower than "all arithmetic on `Knowledge<T>`".
Checked examples show uncertainty metadata being seeded through
`measure(...)`, crossing the `.value` boundary under `with Epistemic`, and
being observed through compiler-supported probes such as `variance_of(...)`.
Package-local helpers such as `knowledge_add` and `knowledge_div` implement
GUM-style propagation inside `packages/epistemic-core`.

## Effect Annotations with Epistemic Values

Effect annotations make side effects explicit and composable with epistemic computation.

```sio
fn read_sensor_value() -> f64 with IO, Epistemic {
    let k: Knowledge<f64> = measure(42.0, uncertainty: 0.5)
    k.value
}

fn main() with IO, Epistemic {
    let value = read_sensor_value()
    println(value)
}
```

Guidelines:
- If a function performs I/O, include `with IO`.
- If GPU kernels or device operations are used, include `with GPU`.
- Keep pure uncertainty transformations effect-free when possible.

## Related References

- Standard library index: `docs/reference/STDLIB_REFERENCE.md`
- Claim registry: `docs/serious-language/public-claim-registry.v1.tsv`
- Evidence matrix: `docs/serious-language/spec-evidence-matrix.v1.tsv`
- Language specification: `docs/spec/LANGUAGE_SPECIFICATION.md`
