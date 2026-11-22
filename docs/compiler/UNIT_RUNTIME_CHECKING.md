<!-- docs:meta
topic_id: repo.docs.compiler.unit-runtime-checking
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.unit-runtime-checking
-->

# Unit Runtime Checking Guide

## 1. Implemented unit representations

Sounio has two separate unit mechanisms. The compiler checks unit annotations
statically; the standard library offers explicit runtime quantities. A unit
annotation does not automatically construct a runtime quantity.

| Mechanism | Representation | Where checks happen |
|-----------|----------------|---------------------|
| Compiler units | Seven SI exponents plus a rational scale, attached through a type's `unit_id` | Type checker; scalar values lower to plain `f64` |
| Library quantities | `Quantity { value, uncertainty, dim }` | Explicit calls to functions in `stdlib/units/lib.sio` |

These are research-language surfaces, not a guarantee of physical correctness
or clinical validity. See [known limitations](KNOWN_LIMITATIONS.md) for current
compiler boundaries.

## 2. Compile-time units

### 2.1 Dimension vector and scale

The compiler's [`UnitDim`](../../self-hosted/check/units.sio) stores:

- `exponents: [i64; 7]`, in the order **mass, length, time, temperature,
  amount, current, luminosity**;
- `scale_num: i64` and `scale_den: i64`, the rational scale relative to the
  registry's base for that dimension.

This is compiler metadata, not a user-facing generic dimension type. The order
of its last four slots is temperature, amount, current, luminosity.

| Physical quantity | Exponent vector in that order | Coherent SI unit |
|-------------------|-------------------------------|------------------|
| Velocity | `[0, 1, -1, 0, 0, 0, 0]` | m/s |
| Acceleration | `[0, 1, -2, 0, 0, 0, 0]` | m/s² |
| Force | `[1, 1, -2, 0, 0, 0, 0]` | kg·m/s² |
| Energy | `[1, 2, -2, 0, 0, 0, 0]` | kg·m²/s² |
| Mass concentration | `[1, -3, 0, 0, 0, 0, 0]` | kg/m³ |
| Amount concentration | `[0, -3, 0, 0, 1, 0, 0]` | mol/m³ |

The compiler registry uses **grams** as its mass scale base: `mg` has scale
`1/1000`, `g` has `1/1`, and `kg` has `1000/1`. This differs from the runtime
library's coherent SI mass convention (kilograms). The exponent vector alone
does not describe a unit's numeric scale.

### 2.2 Checking and lowering

[`self-hosted/check/units.sio`](../../self-hosted/check/units.sio) implements
exponent arithmetic and the named unit registry.
[`self-hosted/check/check.sio`](../../self-hosted/check/check.sio) applies it
to expressions, bindings, calls and casts:

- Multiplication adds exponents and multiplies scales; division subtracts
  exponents and divides scales.
- Addition, subtraction and comparisons require compatible dimensions and
  matching scales. For example, `km + m` is rejected; scales are not silently
  reconciled.
- Unit casts check dimension compatibility. **At baseline `5d386fc`, `as`
  relabels the number without applying a scale conversion:** casting `500.0`
  from `mg` to `kg` still produces `500.0`. Convert the numeric value explicitly
  when changing scales; do not rely on a cast to perform that conversion.

[`self-hosted/ir/lower.sio`](../../self-hosted/ir/lower.sio) lowers scalar unit
values to plain `f64`. Printing such a value does not print a unit name, and
there is no dimension object attached to that scalar at runtime. The compiler
registry is bounded; source presence does not establish unrestricted unit
algebra or validation of every dimensionless-function argument.

## 3. Runtime quantities in the standard library

### 3.1 Data model

[`stdlib/units/lib.sio`](../../stdlib/units/lib.sio) defines a separate
`UnitDim` with seven named `i32` fields: `mass`, `length`, `time`,
`temperature`, `amount`, `current`, and `luminosity`. It has no scale fields.
Its concrete, non-generic `Quantity` struct contains:

| Field | Type | Meaning |
|-------|------|---------|
| `value` | `f64` | Numeric value in the chosen coherent SI unit |
| `uncertainty` | `f64` | Standard uncertainty in the same unit |
| `dim` | `UnitDim` | Runtime dimension exponents |

Use `quantity_new(value, uncertainty, dim)` or helpers such as `q_mass`,
`q_length`, `q_time`, and `q_velocity`. The named constructors expect coherent
SI values and uncertainties: kilograms, meters, seconds, and meters per second.
The struct does not remember whether an incoming number was in grams or
kilograms; callers must normalize external data before construction.

### 3.2 Arithmetic and mismatch handling

`dim_eq(a, b)` compares all seven fields. `quantity_is_compatible(a, b)`
exposes that comparison for quantities. The arithmetic functions behave as
follows:

| Function | Dimension behavior | Uncertainty behavior |
|----------|--------------------|----------------------|
| `quantity_add`, `quantity_sub` | `assert(dim_eq(a.dim, b.dim))`; preserve the input dimension | Root sum of squared standard uncertainties |
| `quantity_mul` | Add exponents | First-order product propagation |
| `quantity_div` | Subtract exponents | First-order quotient propagation |
| `quantity_scale` | Preserve dimension | Multiply uncertainty by the scalar's absolute value |

An incompatible addition or subtraction fails an assertion. There is no
structured unit-mismatch error with suggested conversions in this API. Callers
can use `quantity_is_compatible` before the operation to handle mismatches
explicitly. The uncertainty formulas above do not track input covariance.

### 3.3 Example: distance divided by time

This program uses the explicit runtime library. The wildcard import follows
the module's documented import convention.

```sio
use units::lib::*

fn main() with IO, Mut, Panic, Div {
    let distance = q_length(100.0, 0.5)
    let elapsed = q_time(10.0, 0.1)
    let speed = quantity_div(distance, elapsed)
    assert(dim_eq(speed.dim, dim_velocity()))
    assert(speed.value == 10.0)
    assert(speed.uncertainty > 0.0)
    println("runtime units: PASS")
}
```

## 4. Explicit conversions and boundaries

The runtime module provides fixed scalar helpers including `convert_kg_to_g`,
`convert_g_to_kg`, `convert_m_to_cm`, `convert_cm_to_m`,
`convert_celsius_to_kelvin`, and `convert_kelvin_to_celsius`. These accept and
return `f64`; they do not inspect a `Quantity` or parse unit strings.

For a linear conversion, scale both the value and its standard uncertainty.
For an offset-only temperature conversion, shift the value and retain the
standard uncertainty. Do not pass a Celsius value directly to
`quantity_new(..., dim_temperature())` and expect it to become kelvin.

At file, database, serialization and FFI boundaries, application code must
interpret the unit, validate it, convert the number and uncertainty to the
agreed convention, and construct a `Quantity` explicitly. Neither a plain
`f64` nor a unit cast supplies that validation automatically.

## 5. Verification

Stdlib modules are checked with `bin/souc check`, not run as programs:

```bash
bin/souc check stdlib/units/lib.sio
bin/souc check stdlib/epistemic/gum_supplement1.sio
bash scripts/ci/unit_types_phase1_gate.sh
bash scripts/run_sio_test_suite.sh --filter-exact unit_cast_preserves_value.sio
```

The unit gate covers compiler annotations and refusals. Runtime library
behavior requires an importing caller such as the example above. See
[the test conventions](../../tests/README.md) for run-pass and compile-fail
annotations. No runtime performance measurements are claimed by this guide.

## 6. Proposals (not implemented by the APIs described above)

A generic quantity wrapper, unit-string parsing, automatic scale/offset
conversion with compatibility errors, conversion caching, unit-aware
serialization, and dedicated dimension-assertion helpers could be future
extensions. They are design proposals, not current library APIs. Likewise,
a combined knowledge/refinement/quantity abstraction would need separate
implementation and validation; the concrete `Quantity` already carries its
own numeric standard uncertainty.
