---
title: "Type System"
description: "Sounio's bidirectional type inference with linear types, refinement types, and units of measure."
---

## Type System

Sounio's type system combines **bidirectional type inference**, **linear/affine types**, **refinement types**, and **units of measure** into a unified framework. The type checker (`compiler/src/check/mod.rs`, 7,153 lines) simultaneously performs type inference, effect inference, unit checking, and linearity verification.

### Bidirectional Type Inference

The type checker uses bidirectional inference with constraint-based unification:

1. **Synthesis mode**: Infer the type of an expression bottom-up
2. **Checking mode**: Check an expression against an expected type top-down
3. **Constraint solving**: Unify type variables and resolve constraints

```rust
struct TypeChecker {
    env: TypeEnv,                        // Scoped variable bindings
    next_type_var: u32,                  // Fresh type variable counter
    constraints: Vec<TypeConstraint>,    // Pending constraints
}
```

Type variables are generated fresh during inference and resolved through unification. The environment uses a scope stack for lexical scoping.

### Linear and Affine Types

Linear types enforce single-use semantics for resources:

```sio
linear struct FileHandle { fd: i32 }

fn process(handle: FileHandle) -> Result<()> with IO {
    // handle must be consumed exactly once
    close(handle)  // OK: consumed
}
```

The HIR tracks linearity at the struct level:
- `is_linear: bool` --- Must be used exactly once
- `is_affine: bool` --- May be used at most once (can be dropped)

### Refinement Types

Refinement types add logical predicates to base types, verified by Z3 (when the `smt` feature is enabled):

```sio
type Positive = { x: i32 | x > 0 }
type Probability = { p: f64 | 0.0 <= p && p <= 1.0 }
type OrbitRatio = { r: f64 | r > 0.0 }
```

During HLIR lowering (`hlir/lower.rs:52`), refinement types are resolved to their base type. The predicates are checked statically at type-checking time.

### Units of Measure

**File**: `compiler/src/units/dimension.rs`

Units are tracked through a 7-dimensional SI base:

```rust
pub struct Dimension {
    mass: i8,        // [M]
    length: i8,      // [L]
    time: i8,        // [T]
    current: i8,     // [I]
    temperature: i8, // [Θ]
    amount: i8,      // [N]
    luminosity: i8,  // [J]
}
```

Derived dimensions are computed automatically:
- **Velocity**: `[L T^-1]`
- **Force**: `[M L T^-2]`
- **Energy**: `[M L^2 T^-2]`
- **Concentration**: `[M L^-3]`
- **Clearance**: `[L^3 T^-1]` (PK/PD)

The unit type checker (`units/check.rs`) operates alongside the main type checker, using `UnitType` with inference variables (`UnitVar`) for automatic unit deduction.

### Unit Conversion

**File**: `compiler/src/units/convert.rs`

```rust
pub fn convert<N, From: Unit, To: Unit>(qty: Quantity<N, From>) -> Quantity<N, To>
```

Conversions are checked at compile time via dimension equality. Affine units (temperature scales) use a separate conversion path:

```rust
pub fn convert_affine<From: Unit, To: Unit>(qty: Quantity<N, From>) -> Quantity<N, To>
// Formula: base = value * scale + offset
```

### Supported Units

- **SI Base**: Kilogram, Meter, Second, Ampere, Kelvin, Mole, Candela
- **SI Derived**: Hertz, Joule, Newton, Pascal, Watt, Celsius, Fahrenheit, Molar
- **PK/PD**: MilligramPerLiter, MicrogramPerLiter, MilligramHourPerLiter, PerHour, PerDay

### HLIR Type Representation

At the HLIR level (`hlir/ir.rs:129`), types include:

- **Primitives**: `Void`, `Bool`, `I8`-`I128`, `U8`-`U128`, `F32`, `F64`
- **Aggregates**: `Ptr`, `Array`, `Struct`, `Tuple`, `Function`
- **Linear algebra**: `Vec2`-`Vec4`, `Mat2`-`Mat4`, `Quat`
- **Exotic**: `Octonion` (8D hypercomplex), `Dual` (autodiff)
- **ML**: `QuatLinear`, `QuatConv2d`, `QuatRnnState`, `QuatGate`
- **Epistemic**: `Knowledge { inner, mode, epsilon_bound, provenance_id }`
