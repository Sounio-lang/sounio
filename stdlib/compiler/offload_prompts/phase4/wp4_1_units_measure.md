# WP-4.1: Units of Measure (Dimensional Analysis)

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- NO type suffixes (use `0` not `0i64`)
- Array indexing requires `with Panic`

## Reference Implementation

See: `compiler/src/units/dimension.rs`
See: `compiler/src/units/unit.rs`
See: `compiler/src/units/check.rs`

## Target Output

**File**: `stdlib/compiler/units/dimension.sio`
**Estimated LOC**: ~1,800

## Specification

Implement SI dimensional analysis for scientific computing.

### SI Base Dimensions

```sio
// SI base quantities
fn DIM_MASS() -> i32 { 0 }        // M (kilogram)
fn DIM_LENGTH() -> i32 { 1 }      // L (meter)
fn DIM_TIME() -> i32 { 2 }        // T (second)
fn DIM_CURRENT() -> i32 { 3 }     // I (ampere)
fn DIM_TEMPERATURE() -> i32 { 4 } // Θ (kelvin)
fn DIM_AMOUNT() -> i32 { 5 }      // N (mole)
fn DIM_LUMINOSITY() -> i32 { 6 }  // J (candela)

// Dimensionless
fn DIM_DIMENSIONLESS() -> i32 { 7 }

struct Dimension {
    mass: i32,          // Exponent for M
    length: i32,        // Exponent for L
    time: i32,          // Exponent for T
    current: i32,       // Exponent for I
    temperature: i32,   // Exponent for Θ
    amount: i32,        // Exponent for N
    luminosity: i32,    // Exponent for J
}
```

### Derived Units

```sio
// Create dimension from exponents
fn dimension_new(m: i32, l: i32, t: i32, i: i32, th: i32, n: i32, j: i32) -> Dimension {
    Dimension {
        mass: m, length: l, time: t, current: i,
        temperature: th, amount: n, luminosity: j,
    }
}

// Common derived units
fn dimension_force() -> Dimension {
    // Force: kg⋅m⋅s⁻² (M L T⁻²)
    dimension_new(1, 1, -2, 0, 0, 0, 0)
}

fn dimension_energy() -> Dimension {
    // Energy: kg⋅m²⋅s⁻² (M L² T⁻²)
    dimension_new(1, 2, -2, 0, 0, 0, 0)
}

fn dimension_power() -> Dimension {
    // Power: kg⋅m²⋅s⁻³ (M L² T⁻³)
    dimension_new(1, 2, -3, 0, 0, 0, 0)
}

fn dimension_pressure() -> Dimension {
    // Pressure: kg⋅m⁻¹⋅s⁻² (M L⁻¹ T⁻²)
    dimension_new(1, -1, -2, 0, 0, 0, 0)
}

fn dimension_concentration() -> Dimension {
    // Concentration: mol⋅m⁻³ (N L⁻³)
    dimension_new(0, -3, 0, 0, 0, 1, 0)
}
```

### Unit System

```sio
struct Unit {
    dimension: Dimension,   // Physical dimension
    scale: f64,            // Conversion scale to SI base
    offset: f64,           // Additive offset (for Celsius → Kelvin)
    name: [i8; 32],        // Name (e.g., "mg", "mL", "hour")
}

// Create unit
fn unit_new(name: &str, dim: Dimension, scale: f64, offset: f64) -> Unit {
    // ... copy name
    Unit { dimension: dim, scale: scale, offset: offset, name: [0; 32] }
}

// SI base units
fn unit_kilogram() -> Unit {
    unit_new("kg", dimension_new(1, 0, 0, 0, 0, 0, 0), 1.0, 0.0)
}

fn unit_meter() -> Unit {
    unit_new("m", dimension_new(0, 1, 0, 0, 0, 0, 0), 1.0, 0.0)
}

fn unit_second() -> Unit {
    unit_new("s", dimension_new(0, 0, 1, 0, 0, 0, 0), 1.0, 0.0)
}

// Derived/scaled units
fn unit_gram() -> Unit {
    // 1 g = 0.001 kg
    unit_new("g", dimension_new(1, 0, 0, 0, 0, 0, 0), 0.001, 0.0)
}

fn unit_milligram() -> Unit {
    // 1 mg = 0.000001 kg
    unit_new("mg", dimension_new(1, 0, 0, 0, 0, 0, 0), 0.000001, 0.0)
}

fn unit_hour() -> Unit {
    // 1 hour = 3600 seconds
    unit_new("h", dimension_new(0, 0, 1, 0, 0, 0, 0), 3600.0, 0.0)
}

fn unit_liter() -> Unit {
    // 1 L = 0.001 m³
    unit_new("L", dimension_new(0, 3, 0, 0, 0, 0, 0), 0.001, 0.0)
}

fn unit_milliliter() -> Unit {
    // 1 mL = 0.000001 m³
    unit_new("mL", dimension_new(0, 3, 0, 0, 0, 0, 0), 0.000001, 0.0)
}
```

### Unit Operations

```sio
// Multiply dimensions
fn dimension_multiply(d1: Dimension, d2: Dimension) -> Dimension {
    Dimension {
        mass: d1.mass + d2.mass,
        length: d1.length + d2.length,
        time: d1.time + d2.time,
        current: d1.current + d2.current,
        temperature: d1.temperature + d2.temperature,
        amount: d1.amount + d2.amount,
        luminosity: d1.luminosity + d2.luminosity,
    }
}

// Divide dimensions
fn dimension_divide(d1: Dimension, d2: Dimension) -> Dimension {
    Dimension {
        mass: d1.mass - d2.mass,
        length: d1.length - d2.length,
        time: d1.time - d2.time,
        current: d1.current - d2.current,
        temperature: d1.temperature - d2.temperature,
        amount: d1.amount - d2.amount,
        luminosity: d1.luminosity - d2.luminosity,
    }
}

// Raise to power
fn dimension_power(d: Dimension, exp: i32) -> Dimension {
    Dimension {
        mass: d.mass * exp,
        length: d.length * exp,
        time: d.time * exp,
        current: d.current * exp,
        temperature: d.temperature * exp,
        amount: d.amount * exp,
        luminosity: d.luminosity * exp,
    }
}

// Check compatibility
fn dimensions_compatible(d1: Dimension, d2: Dimension) -> bool {
    (d1.mass == d2.mass) &&
    (d1.length == d2.length) &&
    (d1.time == d2.time) &&
    (d1.current == d2.current) &&
    (d1.temperature == d2.temperature) &&
    (d1.amount == d2.amount) &&
    (d1.luminosity == d2.luminosity)
}

// Is dimensionless?
fn is_dimensionless(d: Dimension) -> bool {
    dimensions_compatible(d, dimension_new(0, 0, 0, 0, 0, 0, 0))
}
```

### PKPD-Specific Units

For pharmacokinetics/pharmacodynamics:

```sio
fn unit_molar() -> Unit {
    // mol/L concentration
    unit_new("M", dimension_new(0, -3, 0, 0, 0, 1, 0), 1.0, 0.0)
}

fn unit_micromolar() -> Unit {
    // µmol/L
    unit_new("µM", dimension_new(0, -3, 0, 0, 0, 1, 0), 0.000001, 0.0)
}

fn unit_nanomolar() -> Unit {
    // nmol/L
    unit_new("nM", dimension_new(0, -3, 0, 0, 0, 1, 0), 0.000000001, 0.0)
}

fn unit_per_minute() -> Unit {
    // 1/min (rate constant)
    unit_new("min⁻¹", dimension_new(0, 0, -1, 0, 0, 0, 0), (1.0 / 60.0), 0.0)
}

fn unit_per_hour() -> Unit {
    // 1/hour
    unit_new("h⁻¹", dimension_new(0, 0, -1, 0, 0, 0, 0), (1.0 / 3600.0), 0.0)
}
```

### Type Annotation Example

```
let dose: mg = 500.0          // 500 milligrams
let volume: mL = 250.0        // 250 milliliters
let conc: mg/mL = dose / volume  // Concentration
// Type system checks: mg / mL is valid (both valid dimensions)
```

### Key Insight

Units are compile-time checked dimensions. Operations on quantities must preserve dimensional consistency: you can't add meters to seconds, but you can divide them. This catches unit errors at compile time.
