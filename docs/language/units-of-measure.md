---
title: Units of Measure
description: Compile-time dimensional analysis for type-safe scientific computing
prerequisites:
  - /docs/getting-started.md
  - /docs/language/types.md
reading_time: 12 minutes
---

# Units of Measure

Sounio provides built-in support for units of measure with compile-time dimensional analysis. This prevents catastrophic errors from unit mismatches - like the $125 million Mars Climate Orbiter loss caused by mixing metric and imperial units, or the approximately 7,000 annual deaths in the US from drug dosing errors.

## Why Units Matter

Consider this seemingly innocent code:

```sio
fn calculate_dose(weight: f64, dose_per_kg: f64) -> f64 {
    return weight * dose_per_kg
}

// Is this correct?
let result = calculate_dose(70.0, 5.0)  // 70 what? 5 what?
```

Without units, we have no way to know if:
- Weight is in kilograms or pounds
- Dose is in mg/kg or mcg/kg
- The result is in mg, g, or something else entirely

With Sounio's unit system:

```sio
fn calculate_dose(weight: kg, dose_per_kg: mg/kg) -> mg {
    return weight * dose_per_kg
}

let result: mg = calculate_dose(70.0_kg, 5.0_mg/kg)  // 350 mg - type safe!

// This would be a compile error:
// let wrong = calculate_dose(70.0_lb, 5.0_mg/kg)  // ERROR: expected kg, got lb
```

## Base Units (SI System)

Sounio's unit system is built on the seven SI base quantities:

| Dimension | Symbol | SI Unit | Sounio Type |
|-----------|--------|---------|-------------|
| Mass | M | kilogram (kg) | `kg` |
| Length | L | meter (m) | `m` |
| Time | T | second (s) | `s` |
| Electric Current | I | ampere (A) | `A` |
| Temperature | Theta | kelvin (K) | `K` |
| Amount of Substance | N | mole (mol) | `mol` |
| Luminous Intensity | J | candela (cd) | `cd` |

### Using Base Units

```sio
// Declare values with units using underscore prefix
let mass: kg = 75.0_kg
let length: m = 1.82_m
let time: s = 60.0_s
let current: A = 2.5_A
let temperature: K = 310.0_K
let amount: mol = 0.5_mol

// Alternative angle-bracket syntax
let mass2: kg = 75.0<kg>
let length2: m = 1.82<m>
```

## Derived Units

Derived units are combinations of base units. Sounio automatically tracks dimensions through calculations.

### Common Derived Units

```sio
// Velocity: length / time = m/s
let velocity: m/s = 10.0_m / 2.0_s  // 5.0 m/s

// Acceleration: velocity / time = m/s^2
let acceleration: m/s^2 = velocity / 5.0_s  // 1.0 m/s^2

// Force: mass * acceleration = kg*m/s^2 (Newton)
let force: N = 10.0_kg * acceleration  // 10.0 N

// Energy: force * distance = kg*m^2/s^2 (Joule)
let energy: J = force * 5.0_m  // 50.0 J

// Power: energy / time = kg*m^2/s^3 (Watt)
let power: W = energy / 10.0_s  // 5.0 W

// Pressure: force / area = kg/(m*s^2) (Pascal)
let area: m^2 = 2.0_m * 2.0_m  // 4.0 m^2
let pressure: Pa = force / area  // 2.5 Pa

// Frequency: 1/time = 1/s (Hertz)
let frequency: Hz = 1.0 / 0.02_s  // 50.0 Hz
```

### Dimensional Type Checking

The compiler tracks dimensions and catches mismatches:

```sio
let mass: kg = 10.0_kg
let length: m = 5.0_m
let time: s = 2.0_s

// Valid operations
let velocity = length / time      // m/s - OK
let momentum = mass * velocity    // kg*m/s - OK
let energy = momentum * velocity  // kg*m^2/s^2 - OK

// Invalid operations - compile errors
// let wrong1 = mass + length     // ERROR: cannot add kg and m
// let wrong2 = mass / mass + length  // ERROR: cannot add dimensionless and m
// let wrong3: kg = length        // ERROR: expected kg, got m
```

## Scientific and Medical Units

Sounio provides a rich set of units for scientific computing, particularly pharmacokinetics and pharmacodynamics.

### Mass Units

```sio
// Mass with metric prefixes
let dose_ng: ng = 500.0_ng        // nanograms
let dose_mcg: mcg = 100.0_mcg     // micrograms
let dose_mg: mg = 50.0_mg         // milligrams
let dose_g: g = 1.5_g             // grams
let dose_kg: kg = 70.0_kg         // kilograms

// Conversions are automatic when dimensions match
let total_mg: mg = dose_g  // Converts 1.5 g to 1500 mg
```

### Volume Units

```sio
let volume_uL: uL = 50.0_uL       // microliters
let volume_mL: mL = 10.0_mL       // milliliters
let volume_dL: dL = 1.0_dL        // deciliters
let volume_L: L = 5.0_L           // liters
```

### Time Units

```sio
let seconds: s = 60.0_s
let minutes: min = 5.0_min
let hours: h = 2.0_h
let days: day = 7.0_day
```

### Concentration Units

```sio
// Mass concentration
let conc_ng_mL: ng/mL = 100.0_ng/mL
let conc_mcg_mL: mcg/mL = 10.0_mcg/mL
let conc_mg_mL: mg/mL = 5.0_mg/mL
let conc_mg_L: mg/L = 50.0_mg/L

// Molar concentration
let molar: mol/L = 0.1_mol/L           // 0.1 M
let millimolar: mmol/L = 50.0_mmol/L   // 50 mM
let micromolar: umol/L = 100.0_umol/L  // 100 uM
let nanomolar: nmol/L = 500.0_nmol/L   // 500 nM
```

### Rate Units

```sio
// Rate constants
let ke: 1/h = 0.1_1/h             // elimination rate constant
let ka: 1/h = 1.5_1/h             // absorption rate constant

// Clearance
let clearance: L/h = 5.0_L/h
let renal_cl: mL/min = 100.0_mL/min

// Infusion rate
let infusion: mg/h = 50.0_mg/h
```

## Pharmacokinetic Calculations

Sounio's unit system is particularly powerful for pharmacokinetic modeling where unit errors can be life-threatening.

### Basic PK Calculations

```sio
struct PKParams {
    cl: L/h,       // Clearance
    v: L,          // Volume of distribution
    ka: 1/h,       // Absorption rate constant
}

fn calculate_ke(params: &PKParams) -> 1/h {
    // ke = CL / V (units check: L/h / L = 1/h)
    return params.cl / params.v
}

fn calculate_half_life(ke: 1/h) -> h {
    // t1/2 = ln(2) / ke
    let ln2: f64 = 0.693147
    return ln2 / ke  // dimensionless / (1/h) = h
}

fn calculate_auc(dose: mg, cl: L/h) -> mg*h/L {
    // AUC = Dose / CL (units: mg / (L/h) = mg*h/L)
    return dose / cl
}
```

### Concentration Calculations

```sio
fn plasma_concentration(
    dose: mg,
    volume: L,
    ke: 1/h,
    time: h
) -> mg/L {
    // C(t) = (Dose/V) * e^(-ke*t)
    let c0: mg/L = dose / volume
    let exponent: f64 = -ke * time  // (1/h) * h = dimensionless
    return c0 * exp(exponent)
}

fn steady_state_concentration(
    infusion_rate: mg/h,
    clearance: L/h
) -> mg/L {
    // Css = R0 / CL
    return infusion_rate / clearance
}
```

### Drug Dosing Example

```sio
fn calculate_loading_dose(
    target_concentration: mg/L,
    volume_of_distribution: L
) -> mg {
    // Loading dose = Cp_target * Vd
    return target_concentration * volume_of_distribution
}

fn calculate_maintenance_dose(
    target_concentration: mg/L,
    clearance: L/h,
    dosing_interval: h
) -> mg {
    // Maintenance dose = Cp_target * CL * tau
    return target_concentration * clearance * dosing_interval
}

// Usage
let target: mg/L = 10.0_mg/L
let vd: L = 50.0_L
let cl: L/h = 5.0_L/h
let tau: h = 12.0_h

let loading: mg = calculate_loading_dose(target, vd)        // 500 mg
let maintenance: mg = calculate_maintenance_dose(target, cl, tau)  // 600 mg
```

## Unit Arithmetic Rules

### Multiplication and Division

When multiplying or dividing quantities, their dimensions combine:

```sio
// Multiplication adds exponents
let force: N = 10.0_kg * 5.0_m/s^2  // kg^1 * m^1 * s^-2 = N

// Division subtracts exponents
let velocity: m/s = 100.0_m / 10.0_s  // m^1 * s^-1

// Repeated multiplication
let volume: m^3 = 2.0_m * 3.0_m * 4.0_m  // m^3 = 24 m^3
```

### Addition and Subtraction

Only quantities with identical dimensions can be added or subtracted:

```sio
// Valid - same dimensions
let total_mass: kg = 5.0_kg + 3.0_kg        // 8.0 kg
let mass_diff: kg = 10.0_kg - 2.5_kg        // 7.5 kg
let total_length: m = 1.5_m + 0.5_m         // 2.0 m

// Invalid - different dimensions
// let invalid = 5.0_kg + 3.0_m            // ERROR: cannot add kg and m
// let invalid2 = 10.0_m/s + 5.0_m         // ERROR: cannot add m/s and m
```

### Powers and Roots

```sio
// Squaring
let area: m^2 = (5.0_m) * (5.0_m)  // 25.0 m^2

// Cubing
let volume: m^3 = (2.0_m) * (2.0_m) * (2.0_m)  // 8.0 m^3

// Square root (dimension exponents must be even)
let side: m = sqrt(25.0_m^2)  // 5.0 m

// This would fail - cannot take sqrt of odd-exponent dimension
// let invalid = sqrt(8.0_m^3)  // ERROR: cannot take sqrt of m^3
```

## Dimensionless Quantities

Some calculations produce dimensionless results:

```sio
// Ratios of same dimension are dimensionless
let ratio: f64 = 100.0_kg / 50.0_kg  // 2.0 (dimensionless)

// Percentages
let efficiency: f64 = output_energy / input_energy  // dimensionless

// Trigonometric functions require dimensionless input
let angle_rad: f64 = 1.57  // radians are dimensionless
let result = sin(angle_rad)  // OK

// let invalid = sin(5.0_m)  // ERROR: expected dimensionless, got m
```

## Integration with Knowledge Types

Units combine naturally with Sounio's epistemic types for uncertainty tracking:

```sio
use epistemic::{EpistemicValue, from_measurement}

// Track both units AND uncertainty
let measured_concentration: EpistemicValue<mg/L> = from_measurement(
    50.0_mg/L,     // measured value with units
    2.5_mg/L,      // uncertainty (same units)
    0.95           // 95% confidence
)

// Unit-safe calculations preserve uncertainty
let volume: EpistemicValue<L> = from_measurement(0.5_L, 0.01_L, 0.99)
let total_drug: EpistemicValue<mg> = measured_concentration * volume
// total_drug automatically propagates uncertainty AND checks units
```

## Type Errors from Unit Mismatches

The compiler provides clear error messages for unit mistakes:

```sio
fn bad_calculation() {
    let mass: kg = 10.0_kg
    let length: m = 5.0_m

    // Attempt to add incompatible units
    let result = mass + length
    // error[E0308]: mismatched units
    //   |
    //   |     let result = mass + length
    //   |                        ^^^^^^ expected `kg`, found `m`
    //   |
    //   = note: cannot add quantities with different dimensions
    //   = note: mass has dimension [M], length has dimension [L]
}
```

## Best Practices

### 1. Always Annotate Physical Quantities

```sio
// Good: explicit units
let patient_weight: kg = 70.0_kg
let dose_per_kg: mg/kg = 5.0_mg/kg
let total_dose: mg = patient_weight * dose_per_kg

// Bad: raw numbers
let weight = 70.0    // 70 what?
let dose = 5.0       // danger!
```

### 2. Define Domain-Specific Types

```sio
// Create type aliases for clarity
type DrugDose = mg
type PlasmaConcentration = mg/L
type Clearance = L/h
type HalfLife = h

fn calculate_pk(dose: DrugDose, cl: Clearance) -> PlasmaConcentration {
    // ...
}
```

### 3. Use Structs for Related Parameters

```sio
struct PatientParams {
    weight: kg,
    height: cm,
    age: years,
    bsa: m^2,        // Body surface area
    crcl: mL/min,    // Creatinine clearance
}

struct DrugParams {
    dose: mg,
    volume: L,
    clearance: L/h,
    bioavailability: f64,  // dimensionless fraction
}
```

### 4. Validate at Boundaries

```sio
fn parse_dose_input(input: string) -> Result<mg, ParseError> {
    // Parse and validate user input has correct units
    let value = parse_float(input)?
    if value < 0.0 {
        return Err(ParseError::NegativeDose)
    }
    // Explicit unit assignment catches data entry errors
    return Ok(value as mg)
}
```

## See Also

- [Refinement Types](/docs/language/refinement-types.md) - Combining units with value constraints
- [Epistemic Types](/docs/language/epistemic.md) - Uncertainty tracking with units
- [Pharmacokinetics Module](/docs/stdlib/pkpd.md) - PK/PD library
- [LLM Programming Guide](/docs/LLM_PROGRAMMING_GUIDE.md) - Complete syntax reference
