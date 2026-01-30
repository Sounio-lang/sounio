# Unit Runtime Checking Guide

## 1. Introduction

The catastrophic failure of NASA's Mars Climate Orbiter in 1999, which resulted in a $125 million loss, serves as a stark reminder of the dangers posed by unit inconsistencies—in this case, a mix-up between feet and meters that sent the spacecraft hurtling off course. Closer to everyday life, the United States alone sees over 7,000 deaths each year due to drug dosing errors, many of which stem from misunderstandings or mismatches in units like milligrams per kilogram or liters per hour. These incidents underscore the critical need for robust unit checking in software systems, particularly in high-stakes fields like aerospace and healthcare. While compile-time checks catch errors early, they alone are insufficient for dynamic scenarios involving user input or external data. This is where runtime checking becomes essential, complementing compile-time safeguards to ensure accuracy and safety. Sounio's approach integrates type-level programming for static validation with runtime mechanisms for dynamic enforcement, providing a comprehensive framework that prevents unit-related disasters at both design and execution stages.

## 2. Dimensional Analysis Fundamentals

Dimensional analysis forms the bedrock of unit checking by ensuring that physical quantities maintain their dimensional integrity throughout computations. This section explores the core concepts, starting with the foundational elements of the International System of Units (SI).

### 2.1 SI Base Quantities

The SI system defines seven base quantities, each with a corresponding unit, to standardize measurements across scientific disciplines. These include mass, denoted as M and measured in kilograms; length, L, in meters; time, T, in seconds; electric current, I, in amperes; temperature, Θ, in kelvin; amount of substance, N, in moles; and luminous intensity, J, in candelas. These base quantities serve as the building blocks for all derived units, allowing complex physical relationships to be expressed consistently and verifiably.

### 2.2 Dimension Type

At the type level, dimensions are represented as a generic type, `Dimension<M, L, T, I, Θ, N, J>`, where the parameters are integer exponents for each base quantity:

```
Dimension<M, L, T, I, Θ, N, J>
          │  │  │  │  │  │  └── J: Luminous Intensity (candela)
          │  │  │  │  │  └───── N: Amount of Substance (mole)
          │  │  │  │  └──────── Θ: Temperature (kelvin)
          │  │  │  └─────────── I: Electric Current (ampere)
          │  │  └────────────── T: Time (second)
          │  └───────────────── L: Length (meter)
          └──────────────────── M: Mass (kilogram)
```

**Examples of Derived Dimensions:**

| Physical Quantity | Dimension                     | SI Unit        |
|-------------------|-------------------------------|----------------|
| Velocity          | `Dimension<0, 1, -1, 0, 0, 0, 0>` | m/s            |
| Acceleration      | `Dimension<0, 1, -2, 0, 0, 0, 0>` | m/s²           |
| Force             | `Dimension<1, 1, -2, 0, 0, 0, 0>` | kg·m/s² (N)    |
| Energy            | `Dimension<1, 2, -2, 0, 0, 0, 0>` | kg·m²/s² (J)   |
| Power             | `Dimension<1, 2, -3, 0, 0, 0, 0>` | kg·m²/s³ (W)   |
| Pressure          | `Dimension<1, -1, -2, 0, 0, 0, 0>` | kg/(m·s²) (Pa) |
| Concentration     | `Dimension<1, -3, 0, 0, 0, 0, 0>` | kg/m³ (mg/L)   |

This type-level representation enables the compiler to enforce dimensional correctness without runtime overhead.

### 2.3 Quantity Type

To wrap scalar values with their units, Sounio employs the Quantity<Value, Unit> generic type. This wrapper facilitates compile-time unit checking, ensuring operations align dimensionally before execution. In release builds, it achieves zero-cost abstraction, meaning no performance penalty after optimization. A practical example is Quantity<f64, Meter>, which pairs a floating-point value with the meter unit for length measurements.

## 3. Type-Level Unit Checking

Type-level unit checking leverages Rust's advanced type system to perform algebraic operations on dimensions, catching errors at compile time and providing mathematical rigor to unit manipulations.

### 3.1 Unit Algebra

The type system models unit operations as algebraic manipulations of dimensions. Addition and subtraction require operands to share identical dimensions, preventing nonsensical combinations like adding mass to length. Multiplication combines dimensions by adding their exponents, while division subtracts them. Exponentiation scales the exponents by the power applied. These rules are enforced through the type system, offering proofs of correctness akin to mathematical derivations, which guarantees that valid physical equations compile successfully while invalid ones fail early.

### 3.2 Common SI Derived Units

Building on base quantities, SI derived units are predefined with their dimensional signatures. Frequency, measured in hertz, equates to the inverse of time (1/T). Force, in newtons, is mass times length over time squared (M·L/T²). Pressure, in pascals, derives from force per area, simplifying to mass over length and time squared (M/(L·T²)). Energy, in joules, is force times length (M·L²/T²), and power, in watts, is energy per time (M·L²/T³). Voltage, in volts, follows as power per current (M·L²/(T³·I)). These definitions ensure seamless integration into type-checked code.

### 3.3 PK/PD Units (Pharmacokinetics/Pharmacodynamics)

In pharmacokinetics and pharmacodynamics, domain-specific units are crucial for modeling drug behavior. Concentration units include milligrams per liter (mg/L), micrograms per milliliter (μg/mL), nanograms per milliliter (ng/mL), and molar concentrations like nanomolar (nM), micromolar (μM), millimolar (mM), and molar (M). Doses are expressed as milligrams (mg), micrograms (μg), or per body weight (mg/kg). Clearance rates use liters per hour (L/h) or milliliters per minute (mL/min), while volume of distribution employs liters (L) or liters per kilogram (L/kg). Area under the curve (AUC) is in units like mg·h/L or ng·h/mL, half-life in hours (h) or minutes (min), and rates such as mg/h, μg/min, or per-day/per-hour. Type-level checking adapts these to prevent dosing miscalculations.

## 4. Runtime Checking Mechanisms

While type-level checks excel in static contexts, runtime mechanisms extend validation to scenarios where units are determined dynamically, such as from user inputs or external sources.

### 4.1 Dynamic Quantities (DynamicQuantity)

The DynamicQuantity type stores dimensions at runtime, accommodating cases where units are not resolvable at compile time. It performs dimension checks during each operation, ensuring compatibility on the fly. This introduces a modest performance overhead of about 10-20% compared to static quantities, but the trade-off enables flexibility in interactive or data-driven applications.

### 4.2 Conversion Validation

Runtime validation distinguishes between affine conversions, which involve both scaling and offsets (e.g., Celsius to Fahrenheit), and linear ones, which use only scaling (e.g., milligrams to micrograms). The system verifies convertibility at runtime and handles errors gracefully for incompatible units, preventing silent failures in mixed-unit environments.

### 4.3 Assertion Macros

To aid developers, Sounio provides assertion macros for explicit runtime checks. For example, assert_dimensionless!(value) verifies that a quantity has no dimensions, useful for scalars in ratios. assert_compatible_units!(a, b) ensures two quantities share dimensions before operations like addition. assert_convertible!(from_unit, to_unit) confirms that a conversion is feasible, catching potential issues early in dynamic code.

## 5. Automatic Unit Conversion

Automatic conversions streamline workflows by handling unit transformations transparently, reducing boilerplate while maintaining accuracy.

### 5.1 Conversion Factors

The system supports SI prefixes like kilo (k), mega (M), giga (G), milli (m), micro (μ), nano (n), and pico (p) for seamless scaling. It also covers common scientific equivalences, such as liters to milliliters (L ↔ mL) or grams to milligrams (g ↔ mg), alongside time units like hours to minutes to seconds (h ↔ min ↔ s). Temperature conversions account for affine relationships, converting kelvin to Celsius or Fahrenheit with appropriate offsets and scales.

### 5.2 Inference and Coercion

Within the same dimension, implicit conversions occur automatically, selecting the most appropriate output unit to avoid precision loss. The system differentiates affine from linear conversions to preserve offsets, ensuring that operations like temperature-based calculations remain faithful to physical reality.

### 5.3 Conversion Cache

To optimize repeated conversions, a thread-safe cache memoizes factors, reducing computation time. This lazy-initialized store enhances performance in loops or high-volume processing without compromising safety.

## 6. PK/PD Examples (Clinical Relevance)

Applying these concepts to pharmacokinetics and pharmacodynamics demonstrates their practical value in clinical simulations, where unit errors can have life-or-death consequences.

### 6.1 Clearance Calculation

Consider calculating drug clearance from dose and AUC:

```sio
let dose: mg = 500.0
let auc: mg·h/L = 45.2
let clearance: L/h = dose / auc
// Runtime check: mg / (mg·h/L) = L/h ✓
```

Runtime validation confirms the dimensional outcome, yielding liters per hour as expected.

### 6.2 Volume of Distribution

Volume of distribution derives from dose divided by concentration:

```sio
let dose: mg = 1000.0
let concentration: mg/L = 12.5
let vd: L = dose / concentration
// Runtime check: mg / (mg/L) = L ✓
```

This ensures the result is in liters, aligning with physiological models.

### 6.3 Infusion Rate

For infusion rates targeting a specific concentration:

```sio
let target_conc: μg/mL = 2.0
let clearance: L/h = 8.5
let infusion_rate: mg/h = target_conc * clearance
// Auto-convert: μg/mL * L/h → mg/h
// Runtime validates: (mass/volume) * (volume/time) = mass/time ✓
```

Automatic prefix and unit adjustments validate the mass-per-time result.

### 6.4 Dose Adjustment

Dose adjustments based on renal function use ratios:

```sio
let creatinine_clearance: mL/min = 45.0
let reference_clearance: mL/min = 120.0
let normal_dose: mg = 500.0
let adjusted_dose: mg = normal_dose * (creatinine_clearance / reference_clearance)
// Runtime check: dimensionless ratio ✓
```

The system confirms the ratio is dimensionless, preserving dose units.

### 6.5 Michaelis-Menten Kinetics

In enzyme kinetics modeling:

```sio
let vmax: mg/h = 100.0
let km: mg/L = 5.0
let conc: mg/L = 10.0
let rate: mg/h = (vmax * conc) / (km + conc)
// Runtime validates each step
```

Step-by-step checks ensure dimensional consistency in the rate equation.

### 6.6 One-Compartment Model

For a simple pharmacokinetic model:

```sio
let dose: mg = 500.0
let vd: L = 50.0
let ke: 1/h = 0.15  // Elimination rate constant
let t: h = 4.0
let conc: mg/L = (dose / vd) * exp(-ke * t)
// Runtime checks:
// - dose/vd: mg/L ✓
// - ke*t: dimensionless ✓
// - exp: dimensionless → dimensionless ✓
```

Validations cover initial concentration, dimensionless exponent, and exponential application.

## 7. Integration with Knowledge Types

Sounio extends unit checking to epistemic reasoning by combining dimensional types with uncertainty modeling.

### 7.1 QuantifiedKnowledge

The QuantifiedKnowledge type merges knowledge representations with quantities, such as Knowledge<Quantity<f64, mg/L>>. This propagates both uncertainty and units through operations, enabling dual validation that accounts for measurement imprecision alongside dimensional accuracy.

### 7.2 Confidence Propagation with Units

Uncertainty propagates naturally with units:

```sio
let dose: Knowledge<mg> = measure(500.0, uncertainty: 5.0)
let volume: Knowledge<mL> = measure(10.0, uncertainty: 0.2)
let conc: Knowledge<mg/mL> = dose / volume
// Both confidence and units propagate correctly
```

Conversions and arithmetic maintain both aspects, supporting reliable clinical inferences.

### 7.3 Unit-Aware Refinement Types

Refinement types incorporate units and constraints, like:

```sio
type TherapeuticConcentration = {
    c: mg/L | 10.0 <= c <= 20.0
} with Knowledge
// Combines: units, refinement, epistemic
```

This enforces therapeutic ranges while tracking units and uncertainty.

## 8. Runtime Error Handling

Robust error handling ensures that unit issues surface clearly, aiding debugging in dynamic contexts.

### 8.1 Unit Mismatch Detection

Runtime checks flag incompatible operations:

```sio
let mass: kg = 70.0
let length: m = 1.75
// This fails at runtime in dynamic mode:
let wrong = mass + length  // ERROR: incompatible dimensions
```

Such detections prevent invalid computations from proceeding.

### 8.2 Conversion Errors

Inconvertible units trigger errors:

```sio
let temp: °C = 37.0
let energy: J = 100.0
// Cannot convert temperature to energy:
let bad = convert(temp, to: J)  // ERROR: not convertible
```

This safeguards against physically meaningless transformations.

### 8.3 Error Messages

Errors provide detailed feedback, including dimension mismatches, suggested conversions, expected versus actual dimensions, and source locations. This facilitates quick resolution in complex codebases.

## 9. Performance Optimization

Balancing correctness with efficiency is key, especially in computationally intensive applications.

### 9.1 Static vs Dynamic Trade-offs

| Approach         | Overhead | Error Detection | Use Case                  | Flexibility |
|------------------|----------|-----------------|---------------------------|-------------|
| Static (Quantity<T, U>) | 0%       | Compile-time    | Core algorithms, libraries | Low - fixed at compile |
| Dynamic (DynamicQuantity) | 10-20%   | Runtime         | User input, config files   | High - resolved at runtime |
| Hybrid           | 2-5%     | Both            | API boundaries             | Medium - static core, dynamic edges |

**Performance Breakdown (1M operations):**

| Operation Type          | Bare f64 | Static Quantity | Dynamic Quantity | Overhead |
|-------------------------|----------|-----------------|------------------|----------|
| Addition                | 1.2 ns   | 1.2 ns          | 14.5 ns          | 12x      |
| Multiplication          | 1.5 ns   | 1.5 ns          | 16.8 ns          | 11x      |
| Unit Conversion         | N/A      | 2.1 ns (inline) | 28.3 ns          | 13x      |
| Dimension Check         | N/A      | 0 ns (compile)  | 8.2 ns           | ∞        |

**Recommendation:**
- Use static `Quantity<f64, Unit>` for 99% of code
- Reserve `DynamicQuantity` for parsing user input or external data
- Apply hybrid approach at API boundaries (accept dynamic, convert to static internally)

### 9.2 Conversion Caching

Precomputed tables and lazy initialization speed up conversions, with thread-local caches minimizing contention in concurrent environments.

### 9.3 SIMD Unit Operations

Vectorized conversions and batch checking leverage SIMD instructions, while parallel propagation accelerates large-scale simulations.

## 10. Usage Examples

These examples illustrate practical application across domains.

### 10.1 Basic Arithmetic with Units

Simple physics computations:

```sio
let distance: m = 100.0
let time: s = 9.58
let velocity: m/s = distance / time
let acceleration: m/s² = velocity / time
```

Runtime confirms derived units like meters per second and per second squared.

### 10.2 Temperature Conversions

Handling affine transforms:

```sio
let celsius: °C = 37.0
let fahrenheit: °F = convert(celsius, to: °F)  // 98.6°F
let kelvin: K = convert(celsius, to: K)        // 310.15 K
```

The system applies offsets and scales accurately.

### 10.3 Drug Concentration Units

Prefix and molar conversions:

```sio
let conc_mg_ml: mg/mL = 25.0
let conc_ug_ml: μg/mL = convert(conc_mg_ml, to: μg/mL)  // 25000.0
let conc_molar: mM = convert(conc_mg_ml, to: mM, mw: 180.16)  // molecular weight
```

Molecular weight integration enables molarity calculations.

### 10.4 Dimensional Analysis in Physics

Kinetic energy validation:

```sio
let mass: kg = 10.0
let velocity: m/s = 5.0
let kinetic_energy: J = 0.5 * mass * velocity * velocity
// Runtime validates: kg · (m/s)² = kg·m²/s² = J ✓
```

Dimensions confirm joule equivalence.

### 10.5 Fluid Dynamics

Mass from density and volume:

```sio
let density: kg/m³ = 1000.0  // Water
let volume: L = 5.0
let mass: kg = density * convert(volume, to: m³)
// Auto-converts L → m³, validates dimensions
```

Automatic unit adjustment yields kilograms.

## 11. FFI and External Integration

Integrating with external systems requires careful unit management to avoid boundary errors.

### 11.1 C Interop

At foreign function interfaces (FFI), units are stripped for C compatibility, with validation reinstated upon return. C functions can be annotated with expected units to guide safe interactions.

### 11.2 Database Storage

Values are stored alongside unit metadata, allowing reconstruction of typed quantities on retrieval. Schema validation ensures consistency across database operations.

### 11.3 JSON/Protocol Buffers

Serialization encodes units as strings, with deserialization performing validation. This maintains backward compatibility in evolving APIs.

## 12. Testing and Validation

Comprehensive testing verifies unit behaviors and conversions.

### 12.1 Unit Test Macros

Targeted tests for PK/PD:

```rust
#[test]
fn test_clearance() {
    let dose = Quantity::new(500.0, Milligram);
    let auc = Quantity::new(45.2, MilligramHourPerLiter);
    let clearance = dose / auc;
    assert_eq!(clearance.unit(), LiterPerHour);
    assert_approx_eq!(clearance.value(), 11.06);
}
```

These confirm units and numerical accuracy.

### 12.2 Property-Based Testing

Using QuickCheck, tests enforce unit laws like associativity and commutativity, alongside round-trip conversion fidelity.

### 12.3 Dimensional Analysis Regression

Tests incorporate physical constants, validated conversions, and reproductions of historical bugs to prevent regressions.

## 13. Debugging Unit Issues

Tools for inspecting and tracing units simplify troubleshooting.

### 13.1 Dimension Inspection

Direct querying:

```sio
println!("Dimension: {}", value.dimension());
// Output: Dimension<1, 1, -2, 0, 0, 0, 0>  (M·L/T² = Newton)
```

This reveals underlying structures.

### 13.2 Conversion Tracing

Detailed paths:

```sio
let traced = convert_with_trace(celsius, to: fahrenheit);
// Shows: °C → K (offset +273.15) → °F (scale 9/5, offset +32)
```

Tracing elucidates transformation steps.

### 13.3 Type Mismatch Diagnosis

Static failures yield compiler messages, while dynamic ones provide stack traces with correction suggestions, streamlining fixes.

## 14. Best Practices

Adopting these practices maximizes reliability and usability.

### 14.1 When to Use Static vs Dynamic

Favor static checking for APIs, libraries, and performance-critical paths; reserve dynamic for user inputs, configurations, or scripts where flexibility is paramount.

### 14.2 Choosing Units

Store in SI base units for universality, display in user-preferred formats, and document expectations in API signatures to guide integrators.

### 14.3 Handling Dimensionless Quantities

Explicitly mark dimensionless types for ratios, percentages, or logarithmic scales like pH and decibels, ensuring they integrate correctly with dimensional operations.

## 15. Common Pitfalls

Awareness of these issues prevents subtle errors.

### 15.1 Forgetting Affine Conversions

Temperatures demand affine handling for offsets; applying linear scales to Celsius-Fahrenheit conversions distorts results.

### 15.2 Precision Loss

Large-to-small conversions (e.g., kg to μg) risk overflow; select numeric types judiciously and monitor rounding in multi-step processes.

### 15.3 Inconsistent Unit Systems

Avoid mixing SI with imperial, CGS versus SI, or natural units in physics; standardize early to sidestep mismatches.

## 16. Advanced Topics

For specialized needs, Sounio supports extensions beyond basics.

### 16.1 Custom Units

Domain-specific units can be defined, the system extended, and conversions registered to accommodate niche applications like astronomy or engineering.

### 16.2 Compound Units

Automatic simplification reduces complex expressions to canonical forms, with pretty-printing for readable output.

### 16.3 Logarithmic Units

Support for decibels (dB), pH, and astronomical magnitudes treats them as transformations of dimensionless ratios, preserving analytical integrity.

## 17. References

This guide draws from established standards: ISO 80000 for quantities and units; NIST SP 811 as a guide for metric practice; the Guide to the Expression of Uncertainty in Measurement (GUM); FDA guidance on units in clinical trials; and the 9th edition of the SI Brochure (2019). These resources provide the foundational authority for Sounio's implementations.