---
title: Refinement Types
description: SMT-verified type constraints for compile-time value validation
prerequisites:
  - /docs/getting-started.md
  - /docs/language/types.md
reading_time: 14 minutes
---

# Refinement Types

Refinement types extend Sounio's type system with logical predicates that constrain values. These predicates are verified at compile time using SMT (Satisfiability Modulo Theories) solving, typically with the Z3 theorem prover. This allows you to express and enforce invariants that would otherwise require runtime checks.

## What Are Refinement Types?

A refinement type takes the form `{ v: T | P }`, where:
- `v` is the refinement variable (the value being constrained)
- `T` is the base type
- `P` is a predicate that must be true for all values of this type

For example:
```sio
// A positive integer is any i32 where the value is greater than 0
type Positive = { x: i32 | x > 0 }
```

Values of type `Positive` are guaranteed to be greater than zero - not by runtime checks, but by compile-time verification.

## Basic Syntax

### Defining Refinement Types

```sio
// Simple predicates
type Positive = { x: i32 | x > 0 }
type NonNegative = { x: i32 | x >= 0 }
type NonZero = { x: i32 | x != 0 }

// Bounded ranges
type Percentage = { x: f64 | 0.0 <= x && x <= 100.0 }
type Probability = { p: f64 | 0.0 <= p && p <= 1.0 }
type UnitInterval = { x: f64 | 0.0 <= x && x <= 1.0 }

// Array length constraints
type NonEmpty<T> = { arr: [T] | len(arr) > 0 }
type Pair<T> = { arr: [T] | len(arr) == 2 }
```

### Using Refinement Types

```sio
// Function with refined parameter
fn sqrt_safe(x: { n: f64 | n >= 0.0 }) -> f64 {
    return sqrt(x)
}

// Function with refined return type
fn abs(x: i32) -> { r: i32 | r >= 0 } {
    if x < 0 {
        return -x
    }
    return x
}

// Literal values - verified at compile time
let p: Positive = 42        // OK: 42 > 0
let prob: Probability = 0.5 // OK: 0 <= 0.5 <= 1

// This would be a compile error
// let bad: Positive = -5   // ERROR: -5 does not satisfy x > 0
```

## Predicate Language

Predicates support arithmetic operations, comparisons, and logical connectives.

### Comparison Operators

```sio
// All comparison operators are supported
type Positive = { x: i32 | x > 0 }
type NonNegative = { x: i32 | x >= 0 }
type Negative = { x: i32 | x < 0 }
type NonPositive = { x: i32 | x <= 0 }
type Zero = { x: i32 | x == 0 }
type NonZero = { x: i32 | x != 0 }
```

### Logical Connectives

```sio
// Conjunction (AND)
type InRange = { x: i32 | x >= 0 && x <= 100 }

// Disjunction (OR)
type Extreme = { x: i32 | x < -100 || x > 100 }

// Negation (NOT)
type NotZero = { x: i32 | !(x == 0) }  // equivalent to x != 0

// Implication
type Constraint = { x: i32 | x > 0 => x < 100 }
```

### Arithmetic in Predicates

```sio
// Arithmetic operations
type EvenPositive = { x: i32 | x > 0 && x % 2 == 0 }
type MultipleOf10 = { x: i32 | x % 10 == 0 }

// Relationships between expressions
type SmallerThanSquare = { x: i32 | x < x * x }

// With other values (dependent refinements)
fn bounded_add(a: i32, b: i32) -> { r: i32 | r == a + b } {
    return a + b
}
```

## Medical Domain Examples

Refinement types are particularly powerful for medical and scientific domains where values must satisfy strict constraints.

### Safe Drug Dosing

```sio
// Maximum daily dose for acetaminophen is 4000 mg
type SafeAcetaminophenDose = { dose: mg | 0.0 < dose && dose <= 4000.0 }

// Pediatric dose often based on weight
type PediatricDose = { dose: mg | 0.0 < dose && dose <= 500.0 }

// Function that only accepts safe doses
fn administer_acetaminophen(dose: SafeAcetaminophenDose, patient_id: i64) with IO {
    record_medication(patient_id, "acetaminophen", dose)
}

// This compiles
administer_acetaminophen(650.0_mg, patient_001)

// This is a compile error
// administer_acetaminophen(5000.0_mg, patient_001)  // ERROR: exceeds safe dose
```

### Vital Sign Ranges

```sio
// Valid ranges for vital signs
type HeartRate = { hr: i32 | 20 <= hr && hr <= 300 }
type SystolicBP = { bp: i32 | 40 <= bp && bp <= 300 }
type DiastolicBP = { bp: i32 | 20 <= bp && bp <= 200 }
type Temperature = { temp: f64 | 25.0 <= temp && temp <= 45.0 }  // Celsius
type SpO2 = { spo2: f64 | 0.0 <= spo2 && spo2 <= 100.0 }

struct VitalSigns {
    heart_rate: HeartRate,
    systolic: SystolicBP,
    diastolic: DiastolicBP,
    temperature: Temperature,
    oxygen_saturation: SpO2,
}

fn record_vitals(vitals: VitalSigns) with IO {
    // All values are guaranteed to be in valid ranges
    store_vitals(vitals)
}
```

### Laboratory Values

```sio
// Creatinine clearance (mL/min) - used for drug dosing
type ValidCrCl = { crcl: f64 | 0.0 < crcl && crcl < 200.0 }

// Serum creatinine (mg/dL)
type ValidSerumCr = { scr: f64 | 0.1 <= scr && scr <= 20.0 }

// Renal function-adjusted dosing
fn adjust_dose_renal(
    base_dose: mg,
    crcl: ValidCrCl
) -> mg {
    if crcl < 30.0 {
        return base_dose * 0.5  // Severe impairment
    } else if crcl < 60.0 {
        return base_dose * 0.75 // Moderate impairment
    }
    return base_dose
}
```

### Therapeutic Ranges

```sio
// Vancomycin trough levels (mcg/mL)
type VancoTrough = { level: f64 | level > 0.0 }
type TherapeuticVancoTrough = { level: f64 | 15.0 <= level && level <= 20.0 }

fn check_vanco_level(level: VancoTrough) -> DosingRecommendation {
    if level < 10.0 {
        return DosingRecommendation::IncreaseDose
    } else if level < 15.0 {
        return DosingRecommendation::SlightIncrease
    } else if level <= 20.0 {
        return DosingRecommendation::Maintain
    } else if level <= 25.0 {
        return DosingRecommendation::SlightDecrease
    } else {
        return DosingRecommendation::HoldDose
    }
}
```

## SMT Verification

The compiler uses Z3 SMT solver to verify refinement type constraints.

### How Verification Works

```sio
type Positive = { x: i32 | x > 0 }

fn double_positive(x: Positive) -> Positive {
    return x * 2  // Compiler must prove: x > 0 => x * 2 > 0
}
```

The compiler generates an SMT query:
```
; Assumptions
(declare-const x Int)
(assert (> x 0))        ; x is Positive

; Goal: prove x * 2 > 0
(assert (not (> (* x 2) 0)))
(check-sat)  ; if UNSAT, the property holds
```

Z3 returns UNSAT (unsatisfiable), proving that `x * 2 > 0` whenever `x > 0`.

### Verification Through Operations

The SMT solver tracks constraints through operations:

```sio
type Positive = { x: f64 | x > 0.0 }

fn scale(p: Positive, factor: Positive) -> Positive {
    // Compiler proves: positive * positive = positive
    return p * factor
}

fn half(p: Positive) -> Positive {
    // Compiler proves: positive / 2 = positive
    return p / 2.0
}

fn square(p: Positive) -> Positive {
    // Compiler proves: positive * positive = positive
    return p * p
}
```

### When Verification Fails

```sio
type Positive = { x: i32 | x > 0 }

fn maybe_negative(x: Positive, y: i32) -> Positive {
    return x + y  // ERROR: cannot prove (x > 0 && true) => x + y > 0
}
// The compiler cannot prove x + y > 0 because y could be negative
```

Error message:
```
error[E0599]: refinement constraint not satisfied
  |
  |     return x + y
  |            ^^^^^ cannot prove `x + y > 0`
  |
  = note: cannot verify that result satisfies constraint `x > 0`
  = note: y has no constraints, could be negative
  = help: add refinement constraint to y: { y: i32 | y >= 0 }
```

## Extracting and Creating Refined Values

### Coercion to Base Type

Refinement types automatically coerce to their base type when needed:

```sio
type OrbitRatio = { r: f64 | 0.25 <= r && r <= 1.0 }

fn compute_interval(ratio: OrbitRatio, margin: f64) -> (f64, f64) {
    // ratio coerces to f64 in arithmetic
    let lower = ratio - margin
    let upper = ratio + margin
    return (lower, upper)
}
```

### Creating Refined Values

Use explicit type annotation - the compiler verifies the constraint:

```sio
// Literal values - verified at compile time
let ratio: OrbitRatio = 0.75  // OK: 0.25 <= 0.75 <= 1.0
// let bad: OrbitRatio = 0.1  // COMPILE ERROR: violates constraint

// From computed values - verified via SMT
fn normalize(x: f64) -> Probability {
    let clamped = clamp(x, 0.0, 1.0)
    return clamped  // OK: clamp guarantees 0.0 <= result <= 1.0
}
```

### Fallible Conversions

When a value might not satisfy the constraint, use `Option`:

```sio
fn try_as_probability(x: f64) -> Option<Probability> {
    if x >= 0.0 && x <= 1.0 {
        return Some(x)
    }
    return None
}

// Usage
let maybe_prob = try_as_probability(user_input)
match maybe_prob {
    Some(p) => use_probability(p),
    None => handle_error(),
}
```

## Inference with Qualifiers

Sounio can infer refinements in some cases using qualifier-based inference (Liquid Types style).

### Automatic Constraint Propagation

```sio
fn process(x: { v: i32 | v > 0 }) -> i32 {
    let y = x + 1      // Inferred: y > 1
    let z = y * 2      // Inferred: z > 2
    return z - 1       // Inferred: result > 1
}
```

### Predefined Qualifiers

The compiler maintains a set of qualifiers for common patterns:

```sio
// Qualifiers for array operations
// len(arr) > 0
// 0 <= i < len(arr)
// len(result) == len(input)

// Qualifiers for numeric operations
// v > 0
// v >= 0
// v != 0
// lo <= v <= hi
```

## Struct Invariants

Refinements can express invariants on struct fields:

```sio
struct BoundedCounter {
    value: i32,

    invariant value >= 0 && value <= 100
}

impl BoundedCounter {
    fn new(initial: { v: i32 | v >= 0 && v <= 100 }) -> Self {
        return BoundedCounter { value: initial }
    }

    fn increment(&!self) {
        if self.value < 100 {
            self.value = self.value + 1
        }
        // Invariant maintained: value still <= 100
    }

    fn decrement(&!self) {
        if self.value > 0 {
            self.value = self.value - 1
        }
        // Invariant maintained: value still >= 0
    }
}
```

## Array and Index Refinements

Refinement types are especially useful for safe array operations:

```sio
// Non-empty array
type NonEmptyArray<T> = { arr: [T] | len(arr) > 0 }

fn first<T>(arr: NonEmptyArray<T>) -> T {
    // Safe: we know len(arr) > 0
    return arr[0]
}

fn last<T>(arr: NonEmptyArray<T>) -> T {
    return arr[len(arr) - 1]  // Safe: index is valid
}

// Valid index type
type ValidIndex<N> = { i: usize | i < N }

fn safe_get<T>(arr: &[T; N], idx: ValidIndex<N>) -> T {
    // idx < N is guaranteed, no bounds check needed
    return arr[idx]
}
```

## Combining with Epistemic Types

Refinement types combine with Sounio's epistemic types for tracking both constraints and uncertainty:

```sio
use epistemic::{EpistemicValue, from_measurement}

type OrbitRatio = { r: f64 | 0.25 <= r && r <= 1.0 }

// Track uncertainty while preserving domain constraints
let ratio: EpistemicValue<OrbitRatio> = from_measurement(
    0.75,           // measured value (must satisfy OrbitRatio)
    0.02,           // uncertainty (+/- 2%)
    0.95            // 95% confidence
)

// Operations preserve both refinement and uncertainty
fn scale_ratio(r: EpistemicValue<OrbitRatio>, factor: f64) -> EpistemicValue<f64> {
    return r * factor  // Uncertainty propagates, refinement checked
}
```

## Error Messages

The compiler provides detailed error messages when refinement verification fails:

```sio
type Positive = { x: i32 | x > 0 }

fn broken(x: Positive) -> Positive {
    return x - 10
}
```

```
error[E0599]: refinement type constraint cannot be verified
   --> src/lib.sio:4:12
    |
  4 |     return x - 10
    |            ^^^^^^
    |
    = error: cannot prove `x - 10 > 0` given `x > 0`
    |
    = note: counterexample found: x = 5 gives result = -5
    = help: the constraint `x > 0` is not sufficient to prove `x - 10 > 0`
    = help: consider adding a stronger precondition: `{ x: i32 | x > 10 }`
```

## Best Practices

### 1. Start with Domain Constraints

Define refinement types that capture your domain's invariants:

```sio
// Pharmacokinetics domain
type Volume = { v: f64 | v > 0.0 }            // L
type Clearance = { cl: f64 | cl > 0.0 }       // L/h
type HalfLife = { t: f64 | t > 0.0 }          // h
type Concentration = { c: f64 | c >= 0.0 }    // mg/L
```

### 2. Use Refinements at API Boundaries

```sio
// Public API validates input
pub fn set_dose(dose: { d: mg | 0.0 < d && d <= MAX_DOSE }) { ... }

// Internal functions can assume valid input
fn calculate_concentration(dose: SafeDose, volume: Volume) -> Concentration {
    return dose / volume  // Both positive, so result is positive
}
```

### 3. Prefer Static Verification

Let the compiler verify constraints rather than adding runtime checks:

```sio
// Good: compile-time verification
fn safe_sqrt(x: { n: f64 | n >= 0.0 }) -> f64 {
    return sqrt(x)  // Always safe
}

// Less ideal: runtime check
fn sqrt_with_check(x: f64) -> Option<f64> {
    if x < 0.0 {
        return None
    }
    return Some(sqrt(x))
}
```

### 4. Document Constraint Rationale

```sio
// Heart rate must be physiologically plausible
// < 20 bpm: incompatible with life
// > 300 bpm: exceeds maximum sinus rate
type HeartRate = { hr: i32 | 20 <= hr && hr <= 300 }
```

## Limitations

Current limitations of Sounio's refinement type system:

1. **Non-linear arithmetic**: Complex polynomial constraints may timeout
2. **Floating-point**: Full IEEE 754 semantics not always modeled precisely
3. **Recursive predicates**: Limited support for recursive refinements
4. **Inference scope**: Cross-function inference is limited

## See Also

- [Units of Measure](/docs/language/units-of-measure.md) - Combining units with refinements
- [Epistemic Types](/docs/language/epistemic.md) - Uncertainty tracking
- [Type System](/docs/language/types.md) - Core type system
- [LLM Programming Guide](/docs/LLM_PROGRAMMING_GUIDE.md) - Complete syntax reference
