<!-- docs:meta
topic_id: repo.docs.contributor-guide.style-guide
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.contributor-guide.style-guide
-->

# Sounio Style Guide

Official style conventions for writing idiomatic Sounio code.

## Table of Contents

1. [Formatting](#formatting)
2. [Naming Conventions](#naming-conventions)
3. [Code Organization](#code-organization)
4. [Epistemic Best Practices](#epistemic-best-practices)
5. [Effect System Guidelines](#effect-system-guidelines)
6. [Units of Measure](#units-of-measure)
7. [Comments and Documentation](#comments-and-documentation)
8. [Error Handling](#error-handling)

---

## Formatting

### Indentation
- Use **4 spaces** for indentation (not tabs)
- Align continuation lines with the opening delimiter

```sio
// Good
fn calculate_result(
    param1: f64,
    param2: f64,
    param3: f64
) -> f64 {
    param1 + param2 + param3
}

// Bad
fn calculate_result(param1: f64,
param2: f64, param3: f64) -> f64 {
  param1 + param2 + param3
}
```

### Line Length
- Target **100 characters** maximum
- Hard limit **120 characters**
- Break long lines at logical points

### Braces
- Opening brace on same line as declaration
- Closing brace on its own line

```sio
// Good
fn example() -> i32 {
    if condition {
        do_something()
    } else {
        do_something_else()
    }
}

// Bad - K&R style not used
fn example() -> i32
{
    if condition
    {
        do_something()
    }
}
```

### Whitespace
- One space after keywords (`if`, `while`, `for`)
- No space between function name and parentheses
- Space around binary operators

```sio
// Good
if x > 0 {
    print(x)
}

// Bad
if x>0{
    print (x)
}
```

---

## Naming Conventions

### General Rules
- Use **descriptive names** - clarity over brevity
- Avoid abbreviations unless universally understood
- No Hungarian notation

### Variables and Functions
- **snake_case** for variables and functions

```sio
// Good
let user_count = 42
fn calculate_dose_adjustment() { }

// Bad
let UserCount = 42
let usrCnt = 42
fn CalculateDoseAdjustment() { }
```

### Types and Structs
- **PascalCase** for types, structs, enums

```sio
// Good
struct PatientData { }
enum ResultStatus { }
type ConcentrationValue = Knowledge<mg/mL>

// Bad
struct patient_data { }
enum result_status { }
```

### Constants
- **SCREAMING_SNAKE_CASE** for module-level constants

```sio
// Good
const MAX_DOSE: mg = 1000.0
const PI: f64 = 3.14159

// Bad
const maxDose: mg = 1000.0
const pi: f64 = 3.14159
```

### Effect Names
- **PascalCase** for custom effects

```sio
effect DatabaseAccess {
    fn query(sql: string) -> Result
}
```

---

## Code Organization

### Module Structure
```sio
// 1. Imports (grouped by category)
import stdlib.epistemic::*        // Standard library
import stdlib.units::*
import local.models::*            // Local modules

// 2. Type definitions
struct Data { }
enum Status { }

// 3. Constants
const DEFAULT_TIMEOUT: s = 30.0

// 4. Functions (public first, then private)
fn public_api() { }
fn internal_helper() { }

// 5. Main (if applicable)
fn main() { }
```

### Import Organization
- Group imports by source (stdlib, external, local)
- Sort alphabetically within groups
- Use wildcard (`*`) sparingly

```sio
// Good - organized groups
import stdlib.epistemic::*
import stdlib.math::{sin, cos, sqrt}
import stdlib.units::*

import external.plotting::*

import local.models::pk_model
import local.utils::validation

// Bad - unsorted, mixed
import local.utils::validation
import stdlib.math::{sqrt}
import external.plotting::*
import stdlib.epistemic::*
```

---

## Epistemic Best Practices

### Always Track Uncertainty
```sio
// Good - explicit uncertainty
let measurement = Knowledge::new(
    value: 42.0,
    uncertainty: 0.5,
    source: "instrument_A"
)

// Bad - raw value loses uncertainty
let measurement = 42.0  // Where did this come from? How precise?
```

### Use Confidence Gates
```sio
// Good - confidence-based execution
fn process_measurement(value: Knowledge<f64>) with IO {
    if value.confidence > 0.95 {
        critical_operation(value)
    } else {
        require_manual_review(value)
    }
}

// Bad - ignoring confidence
fn process_measurement(value: Knowledge<f64>) with IO {
    critical_operation(value)  // What if confidence is low?
}
```

### Preserve Provenance
```sio
// Good - maintain source information
fn calibrate(raw: Knowledge<T>) -> Knowledge<T> {
    let calibrated_value = raw.value * CALIBRATION_FACTOR
    Knowledge::new(
        value: calibrated_value,
        uncertainty: raw.uncertainty * CALIBRATION_FACTOR,
        source: Source {
            original: raw.provenance,
            transformation: "calibration_v2.1",
            timestamp: now(),
        }
    )
}

// Bad - losing provenance
fn calibrate(raw: Knowledge<T>) -> Knowledge<T> {
    Knowledge::new(calibrated_value, uncertainty: new_unc)
    // Where did this come from?
}
```

---

## Effect System Guidelines

### Explicit Effect Annotations
```sio
// Good - clear about effects
fn read_config() -> Config with IO {
    fs.read_to_string("config.toml")
}

fn update_counter(x: &! i32) with Mut {
    *x = *x + 1
}

// Bad - missing effect annotations
fn read_config() -> Config {  // Should have 'with IO'
    fs.read_to_string("config.toml")
}
```

### Minimal Effect Surface
```sio
// Good - separate pure computation from effects
fn compute_result(data: [f64]) -> f64 {
    // Pure function - no effects
    data.sum() / data.len() as f64
}

fn load_and_compute(path: string) -> f64 with IO {
    // Effects isolated to I/O boundary
    let data = load_data(path)
    compute_result(data)
}

// Bad - mixing computation with effects
fn compute_result(path: string) -> f64 with IO {
    let data = load_data(path)  // Effect mixed with computation
    data.sum() / data.len() as f64
}
```

### Effect Propagation
```sio
// Good - effects propagate naturally
fn process() -> Result with IO, Panic {
    let data = load()    // IO
    validate(data)       // Panic
    transform(data)
}

// Bad - catching and hiding effects
fn process() -> Result {
    try {
        let data = load()    // Hiding IO effect
        Ok(data)
    } catch {
        Err("failed")
    }
}
```

---

## Units of Measure

### Always Annotate Physical Quantities
```sio
// Good - explicit units
fn calculate_velocity(distance: m, time: s) -> m/s {
    distance / time
}

let dose: mg = 500.0
let volume: mL = 250.0
let concentration: mg/mL = dose / volume

// Bad - dimensionless numbers
fn calculate_velocity(distance: f64, time: f64) -> f64 {
    distance / time  // Units? m/s? km/h? 
}
```

### Let the Compiler Check Dimensions
```sio
// Good - compiler catches errors
let distance: m = 100.0
let time: s = 10.0
let velocity: m/s = distance / time  // ✓ Type checks

// Bad - would be compile error
// let invalid: m = distance + time  // ✗ Can't add m to s
```

### Use Custom Units for Domain-Specific Quantities
```sio
// Pharmacology
type Clearance = L/h
type AUC = mg*h/L
type Dose = mg

fn calculate_clearance(dose: Dose, auc: AUC) -> Clearance {
    dose / auc  // Units automatically: mg / (mg*h/L) = L/h
}
```

---

## Comments and Documentation

### Module-Level Documentation
```sio
/// Pharmacokinetic modeling for one-compartment models.
///
/// This module implements standard PK equations with epistemic
/// uncertainty propagation.
///
/// # Examples
/// ```
/// let model = OneCompartment::new(CL: 10.0 L/h, V: 100.0 L)
/// let conc = model.predict(dose: 1000.0 mg, time: 2.0 h)
/// ```
module pk_one_compartment
```

### Function Documentation
```sio
/// Calculate drug concentration over time using epistemic propagation.
///
/// # Parameters
/// - `dose`: Administered dose with measurement uncertainty
/// - `time`: Time since administration
/// - `clearance`: Drug clearance rate
/// - `volume`: Volume of distribution
///
/// # Returns
/// Predicted concentration with propagated uncertainty
///
/// # Effects
/// None (pure computation)
fn predict_concentration(
    dose: Knowledge<mg>,
    time: h,
    clearance: Knowledge<L/h>,
    volume: Knowledge<L>
) -> Knowledge<mg/L> {
    // Implementation
}
```

### Inline Comments
- Use `//` for single-line comments
- Explain **why**, not **what**
- Keep comments up to date with code

```sio
// Good - explains reasoning
// Use RK45 instead of Euler for stiff equations
let solution = solve_ode(method: RK45)

// Bad - obvious from code
// Set x to 10
let x = 10
```

### TODO Comments
```sio
// TODO(username): Brief description of what needs to be done
// FIXME: Brief description of the bug
// HACK: Explanation of why this hack is necessary
```

---

## Error Handling

### Use Result Types
```sio
// Good - explicit error handling
fn parse_config(path: string) -> Result<Config, ParseError> with IO {
    let content = fs.read_to_string(path)?
    toml.parse(content)
}

// Bad - panic on error
fn parse_config(path: string) -> Config with IO, Panic {
    let content = fs.read_to_string(path).unwrap()
    toml.parse(content).unwrap()
}
```

### Provide Context in Errors
```sio
// Good - detailed error
if dose < MINIMUM_DOSE {
    return Err(Error {
        kind: ValidationError,
        message: "Dose {dose} below minimum {MINIMUM_DOSE}",
        context: context!(dose, MINIMUM_DOSE),
    })
}

// Bad - vague error
if dose < MINIMUM_DOSE {
    return Err("invalid dose")
}
```

### Early Returns for Error Cases
```sio
// Good - fail fast
fn process(data: Data) -> Result<Output> {
    if data.is_empty() {
        return Err("empty data")
    }
    if !data.is_valid() {
        return Err("invalid data")
    }
    
    // Happy path not nested
    Ok(transform(data))
}

// Bad - deeply nested
fn process(data: Data) -> Result<Output> {
    if !data.is_empty() {
        if data.is_valid() {
            Ok(transform(data))
        } else {
            Err("invalid")
        }
    } else {
        Err("empty")
    }
}
```

---

## Additional Guidelines

### Prefer Immutability
```sio
// Good
let x = calculate()
let y = transform(x)

// Avoid when possible
var x = calculate()
x = transform(x)
```

### Small, Focused Functions
- Each function should do one thing well
- Aim for < 50 lines per function
- Extract complex logic into helper functions

### Avoid Magic Numbers
```sio
// Good
const BOILING_POINT_CELSIUS: celsius = 100.0
const FREEZING_POINT_CELSIUS: celsius = 0.0

if temperature > BOILING_POINT_CELSIUS {
    // ...
}

// Bad
if temperature > 100.0 {  // What is 100.0?
    // ...
}
```

---

## Tools

### Automatic Formatting
```bash
# Format a single file (prints the formatted source on stdout; it does not
# rewrite the file in place)
souc fmt file.sio
```

There is no whole-project formatting flag -- `souc fmt --all` is rejected with
`error: unknown flag: --all`. Loop over the files you touched instead.

### Linting
There is no `souc lint` subcommand. Linting is a separate script,
`scripts/dev/sounio-lint.py` (the grammar enforcer that catches Rust-isms):

```bash
# Lint a single file
python3 scripts/dev/sounio-lint.py file.sio

# Errors only, machine-readable
python3 scripts/dev/sounio-lint.py --errors-only --json file.sio

# Print the auto-fixed source on stdout (it does not edit the file in place)
python3 scripts/dev/sounio-lint.py --fix file.sio
make lint-fix FILE=file.sio          # same thing, via the Makefile

# Sweep the tree the way CI does (tests/stdlib and examples)
make lint
```

---

*Code is read more often than written. Write for clarity, not cleverness.*
