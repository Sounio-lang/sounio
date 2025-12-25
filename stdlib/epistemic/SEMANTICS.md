# Epistemic Computing Semantics

This document defines the formal semantics and invariants for Demetrios epistemic types. These invariants are non-negotiable and enforced by the type system and runtime.

---

## Core Principle: Two Orthogonal Channels

Epistemic computing in Demetrios maintains **two orthogonal channels** that must never be conflated:

### Channel A: Uncertainty of the Value (Metrology)

This is the **uncertainty of the measured/computed quantity itself**:
- Standard uncertainty (σ, standard deviation)
- Interval bounds (guaranteed enclosure)
- Full distribution (when needed)

**Propagation rule**: GUM (Guide to the Expression of Uncertainty in Measurement, JCGM 100:2008)
- For independent inputs: variances add in quadrature
- For correlated inputs: include covariance terms
- For arbitrary functions: use sensitivity coefficients (partial derivatives)

**Reference**: [JCGM 100:2008 - GUM](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf)

### Channel B: Confidence in the Claim (Epistemic Trust)

This is **belief in the validity of the claim**, independent of measurement precision:
- Source reliability
- Method quality
- Calibration status
- Protocol adherence

**Propagation rule**: Monotone non-increasing under pure transformations
- Confidence can only decrease through computation
- Increasing confidence requires explicit `boost(reason, citation)` with provenance

---

## The Knowledge Type

```d
struct Knowledge<T> {
    value: T,

    // Channel A: uncertainty of the value (metrology)
    uncert: Uncertainty<T>,

    // Channel B: confidence in provenance/assumptions (epistemic trust)
    conf: f64,  // invariant: 0.0 <= conf <= 1.0

    provenance: Vec<Provenance>,
    strategy: CombinationStrategy,
}

enum Uncertainty<T> {
    // Standard uncertainty (GUM-style)
    StdDev { u: T },

    // Guaranteed enclosure (IEEE 1788)
    Interval { lo: T, hi: T },

    // No uncertainty tracked (exact value)
    Exact,
}
```

---

## Non-Negotiable Invariants

### Invariant 1: Confidence Monotonicity

**Under pure transformations, confidence NEVER increases.**

```
∀ f: T → U, ∀ k: Knowledge<T>:
    (f.map(k)).conf ≤ k.conf
```

For combining multiple inputs:
```
∀ f: (T, U) → V, ∀ k1: Knowledge<T>, k2: Knowledge<U>:
    (f.combine(k1, k2)).conf ≤ min(k1.conf, k2.conf)
```

**Rationale**: Computation cannot create trust. If you don't fully trust your inputs, you cannot fully trust your outputs.

**Exception**: Explicit `boost(reason, citation, method)` can increase confidence, but:
1. Must provide a reason string
2. Must provide a citation or method reference
3. Is recorded in provenance
4. Should be rare and justified

### Invariant 2: Uncertainty Non-Contraction

**Under pure transformations, uncertainty NEVER shrinks.**

For StdDev (independent inputs):
```
∀ f: (T, U) → V:
    u(f(a,b))² ≥ (∂f/∂a · u(a))² + (∂f/∂b · u(b))²
```

For Interval:
```
∀ f: (T, U) → V, ∀ a ∈ [a_lo, a_hi], b ∈ [b_lo, b_hi]:
    f(a, b) ∈ [f_lo, f_hi]  (guaranteed enclosure)
```

**Rationale**: This is the mathematical definition of proper uncertainty propagation. Violating this means your uncertainty bounds are lies.

**Exception**: Explicit evidence fusion (e.g., combining independent measurements of the same quantity) can reduce uncertainty, but this requires:
1. Explicit `fuse()` operation
2. Independence assumptions documented
3. Recorded in provenance

### Invariant 3: Provenance Growth

**Provenance records are append-only under transformations.**

```
∀ operation:
    result.provenance ⊇ inputs.provenance
```

**Rationale**: The computational history is part of the epistemic state. Dropping provenance is dropping information about why we believe what we believe.

### Invariant 4: No Silent Unwrap

**Extracting a value from Knowledge<T> requires explicit acknowledgment.**

```d
// ILLEGAL - no implicit conversion
let x: f64 = knowledge_value  // ERROR

// LEGAL - explicit unwrap with reason
let x: f64 = knowledge_value.unwrap("accepted for clinical use")

// LEGAL - explicit unsafe block
let x: f64 = unsafe { knowledge_value.value }
```

**Rationale**: Discarding epistemic metadata should be a conscious decision, not an accident.

---

## Uncertainty Propagation Rules

### StdDev Propagation (GUM First-Order)

For a function y = f(x₁, x₂, ..., xₙ) with independent inputs:

```
u(y)² = Σᵢ (∂f/∂xᵢ)² · u(xᵢ)²
```

**Special cases (exact formulas):**

| Operation | Formula | Combined Uncertainty |
|-----------|---------|---------------------|
| y = a + b | u(y)² = u(a)² + u(b)² | Quadrature sum |
| y = a - b | u(y)² = u(a)² + u(b)² | Quadrature sum |
| y = a · b | u_rel(y)² = u_rel(a)² + u_rel(b)² | Relative quadrature |
| y = a / b | u_rel(y)² = u_rel(a)² + u_rel(b)² | Relative quadrature |
| y = k · a | u(y) = |k| · u(a) | Scaled |
| y = aⁿ | u_rel(y) = |n| · u_rel(a) | Power rule |

Where u_rel(x) = u(x)/|x| is the relative uncertainty.

### Interval Propagation (IEEE 1788 Style)

For guaranteed enclosure, use interval arithmetic:

| Operation | Formula |
|-----------|---------|
| [a,b] + [c,d] | [a+c, b+d] |
| [a,b] - [c,d] | [a-d, b-c] |
| [a,b] · [c,d] | [min(ac,ad,bc,bd), max(ac,ad,bc,bd)] |
| [a,b] / [c,d] | [a,b] · [1/d, 1/c] if 0 ∉ [c,d] |

**Reference**: IEEE 1788-2015 Standard for Interval Arithmetic

---

## Confidence Propagation Rules

### Default: Conservative (Minimum)

```d
result.conf = min(input1.conf, input2.conf, ...)
```

**Rationale**: We can only be as confident in the output as we are in our least trusted input.

### Alternative: Independence (Product)

```d
result.conf = input1.conf * input2.conf * ...
```

**Rationale**: If confidence represents independent probabilities of correctness, they multiply.

### Which to Use?

- Use `min` when inputs might share failure modes
- Use `*` when inputs are genuinely independent
- Document your choice in the operation

---

## Dempster-Shafer: Scoped to Propositions

Dempster-Shafer belief functions are appropriate for:
- `Knowledge<bool>` - binary propositions
- Categorical hypotheses - which of N options is true
- Evidence fusion with explicit conflict

DS is **NOT** a drop-in combiner for numeric measurements. For numeric values with DS, you must:
1. Define hypothesis sets (e.g., intervals)
2. Map measurements to mass functions
3. Apply DS combination with conflict handling
4. Map back to a numeric representation

---

## Provenance Types

```d
enum Provenance {
    // Primary sources
    Measured { sensor: string, timestamp: datetime, calibration: CalibrationRecord },
    Literature { doi: string, table: string, conditions: string },
    Input { source: string, validated: bool },

    // Derived sources
    Computed { operation: string, inputs: Vec<ProvenanceId> },
    Interpolated { method: string, points: u32 },
    Extrapolated { method: string, distance: f64 },  // distance from known data

    // Trust modifications
    Boosted { reason: string, citation: string, amount: f64 },
    Degraded { reason: string, amount: f64 },
}
```

---

## Examples

### Correct: GUM Propagation

```d
let mass = Knowledge::measured(75.0, StdDev(0.5), "scale_001")
let height = Knowledge::measured(1.75, StdDev(0.01), "stadiometer")

// BMI = mass / height²
// u_rel(BMI)² = u_rel(mass)² + 4·u_rel(height)²
let bmi = mass / (height * height)

// bmi.uncert correctly propagated via GUM
// bmi.conf = min(mass.conf, height.conf) - cannot increase
```

### Correct: Interval Enclosure

```d
let a = Knowledge::interval(1.0, 2.0)  // a ∈ [1, 2]
let b = Knowledge::interval(3.0, 4.0)  // b ∈ [3, 4]

let c = a + b  // c ∈ [4, 6] - guaranteed enclosure
let d = a * b  // d ∈ [3, 8] - guaranteed enclosure
```

### Incorrect: Confidence Increase (Rejected)

```d
let measurement = Knowledge::new(100.0, StdDev(5.0), 0.8)

// ILLEGAL - confidence cannot increase without justification
let better = measurement.with_conf(0.95)  // ERROR

// LEGAL - explicit boost with reason
let validated = measurement.boost(
    0.95,
    "cross-validated against reference standard",
    "ISO 17025 calibration"
)
```

### Incorrect: Silent Unwrap (Rejected)

```d
let k: Knowledge<f64> = get_measurement()

// ILLEGAL - implicit conversion discards metadata
let x: f64 = k  // ERROR

// LEGAL - explicit acknowledgment
let x: f64 = k.unwrap("used in non-critical calculation")
```

---

## Summary

1. **Uncertainty ≠ Confidence**: They're orthogonal. Don't conflate them.
2. **Uncertainty propagates via GUM**: Variances add in quadrature (or worse).
3. **Confidence is monotone non-increasing**: Computation creates doubt, not trust.
4. **Provenance is append-only**: Never silently drop history.
5. **No magic**: Explicit operations for anything that violates the "natural" direction.

This makes it **impossible to pretend certainty you don't have**.
