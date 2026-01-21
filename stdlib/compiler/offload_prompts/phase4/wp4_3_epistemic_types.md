# WP-4.3: Epistemic Types (Uncertainty Tracking)

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- NO type suffixes
- Array indexing requires `with Panic`

## Reference Implementation

See: `compiler/src/epistemic/knowledge.rs`
See: `compiler/src/epistemic/confidence.rs`
See: `compiler/src/epistemic/propagate.rs`

## Target Output

**File**: `stdlib/compiler/epistemic/knowledge.sio`
**Estimated LOC**: ~1,500

## Specification

Implement epistemic type system for uncertainty tracking in scientific computing.

### Knowledge Type

```sio
// Knowledge<T> represents a value with uncertainty bounds
struct Knowledge {
    type_idx: i64,           // Type of inner value (T)
    confidence: f64,         // ε (epsilon) - confidence bound
    provenance: Provenance,  // Where did this knowledge come from?
}

struct Provenance {
    kind: i32,       // Measured, Computed, Assumed, Inferred
    source: i64,     // Reference to source value/function
    derivation_depth: i64,  // How many steps from original source
}

// Provenance kinds
fn PROVENANCE_MEASURED() -> i32 { 0 }   // Empirical measurement
fn PROVENANCE_COMPUTED() -> i32 { 1 }   // Derived computation
fn PROVENANCE_ASSUMED() -> i32 { 2 }    // Assumption/approximation
fn PROVENANCE_INFERRED() -> i32 { 3 }   // Inferred from other values
```

### Confidence Bounds

```sio
// Construct Knowledge value
fn knowledge_new(type_idx: i64, confidence: f64, provenance: Provenance) -> Knowledge {
    Knowledge {
        type_idx: type_idx,
        confidence: confidence,
        provenance: provenance,
    }
}

// Measured value: ε represents measurement error
// Example: temperature = 37.0°C ± 0.5°C
fn knowledge_measured(type_idx: i64, epsilon: f64) -> Knowledge {
    let prov = Provenance {
        kind: PROVENANCE_MEASURED(),
        source: -1,
        derivation_depth: 0,
    };
    knowledge_new(type_idx, epsilon, prov)
}

// Assumed value: ε is assumption uncertainty
// Example: assume drug half-life = 2.5 hours ± 0.3 hours
fn knowledge_assumed(type_idx: i64, epsilon: f64) -> Knowledge {
    let prov = Provenance {
        kind: PROVENANCE_ASSUMED(),
        source: -1,
        derivation_depth: 0,
    };
    knowledge_new(type_idx, epsilon, prov)
}

// Confidence (inverse of ε)
// If ε = 0.1, confidence = 90%
fn knowledge_confidence(k: Knowledge) -> f64 {
    1.0 - k.confidence
}

// Combine confidence bounds (widest envelope)
fn knowledge_union(k1: Knowledge, k2: Knowledge) -> Knowledge {
    // Union = wider bound (conservative estimate)
    var result = k1;
    if k2.confidence > k1.confidence {
        result.confidence = k2.confidence;
    }
    result
}

// Intersect confidence bounds (narrowest range)
fn knowledge_intersect(k1: Knowledge, k2: Knowledge) -> Knowledge {
    // Intersect = narrower bound (optimistic estimate)
    var result = k1;
    if k2.confidence < k1.confidence {
        result.confidence = k2.confidence;
    }
    result
}
```

### Uncertainty Propagation

```sio
// Propagate uncertainty through arithmetic operations

// Addition: ε_result = ε_a + ε_b
fn knowledge_add(k_a: Knowledge, k_b: Knowledge) -> Knowledge {
    let result_eps = k_a.confidence + k_b.confidence;
    var result = k_a;
    result.confidence = result_eps;
    result.provenance.kind = PROVENANCE_COMPUTED();
    result.provenance.derivation_depth = max(k_a.provenance.derivation_depth, k_b.provenance.derivation_depth) + 1;
    result
}

// Subtraction: same as addition
fn knowledge_sub(k_a: Knowledge, k_b: Knowledge) -> Knowledge {
    knowledge_add(k_a, k_b)  // ε propagates same way
}

// Multiplication: ε_result ≈ |a|⋅ε_b + |b|⋅ε_a
// Simplified: relative error multiplies
fn knowledge_mul(k_a: Knowledge, k_b: Knowledge, value_a: f64, value_b: f64) -> Knowledge with Panic, Div {
    // Relative errors
    var rel_eps_a: f64 = 0.0;
    if value_a != 0.0 {
        rel_eps_a = k_a.confidence / value_a;
    }

    var rel_eps_b: f64 = 0.0;
    if value_b != 0.0 {
        rel_eps_b = k_b.confidence / value_b;
    }

    // Combine relative errors
    let combined_rel_eps = rel_eps_a + rel_eps_b;
    var result = k_a;
    result.confidence = combined_rel_eps * value_a * value_b;
    result.provenance.kind = PROVENANCE_COMPUTED();
    result.provenance.derivation_depth = max(k_a.provenance.derivation_depth, k_b.provenance.derivation_depth) + 1;
    result
}

// Division: ε_result ≈ ε_a/|b| + |a|⋅ε_b/b²
fn knowledge_div(k_a: Knowledge, k_b: Knowledge, value_a: f64, value_b: f64) -> Knowledge with Panic, Div {
    // Propagate uncertainties through division
    // Simplified: relative errors add
    if value_b == 0.0 {
        // Error: division by zero
    }

    var result = k_a;
    result.confidence = k_a.confidence / value_b + value_a * k_b.confidence / (value_b * value_b);
    result.provenance.kind = PROVENANCE_COMPUTED();
    result.provenance.derivation_depth = max(k_a.provenance.derivation_depth, k_b.provenance.derivation_depth) + 1;
    result
}

fn max(a: i64, b: i64) -> i64 {
    if a > b { a } else { b }
}
```

### Unwrap with Justification

```sio
// Safe unwrap requires justification
fn knowledge_unwrap(k: Knowledge, reason: &str) -> f64 {
    // reason must be a non-empty justification
    // Examples: "measurement error is acceptable", "95% confidence is sufficient"
    // Compiler checks that reason is provided
    // Runtime could log/warn about uncertainty loss

    // For now, just extract value (compiler enforces reason)
    k.confidence  // Placeholder: would be actual value
}

// Example usage:
// let measured_dose: Knowledge = knowledge_measured(DOSE_TYPE, 0.05);  // 5% error
// let dose = knowledge_unwrap(measured_dose, "clinical standards allow 5% error");
```

### Type Checking Rules

```sio
// Knowledge<T> can be:
// 1. Propagated (combined with other Knowledge<T>)
// 2. Unwrapped (extract T, requires justification)
// 3. Refined (narrow bounds)

// Cannot directly use Knowledge<T> where T expected
// Example:
//   let x: i32 = knowledge_value;  // ERROR
//   let x: i32 = knowledge_unwrap(knowledge_value, "reason");  // OK

struct KnowledgeCheckResult {
    is_valid: bool,
    reason_required: bool,
    min_confidence: f64,  // Minimum acceptable confidence
}

fn check_knowledge_use(k: Knowledge, context: TypeContext) -> KnowledgeCheckResult {
    // Check if use of Knowledge<T> is justified
    // Require reason if confidence is low
    var result = KnowledgeCheckResult {
        is_valid: true,
        reason_required: k.confidence > 0.1,  // >10% error
        min_confidence: 0.95,  // 95% confidence
    };
    result
}
```

### PKPD Integration

For pharmacokinetics/pharmacodynamics:

```sio
// Measured concentration
fn measure_concentration(value: f64, measurement_error: f64) -> Knowledge {
    knowledge_measured(CONC_TYPE, measurement_error)
}

// Computed PK parameter (Clearance)
fn compute_clearance(dose: Knowledge, auc: Knowledge) -> Knowledge {
    // CL = Dose / AUC
    knowledge_div(dose, auc, dose.confidence, auc.confidence)
}

// Example workflow:
//   let dose: Knowledge = knowledge_measured(DOSE_TYPE, 0.02);  // 2% error
//   let auc: Knowledge = knowledge_measured(AUC_TYPE, 0.05);   // 5% error
//   let cl = compute_clearance(dose, auc);  // Propagates to ~7% error
//   let clearance_value = knowledge_unwrap(cl, "CL needed for dosing adjustment");
```

### Reflection

Query knowledge metadata:

```sio
// Get provenance info
fn provenance_string(prov: Provenance) -> &str {
    // Return human-readable provenance
    // "Measured (direct)", "Computed (5 steps)", "Assumed (with justification)"
}

// Check if measurement-derived
fn is_measured(k: Knowledge) -> bool {
    k.provenance.kind == PROVENANCE_MEASURED()
}

// Check confidence level
fn is_confident(k: Knowledge, threshold: f64) -> bool {
    // True if confidence > threshold
    knowledge_confidence(k) > threshold
}
```

### Key Insight

Epistemic types track not just values but also uncertainty bounds and provenance. This enables principled reasoning about measurement error and computational precision in scientific code, catching precision loss errors at compile time.
