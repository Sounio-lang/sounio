---
title: Confidence Gates
description: Control flow based on confidence levels - making decisions under uncertainty
prerequisites: [docs/epistemic/knowledge-type.md, docs/epistemic/uncertainty-propagation.md]
reading_time: 12 minutes
---

# Confidence Gates

Confidence gates are control flow constructs that make decisions based on the epistemic status of values. Not all computations should proceed blindly - when confidence drops below a threshold, execution should pause, warn, or take alternative paths.

This is not defensive programming. This is **epistemic programming** - the system knows what it does not know.

## The Core Concept

In traditional programming, you might check if a value is null or within bounds. In epistemic programming, you also check if you *trust* the value enough to act on it.

```sio
fn make_decision(data: Knowledge<f64>) -> Action {
    // Check confidence before proceeding
    if data.conf().mean() < 0.90 {
        return Action::RequestMoreData
    }

    if data.conf().mean() < 0.95 {
        return Action::ProceedWithCaution(data)
    }

    Action::Proceed(data)
}
```

## Basic Confidence Checks

### Checking Mean Confidence

```sio
let measurement = Knowledge::measured(42.0, variance: 4.0, instrument: "sensor")

// Direct confidence check
if measurement.conf().mean() > 0.95 {
    // High confidence - proceed normally
    process_value(measurement)
} else {
    // Low confidence - take alternative action
    request_verification(measurement)
}
```

### Checking Probability Thresholds

For `Knowledge<f64>`, you can check the probability of the true value meeting certain criteria:

```sio
let drug_efficacy = Knowledge::measured(0.75, variance: 0.0025, instrument: "clinical_trial")

// P(efficacy > 0.70) with 95% confidence?
if drug_efficacy.prob_gt(0.70) > 0.95 {
    approve_treatment(drug_efficacy)
} else {
    request_additional_trials()
}
```

### Checking Uncertainty Needs

```sio
// Should we collect more data?
if measurement.conf().needs_exploration(0.01) {
    // High variance in confidence estimate - need more evidence
    schedule_additional_measurements()
}

// Alternative: check uncertainty score directly
if measurement.conf().uncertainty() > 0.05 {
    // Effective sample size too low
    collect_more_data()
}
```

## Guard Patterns

### Early Return on Low Confidence

```sio
fn process_critical_data(data: Knowledge<f64>) -> Result<Output, EpistemicError> {
    // Guard: require minimum confidence
    if data.conf().mean() < 0.90 {
        return Err(EpistemicError::InsufficientConfidence {
            required: 0.90,
            actual: data.conf().mean(),
            source: data.prov().to_string()
        })
    }

    // Guard: require minimum precision
    if data.std() / data.value() > 0.10 {
        return Err(EpistemicError::InsufficientPrecision {
            relative_uncertainty: data.std() / data.value(),
            threshold: 0.10
        })
    }

    // All guards passed - proceed
    Ok(compute_result(data))
}
```

### Match on Confidence Levels

```sio
enum ConfidenceLevel {
    VeryHigh,   // > 0.99
    High,       // > 0.95
    Medium,     // > 0.90
    Low,        // > 0.80
    VeryLow,    // <= 0.80
}

fn classify_confidence(k: &Knowledge<f64>) -> ConfidenceLevel {
    let c = k.conf().mean()
    if c > 0.99 { ConfidenceLevel::VeryHigh }
    else if c > 0.95 { ConfidenceLevel::High }
    else if c > 0.90 { ConfidenceLevel::Medium }
    else if c > 0.80 { ConfidenceLevel::Low }
    else { ConfidenceLevel::VeryLow }
}

fn handle_measurement(measurement: Knowledge<f64>) -> Action {
    match classify_confidence(&measurement) {
        ConfidenceLevel::VeryHigh => Action::AutoApprove(measurement),
        ConfidenceLevel::High => Action::Approve(measurement),
        ConfidenceLevel::Medium => Action::ReviewRequired(measurement),
        ConfidenceLevel::Low => Action::AdditionalDataNeeded(measurement),
        ConfidenceLevel::VeryLow => Action::Reject(measurement),
    }
}
```

## Decision-Making Under Uncertainty

### Medical/Clinical Decisions

```sio
fn evaluate_treatment(efficacy: Knowledge<f64>, safety: Knowledge<f64>) -> Decision {
    // Both efficacy AND safety must meet thresholds
    let efficacy_ok = efficacy.prob_gt(0.20) > 0.80  // P(efficacy > 20%) with 80% confidence
    let safety_ok = safety.prob_lt(0.05) > 0.95      // P(adverse events < 5%) with 95% confidence

    if efficacy_ok && safety_ok {
        Decision::Approve
    } else if !safety_ok {
        Decision::RequireMoreSafetyData
    } else {
        Decision::RequireMoreEfficacyData
    }
}
```

### Tiered Approval System

```sio
fn regulatory_decision(
    primary_endpoint: Knowledge<f64>,
    secondary_endpoints: Vec<Knowledge<f64>>,
    safety_profile: Knowledge<f64>
) -> RegulatoryAction {
    // Primary must be highly confident
    if primary_endpoint.conf().mean() < 0.95 {
        return RegulatoryAction::InsufficientEvidence
    }

    // Safety must be very high confidence
    if safety_profile.conf().mean() < 0.99 {
        return RegulatoryAction::RequireAdditionalSafetyStudy
    }

    // Check secondary endpoints
    var secondary_pass = 0
    for endpoint in secondary_endpoints {
        if endpoint.conf().mean() > 0.90 {
            secondary_pass = secondary_pass + 1
        }
    }

    if secondary_pass >= secondary_endpoints.len() / 2 {
        RegulatoryAction::Approve
    } else {
        RegulatoryAction::ApproveWithRestrictions
    }
}
```

### Financial/Risk Decisions

```sio
fn evaluate_investment(expected_return: Knowledge<f64>, risk: Knowledge<f64>) -> Action {
    // Must be confident enough in our estimates
    let min_confidence = 0.85

    if expected_return.conf().mean() < min_confidence ||
       risk.conf().mean() < min_confidence {
        return Action::GatherMoreData
    }

    // Risk-adjusted decision
    let prob_positive = expected_return.prob_gt(0.0)
    let prob_acceptable_risk = risk.prob_lt(0.10)

    if prob_positive > 0.90 && prob_acceptable_risk > 0.80 {
        Action::Invest
    } else if prob_positive > 0.75 {
        Action::InvestConservatively
    } else {
        Action::Pass
    }
}
```

## Combining Confidence from Multiple Sources

### Minimum Confidence

When all sources must be reliable:

```sio
fn require_all_confident(sources: Vec<Knowledge<f64>>, threshold: f64) -> bool {
    for source in sources {
        if source.conf().mean() < threshold {
            return false
        }
    }
    return true
}
```

### Weighted Confidence

When sources have different reliability:

```sio
fn weighted_confidence(
    sources: Vec<Knowledge<f64>>,
    weights: Vec<f64>
) -> f64 {
    var total_weight = 0.0
    var weighted_sum = 0.0

    for i in 0..sources.len() {
        total_weight = total_weight + weights[i]
        weighted_sum = weighted_sum + sources[i].conf().mean() * weights[i]
    }

    weighted_sum / total_weight
}
```

### Evidence Fusion

When combining independent evidence:

```sio
fn fuse_evidence(sources: Vec<Knowledge<f64>>) -> Knowledge<f64> {
    if sources.len() == 0 {
        return Knowledge::constant(0.0)
    }

    // Start with first source
    var fused = sources[0]

    // Combine with remaining sources
    for i in 1..sources.len() {
        fused = fuse_measurements(fused, sources[i])
    }

    fused
}

fn fuse_measurements(a: Knowledge<f64>, b: Knowledge<f64>) -> Knowledge<f64> {
    // Inverse-variance weighting
    let var_a = a.var()
    let var_b = b.var()

    // Handle exact values
    if var_a < 1.0e-15 && var_b < 1.0e-15 {
        let value = (a.value() + b.value()) / 2.0
        let conf = a.conf().combine(&b.conf())
        return Knowledge::new(value, 0.0, conf, Source::Computed { operation: "fusion" })
    }

    let w_a = 1.0 / var_a
    let w_b = 1.0 / var_b
    let w_total = w_a + w_b

    let value = (w_a * a.value() + w_b * b.value()) / w_total
    let variance = 1.0 / w_total
    let conf = a.conf().combine(&b.conf())  // Combined evidence

    Knowledge::new(value, variance, conf, Source::Computed { operation: "fusion" })
}
```

## Minimum Confidence Requirements

### Function-Level Requirements

```sio
/// Process data only if confidence exceeds threshold
///
/// # Panics
/// Panics if confidence < 0.95
fn critical_process(data: Knowledge<f64>) -> f64 {
    if data.conf().mean() < 0.95 {
        panic("Insufficient confidence for critical process")
    }
    compute_critical_result(data)
}
```

### Type-Level Requirements (Refinement Types)

```sio
// Define a refined type that requires high confidence
type HighConfidence<T> = { k: Knowledge<T> | k.conf().mean() >= 0.95 }

fn process_verified(data: HighConfidence<f64>) -> f64 {
    // Compiler guarantees confidence >= 0.95
    data.value() * 2.0
}

// Usage
let measurement = Knowledge::measured(42.0, variance: 4.0, instrument: "sensor")

// This will fail if confidence < 0.95
let verified: HighConfidence<f64> = measurement  // Compile-time check
process_verified(verified)
```

## Error Handling When Confidence is Too Low

### Custom Error Type

```sio
pub enum EpistemicError {
    InsufficientConfidence {
        required: f64,
        actual: f64,
        source: string,
    },
    InsufficientPrecision {
        relative_uncertainty: f64,
        threshold: f64,
    },
    ProvenanceUnknown,
    ConflictingEvidence {
        source_a: string,
        source_b: string,
        discrepancy: f64,
    },
}
```

### Result-Based Handling

```sio
fn validate_for_publication(
    data: Knowledge<f64>
) -> Result<Knowledge<f64>, EpistemicError> {
    // Check confidence
    if data.conf().mean() < 0.95 {
        return Err(EpistemicError::InsufficientConfidence {
            required: 0.95,
            actual: data.conf().mean(),
            source: data.prov().to_string()
        })
    }

    // Check precision
    let rel_u = data.std() / data.value().abs()
    if rel_u > 0.05 {
        return Err(EpistemicError::InsufficientPrecision {
            relative_uncertainty: rel_u,
            threshold: 0.05
        })
    }

    // Check provenance
    match data.prov().source {
        Source::Unknown => {
            return Err(EpistemicError::ProvenanceUnknown)
        },
        _ => {}
    }

    Ok(data)
}
```

### Graceful Degradation

```sio
fn compute_with_fallback(
    primary: Knowledge<f64>,
    fallback: Knowledge<f64>
) -> Knowledge<f64> {
    // Use primary if sufficiently confident
    if primary.conf().mean() > 0.90 && primary.std() / primary.value() < 0.10 {
        return primary.with_provenance("selected_primary")
    }

    // Otherwise use fallback
    if fallback.conf().mean() > 0.80 {
        return fallback.with_provenance("selected_fallback")
    }

    // Last resort: combine both
    fuse_measurements(primary, fallback).with_provenance("fused_fallback")
}
```

## Practical Patterns

### Logging Epistemic Decisions

```sio
fn log_decision(
    decision: string,
    basis: Knowledge<f64>,
    threshold: f64,
    passed: bool
) with IO {
    println("[EPISTEMIC] Decision: " + decision)
    println("  Value: " + basis.value().to_string())
    println("  Confidence: " + basis.conf().mean().to_string())
    println("  Threshold: " + threshold.to_string())
    println("  Result: " + if passed { "PASS" } else { "FAIL" })
    println("  Provenance: " + basis.prov().to_string())
}
```

### Confidence Decay Tracking

```sio
fn trace_confidence_decay(
    initial: Knowledge<f64>,
    operations: Vec<string>
) with IO {
    var current = initial
    println("Initial confidence: " + current.conf().mean().to_string())

    for op in operations {
        current = apply_operation(current, op)
        println("After " + op + ": " + current.conf().mean().to_string())
    }

    let total_decay = initial.conf().mean() - current.conf().mean()
    println("Total decay: " + total_decay.to_string())

    if total_decay > 0.10 {
        println("WARNING: Significant confidence loss through computation chain")
    }
}
```

### Asserting Epistemic Requirements

```sio
fn epistemic_assert(
    condition: bool,
    measurement: Knowledge<f64>,
    message: string
) {
    if !condition {
        panic("Epistemic assertion failed: " + message +
              "\n  Value: " + measurement.value().to_string() +
              "\n  Uncertainty: " + measurement.std().to_string() +
              "\n  Confidence: " + measurement.conf().mean().to_string() +
              "\n  Provenance: " + measurement.prov().to_string())
    }
}

// Usage
epistemic_assert(
    data.conf().mean() > 0.95,
    data,
    "Insufficient confidence for safety-critical calculation"
)
```

## Complete Example: Drug Dosing Decision

```sio
use epistemic::{Knowledge, BetaConfidence}
use units::{mg, kg, mg_per_kg}

struct DosingDecision {
    recommended_dose: Knowledge<mg>,
    confidence_level: string,
    warnings: Vec<string>,
    approved: bool,
}

fn calculate_dose(
    patient_weight: Knowledge<kg>,
    target_concentration: Knowledge<f64>,
    pk_parameters: PKParams
) -> DosingDecision with IO {
    var warnings: Vec<string> = Vec::new()

    // Check input confidence levels
    if patient_weight.conf().mean() < 0.90 {
        warnings.push("Patient weight has low confidence - verify measurement")
    }

    if target_concentration.conf().mean() < 0.95 {
        warnings.push("Target concentration confidence below 95%")
    }

    // Calculate dose with uncertainty propagation
    let dose_per_kg = target_concentration * pk_parameters.volume / pk_parameters.bioavailability
    let total_dose = dose_per_kg * patient_weight

    // Determine confidence level and approval
    let dose_conf = total_dose.conf().mean()
    let dose_precision = total_dose.std() / total_dose.value()

    let (confidence_level, approved) = if dose_conf > 0.99 && dose_precision < 0.05 {
        ("Very High", true)
    } else if dose_conf > 0.95 && dose_precision < 0.10 {
        ("High", true)
    } else if dose_conf > 0.90 && dose_precision < 0.15 {
        warnings.push("Dose approved with caution - consider monitoring")
        ("Medium", true)
    } else if dose_conf > 0.80 {
        warnings.push("Low confidence - pharmacist review required")
        ("Low", false)
    } else {
        warnings.push("Very low confidence - do not use this dose")
        ("Very Low", false)
    }

    // Log the decision
    println("Dosing Decision:")
    println("  Calculated dose: " + total_dose.value().to_string() + " mg")
    println("  95% CI: " + total_dose.ci95().to_string())
    println("  Confidence level: " + confidence_level)
    println("  Approved: " + approved.to_string())

    for warning in &warnings {
        println("  WARNING: " + warning)
    }

    DosingDecision {
        recommended_dose: total_dose,
        confidence_level: confidence_level,
        warnings: warnings,
        approved: approved,
    }
}
```

## See Also

- [Knowledge Type](knowledge-type.md) - The core `Knowledge<T>` structure
- [Uncertainty Propagation](uncertainty-propagation.md) - How uncertainty flows through computations
- [stdlib/epistemic/policy.sio](/stdlib/epistemic/policy.sio) - Epistemic policy enforcement
- [stdlib/epistemic/active.sio](/stdlib/epistemic/active.sio) - Active inference and exploration
