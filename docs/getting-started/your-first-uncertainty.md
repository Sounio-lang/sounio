---
title: Your First Uncertainty
description: Learn how Sounio tracks uncertainty through computations automatically
prerequisites: hello-world.md
reading_time: 15 minutes
---

# Your First Uncertainty

This tutorial introduces Sounio's core innovation: **epistemic computing**. You will learn how to create values that carry their uncertainty and confidence, how uncertainty propagates automatically through calculations, and how to make decisions based on data quality.

## Why Uncertainty Matters

### The Mars Climate Orbiter Disaster

In 1999, NASA lost the $125 million Mars Climate Orbiter. The cause? A unit conversion error: one team used metric units, another used imperial. The software did not track that a value of "4.45" was in Newton-seconds rather than pound-force-seconds.

If the type system had tracked units and uncertainty, the discrepancy would have been caught at compile time.

### The Reproducibility Crisis

Between 2011 and 2021, an estimated **$28 billion** was wasted on irreproducible preclinical research in the United States alone. One major cause: loss of uncertainty information.

When a measurement of `5.23 mg/L` is passed between systems, stored in databases, and used in calculations, the `+/- 0.15` often disappears. Downstream analyses treat it as exact. Conclusions are drawn that the original uncertainty would have precluded.

### The Sounio Solution

Sounio makes uncertainty **infectious**. You cannot accidentally drop it. Every value knows:

1. **Its uncertainty** - How precisely do we know the value?
2. **Its confidence** - How much do we trust this information?
3. **Its provenance** - Where did this value come from?

## Creating Values with Uncertainty

### The EpistemicValue Type

In Sounio, values with uncertainty are represented using `EpistemicValue`:

```sio
import stdlib.epistemic.core::*

fn main() -> i32 {
    // A measurement with standard uncertainty
    let mass = epistemic_std(
        75.0,    // value: 75.0 kg
        0.5,     // uncertainty: +/- 0.5 kg
        0.95     // confidence: 95%
    )

    print("Mass: ")
    print(get_value(mass))
    print(" +/- ")
    print(get_std_uncertainty(mass))
    println()

    0
}
```

Output:

```
Mass: 75.0 +/- 0.5
```

### Three Types of Uncertainty

Sounio supports three ways to express uncertainty:

#### 1. Standard Uncertainty (Most Common)

The value plus/minus a standard deviation:

```sio
// 100.0 +/- 2.5, with 95% confidence
let measurement = epistemic_std(100.0, 2.5, 0.95)
```

#### 2. Interval Bounds

When you know the possible range but not the distribution:

```sio
// Value is between 98.0 and 102.0, with 99% confidence
let bounded = epistemic_interval(98.0, 102.0, 0.99)

// The "value" is the midpoint: 100.0
// The interval is [98.0, 102.0]
```

#### 3. Exact Values

When a value has no measurement uncertainty (e.g., physical constants):

```sio
// Speed of light: exact value, 100% confidence
let c = epistemic_exact(299792458.0, 1.0)
```

## Automatic Uncertainty Propagation

The true power of Sounio emerges when you perform calculations. Uncertainty propagates automatically using the **GUM** (Guide to the Expression of Uncertainty in Measurement) standard.

### Addition and Subtraction

When adding or subtracting, uncertainties combine in quadrature:

```sio
let a = epistemic_std(100.0, 2.0, 0.95)  // 100 +/- 2
let b = epistemic_std(50.0, 1.5, 0.95)   // 50 +/- 1.5

let sum = add_epistemic(a, b)

print("Sum: ")
print(get_value(sum))          // 150.0
print(" +/- ")
print(get_std_uncertainty(sum)) // 2.5 (sqrt(2^2 + 1.5^2))
println()
```

The combined uncertainty is `sqrt(2^2 + 1.5^2) = 2.5`, not `2 + 1.5 = 3.5`. This is because independent errors partially cancel.

### Multiplication and Division

For multiplication and division, relative uncertainties combine:

```sio
let mass = epistemic_std(10.0, 0.5, 0.95)    // 10.0 +/- 0.5 (5% relative)
let volume = epistemic_std(2.0, 0.1, 0.95)   // 2.0 +/- 0.1 (5% relative)

let density = div_epistemic(mass, volume)

print("Density: ")
print(get_value(density))           // 5.0
print(" +/- ")
print(get_std_uncertainty(density)) // ~0.35 (7.07% relative)
println()
```

The relative uncertainty is `sqrt(5%^2 + 5%^2) = 7.07%`.

### Confidence Monotonicity

A critical invariant in Sounio: **confidence can only decrease through pure transformations**.

```sio
let a = epistemic_std(100.0, 2.0, 0.95)  // 95% confidence
let b = epistemic_std(50.0, 1.5, 0.90)   // 90% confidence

let result = add_epistemic(a, b)

print("Result confidence: ")
print(get_confidence(result))  // 0.90 (minimum of inputs)
println()
```

The result's confidence is capped at the minimum of the inputs. You cannot gain confidence through computation.

## Confidence Gates

Sounio enables **confidence-based control flow**. This is a paradigm shift: instead of just checking values, you check data quality.

### Basic Confidence Checks

```sio
fn process_measurement(m: EpistemicValue) {
    let conf = get_confidence(m)

    if conf >= 0.95 {
        print("High confidence - proceeding automatically")
        proceed(m)
    } else if conf >= 0.80 {
        print("Moderate confidence - flagging for review")
        flag_for_review(m)
    } else {
        print("Low confidence - requesting new measurement")
        request_remeasurement()
    }
    println()
}
```

### Threshold-Based Decisions

```sio
fn clinical_decision(concentration: EpistemicValue, threshold: f64) {
    let value = get_value(concentration)
    let uncertainty = get_std_uncertainty(concentration)
    let conf = get_confidence(concentration)

    // Check if the value is definitively above threshold
    // considering uncertainty
    let lower_bound = value - 2.0 * uncertainty

    if lower_bound > threshold {
        print("Definitely above threshold")
    } else if value > threshold {
        print("Probably above threshold (check uncertainty)")
    } else {
        print("Below or uncertain relative to threshold")
    }

    if conf < 0.90 {
        print(" [LOW CONFIDENCE - VERIFY]")
    }
    println()
}
```

## Evidence Fusion

When you have multiple measurements of the same quantity, Sounio can combine them intelligently:

```sio
// Two independent measurements of the same quantity
let m1 = epistemic_std(100.0, 5.0, 0.90)  // 100 +/- 5, 90% confidence
let m2 = epistemic_std(102.0, 5.0, 0.85)  // 102 +/- 5, 85% confidence

// Fuse the measurements using inverse-variance weighting
let combined = fuse_measurements(m1, m2)

print("Fused value: ")
print(get_value(combined))           // ~101.0 (weighted average)
print(" +/- ")
print(get_std_uncertainty(combined)) // ~3.54 (less than either input!)
print(" (conf: ")
print(get_confidence(combined))      // 0.90 (max of inputs)
print(")")
println()
```

Key insight: **fusion reduces uncertainty**. Two independent measurements of the same thing give you more information than either alone. The combined uncertainty is less than either input.

## A Complete Example: Laboratory Analysis

Here is a realistic example computing drug concentration with uncertainty:

```sio
import stdlib.epistemic.core::*

// Calculate drug concentration from mass and volume
fn calculate_concentration(
    mass: EpistemicValue,
    volume: EpistemicValue
) -> EpistemicValue {
    return div_epistemic(mass, volume)
}

// Determine if concentration is within therapeutic range
fn check_therapeutic_range(
    concentration: EpistemicValue,
    min_therapeutic: f64,
    max_therapeutic: f64
) {
    let value = get_value(concentration)
    let u = get_std_uncertainty(concentration)
    let conf = get_confidence(concentration)

    // 95% confidence interval
    let lower = value - 2.0 * u
    let upper = value + 2.0 * u

    print("Concentration: ")
    print(value)
    print(" +/- ")
    print(u)
    print(" (95% CI: [")
    print(lower)
    print(", ")
    print(upper)
    print("])")
    println()

    // Check if definitely in range
    if lower >= min_therapeutic && upper <= max_therapeutic {
        print("RESULT: Definitely within therapeutic range")
    } else if upper < min_therapeutic {
        print("RESULT: Below therapeutic range")
    } else if lower > max_therapeutic {
        print("RESULT: Above therapeutic range - potential toxicity")
    } else {
        print("RESULT: Uncertain - straddles therapeutic boundary")
    }
    println()

    // Confidence warning
    if conf < 0.90 {
        print("WARNING: Measurement confidence is only ")
        print(conf * 100.0)
        print("%. Consider re-measurement.")
        println()
    }
}

fn main() -> i32 {
    // Measured mass: 500 mg +/- 2.5 mg, 95% confidence
    let mass = epistemic_std(500.0, 2.5, 0.95)

    // Measured volume: 10.0 mL +/- 0.2 mL, 95% confidence
    let volume = epistemic_std(10.0, 0.2, 0.95)

    // Calculate concentration (mg/mL)
    let concentration = calculate_concentration(mass, volume)

    // Therapeutic range: 45-55 mg/mL
    check_therapeutic_range(concentration, 45.0, 55.0)

    0
}
```

Output:

```
Concentration: 50.0 +/- 1.12 (95% CI: [47.76, 52.24])
RESULT: Definitely within therapeutic range
```

## Key Concepts Summary

### Uncertainty vs. Confidence

These are **orthogonal** concepts:

| Concept | Question | Example |
|---------|----------|---------|
| **Uncertainty** | How precisely do we know the value? | +/- 0.5 mg |
| **Confidence** | How much do we trust this measurement? | 95% |

A measurement can have:
- Low uncertainty, high confidence (precise lab measurement)
- High uncertainty, high confidence (known rough estimate)
- Low uncertainty, low confidence (suspicious precise claim)
- High uncertainty, low confidence (unreliable rough estimate)

### The GUM Standard

Sounio implements the [Guide to the Expression of Uncertainty in Measurement](https://www.bipm.org/en/committees/jc/jcgm/publications), the international standard for uncertainty propagation:

- Addition/subtraction: variances add
- Multiplication/division: relative variances add
- Correlations are handled (when specified)

### Invariants

Sounio enforces these invariants:

1. **Confidence is bounded**: `0.0 <= confidence <= 1.0`
2. **Confidence is monotonic**: Pure transformations never increase confidence
3. **Uncertainty is non-negative**: `uncertainty >= 0.0`
4. **Intervals are ordered**: `lower_bound <= upper_bound`

These are checked at compile time where possible, and at runtime otherwise.

## Next Steps

You now understand Sounio's core innovation. Continue with:

- [Project Structure](./project-structure.md) - Organize larger Sounio projects
- [Editor Setup](./editor-setup.md) - Configure your development environment

## See Also

- [Language Reference](../LLM_PROGRAMMING_GUIDE.md) - Complete syntax including epistemic types
- [Manifesto](../../MANIFESTO.md) - The philosophy behind epistemic computing
- [stdlib/epistemic/](../../stdlib/epistemic/) - Full epistemic library source
