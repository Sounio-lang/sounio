---
title: Epistemic Computing in Sounio
description: Every value knows its uncertainty - Sounio's core philosophy of first-class uncertainty quantification
prerequisites: [docs/getting-started.md]
reading_time: 10 minutes
---

# Epistemic Computing in Sounio

**Every value knows its uncertainty. Every computation propagates variance. Every result carries its provenance.**

## The Problem: Science's Reproducibility Crisis

Between 2011 and 2021, an estimated **$28 billion** was wasted on irreproducible preclinical research in the United States alone. The causes are many, but one stands out: **loss of uncertainty information**.

When a measurement of `5.23 mg/L` passes between systems, gets stored in databases, and flows through calculations, the critical context often disappears:
- What was the measurement error?
- What instrument made this measurement?
- How confident are we in this value?
- How did uncertainty propagate through subsequent calculations?

Traditional programming languages treat numbers as perfect: `3.14159` is exactly that, no more, no less. But science does not work this way. Every measurement has error. Every model has uncertainty. Every prediction has confidence bounds.

## The Solution: Epistemic Computing

**Sounio** is built on a radical premise: **uncertainty is not a bug - it is a feature**.

Epistemic computing means that every value in your program carries not just data, but *knowledge about itself*:

```sio
// Traditional approach - pretending we know exactly
let concentration = 5.23  // mg/L... but really?

// Epistemic approach - acknowledging reality
let concentration = Knowledge::new(
    value: 5.23,
    uncertainty: 0.15,
    confidence: 0.95,
    source: "lab_analyzer_001"
)
```

The `Knowledge<T>` type is not a wrapper with metadata bolted on. It is a **fundamental epistemic primitive** that changes how computation works:

1. **Uncertainty propagates automatically** - When you compute `dose / volume`, the result carries propagated uncertainty following GUM (Guide to Uncertainty in Measurement) standards

2. **Confidence decays through transformations** - Each computation slightly reduces confidence, reflecting that derived values are inherently less certain than direct measurements

3. **Provenance accumulates** - Every operation adds to the transformation history, creating an auditable trail from raw data to final result

## The Five Principles

### 1. All Knowledge is Uncertain

In the physical world, there is no such thing as a perfect measurement. The Heisenberg uncertainty principle is not a limitation of our instruments - it is a fundamental property of reality. Even macroscopic measurements carry noise, calibration error, and finite precision.

```sio
// Every measured value has uncertainty
let mass = Knowledge::measured(75.0, variance: 0.25, instrument: "scale_A")

// Even "constants" may have uncertainty in practice
let avogadro = Knowledge::new(
    value: 6.02214076e23,
    uncertainty: 0.0,  // Defined exactly since 2019
    confidence: 1.0,
    source: "SI_definition_2019"
)
```

### 2. Provenance is Non-Negotiable

Data without origin is data without trust. When a regulatory agency asks "where did this number come from?", you should have an answer that traces back to primary sources.

```sio
let clearance = Knowledge::new(
    value: 10.5,        // L/h
    uncertainty: 1.2,   // L/h
    confidence: 0.95,
    source: "Phase_III_NCT04123456"
)

// Query the provenance chain
let trail = clearance.provenance().to_string()
// "Phase_III_NCT04123456 -> population_pk_analysis -> covariate_adjustment"
```

### 3. Uncertainty Propagates Automatically

Manual uncertainty propagation is tedious and error-prone. Sounio implements the GUM (Guide to the Expression of Uncertainty in Measurement) automatically through operator overloading.

```sio
let mass = Knowledge::measured(100.0, variance: 0.25, instrument: "balance")
let volume = Knowledge::measured(50.0, variance: 0.04, instrument: "pipette")

// Density calculation with automatic propagation
let density = mass / volume
// density.value = 2.0
// density.variance computed via GUM: (1/V)^2 * Var(m) + (m/V^2)^2 * Var(V)
```

You write the physics. The compiler handles the statistics.

### 4. Confidence Gates Execution

Not all computations should proceed blindly. When confidence drops below a threshold, execution should pause, warn, or take alternative paths.

```sio
fn critical_decision(data: Knowledge<f64>) -> Action {
    if data.confidence < 0.90 {
        return Action::RequestMoreData
    }

    if data.confidence < 0.95 {
        return Action::ProceedWithCaution(data)
    }

    Action::Proceed(data)
}
```

This is not defensive programming - it is *epistemic programming*. The system knows what it does not know.

### 5. Standards Compliance by Design

Science has standards for a reason. Sounio is built to comply with:

- **GUM** - ISO Guide to the Expression of Uncertainty in Measurement
- **ISO 17025** - Competence of testing and calibration laboratories
- **21 CFR Part 11** - Electronic records and signatures (FDA)
- **FAIR Principles** - Findable, Accessible, Interoperable, Reusable data

These are not afterthoughts - they are architectural foundations.

## Why "Epistemic"?

The term comes from *epistemology*, the branch of philosophy concerned with knowledge - its nature, scope, and limits. In Sounio:

- **Epistemic** = relating to knowledge and the degree of its validation
- **Epistemic status** = how confident we are in a claim
- **Epistemic honesty** = acknowledging what we do not know

Traditional computing is *syntactic* - it manipulates symbols. Epistemic computing is *semantic* - it reasons about what those symbols mean, including their uncertainty.

## Learning Path

This documentation covers Sounio's epistemic computing in depth:

1. **[Knowledge Type](knowledge-type.md)** - The core `Knowledge<T>` type, its structure, constructors, and methods

2. **[Uncertainty Propagation](uncertainty-propagation.md)** - GUM-compliant automatic propagation through arithmetic operations

3. **[Confidence Gates](confidence-gates.md)** - Control flow based on confidence levels and decision-making under uncertainty

## Quick Example

A complete example showing epistemic computing in action:

```sio
use epistemic::{Knowledge, BetaConfidence}
use units::{mg, mL}

fn calculate_drug_concentration() -> Knowledge<f64> with IO {
    // Measure dose with uncertainty
    let dose: Knowledge<mg> = Knowledge::measured(
        500.0,
        variance: 625.0,  // 25 mg standard deviation
        instrument: "analytical_balance"
    )

    // Measure volume with uncertainty
    let volume: Knowledge<mL> = Knowledge::measured(
        10.0,
        variance: 0.01,  // 0.1 mL standard deviation
        instrument: "volumetric_pipette"
    )

    // Calculate concentration - uncertainty auto-propagates
    let concentration = dose / volume

    // Make decisions based on confidence
    if concentration.prob_gt(45.0) > 0.95 {
        println("Therapeutic range achieved with 95% confidence")
    } else if concentration.confidence().needs_exploration(0.01) {
        println("High uncertainty - consider additional measurements")
    }

    return concentration
}
```

## Integration with Other Sounio Features

Epistemic types compose naturally with other Sounio features:

### Units of Measure

```sio
let dose: Knowledge<mg> = Knowledge::measured(500.0, variance: 25.0, instrument: "scale")
let volume: Knowledge<mL> = Knowledge::measured(10.0, variance: 0.01, instrument: "pipette")
let concentration: Knowledge<mg/mL> = dose / volume  // Units are type-checked
```

### Effects System

```sio
fn sample_posterior() -> Knowledge<f64> with Prob {
    // Probabilistic operations that return epistemic values
    let theta = sample(Beta(1.0, 1.0))
    return Knowledge::from_samples(theta, n_samples: 10000)
}
```

### Refinement Types

```sio
type Probability = { p: f64 | 0.0 <= p && p <= 1.0 }

// Combine refinement with epistemic tracking
let risk: Knowledge<Probability> = Knowledge::new(
    value: 0.15,
    uncertainty: 0.02,
    confidence: 0.90,
    source: "risk_model_v2"
)
```

## See Also

- [MANIFESTO.md](/MANIFESTO.md) - The philosophical foundations of Sounio
- [stdlib/epistemic/README.md](/stdlib/epistemic/README.md) - Standard library module overview
- [LLM_PROGRAMMING_GUIDE.md](/docs/LLM_PROGRAMMING_GUIDE.md) - Complete Sounio syntax reference
- [GUM (JCGM 100:2008)](https://www.bipm.org/en/publications/guides/gum.html) - ISO uncertainty measurement guide
