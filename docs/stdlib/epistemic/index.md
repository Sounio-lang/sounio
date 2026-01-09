# Epistemic Module

**Every value knows its uncertainty. Every computation propagates variance. Every result carries its provenance.**

The `epistemic` module is the flagship feature of Sounio, implementing first-class uncertainty quantification that makes epistemic computing a reality. Unlike traditional programming where numbers are treated as exact, epistemic types track variance, confidence, and provenance through all computations.

## Philosophy

Traditional programming:
```sio
let mass = 3.14
```

Epistemic programming:
```sio
let mass: Knowledge<f64> = Knowledge::measured(3.14, 0.01, "scale_A")
```

The difference is profound:
- **Variance** tells us how precisely we know the value
- **Confidence** tells us how much we trust our uncertainty estimate
- **Provenance** tells us where the knowledge came from

## Module Structure

The epistemic module contains 42 files organized by functionality:

### Core Types

| File | Description |
|------|-------------|
| `knowledge.sio` | `Knowledge<T>` - the fundamental epistemic type |
| `core.sio` | `EpistemicValue`, `Uncertainty`, basic operations |
| `invariants.sio` | Semantic invariants for epistemic types |

### Uncertainty Propagation

| File | Description |
|------|-------------|
| `propagate.sio` | Delta method variance propagation |
| `gum.sio` | GUM (Guide to Uncertainty in Measurement) implementation |
| `montecarlo.sio` | Monte Carlo propagation for nonlinear functions |
| `interval_ieee.sio` | IEEE 1788 interval arithmetic |

### Statistical Inference

| File | Description |
|------|-------------|
| `mcmc.sio` | NUTS and Metropolis-Hastings MCMC samplers |
| `meta.sio` | Fixed/random effects meta-analysis |
| `stats.sio` | Statistical functions with uncertainty |
| `multivariate.sio` | Multivariate uncertainty |
| `correlation.sio` | Correlation handling in propagation |

### Active Learning

| File | Description |
|------|-------------|
| `active.sio` | Active inference and exploration/exploitation |
| `discovery.sio` | Uncertainty-guided discovery |
| `optimization.sio` | Bayesian optimization with uncertainty |

### Provenance & Audit

| File | Description |
|------|-------------|
| `prov.sio` | Provenance tracking |
| `merkle.sio` | Cryptographic provenance chains |
| `ledger.sio` | Audit ledger for knowledge |
| `traceability.sio` | End-to-end traceability |
| `slsa.sio` | SLSA compliance |

### Specialized Domains

| File | Description |
|------|-------------|
| `causal.sio` | Causal inference with uncertainty |
| `timeseries.sio` | Time series with uncertainty |
| `ode.sio` | ODEs with uncertain parameters |
| `pde.sio` | PDEs with uncertain parameters |
| `linalg.sio` | Linear algebra with uncertainty |

### Analysis & Policy

| File | Description |
|------|-------------|
| `sobol.sio` | Sobol sensitivity analysis |
| `coverage.sio` | Coverage probability verification |
| `policy.sio` | Epistemic policy rules |
| `budget.sio` | Uncertainty budget management |
| `roi.sio` | Return on information investment |

## Key Concepts

### 1. Uncertainty vs Confidence (Orthogonal Concepts)

```sio
//! KEY INSIGHT: Uncertainty and Confidence are ORTHOGONAL concepts:
//! - Uncertainty: How precisely do we know the VALUE? (metrology)
//! - Confidence: How much do we TRUST the claim? (epistemology)
```

A measurement can have:
- **Low uncertainty, high confidence**: Precise calibrated instrument
- **High uncertainty, low confidence**: Wide error bars from untested source
- **Low uncertainty, low confidence**: Precise but suspicious data

### 2. Variance Propagation

Arithmetic operations automatically propagate uncertainty using the delta method:

| Operation | Variance Formula |
|-----------|-----------------|
| `X + Y` | `Var(X) + Var(Y)` |
| `X - Y` | `Var(X) + Var(Y)` |
| `X * Y` | `Y^2 Var(X) + X^2 Var(Y)` |
| `X / Y` | `Var(X)/Y^2 + X^2 Var(Y)/Y^4` |

### 3. Confidence Decay

Confidence never increases through pure computation. Each transformation applies a decay factor:

```sio
let result = a + b  // conf(result) <= min(conf(a), conf(b))
```

### 4. Provenance Tracking

Every `Knowledge<T>` value carries its complete computational history:

```sio
let x = Knowledge::measured(10.0, 0.5, "sensor_A")
let y = x.sqrt()  // Provenance: sensor_A -> sqrt
```

## Quick Start

```sio
use epistemic::{Knowledge, BetaConfidence}

// Create epistemic values
let dose = Knowledge::measured(500.0, 25.0, "scale_A")
let volume = Knowledge::measured(10.0, 0.01, "pipette_B")

// Arithmetic automatically propagates variance
let concentration = dose / volume
// Result: 50.0 with variance automatically computed

// Query probability statements
let prob_therapeutic = concentration.prob_gt(45.0)
println("P(conc > 45) = {}", prob_therapeutic)

// Get confidence intervals
let (lower, upper) = concentration.ci95()
println("95% CI: [{}, {}]", lower, upper)
```

## Type Hierarchy

```
Knowledge<T>
+-- value: T              -- Point estimate
+-- variance: f64         -- Uncertainty (standard deviation squared)
+-- confidence: BetaConfidence
|   +-- alpha: f64        -- Beta posterior parameter
|   +-- beta: f64         -- Beta posterior parameter
+-- provenance: Provenance
    +-- source: Source
    +-- steps: Vec<ProvenanceStep>

BetaConfidence
+-- mean() -> f64         -- E[confidence]
+-- variance() -> f64     -- Var[confidence] ("uncertainty about uncertainty")
+-- concentration() -> f64 -- alpha + beta (evidence amount)
+-- needs_exploration(threshold) -> bool
```

## Integration with Units

Epistemic types compose with Sounio's units of measure:

```sio
use epistemic::Knowledge
use units::{mg, mL}

let dose: Knowledge<mg> = Knowledge::measured(500.0_mg, 25.0, "scale")
let volume: Knowledge<mL> = Knowledge::measured(10.0_mL, 0.01, "pipette")
let concentration: Knowledge<mg/mL> = dose / volume
```

## Design Principles

1. **Variance over Error Bars**: We track variance (sigma squared) not plus/minus because variance is additive for independent variables

2. **Confidence is a Distribution**: `BetaConfidence` captures "how sure are we about being sure?" using Beta posteriors

3. **Provenance is First-Class**: Every `Knowledge` value knows its complete computational history

4. **Decay is Explicit**: Transformations decay confidence at known rates, never increase it

5. **GUM Compliant**: Implements JCGM 100:2008 (Guide to the Expression of Uncertainty in Measurement)

## References

- JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement (GUM)
- JCGM 101:2008 - Supplement 1 to the GUM (Monte Carlo methods)
- Taylor, J.R. "Introduction to Error Analysis"
- Gelman, A. et al. "Bayesian Data Analysis"
- Friston, K. "Active Inference and Free Energy"
- Pearl, J. "Causality: Models, Reasoning, and Inference"

## See Also

- [Knowledge<T> API Reference](knowledge.md) - Complete type documentation
- [Variance Propagation](propagate.md) - Delta method and Monte Carlo
- [MCMC Sampling](mcmc.md) - Bayesian posterior inference
- [Meta-Analysis](meta.md) - Combining evidence across studies
- [Sequential Monte Carlo](smc.md) - Particle filtering
