# Epistemic Computing Standard Library

**Every value knows its uncertainty. Every computation propagates variance. Every result carries its provenance.**

## Overview

The `epistemic` module provides first-class uncertainty quantification for Sounio. Unlike traditional programming where numbers are treated as exact, epistemic types track:

- **Variance**: How uncertain is this value?
- **Confidence**: How reliable is our uncertainty estimate?
- **Provenance**: Where did this knowledge come from?

## Modules

### `knowledge.d` - Core Types

```sounio
use epistemic::{Knowledge, BetaConfidence}

// Create epistemic values
let dose = Knowledge::measured(500.0, 25.0, "scale_A")
let volume = Knowledge::measured(10.0, 0.01, "pipette_B")

// Arithmetic automatically propagates variance
let concentration = dose / volume
// Result: 50.0 ± 0.71 mg/mL

// Query probability statements
println("P(conc > 45) = {}", concentration.prob_gt(45.0))
println("95% CI: {:?}", concentration.ci95())
```

### `propagate.d` - Variance Propagation

Implements the delta method for variance propagation through arbitrary functions:

```sounio
use epistemic::propagate

let x = Knowledge::measured(2.0, 0.1, "sensor")

// Explicit propagation
let exp_x = propagate::exp(x)      // Var(e^X) ≈ e^(2X)·Var(X)
let log_x = propagate::ln(x)       // Var(ln X) ≈ Var(X)/X²
let sqrt_x = propagate::sqrt(x)    // Var(√X) ≈ Var(X)/(4X)

// Monte Carlo for complex functions
let complex = propagate::monte_carlo(x, |v| my_complex_fn(v), 10000)
```

**Propagation Rules:**

| Function | Variance Formula |
|----------|------------------|
| `X + Y` | `Var(X) + Var(Y)` |
| `X - Y` | `Var(X) + Var(Y)` |
| `X * Y` | `Y²Var(X) + X²Var(Y)` |
| `X / Y` | `Var(X)/Y² + X²Var(Y)/Y⁴` |
| `e^X` | `e^(2X) · Var(X)` |
| `ln(X)` | `Var(X) / X²` |
| `√X` | `Var(X) / (4X)` |
| `X²` | `4X² · Var(X)` |

### `meta.d` - Meta-Analysis

Combine results across multiple studies:

```sounio
use epistemic::meta

let trial1 = Knowledge::measured(0.35, 0.04, "RCT_2021")
let trial2 = Knowledge::measured(0.42, 0.06, "RCT_2022")
let trial3 = Knowledge::measured(0.38, 0.03, "RCT_2023")

// Fixed-effects pooling (assumes homogeneity)
let fe = meta::fixed_effects([trial1, trial2, trial3])

// Random-effects pooling (accounts for heterogeneity)
let re = meta::random_effects([trial1, trial2, trial3])

// Check heterogeneity
println("I² = {}%", re.heterogeneity.i_squared * 100)
println("Interpretation: {}", re.heterogeneity.interpretation())

// Bayesian hierarchical pooling with prior
let bayes = meta::bayesian_pool([trial1, trial2, trial3], 0.40, 0.1)
```

### `active.d` - Active Inference

Exploration/exploitation based on uncertainty:

```sounio
use epistemic::active

// Which variable needs more data?
let priority = active::exploration_priority([drug_a, drug_b, drug_c])

// UCB (Upper Confidence Bound) selection
let chosen = active::ucb_select([opt_a, opt_b], 1.0)

// Belief updating with new observation
let posterior = active::update_belief(&prior, observation, obs_variance)

// Expected Free Energy for action selection
let efe = active::expected_free_energy(&current, 0.5, 1.0, 1.0)
```

### `merkle.d` - Cryptographic Provenance

Tamper-evident audit trails:

```sounio
use epistemic::merkle::{MerkleDAG, Hash256}

var dag = MerkleDAG::new()

// Track data lineage
let raw = dag.add_leaf(measurement, "raw CT scan data")
let processed = dag.add_transform("normalize", [raw], normalized, "normalized")
let result = dag.add_transform("analyze", [processed], analysis, "KEC analysis")

// Verify nothing was tampered
assert(dag.verify_chain(&result))

// Export audit trail
let trail = dag.audit_trail(&result)
for entry in trail {
    println(entry.to_string())
}
```

## Type Hierarchy

```
Knowledge<T>
├── value: T              -- Point estimate
├── variance: f64         -- Uncertainty (σ²)
├── confidence: BetaConfidence
│   ├── alpha: f64        -- Beta posterior parameter
│   └── beta: f64         -- Beta posterior parameter
└── provenance: Provenance
    ├── source: Source
    └── steps: Vec<ProvenanceStep>

BetaConfidence
├── mean() -> f64         -- E[confidence]
├── variance() -> f64     -- Var[confidence] ("uncertainty about uncertainty")
├── concentration() -> f64 -- α + β (evidence amount)
└── needs_exploration(threshold) -> bool
```

## Integration with Units

Epistemic types compose with Sounio units of measure:

```sounio
use epistemic::Knowledge
use units::{mg, mL}

let dose: Knowledge<mg> = Knowledge::measured(500.0_mg, 25.0, "scale")
let volume: Knowledge<mL> = Knowledge::measured(10.0_mL, 0.01, "pipette")
let concentration: Knowledge<mg/mL> = dose / volume
```

## Design Principles

1. **Variance over Error Bars**: We track σ² not ± because variance is additive
2. **Confidence is a Distribution**: BetaConfidence captures "how sure are we about being sure?"
3. **Provenance is First-Class**: Every Knowledge knows its computational history
4. **Decay is Explicit**: Transformations decay confidence at known rates

## References

- Taylor, J.R. "Introduction to Error Analysis"
- Gelman, A. et al. "Bayesian Data Analysis"
- Friston, K. "Active Inference and Free Energy"
- Pearl, J. "Causality: Models, Reasoning, and Inference"

## License

MIT / Apache-2.0 (same as Sounio)
