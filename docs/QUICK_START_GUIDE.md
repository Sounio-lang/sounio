<!-- docs:meta
topic_id: repo.docs.quick-start-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.quick-start-guide
-->

# Sounio Quick Start Guide

> **Canonical epistemic API.** The checked public surface for epistemic values
> is `epistemic::knowledge` (free-fn form: `ep_measured`, `ep_val`, `ep_std`,
> `ep_add`, `ep_mul`, `ep_div`, `ep_merge`, `ep_is_credible`, `ep_*_cov` for
> covariance-aware GUM), anchored by
> `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio` and
> `tests/run-pass/ep_gum_covariance.sio`. The legacy
> `epistemic::lib` surface (`epistemic_std`, `add_epistemic`, `mul_epistemic`,
> `fuse_measurements`) is exercised by
> `tests/stdlib/epistemic/test_core_e2e.sio` (a `//@ run-pass` test collected by
> `scripts/dev/run_sio_test_suite.sh`) but is a legacy surface separate from the
> canonical `epistemic::knowledge` API. `with_confidence` operators and
> units-as-type-parameters are aspirational in source and absent from the
> checked public surface — see `docs/compiler/KNOWN_LIMITATIONS.md`.

> **Other guides**: [LLM Quick Start](guide/SOUNIO_QUICK_START.md) (for AI assistants) | [General Getting Started](guide/getting-started.md) | [Conservative contract](guide/MINIMUM_VIABLE_SOUNIO.md)

## For Scientists & Domain Experts (Non-Programmers)

### The 5-Minute Introduction

Sounio lets you write scientific code that **automatically tracks uncertainty**.

#### Before (Traditional Python):
```python
# Uncertainty gets lost!
dose = 500.0  # mg ± ??
volume = 50.0  # mL ± ??
concentration = dose / volume  # What's the error?
```

#### After (Sounio):
```sio
use epistemic::knowledge::{ep_measured, ep_div}

// Uncertainty is tracked automatically; std-dev form (ep_measured stores variance = std^2).
let dose = ep_measured(500.0, 2.5)  // 500, σ=2.5, default confidence 900/1000
let volume = ep_measured(50.0, 0.2)  // 50, σ=0.2, default confidence 900/1000
let concentration = ep_div(&dose, &volume)  // GUM δ-method: concentration = dose / volume
```

### Your First Sounio Program

1. **Installation**:
```bash
# Coming soon - package manager
# For now, clone the repo and use the checked compiler artifact
git clone https://github.com/sounio-lang/sounio
cd sounio
bin/souc info
```

2. **Create `hello_uncertainty.sio`**:
```sio
use epistemic::knowledge::{
    ep_measured, ep_val, ep_std, ep_confidence, ep_mul
}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Every measurement knows its uncertainty (canonical free-fn form).
    let temperature = ep_measured(25.5, 0.3)
    let pressure = ep_measured(101.3, 0.5)

    // Multiplication propagates variance (GUM δ-method, uncorrelated).
    let combined = ep_mul(&temperature, &pressure)

    println("Temperature: {} ± {} (conf {}/1000)",
            ep_val(&temperature), ep_std(&temperature),
            ep_confidence(&temperature))
    println("Pressure: {} ± {} (conf {}/1000)",
            ep_val(&pressure), ep_std(&pressure),
            ep_confidence(&pressure))
    println("Combined: {} ± {} (conf {}/1000)",
            ep_val(&combined), ep_std(&combined),
            ep_confidence(&combined))

    0
}
```

3. **Run it**:
```bash
bin/souc run hello_uncertainty.sio
```

### Key Concepts for Scientists

#### 1. Epistemic Values
Every measurement has three parts:
- **Value**: The best estimate (e.g., 25.5°C)
- **Uncertainty**: The error range (e.g., ±0.3°C)
- **Confidence**: How sure we are (e.g., 95%)

#### 2. Automatic Propagation
When you add/multiply/divide measurements:
- Sounio calculates the new uncertainty using GUM rules
- Confidence may decrease (more operations = less certainty)

#### 3. Confidence Gates
```sio
use epistemic::knowledge::{ep_is_credible, ep_confidence}

// Only proceed if we are confident enough (integer threshold 950 ≈ 95%).
if ep_is_credible(&concentration, 950) {
    administer_drug(concentration)
} else {
    println("Warning: Low confidence (current: {}/1000)", ep_confidence(&concentration))
    request_more_measurements()
}
```

### Common Patterns

#### Pharmacokinetics Example:
```sio
use epistemic::knowledge::{Epistemic, ep_div, ep_is_credible, ep_confidence}

// Simple PK model with uncertainty (canonical Epistemic; see Quantity for units).
fn calculate_auc(dose: Epistemic, clearance: Epistemic) -> Epistemic with IO, Div, Panic {
    // AUC = Dose / Clearance (with GUM variance propagation).
    let auc = ep_div(&dose, &clearance)

    if !ep_is_credible(&auc, 800) {
        println("Warning: AUC low confidence ({}/1000)", ep_confidence(&auc))
    }

    return auc
}
```

> **Units-as-type-parameters** (`Epistemic<mg>`, `Epistemic<L/h>`) are **not**
> in the checked surface. The dimensional story lives in `stdlib/units/lib.sio`
> via `Quantity { value, uncertainty, dim: UnitDim }` and `dim_mass()`,
> `dim_length()`, `dim_time()`. See `tests/stdlib/units/test_units_stdlib.sio`
> for the canonical shape.

#### Experimental Data Analysis:
```sio
use epistemic::knowledge::{Epistemic, ep_merge}

// Inverse-variance-weighted fusion of multiple measurements (reduces σ).
fn analyze_experiment(measurements: [Epistemic; 8]) -> Epistemic with IO, Div, Panic {
    var result = measurements[0]
    for i in 1..8 {
        result = ep_merge(&result, &measurements[i])
    }
    return result
}
```

> Note: `fuse_measurements` (legacy `stdlib/epistemic/lib.sio`) becomes
> `ep_merge` (canonical `knowledge.sio`); `Epistemic<f64>` is replaced by the
> non-generic `Epistemic` from `knowledge.sio` (struct fields `val`, `variance`,
> `confidence`). `len()` is not a free stdlib function, but arrays and vectors
> expose a checked `.len()` method; this illustrative snippet instead uses a
> fixed-length `[Epistemic; 8]` array with an explicit `1..8` loop for clarity.

### Next Steps

1. **Try the examples**:
```bash
cd examples/epistemic
../../bin/souc run core_demo.sio
# Note: core_demo.sio uses the legacy epistemic::lib surface
# (epistemic_std, add_epistemic); for canonical patterns see
# ../units/dimensional_report.sio and tests/run-pass/ep_gum_covariance.sio.
```

2. **Explore your domain**:
   - `examples/pbpk/` - Pharmacokinetics
   - `examples/fmri/` - Neuroimaging
   - `examples/science/` - General scientific computing

3. **Read the manifesto** to understand the philosophy

### Getting Help

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Community**: GitHub Discussions (coming soon)

### Remember: You're Not "Programming"
You're **specifying scientific computations** in a way that preserves uncertainty information. The computer handles the implementation details.

---
*"All measurements are uncertain. Sounio helps you compute with that uncertainty."*
```
