<!-- docs:meta
topic_id: repo.docs.archived.getting-started-duplicates.quick-start-guide
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.getting-started-duplicates.quick-start-guide
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Quick Start Guide

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
// Uncertainty is tracked automatically
let dose = epistemic_std(500.0, 2.5, 0.95)  // 500mg ± 2.5mg, 95% confidence
let volume = epistemic_std(50.0, 0.2, 0.90)  // 50mL ± 0.2mL, 90% confidence
let concentration = add_epistemic(dose, volume)  // Error automatically calculated!
```

### Your First Sounio Program

1. **Installation**:
```bash
# Coming soon - package manager
# For now, clone and build from source
git clone https://github.com/sounio-lang/sounio
cd sounio
./build.sh
```

2. **Create `hello_uncertainty.sio`**:
```sio
fn main() -> i32 {
    // Every measurement knows its uncertainty
    let temperature = epistemic_std(25.5, 0.3, 0.95)
    let pressure = epistemic_std(101.3, 0.5, 0.90)
    
    // Calculations propagate uncertainty automatically
    let combined = mul_epistemic(temperature, pressure)
    
    println("Temperature: {} ± {}", temperature.value, temperature.uncertainty)
    println("Pressure: {} ± {}", pressure.value, pressure.uncertainty)
    println("Combined: {} ± {} ({}% confidence)", 
            combined.value, combined.uncertainty, combined.confidence * 100.0)
    
    0
}
```

3. **Run it**:
```bash
./souc run hello_uncertainty.sio
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
// Only proceed if we're confident enough
if concentration.confidence > 0.95 {
    administer_drug(concentration)
} else {
    println("Warning: Low confidence ({})", concentration.confidence)
    request_more_measurements()
}
```

### Common Patterns

#### Pharmacokinetics Example:
```sio
// Simple PK model with uncertainty
fn calculate_auc(dose: Epistemic<mg>, clearance: Epistemic<L/h>) -> Epistemic<mg*h/L> {
    // AUC = Dose / Clearance (with uncertainty propagation)
    let auc = div_epistemic(dose, clearance)
    
    // Check if result is reliable enough
    if auc.confidence < 0.80 {
        println("Warning: AUC confidence only {}%", auc.confidence * 100)
    }
    
    return auc
}
```

#### Experimental Data Analysis:
```sio
fn analyze_experiment(measurements: [Epistemic<f64>]) -> Epistemic<f64> {
    // Fuse multiple measurements (reduces uncertainty!)
    var result = measurements[0]
    for i in 1..len(measurements) {
        result = fuse_measurements(result, measurements[i])
    }
    
    return result
}
```

### Next Steps

1. **Try the examples**:
```bash
cd examples/epistemic
../souc run core_demo.sio
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
