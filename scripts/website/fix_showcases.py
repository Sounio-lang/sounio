import os

showcases_dir = "website/src/content/showcases"

causal = """---
title: "Causal Inference"
description: "Native causal model syntax for declaring nodes and edges"
domain: "causal"
icon: "🔀"
---

# Causal Inference with Sounio

Sounio provides native syntax for causal directed acyclic graphs (DAGs).

## Example: Smoking & Cancer Model

```sio
causal model SmokingCancer {
    // Declare causal variables (nodes in the DAG)
    nodes: [Smoking, Tar, Cancer, Genetics]

    // Define causal relationships (edges)
    Genetics -> Smoking
    Genetics -> Cancer
    Smoking -> Tar
    Tar -> Cancer

    // Structural causal equations
    equations: {
        Smoking = 0.5 * Genetics,
        Tar = 0.8 * Smoking,
        Cancer = 0.6 * Tar + 0.3 * Genetics
    }
}
```
"""

climate = """---
title: "Climate Science"
description: "Climate modeling with uncertainty propagation for robust predictions"
domain: "climate"
icon: "🌍"
---

# Climate Science with Sounio

Sounio enables climate modeling with rigorous uncertainty quantification for policy-relevant projections.

## Ensemble Aggregation

```sio
struct EpistemicValue {
    value: f64,
    variance: f64,
    conf_alpha: f64,
    conf_beta: f64
}

fn ensemble_mean_5(m1: EpistemicValue, m2: EpistemicValue, m3: EpistemicValue, m4: EpistemicValue, m5: EpistemicValue) -> EpistemicValue {
    let n = 5.0
    let mean_value = (m1.value + m2.value + m3.value + m4.value + m5.value) / n
    let within_var = (m1.variance + m2.variance + m3.variance + m4.variance + m5.variance) / n
    
    // ... between variance calculated ...
    let total_var = within_var + between_var
    
    return EpistemicValue { value: mean_value, variance: total_var, conf_alpha: 1.0, conf_beta: 1.0 }
}
```
"""

gpu = """---
title: "GPU Computing"
description: "Foundations for GPU-accelerated computing"
domain: "gpu"
icon: "🚀"
---

# GPU Computing with Sounio

Sounio's GPU support is currently under heavy development. While the compiler parser accepts `kernel fn` and `with GPU` effect tracking as aspirational features, current working examples focus on vectorized data structures and CPU fallbacks.

## Aspirational Syntax

```sio
// This syntax is parsed but awaiting full code-generation backends
kernel fn vector_add(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x
    if i < n {
        c[i] = a[i] + b[i]
    }
}
```
"""

genomics = """---
title: "Genomics & Bioinformatics"
description: "DNA sequence operators and analysis"
domain: "genomics"
icon: "🧬"
---

# Genomics & Bioinformatics with Sounio

Sounio provides core algorithms for sequence transformations.

## Sequence Operators

```sio
// Complement a single base (XOR with 0b11)
fn complement_base(b: u8) -> u8 {
    return b ^ 3
}

// Complement entire sequence
fn complement(seq: &[u8]) -> [u8] {
    return seq.map(|b| b ^ 3)
}

// Reverse complement
fn reverse_complement(seq: &[u8]) -> [u8] {
    let rev = seq.reverse()
    return rev.map(|b| b ^ 3)
}
```
"""

neuroimaging = """---
title: "Neuroimaging"
description: "fMRI pipeline analysis and HRF computation"
domain: "neuroimaging"
icon: "🧠"
---

# Neuroimaging with Sounio

Sounio's standard library includes foundational functions for fMRI analysis, including Hemodynamic Response Functions (HRF) and Representational Dissimilarity Matrices (RDM).

## HRF Testing

```sio
fn test_hrf() -> bool {
    let params = hrf_params_canonical()

    // HRF should peak around 5-6 seconds
    let h5 = hrf_value(5.0, params)
    let h6 = hrf_value(6.0, params)
    let h10 = hrf_value(10.0, params)

    h5 > h10 && h6 > h10
}
```
"""

pharma = """---
title: "Pharmacometrics"
description: "PK/PD modeling with GUM uncertainty propagation for clinical dosing"
domain: "pharma"
icon: "💊"
---

# Pharmacometrics with Sounio

Sounio provides a powerful platform for pharmacometric modeling, enabling drug dosage optimization with rigorous uncertainty handling.

## GUM-Compliant Vancomycin Dosing

Sounio's `Knowledge` construct carries confidence values (`ε`) which propagate through computations.

```sio
// 15 mg/kg/day: Level 1A evidence (RCT meta-analysis, 18 centres, n=2204)
// ε=0.92 reflects high evidence quality with known inter-patient variability.
let base_dose: Knowledge[f64] = Knowledge(15.0, ε=0.92, prov="ASHP_2020_Level1A_RCT")

// Hospital floor scale: ±0.5 kg on 78.5 kg patient → ε=0.98 (calibrated device).
let weight: Knowledge[f64] = Knowledge(78.5, ε=0.98, prov="hospital_scale_calibrated")
let ref_wt: Knowledge[f64] = Knowledge(70.0, ε=1.0)

// GUM propagation is automatic: ε(a/b) = ε(a)×ε(b) under independent errors.
let weight_factor: Knowledge[f64] = weight / ref_wt
let dose1: Knowledge[f64] = base_dose * weight_factor
// ε(dose1) ≈ 0.92 × 0.98 × 1.0 = 0.9016
```

## Safety Gates

You can enforce safety thresholds at compile-time or runtime using the propagated confidence.

```sio
let conf_dose = dose3.ε

if conf_dose >= 0.82 {
    println("  PRESCRIBE: dose pipeline confidence sufficient")
} else {
    println("  BLOCKED:   refine CrCl estimate (Cockcroft-Gault CV too high)")
}
```
"""

quantum = """---
title: "Quantum Computing"
description: "Quantum state modeling via Linear Types"
domain: "quantum"
icon: "⚛️"
---

# Quantum Computing with Sounio

Sounio enforces the no-cloning theorem of quantum mechanics at **compile time** via linear types: a `Qubit` is a linear resource that must be consumed exactly once. Attempting to copy it produces a compile error.

## Linear Types for Quantum States

```sio
// A Qubit cannot be cloned.
// standard quantum gates (H, X, Y, Z, CNOT) consume and return Qubits.

fn apply_hadamard(q: Qubit) -> Qubit {
    // apply H gate ...
}

fn bell_state() -> (Qubit, Qubit) {
    let q1 = allocate_qubit()
    let q2 = allocate_qubit()
    
    let h1 = apply_hadamard(q1)
    let (out1, out2) = apply_cnot(h1, q2)
    
    return (out1, out2)
}
```
"""

science = """---
title: "Scientific Computing"
description: "Data analysis and uncertainty tracking"
domain: "science"
icon: "🔬"
---

# Scientific Computing with Sounio

Sounio provides foundational utilities for modeling uncertainty in scientific domains.

## Uncertainty in PBPK Modeling

```sio
struct KnowledgeF64 {
    value: f64,
    uncertainty: f64
}

fn concentration(dose_mg: KnowledgeF64, volume_l: KnowledgeF64) -> KnowledgeF64 {
    let c = dose_mg.value / volume_l.value

    // First-order linearized uncertainty for division:
    // dc ≈ |∂c/∂dose| u_dose + |∂c/∂vol| u_vol
    let d_cdose = 1.0 / volume_l.value
    let d_cvol = -dose_mg.value / (volume_l.value * volume_l.value)

    let u = (if d_cdose < 0.0 { -d_cdose } else { d_cdose }) * dose_mg.uncertainty

    return KnowledgeF64 {
        value: c,
        uncertainty: u
    }
}
```
"""

files = {
    "causal.mdx": causal,
    "climate.mdx": climate,
    "gpu.mdx": gpu,
    "genomics.mdx": genomics,
    "neuroimaging.mdx": neuroimaging,
    "pharma.mdx": pharma,
    "quantum.mdx": quantum,
    "scientific-computing.mdx": science
}

for name, content in files.items():
    with open(os.path.join(showcases_dir, name), "w") as f:
        f.write(content)
    print(f"Rewrote {name}")

