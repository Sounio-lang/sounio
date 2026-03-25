# Sounio Ecosystem

Welcome to the **Sounio Ecosystem** — a unified platform for epistemic computing, scientific programming, and drug discovery research.

## What is Sounio?

**Sounio** is a systems language for epistemic computing — the art of reasoning about measurement uncertainty and knowledge. Unlike traditional programming languages that ignore uncertainty, Sounio treats it as a first-class value:

```python
import sounio

# Measured dose with GUM (Guide to the Expression of Uncertainty in Measurement)
dose = sounio.Knowledge(500.0, epsilon=2.5, provenance="calibrated syringe")

# Uncertainty propagates through calculations
adjusted = dose * 1.1  # Scales uncertainty by same factor
print(adjusted)  # Knowledge(550.000 ± 2.750, prov='(calibrated syringe)*(1.1)')
```

Sounio integrates with Python, Jupyter, and specialized scientific pipelines to bring epistemic reasoning to:

- **Pharmacokinetics/Pharmacodynamics (PK/PD)** modeling
- **Virtual drug screening** (Lipinski's Rule of Five)
- **Clinical trial simulation** with uncertainty quantification
- **Real-time measurement systems** where accuracy matters

## The Triple Ecosystem

### 🐍 **sounio-py**: Python Bindings

Direct Python access to Sounio's epistemic computing via the `Knowledge` class and native extensions.

**Key features:**
- GUM-compliant uncertainty propagation (addition, multiplication, division)
- Integration with NumPy (UncertainArray) and pandas (EpistemicDataFrame)
- Measure any quantity and propagate uncertainty automatically
- Call Sounio code from Python for high-performance computing

**Use when:** You're a Python researcher/engineer who wants uncertainty-aware computation without learning a new language.

[Get started →](sounio-py/quickstart.md)

### 📓 **sounio-jupyter**: Interactive Notebooks

A full Jupyter kernel that brings Sounio into your notebook workflow, with magics, completion, and rich epistemic visualization.

**Key features:**
- Write Sounio directly in notebook cells (auto-wrapped)
- Tab completion for keywords, functions, and variables
- Magic commands: `%check`, `%time`, `%types`, `%ast`
- Rich HTML rendering of Knowledge values with uncertainty bands
- Integrate with Python cells seamlessly

**Use when:** You want interactive Sounio development in a notebook environment.

[Get started →](sounio-jupyter/overview.md)

### 💊 **drug-discovery**: Pure Sounio Pipeline

A 3-stage epistemic pipeline for computational drug discovery, written entirely in Sounio.

**Stages:**
1. **Virtual Screening** — Lipinski filtering + SMILES parsing
2. **PK/PD Modeling** — Absorption, metabolism, elimination with uncertainty
3. **Clinical Simulation** — Trial outcome prediction with epistemic reasoning

**Use when:** You're building scientific applications that need integrated uncertainty reasoning from screening through simulation.

[Get started →](drug-discovery/tutorial.md)

---

## Quick Start

### Installation

=== "Python (sounio-py)"
    ```bash
    pip install sounio
    ```

    ```python
    import sounio
    x = sounio.Knowledge(36.5, epsilon=0.1, provenance="thermometer")
    y = sounio.Knowledge(1.5, epsilon=0.05, provenance="barometer")
    z = x + y
    print(z)  # Knowledge(38.000 ± 0.112, prov='...')
    ```

=== "Jupyter Kernel"
    ```bash
    pip install sounio-jupyter
    jupyter kernelspec list  # Verify "sounio" appears
    jupyter notebook
    ```

    Create a new notebook with **Sounio** kernel:

    ```sounio
    let measurement: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
    print(measurement)
    ```

=== "Sounio Compiler"
    Download the pre-built binary:

    ```bash
    export SOUC=./bin/souc
    $SOUC run examples/hello.sio
    ```

    Or build from source — see [Sounio main repository](https://github.com/sounio-org/sounio).

---

## Core Concepts

### Knowledge<T>: Epistemic Values

Every measurement comes with uncertainty. Sounio's `Knowledge<T>` type captures both:

- **value** — central estimate (e.g., dose in mg)
- **epsilon** — standard uncertainty under GUM (1σ)
- **provenance** — source/method label for traceability

```python
import sounio

# Measurement from a calibrated instrument
clearance = sounio.Knowledge(
    value=120.0,
    epsilon=5.0,
    provenance="LC-MS assay (CV=4.2%)"
)

# Uncertainty propagates through operations
dose_scaled = clearance * 1.5  # epsilon also scales
print(dose_scaled)  # Knowledge(180.000 ± 7.500, prov='...')
```

### GUM Arithmetic

Sounio follows the **Guide to the Expression of Uncertainty in Measurement** (ISO/IEC GUIDE 98-3:2008):

| Operation | Rule | Example |
|-----------|------|---------|
| **Addition** | εz = √(εa² + εb²) | 10 ± 0.5 + 20 ± 0.3 = 30 ± 0.583 |
| **Multiplication** | εz/z = √((εa/a)² + (εb/b)²) | (10 ± 1) × (20 ± 2) = 200 ± 28.3 |
| **Division** | εz/z = √((εa/a)² + (εb/b)²) | (100 ± 10) / (5 ± 0.5) = 20 ± 4.47 |
| **Scalar** | εz = \|c\| · εa | (10 ± 1) × 2.5 = 25 ± 2.5 |

These rules automatically propagate measurement uncertainty through calculations, giving you **confidence intervals for free**.

### Effects System

Sounio tracks computational effects (I/O, mutation, division-by-zero risk) in the type system:

```sounio
// This function might perform I/O or mutate state
fn compute_pk(dose: f64) -> f64 with IO, Mut, Div {
    // ...
}

// Type-safe: can only call from a context that allows these effects
fn main() with IO, Mut, Div {
    let result = compute_pk(500.0)
    print_f64(result)
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         User Application (Python/Jupyter)       │
├─────────────────────────────────────────────────┤
│  sounio-py          │  sounio-jupyter          │
│  Knowledge class    │  Kernel + magics         │
│  UncertainArray     │  Completion              │
│  EpistemicDataFrame │  Rich display            │
├─────────────────────────────────────────────────┤
│              Sounio Type Checker                │
│         (Bidirectional inference)               │
├─────────────────────────────────────────────────┤
│           IR Lowering + E-Graph Opt             │
│              (Epistemic saturation)             │
├─────────────────────────────────────────────────┤
│              Codegen → x86-64 ELF               │
│          (JIT for rapid iteration)              │
├─────────────────────────────────────────────────┤
│   Sounio Standard Library (epistemic, math)     │
│  sqrt, exp, log, Knowledge<T>, measure(), etc.  │
└─────────────────────────────────────────────────┘
```

---

## Common Tasks

### I want to...

- **Compute with uncertainty in Python** → [sounio-py Quickstart](sounio-py/quickstart.md)
- **Use Knowledge in pandas DataFrames** → [sounio-py API: EpistemicDataFrame](sounio-py/api.md#epistemichidden)
- **Write interactive Sounio notebooks** → [sounio-jupyter Usage](sounio-jupyter/usage.md)
- **Build a drug discovery pipeline** → [Drug Discovery Tutorial](drug-discovery/tutorial.md)
- **Learn the Sounio language** → [Sounio Programming Guide](https://github.com/sounio-org/sounio/docs/LLM_PROGRAMMING_GUIDE.md)
- **Contribute to the ecosystem** → [GitHub Issues](https://github.com/sounio-org/sounio/issues)

---

## Why Epistemic Computing?

Traditional software treats numbers as exact. In science, every measurement has uncertainty:

- **Thermometer ±0.5°C**
- **LC-MS assay CV=4.2%**
- **Population PK model ±15% clearance**

Sounio makes uncertainty **automatic and correct**. When you compute with `Knowledge` values, uncertainty propagates through your code via proven mathematical rules. No more manual error propagation — no more forgotten uncertainty bars on plots.

This is especially critical in **drug discovery**, where dosing errors or PK model uncertainty can affect patient safety.

---

## Roadmap

- [ ] **Q2 2026**: Full NumPy integration with WMMA GPU acceleration
- [ ] **Q3 2026**: Clinical trial simulator with dosing optimization
- [ ] **Q4 2026**: Multi-objective epistemic inference (Pareto frontiers)
- [ ] **2027**: Bayesian network epistemic reasoning

---

## Support

- **Documentation**: Read the guides for each project
- **Issues**: [GitHub Issues](https://github.com/sounio-org/sounio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sounio-org/sounio/discussions)
- **License**: Apache 2.0

---

**Ready to start?** Pick your entry point:

- 🐍 [Python developers: Start with sounio-py](sounio-py/quickstart.md)
- 📓 [Jupyter users: Start with sounio-jupyter](sounio-jupyter/overview.md)
- 💊 [Drug discovery: Start with the tutorial](drug-discovery/tutorial.md)
