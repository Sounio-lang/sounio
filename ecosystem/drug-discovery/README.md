# Epistemic Drug Discovery Pipeline

Pure Sounio implementation of a three-stage drug discovery pipeline with uncertainty quantification at every step. Demonstrates epistemic computing for pharmaceutical research: molecular screening, pharmacokinetic/pharmacodynamic (PK/PD) modeling, and Monte Carlo clinical trial simulation.

## Overview

This is a **showcase application** of Sounio's epistemic type system. Rather than classical point estimates, every computed value carries quantified uncertainty via the `Knowledge<T>` type. The pipeline follows GUM (Guide to the Expression of Uncertainty in Measurement) principles for proper uncertainty propagation.

```
Candidate Molecule
  ↓ (STAGE 1: Lipinski filter)
Virtual Screening → Knowledge { value: pass/fail, epsilon: confidence }
  ↓ (STAGE 2: one-compartment oral PK)
PK/PD Modeling → { half_life, Tmax, Cmax, AUC } with propagated uncertainty
  ↓ (STAGE 3: Monte Carlo simulation)
Clinical Trial → { efficacy_rate, adverse_rate, therapeutic_index }
  ↓
Decision: PROCEED / HALT (with epistemic confidence)
```

## Architecture

**Three Pure-Sounio Stages**

1. **Virtual Screening (Lipinski's Rule of 5)**
   - Molecular weight < 500 Da
   - LogP < 5
   - H-bond donors ≤ 5
   - H-bond acceptors ≤ 10
   - Confidence = product of measurement uncertainties

2. **PK/PD Modeling (One-Compartment Oral)**
   - Input: bioavailability, absorption rate (Ka), clearance (CL), volume of distribution (Vd)
   - Output: half-life, Tmax (time to peak), Cmax (peak concentration), AUC (area under curve)
   - Uses first-order kinetics with GUM uncertainty propagation

3. **Monte Carlo Clinical Trial**
   - 100–1000 virtual patients
   - Stochastic PK parameters (±15% variability around mean)
   - Efficacy = % of patients with plasma concentration > MEC (minimum effective concentration)
   - Adverse = % of patients exceeding toxic threshold
   - Therapeutic Index = toxic / MEC

4. **Final Decision**
   - Combines epistemic confidence from all three stages
   - PROCEED if efficacy > 60%, adverse < 10%, confidence > 50%
   - HALT otherwise

## Installation

### Prerequisites

- Sounio compiler: `souc` binary from [sounio.dev](https://sounio.dev)
- Export stdlib path: `export SOUNIO_STDLIB_PATH=$(pwd)/stdlib`

### From Repository

```bash
cd ecosystem/drug-discovery
```

No build needed—pure Sounio source.

## Quick Start

### Run Full Pipeline

```bash
export SOUC=./bin/souc
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

$SOUC run examples/full_pipeline.sio
```

Expected output (3–5 seconds):

```
========================================
  DRUG DISCOVERY PIPELINE
  Full End-to-End Demo
========================================

STAGE 1: Virtual Screening
--------------------------
  MW  = 129.160 Da, LogP = -1.430, HBD = 2, HBA = 5
  Knowledge { value: 1.000 epsilon: 0.944 prov: "lipinski_screen" }
  Result: PASS

STAGE 2: PK/PD Modeling
-----------------------
  Knowledge { value: 4.620 epsilon: 0.834 prov: "pk_half_life" }
  Knowledge { value: 0.467 epsilon: 0.854 prov: "pk_tmax" }
  Knowledge { value: 12.456 epsilon: 0.821 prov: "pk_cmax" }
  Knowledge { value: 56.667 epsilon: 0.771 prov: "pk_auc" }

STAGE 3: Clinical Trial Simulation
-----------------------------------
  Patients: 100
  Knowledge { value: 0.730 epsilon: 0.950 prov: "trial_efficacy" }
  Knowledge { value: 0.050 epsilon: 0.950 prov: "trial_adverse" }
  Knowledge { value: 4.000 epsilon: 0.900 prov: "therapeutic_index" }

FINAL DECISION
--------------
  PROCEED -- advance to Phase II
  Knowledge { value: 0.690 epsilon: 0.690 prov: "pipeline_decision" }

========================================
  Pipeline complete.
========================================
```

### Type-Check Without Running

```bash
$SOUC check examples/full_pipeline.sio
```

### Dump AST/Types

```bash
$SOUC check examples/full_pipeline.sio --show-ast
$SOUC check examples/full_pipeline.sio --show-types
```

## Knowledge Type Format

Every computed value is output as:

```
Knowledge { value: X epsilon: Y prov: "Z" }
```

Where:
- **value** — Central estimate (point value)
- **epsilon** — Standard uncertainty (1-sigma, GUM)
- **prov** — Provenance label (source/computation trace)

Example:
```
Knowledge { value: 12.456 epsilon: 0.821 prov: "pk_cmax" }
```

Interpretation: *Peak plasma concentration is 12.456 µg/mL with standard uncertainty of 0.821 µg/mL, derived from PK/PD modeling.*

## Running from Python

Use the high-level API in `sounio-py`:

```python
import sounio

pipeline = sounio.DrugDiscoveryPipeline()

# Stage 1: Virtual screening
screening = pipeline.screen_molecule(
    name="metformin",
    mol_weight=129.16,
    mw_uncertainty=0.99,
    logp=-1.43,
    logp_uncertainty=0.95,
    h_donors=2,
    h_acceptors=5
)
print(f"Screening: {screening}")

# Stage 2: PK/PD
pk = pipeline.fit_pkpd(
    bioavailability=0.85,
    bioavail_uncertainty=0.92,
    elimination_rate=0.15,
    volume_distribution=50.0
)
print(f"Half-life: {pk.half_life}")

# Stage 3: Monte Carlo
trial = pipeline.simulate_trial(
    n_patients=100,
    target_conc=5.0,
    toxic_threshold=20.0,
    seed=42
)
print(f"Efficacy: {trial.efficacy_rate}")
print(f"Adverse: {trial.adverse_rate}")

# Decision
if trial.efficacy_rate.value > 0.6 and trial.adverse_rate.value < 0.1:
    print("PROCEED to Phase II")
else:
    print("HALT — insufficient evidence")
```

## File Structure

```
drug-discovery/
├── README.md                           # This file
├── sounio.toml                         # Package manifest
├── src/
│   ├── lib.sio                         # Shared types & helpers
│   ├── pipeline/
│   │   ├── virtual_screening.sio       # Stage 1: Lipinski filter
│   │   ├── pkpd_modeling.sio           # Stage 2: One-compartment model
│   │   └── clinical_trial.sio          # Stage 3: Monte Carlo sim
│   └── data_models/
│       ├── molecule.sio                # Mol properties struct
│       ├── patient.sio                 # Patient record struct
│       └── knowledge.sio               # Knowledge[T] helper (if local)
├── examples/
│   └── full_pipeline.sio               # Complete end-to-end demo (308 lines)
├── tests/
│   ├── test_lipinski.sio               # Virtual screening tests
│   ├── test_pkpd.sio                   # PK/PD model tests
│   └── test_trial.sio                  # Trial simulation tests
└── paper/
    ├── paper.md                        # Reproducible results write-up
    └── reproducibility.sio             # Exact demo script
```

## Key Functions in full_pipeline.sio

### Mathematical Utilities
- `fp_abs(x)` — absolute value
- `fp_exp(x)` — exponential (Taylor series)
- `fp_log(x)` — natural logarithm

### Knowledge Arithmetic
- `fk_mul(a, b)` — multiplication with GUM propagation
- `fk_div(a, b)` — division with GUM propagation
- `fk_sub(a, b)` — subtraction with GUM propagation
- `fk_scale(a, s)` — scalar multiplication
- `fk_exp(a)` — exponential of Knowledge value

### Display
- `print_f64_3(v)` — print float with 3 decimal places
- `print_k(v, eps, label)` — print Knowledge value with provenance

### Randomness
- `FRng` struct — 32-bit linear congruential PRNG
- `frng_next(rng)` — advance RNG state
- `frng_range(rng, lo, hi)` — uniform random in [lo, hi)

## Uncertainty Quantification

All outputs propagate uncertainty via GUM rules:

**Addition/Subtraction**
```
ε(a ± b) = √(εa² + εb²)
```

**Multiplication/Division**
```
ε(a × b) = |a × b| × √((εa/a)² + (εb/b)²)
ε(a / b) = |a / b| × √((εa/a)² + (εb/b)²)
```

**Scalar Multiplication**
```
ε(s × a) = |s| × εa
```

The pipeline computes confidence at each stage, then combines them for the final decision.

## Testing

```bash
# Run individual stage tests
$SOUC run tests/test_lipinski.sio
$SOUC run tests/test_pkpd.sio
$SOUC run tests/test_trial.sio

# Run full pipeline
$SOUC run examples/full_pipeline.sio
```

## Performance

- **JIT compilation**: 500 ms (one-time)
- **Full pipeline**: 3–5 seconds (100 patients)
- **Per-patient trial simulation**: ~30 ms per 100 patients

Native compilation (via self-hosted Sounio compiler) yields <1s total runtime.

## Extending the Pipeline

To add a new stage or modify parameters:

1. **Edit `examples/full_pipeline.sio`** directly (single-file demo)
2. Or structure as modules under `src/pipeline/` and import via `let x = library("module.sio")`

Example: adding PD (pharmacodynamics) stage after PK:

```sounio
fn pd_effect(conc: FK) -> FK with Div, Panic, Mut {
    // E_max model: effect = (E_max * conc) / (EC_50 + conc)
    let e_max = FK { value: 100.0, epsilon: 0.95, prov_id: 70 }
    let ec_50 = FK { value: 2.5, epsilon: 0.90, prov_id: 71 }
    let numerator = fk_mul(e_max, conc)
    let denominator = fk_add(ec_50, conc)
    let effect = fk_div(numerator, denominator)
    return effect
}
```

## Publications

Results from this pipeline can be cited as:

> *Epistemic Drug Discovery: A Case Study in Uncertainty Quantification*. Sounio Epistemic Computing, 2026. Source: `ecosystem/drug-discovery/`

## References

- **GUM Standard**: [JCGM 100:2008](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf) — Guide to the Expression of Uncertainty in Measurement
- **Lipinski's Rule of 5**: Lipinski, C. A., et al. "Experimental and computational approaches to estimate solubility and permeability in drug discovery." *Advanced Drug Delivery Reviews*, 1997, 23(1), 3–25.
- **One-Compartment PK Model**: Gibaldi, M., & Perrier, D. "Pharmacokinetics" (2nd ed.). CRC Press, 1982.
- **Sounio Language**: [docs.sounio.dev](https://docs.sounio.dev)

## License

Apache-2.0. See LICENSE in the Sounio repository.
