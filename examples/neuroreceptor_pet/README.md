# Executable Epistemic PET Kinetics: GUM-Compliant Propagation of PBPK-Informed Uncertainty into Receptor Binding Metrics

**NRM 2026 Late-Breaking Abstract Vertical Slice**

This example implements a minimal, reproducible **two-tissue compartment model (2TCM)** for PET neuroreceptor imaging with full **GUM-compliant uncertainty propagation** (JCGM 100:2008 §5.1.3) using finite-difference Jacobian, exactly as demonstrated in `stdlib/darwin_pbpk/epistemic_pbpk14.sio`.

## Scientific Motivation

PET receptor binding studies are highly sensitive to both model structure and parameter uncertainty. Traditional error propagation (e.g. bootstrap or asymptotic covariance) often underestimates real-world epistemic uncertainty coming from input function, plasma protein binding, and kinetic rate constants. 

This vertical slice demonstrates how **epistemic priors** (mean + variance + per-parameter confidence) can be propagated through the standard 2TCM equations to yield **uncertainty on derived metrics** (V_T, BP_ND, TAC AUC, peak) using the same numerical GUM method already validated on the 14-compartment PBPK model.

## Model Equations

```
dC₁/dt = K₁·Cₚ(t) − (k₂ + k₃)·C₁ + k₄·C₂
dC₂/dt = k₃·C₁ − k₄·C₂
C_T(t) = C₁(t) + C₂(t)          (tissue time-activity curve)

BP_ND = k₃ / k₄
V_T   = (K₁ / k₂) · (1 + k₃/k₄)
```

Plasma input (synthetic, analytically stable):
```
Cₚ(t) = cp_amp · exp_approx(−cp_decay·t)
```

## Relation to Darwin PBPK

This example **does not modify** the core PBPK14 machinery. It reuses:
- The exact GUM finite-difference sensitivity pattern (`h = max(1e-6·|μ|, 1e-2·σ)`)
- The same style of epistemic prior struct and evidence-weighted confidence calculation
- Identical acceptance test philosophy (monotonicity, variance > 0, sensitivity normalization)

It serves as a **neuroimaging companion** to the existing `epistemic_pbpk14.sio` and `tsit5_pbpk14.sio` modules.

## How to Run

```bash
cd /workspace/sounio
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

$SOUC_BIN run examples/neuroreceptor_pet/pet_2tcm_epistemic.sio
```

Or via the test harness if available:
```bash
bash scripts/run_sio_test_suite.sh --example neuroreceptor_pet
```

## Expected Output

The program prints:
- Header identifying NRM 2026 proof-of-concept
- Synthetic epistemic priors
- TAC AUC, TAC peak, V_T, BP_ND with propagated standard deviations
- Evidence-weighted confidence score
- Normalized sensitivity fractions for each parameter
- 10 acceptance tests (including causal monotonicity and sensitivity normalization)

All tests are designed to pass with the provided demonstration priors.

## Limitations (Honest Status)

- Proof-of-concept only. Not a validated clinical fitting tool.
- Uses fixed-step RK4 (conservative). No adaptive solver in this slice.
- Synthetic plasma input function (no real arterial sampling).
- No fitting to real PET data in this vertical slice.
- No GPU, no patient-level hierarchical modeling, no tracer-specific validation.

This is **executable epistemic modeling** — a demonstration that uncertainty propagation through PET kinetics can be made fully transparent, reproducible, and auditable in Sounio.

## NRM 2026 Framing

> "We present the first executable epistemic implementation of the gold-standard 2TCM for PET neuroreceptor imaging, propagating GUM-compliant uncertainty from priors through to binding potential (BP_ND) and total distribution volume (V_T). This work establishes a new standard for transparent propagation of epistemic uncertainty in quantitative neuroimaging."

**Repository**: Sounio-lang/darwin-pbpk  
**File**: `examples/neuroreceptor_pet/pet_2tcm_epistemic.sio`

---

*This implementation strictly follows the style, safety patterns, and GUM methodology of `epistemic_pbpk14.sio`. No core compiler or PBPK14 modules were modified.*
