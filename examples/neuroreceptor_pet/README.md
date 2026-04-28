# Executable Epistemic PET Kinetics: GUM-Compliant Propagation of PBPK-Informed Uncertainty into Receptor Binding Metrics

**NRM 2026 Late-Breaking Abstract — Vertical Slice (Numerically Audited)**

This example implements a minimal, reproducible **two-tissue compartment model (2TCM)** for PET neuroreceptor imaging with **GUM-compliant uncertainty propagation** (JCGM 100:2008 §5.1.3) via a finite-difference Jacobian, following the methodology established in `stdlib/darwin_pbpk/epistemic_pbpk14.sio`.

## Scientific Motivation

PET receptor binding studies are highly sensitive to both model structure and parameter uncertainty. Traditional error propagation (bootstrap, asymptotic covariance) often underestimates epistemic uncertainty from the input function, plasma protein binding, and kinetic rate constants.

This vertical slice demonstrates how **epistemic priors** (mean + variance + per-parameter confidence) can be propagated through standard 2TCM equations to yield **uncertainty on derived metrics** (V_T, BP_ND, TAC AUC, peak) using the same numerical GUM method validated on the 14-compartment PBPK model.

## Model Equations

```
dC₁/dt = K₁·Cₚ(t) − (k₂ + k₃)·C₁ + k₄·C₂
dC₂/dt = k₃·C₁ − k₄·C₂
C_T(t) = C₁(t) + C₂(t)          (tissue time-activity curve)

BP_ND = k₃ / k₄
V_T   = (K₁ / k₂) · (1 + k₃/k₄)
```

Plasma input (synthetic, analytically decaying):
```
Cₚ(t) = cp_amp · exp(−cp_decay · t)
```

`exp` is implemented locally via aggressive range reduction to `|x| ≤ 0.5` followed by a 20-term Taylor series. This keeps Cp(t) physically correct across the full 0–60 minute window (validated against expected values to ~1e-6 at t=60).

## Numerical Audit Results

The example runs an internal audit on every execution. Current results with the spec priors (K1=0.15, k2=0.20, k3=0.10, k4=0.05, cp_amp=1.0, cp_decay=0.20, t=0..60 min, dt=0.05):

| Quantity | Computed | Analytic / Expected | Status |
|----------|---------:|--------------------:|:------:|
| `Cp(0)` | 1.000000 | 1.0 | ✓ |
| `Cp(1)` | 0.818731 | 0.8187 | ✓ |
| `Cp(5)` | 0.367879 | 0.3679 | ✓ |
| `Cp(10)` | 0.135335 | 0.1353 | ✓ |
| `Cp(60)` | 0.000006 | 6.14e-6 | ✓ |
| **TAC AUC** | **9.4695** | ~9.47 (RK4 ref) | ✓ |
| **TAC Peak** | **0.3082** | ~0.31 | ✓ |
| `V_T` mean | 2.2500 | 2.25 (exact) | ✓ |
| `BP_ND` mean | 2.0000 | 2.00 (exact) | ✓ |
| `V_T` SD | 0.6190 | ~0.62 (delta) | ✓ |
| `BP_ND` SD | 0.5651 | ~0.566 (delta) | ✓ |
| `d(BP_ND)/dk₃` | 20.000 | 20.0 | ✓ |
| `d(BP_ND)/dk₄` | −39.92 | −40.0 | ✓ |
| `d(V_T)/dK₁` | 15.000 | 15.0 | ✓ |
| `d(V_T)/dk₂` | −11.23 | −11.25 | ✓ |
| `d(V_T)/dk₃` | 15.000 | 15.0 | ✓ |
| `d(V_T)/dk₄` | −29.94 | −30.0 | ✓ |
| Sensitivity sum | 1.000000 | 1.0 | ✓ |

All 10 acceptance tests pass. Finite-difference sensitivities agree with the delta-method analytic predictions for V_T and BP_ND to better than 0.5%.

## Relation to Darwin PBPK

This example **does not modify** core PBPK14 machinery. It reuses:
- The exact GUM finite-difference pattern (`h = max(1e-6·|μ|, 1e-2·σ)`)
- The same epistemic prior struct layout and evidence-weighted confidence calculation
- The same testing philosophy (exact analytic checks + monotonicity + sensitivity normalization)

It serves as a **neuroimaging companion** to `epistemic_pbpk14.sio` and `tsit5_pbpk14.sio`.

## How to Run

```bash
cd /workspace/sounio
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" check examples/neuroreceptor_pet/pet_2tcm_epistemic.sio
"$SOUC_BIN" run   examples/neuroreceptor_pet/pet_2tcm_epistemic.sio
```

The program exits with code 0 if all audits pass, 1 otherwise.

## Honest Scientific Status

This is a **numerically audited proof-of-concept vertical slice**. Internal tests verify:

1. Cp(t) decays correctly to ~1e-6 at t=60 min.
2. V_T and BP_ND reproduce the exact analytic values (2.25 and 2.00).
3. TAC AUC and peak fall inside the expected physical range and match the RK4 reference value.
4. GUM-propagated SDs for V_T and BP_ND agree with the delta-method prediction.
5. Each finite-difference sensitivity matches its analytic partial derivative.

### What this example **does not** claim:

- Not a validated clinical fitting tool.
- Priors are literature-inspired but **not fitted** to any real human or primate PET dataset.
- Fixed-step RK4 only (no adaptive Tsit5 in this slice).
- No parameter estimation, no hierarchical modeling, no partial volume correction, no vascular fraction, no arterial dispersion.
- No GPU, no population inference, no tracer-specific validation.
- Not intended for clinical use or regulatory decision-making.

The implementation strictly follows the style, safety patterns, and GUM methodology of `stdlib/darwin_pbpk/epistemic_pbpk14.sio`. No compiler or core PBPK infrastructure was modified.

## NRM 2026 Strategic Framing

The contribution is **not** a new PET fitting algorithm. It is the demonstration that a strongly-typed, self-hosted language with built-in effects and an auditable numerical stack can produce a 2TCM implementation whose uncertainty propagation:

- Matches analytic expectations to ~0.5%.
- Is fully transparent and reproducible from source.
- Exposes the same epistemic prior/confidence structure already validated on PBPK14.
- Provides a viable path for future coupling of PBPK plasma priors into neuroreceptor kinetic metrics.

---

**Repository**: Sounio-lang/darwin-pbpk
**File**: `examples/neuroreceptor_pet/pet_2tcm_epistemic.sio`
**Audit pass count**: 10/10 (as of 2026-04-28)
