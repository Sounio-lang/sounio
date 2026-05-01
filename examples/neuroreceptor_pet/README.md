<!-- docs:meta
topic_id: repo.examples.neuroreceptor-pet.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.neuroreceptor-pet.readme
-->

# Executable Epistemic PET Kinetics: GUM-Compliant Propagation of PBPK-Informed Uncertainty into Receptor Binding Metrics

**NRM 2026 Late-Breaking Abstract — Proof-of-Concept Vertical Slice**

> **Scope note.** This is a minimal, reproducible **proof-of-concept** for
> executable epistemic PET modelling in Sounio. It is explicitly **not** a
> clinical PET fitting package, not equivalent to PMOD/AMIDE/PNEURO, and
> has **not** been fitted to any patient dataset. All quantitative claims
> are audited in [`AUDIT_REPORT.md`](AUDIT_REPORT.md).

This example implements a minimal **two-tissue compartment model (2TCM)** with
**GUM-compliant uncertainty propagation** (JCGM 100:2008 §5.1.3) across eight
epistemic priors, including the PBPK-informed scalars `fu_plasma` and
`bbb_scalar`. The methodology mirrors `stdlib/darwin_pbpk/epistemic_pbpk14.sio`.

## Files

| File | Role |
|------|------|
| `pet_2tcm_epistemic.sio` | Main 2TCM + GUM audit (12 numerical acceptance tests) |
| `pet_2tcm_export.sio` | CSV exporter for the synthetic TAC curve (stdout → file) |
| `pet_tracer_variants.sio` | Same 2TCM+GUM code exercised with four literature-anchored prior sets (illustrative, not tracer-specific validation) |
| `pet_srtm.sio` | SRTM (Lammertsma & Hume 1996) solver with 2TCM side-by-side in both rapid-equilibrium and slow-binding regimes |
| `pet_fit_validation.sio` | Single-realisation parameter-recovery stress test (coordinate-descent) |
| `pet_fit_montecarlo.sio` | 20-realisation Monte-Carlo stress test (LCG + Irwin-Hall noise) |
| `pet_lammertsma1996_analysis.sio` | **Algebraic consistency check** of the Innis 2007 formula `BP = V_T_tar/V_T_ref − 1` against the published aggregate V_T/BP summary values in Lammertsma 1996 Tables 2 and 3. **Not** a fit to real dynamic TAC data. |
| `NRM2026_ABSTRACT.md` | Late-breaking abstract draft |
| `LITERATURE_VALIDATION.md` | Prior ranges vs [11C]raclopride literature |
| `AUDIT_REPORT.md` | Audit of every numerical claim in this folder |
| `results/*.txt` | Captured stdout for each acceptance run |
| `results/tac_curve.csv` | Generated synthetic TAC curve |

## Model

```
dC₁/dt = K₁_eff·Cₚ(t) − (k₂ + k₃)·C₁ + k₄·C₂
dC₂/dt = k₃·C₁ − k₄·C₂
C_T(t) = C₁(t) + C₂(t)

K₁_eff = K₁ · fu_plasma · bbb_scalar       (PBPK coupling)

BP_ND = k₃ / k₄                             (independent of fu, bbb)
V_T   = (K₁_eff / k₂) · (1 + k₃/k₄)
Cₚ(t) = cp_amp · exp(−cp_decay · t)
```

`exp` implemented locally via aggressive range reduction to `|x| ≤ 0.5` + 20-term Taylor; accurate to ~1e-10 across 0..60 min.

## Eight Epistemic Priors

| Idx | Parameter    | Mean  | Variance | Role |
|-----|--------------|-------|----------|------|
| 0   | K1           | 0.15  | 0.0004   | Plasma→tissue influx |
| 1   | k2           | 0.20  | 0.0009   | Tissue→plasma efflux |
| 2   | k3           | 0.10  | 0.0004   | Specific binding |
| 3   | k4           | 0.05  | 0.0001   | Dissociation |
| 4   | cp_amp       | 1.00  | 0.0025   | Input amplitude |
| 5   | cp_decay     | 0.20  | 0.0004   | Input decay |
| 6   | fu_plasma    | 1.00  | 0.01     | Unbound plasma fraction (PBPK) |
| 7   | bbb_scalar   | 1.00  | 0.01     | BBB transport scalar (PBPK) |

## Audit Results (12/12 PASS)

| Quantity | Computed | Analytic | Δ |
|----------|---------:|---------:|----:|
| `Cp(0)` | 1.000000 | 1.0 | 0 |
| `Cp(5)` | 0.367879 | 0.3679 | 1e-6 |
| `Cp(60)` | 0.000006 | 6.14e-6 | < 1e-7 |
| `TAC AUC` | 9.4695 | ~9.47 | < 0.1 % |
| `TAC Peak` | 0.3082 | ~0.31 | < 1 % |
| `V_T` mean | 2.2500 | 2.25 | 0 |
| `BP_ND` mean | 2.0000 | 2.00 | 0 |
| `V_T` SD | 0.6960 | ~0.695 | 0.15 % |
| `BP_ND` SD | 0.5651 | ~0.566 | 0.15 % |
| `d(BP_ND)/dk₃` | 20.000 | 20.0 | 0 |
| `d(BP_ND)/dk₄` | −39.92 | −40.0 | 0.2 % |
| `d(BP_ND)/dfu` | **0.000** | **0.0** | exact |
| `d(BP_ND)/dbbb` | **0.000** | **0.0** | exact |
| `d(V_T)/dK₁` | 15.000 | 15.0 | 0 |
| `d(V_T)/dfu` | 2.250 | 2.25 | 0 |
| `d(V_T)/dbbb` | 2.250 | 2.25 | 0 |
| sensitivity sum | 1.000000 | 1.0 | 0 |

### Sensitivity Fractions (V_T)

| Parameter | Fraction |
|-----------|---------:|
| K1        | 18.6 % |
| k2        | 23.4 % |
| k3        | 18.6 % |
| k4        | 18.5 % |
| fu_plasma | **10.4 %** (PBPK) |
| bbb_scalar| **10.4 %** (PBPK) |
| cp_amp    | 0 % |
| cp_decay  | 0 % |

Evidence-weighted confidence: **0.626**.

The **structural insensitivity of BP_ND to fu and bbb** is recovered exactly — a scientifically meaningful result separating kinetic-binding uncertainty from PBPK-input uncertainty.

## How to Run

```bash
cd /workspace/sounio
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

# 12-test audit (returns exit code 0 if all pass)
"$SOUC_BIN" run examples/neuroreceptor_pet/pet_2tcm_epistemic.sio

# Export TAC curve to CSV
"$SOUC_BIN" run examples/neuroreceptor_pet/pet_2tcm_export.sio \
    > examples/neuroreceptor_pet/results/tac_curve.csv
```

## Literature Anchoring

All synthetic priors fall inside published ranges for **[11C]raclopride in human striatum** (2TCM method A). See `LITERATURE_VALIDATION.md` for the full comparison table, including:

- K1 = 0.15 matches Lammertsma 1996 (range 0.10–0.15)
- BP_ND = 2.00 matches published striatum range (1.5–3.0)
- V_T = 2.25 matches published range (2–4 ml·cm⁻³)
- Model equations are exactly the Lammertsma/Innis 2TCM

## Honest Scientific Status

**Numerically audited.** `pet_2tcm_epistemic.sio` passes 12/12 internal
acceptance tests. Finite-difference derivatives agree with analytic delta-method
predictions at sub-percent level. See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) for
per-file pass/fail details.

**Literature-anchored priors.** Priors are set inside published [11C]raclopride
in human striatum ranges (Lammertsma 1996; Farde 1989; Innis 2007). Priors are
*not* fitted to any patient or phantom data.

**What this package is not.**

- Not a clinical PET fitting package.
- Not equivalent to PMOD / AMIDE / PNEURO.
- Not validated against real dynamic TAC data.
- Not a substitute for peer-reviewed kinetic modelling software.
- Fixed-step RK4 only, synthetic exponential plasma input, no hierarchical
  modelling, no partial-volume correction, no metabolite correction, no delay /
  dispersion modelling, no frame weighting.
- The Monte-Carlo file uses a simple coordinate-descent optimiser on a discrete
  multiplier grid — not Levenberg-Marquardt or a production nonlinear fitter.
- The `pet_lammertsma1996_analysis.sio` script is a closed-form algebraic
  consistency check on *published aggregate summary statistics* (Lammertsma
  1996 Tables 2 and 3), not a re-analysis of raw dynamic PET data.

The implementation follows the style, safety patterns, and GUM methodology of
`stdlib/darwin_pbpk/epistemic_pbpk14.sio`. No compiler or core PBPK
infrastructure was modified.

## NRM 2026 Framing

See `NRM2026_ABSTRACT.md` for the late-breaking abstract draft. Core contribution: demonstration that a strongly-typed, self-hosted language can produce a PET kinetic model whose uncertainty propagation (including PBPK coupling) is auditable against analytic expectations and fully reproducible from source.

---

**Repository:** Sounio-lang/sounio (https://github.com/Sounio-lang/sounio)
**Branch:** `integration/sounio-dev-ready-base`
**Audited commit:** `2e817fcbde01b14ac3524c09e4ae0d88d72d83c3`
**Audit status:** 12/12 PASS as of 2026-04-28
