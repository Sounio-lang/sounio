# Sounio Epistemic Computing Demo Report

**Author:** Sounio Ecosystem Demo  
**Date:** 2026-03-18

---

## Introduction

This report demonstrates the Sounio epistemic computing pipeline, combining PubChem molecular data, GUM-based uncertainty propagation, and drug discovery pipeline execution.


## Molecule Properties (PubChem)

| Parameter | Value | Uncertainty | Rel. Unc. | Provenance |
|-----------|-------|-------------|-----------|------------|
| Molecular Weight | 180.159 | ± 0.001 | 0.00% | pubchem |
| LogP (XLogP3) | 1.2 | ± 0.5 | 41.67% | pubchem_xlogp3 |


## Dosing Calculation

| Parameter | Value | Uncertainty | Rel. Unc. | Provenance |
|-----------|-------|-------------|-----------|------------|
| Prescribed Dose | 500 | ± 10 | 2.00% | prescribed_dose_mg |
| Patient Weight | 70 | ± 0.5 | 0.71% | patient_weight_kg |
| Dose per kg | 7.14286 | ± 0.151695 | 2.12% | (prescribed_dose_mg)/(patient_weight_kg) |
| Estimated Plasma Concentration | 10 | ± 1.0198 | 10.20% | (prescribed_dose_mg)/(pk_vd_estimate) |


## Pipeline Results

| Parameter | Value | Uncertainty | Rel. Unc. | Provenance |
|-----------|-------|-------------|-----------|------------|
| lipinski_screen | 1 | ± 0.94 | 94.00% | lipinski_screen |
| pk_half_life | 4.62 | ± 0.767 | 16.60% | pk_half_life |
| pk_tmax | 1.98 | ± 0.446 | 22.53% | pk_tmax |
| pk_cmax | 6.315 | ± 0.322 | 5.10% | pk_cmax |
| pk_auc | 56.666 | ± 0.776 | 1.37% | pk_auc |
| trial_efficacy | 0 | ± 0.95 | 0.00% | trial_efficacy |
| trial_adverse | 0 | ± 0.95 | 0.00% | trial_adverse |
| therapeutic_index | 4 | ± 0.9 | 22.50% | therapeutic_index |
| pipeline_decision | 0.221 | ± 0.221 | 100.00% | pipeline_decision |


## Methodology

All epistemic calculations use the Guide to the Expression of Uncertainty in Measurement (GUM) standard. Relative uncertainties are propagated through arithmetic operations using:  
- Addition/Subtraction: ε = √(ε₁² + ε₂²)  
- Multiplication/Division: ε = |result| × √((ε₁/val₁)² + (ε₂/val₂)²)  
- Scalar operations: ε = |factor| × ε₁  


