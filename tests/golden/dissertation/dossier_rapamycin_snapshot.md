# PBPK Dossier — Rapamycin (Sirolimus)

## §1. Subject of submission

- Drug: Rapamycin (Sirolimus)
- Model: PBPK14 + Cypher DES coupling

## §2. Model card

Compartmental PBPK system: PBPK14 + Cypher DES coupling

## §3. Parameter priors

| name | units | value | rel_u | confidence |
|---|---|---|---|---|
| CL_hep | L/h | 12.600000 | 0.180000 | 0.612000 |
| Kpuu_brain | - | 0.045000 | 0.450000 | 0.503000 |

## §4. Numerical method

- Integrator: Tsit5 adaptive
- abs_tol: 0.000001
- rel_tol: 0.000100
- h_min: 0.001000
- h_max: 0.500000

## §5. ISO 17025 GUM budget (1st order)

| source | A/B | u_i | contribution |
|---|---|---|---|
| rapamycin_iv_dose | B | 0.300000 | 0.530000 |
| population_CL | A | 0.226000 | 0.301000 |
| Kpuu_brain_extrap | B | 0.205000 | 0.169000 |

- combined u_c: 0.409000
- expanded U_95: 0.818000

## §6. ISO 17025 GUM budget (2nd order)

| source | A/B | u_i | contribution |
|---|---|---|---|
| CL_x_Kpuu_cross | B | 0.041000 | 0.620000 |
| Kpuu_x_fu_cross | B | 0.032000 | 0.380000 |

- combined u_c: 0.058000
- expanded U_95: 0.116000

## §7. Confidence gate evidence (Phase J)

- min_conf threshold: 0.500000
- observed confidence: 0.612000
- verdict: ADMIT

## §8. Clinical validation

| endpoint | units | sounio | reference | pct_diff |
|---|---|---|---|---|
| AUC_brain | ng·h/mL | 184.200000 | 178.500000 | 3.190000 |

## §9. Audit trail

- commit_sha: 0000000000000000000000000000000000000000
- generated_at_utc: 2026-05-10T00:00:00Z
- sounio_version: lane-8c-test

PASS dossier_smoke
