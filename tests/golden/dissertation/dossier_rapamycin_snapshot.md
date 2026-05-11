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

## §10. Live interactive viewer — multi-drug A/B (Stage G)

A 3D, browser-native interactive view of this model is hosted at:

> **https://www.souniolang.org/dissertation/**

The Stage-G viewer is drug-agnostic: a chip-row selector at the top of the
canvas (or the `D` keyboard shortcut) toggles between rapamycin (Cypher
coronary stent — Higuchi diffusion release) and semaglutide (subcutaneous
depot — first-order absorption, k_a = ln 2 / 60 h, F = 0.89 per Overgaard
2019 / Carlsson 2020). Release-source visual, receptor-occupancy bars, PD
readout panel, Phase J evidence band, patient-profile dropdown and the
release-scale slider all swap with the active drug.

The PBPK kernel is PBPK28 — permeability-limited (C_v, C_t) per organ with
a PS coupling — plus per-organ TMDD blocks (Mager 2004) and the drug's PD
ODE. The fully-coupled Crank-Nicolson step on the 27-state arrow matrix is
parity-locked (<= 1% RMSE per organ) against the Sounio reference solver
by scripts/ci/dissertation_pbpk28_parity_gate.sh, which runs nine hard
cases in a single pass (PBPK28 Node<->Sounio, QSS analytical, PBPK14
model-form reporting, mass-conservation monotonicity, rapamycin TMDD/PD,
semaglutide PBPK28/TMDD/PD).

Six narrated tours are available — three per drug. Press `T` to cycle
within the active drug; `D` cycles drug and resets tour selection:

Rapamycin:
1. **Cypher -> blood -> liver** (30 s) — Higuchi release; Kp = 5.4 hepatic
   accumulation.
2. **BBB closeup — Kp = 0.10** (20 s) — P-gp efflux; why BPR stays < 0.15.
3. **GUM cone widening under CL_hep variability** (20 s) — direct visual
   proof of contribution #1 (GUM-through-ODE).

Semaglutide:
4. **SC depot -> blood -> pancreas** (30 s) — first-order absorption from
   the abdominal depot, slow systemic distribution.
5. **GLP-1R occupancy — brain, gut, pancreas** (24 s) — TMDD bars filling
   at three sites; appetite, satiety, insulinotropic pathways narrated.
6. **Bergman PD — DG falls as DI rises** (22 s) — pancreatic occupancy
   drives insulin secretion which suppresses plasma glucose; switches
   mid-tour to "slow CL" so the PD GUM cone widens visibly.

For committee handouts the viewer's `Snapshot PNG` button (or the `S`
keyboard shortcut) exports a print-quality PNG of the current frame,
file-named with the active drug so multi-drug demos remain unambiguous.

PASS dossier_smoke
