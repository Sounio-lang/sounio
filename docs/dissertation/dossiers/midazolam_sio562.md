<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.midazolam-sio562
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.midazolam-sio562
-->

# Scenario dossier — midazolam (SIO-562)

Contract: EPIC acceptance (15-drug). Computational coverage and predictive
evaluation reported separately; no scenario is called validated by parity alone.

NOTE (EPIC correction 31/10): midazolam is a PBPK14 first-pass/DDI case,
not PBPK28. It must not be grouped as PBPK28 validation.

## 1. Scientific question and sustained claim

Does one hepatic intrinsic clearance yield consistent first-pass survival
and systemic clearance, so ketoconazole inhibition moves F, CL_h and AUCR
together? Sustained: the mechanistic consistency + computed AUCR.
NOT sustained: clinical DDI prediction.

## 2. Route, formulation, dose, population, horizon

- Single oral 5 mg, gut depot + lag, 36 h horizon, fixed-step RK4
  dt=0.002; DDI arm adds static steady-state ketoconazole.
- Mechanism `ddi/cyp3a_competitive.sio`: one CLint_h gives both FH and
  CL_h (no double count). PGx-adjusted via `pgx/cyp3a45_midazolam`.
- Population: reference adult. No covariates beyond CYP3A inhibition.

## 3. Partitioning, blood/plasma, free/total, units

- Systemic Kp are NOMINAL midazolam-plausible values (Vss ~1.4 L/kg,
  header-explicit) — ASSUMED by construction.
- fu ~0.03 (Smith 1981). Renal CL negligible (<1%). Units mg/L.

## 4. Parameters — source, unit, class

| param | value | source | class |
| --- | --- | --- | --- |
| oral F | ~0.4 | Heizmann 1984 | measured |
| gut+hepatic first-pass | mechanism | Thummel 1996, Yang 2007 (Qgut) | measured (mechanism form) |
| keto DDI | ~15x oral AUC | Olkkola 1994, 1996; Gorski 1998 | measured (literature band) |
| fu | 0.03 | Smith 1981 | measured |
| systemic Kp | nominal Vss 1.4 | scenario convention | assumed (header-explicit) |

## 5. Fit versus evaluation data

No fit performed here. Evaluation: solo↔inhibited lockstep parity;
min state 0.000000 both arms (SIO-504); dt vs dt/2 exact late agreement
(SIO-504); AUCR closed-form grid (case 19). F, CL_h, AUCR are
distribution-independent, so nominal Kp do not leak into them.

## 6. External comparison or explicit pendency

- Olkkola ~15x band is the reference band for AUCR; quantitative
  overlay against the band NOT executed → PENDENCY (comparison, not
  calibration).
- SIO-505: solo/inh plasma t=0.25 h legends CONFIRM current output.

## 7. Three columns

- Computational: VERIFIED (parity, positivity, convergence datum,
  closed-form AUCR grid).
- External evaluation: PENDING.
- Permitted claim: verified consistent first-pass/DDI computation;
  no clinical DDI prediction claim.

## 8. Mechanism coverage

Mechanistic first-pass + competitive CYP3A DDI + PGx CLint scaling,
executed. Covered by executed model.

## 9. Figure/table link

MDZ solo/inh + igrid AUCR; SIO-504 addendum (RK4 row); SIO-505.

## 10. Versions

Self-contained ref
`tests/run-pass/dissertation_pbpk28_parity_ref_midazolam.sio`;
production `scenarios/oral_midazolam_ddi.sio`. Measured at 55c141ce.
