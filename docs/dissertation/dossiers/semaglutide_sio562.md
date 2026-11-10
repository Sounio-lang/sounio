<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.semaglutide-sio562
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.semaglutide-sio562
-->

# Scenario dossier — semaglutide (SIO-562)

Contract: EPIC acceptance (15-drug). Computational coverage and predictive
evaluation reported separately; no scenario is called validated by parity alone.

## 1. Scientific question and sustained claim

Does the same PBPK28 integrator carry a large peptide (SC depot,
proteolytic clearance, vascular-confined distribution) through GLP-1R
binding to glucose-insulin PD with bit-exact cross-engine agreement?
Sustained: the DrugProfile abstraction + numerical agreement.
NOT sustained: clinical PK/PD prediction.

## 2. Route, formulation, dose, population, horizon

- SC depot 1.0 mg weekly; release ka = ln2/60 h, F = 0.89 (Carlsson 2020);
  proteolytic CL 0.077 L/h (Overgaard 2019).
- 30 h horizon in the parity ref, dt=0.001.
- Population: reference 70 kg adult. No covariates.

## 3. Partitioning, blood/plasma, free/total, units

- Kp < 1 everywhere (vascular-confined peptide); BBB PS ~1 (nearly
  impermeable, steric exclusion). Empirical Kp — Rodgers-Rowland NOT
  applicable to peptides (header-explicit framework boundary).
- fu_plasma 0.01, 99% albumin-bound (Tougaard 2022). Units mg/L;
  TMDD in nM; PD deviations mg/dL (G) and mU/L (I).

## 4. Parameters — source, unit, class

| param | value | source | class |
| --- | --- | --- | --- |
| CL proteolytic | 0.077 L/h | Overgaard 2019 popPK n=72, CL CV 15% | measured |
| depot ka / F | ln2/60 h, 0.89 | Carlsson 2020 depot model | measured (model fit by authors) |
| GLP-1R Kd | 0.4 nM | Lau 2015 | measured |
| fu | 0.01 | Tougaard 2022 | measured |
| early clinical PK | profiles | Kapitza 2015 | measured (not run against) |
| Kp/PS | file vectors, Kp<1 | empirical peptide convention | assumed-class — SIO-550 |

## 5. Fit versus evaluation data

Depot/popPK fits belong to Overgaard/Carlsson (by their authors).
Evaluation here: Node↔Sounio RMSE exactly 0.000000e+00 on 14 organs
(case 7), 3 TMDD organs (case 8), Bergman PD (case 9). No independent
clinical dataset run.

## 6. External comparison or explicit pendency

- Overgaard 2019 / Kapitza 2015 profiles: comparison NOT executed →
  PENDENCY.
- SIO-505: no semaglutide legend value cited in v28 notes (no check
  due).

## 7. Three columns

- Computational: VERIFIED (bit-exact cross-engine, 0 clamp fires per
  SIO-504).
- External evaluation: PENDING.
- Permitted claim: verified peptide PK→TMDD→PD computation and
  DrugProfile portability; no clinical prediction claim.

## 8. Mechanism coverage

SC depot release + GLP-1R TMDD (brain/gut/pancreas) + Bergman PD,
executed. Covered by executed model.

## 9. Figure/table link

Gate cases 7, 8, 9; SIO-504 addendum (CN row); viewer tour (drug B).

## 10. Versions

Refs `tests/run-pass/dissertation_pbpk28_parity_ref_semaglutide.sio`;
production `scenarios/semaglutide_sc_depot.sio` (clamped CN).
Measured at 55c141ce.
