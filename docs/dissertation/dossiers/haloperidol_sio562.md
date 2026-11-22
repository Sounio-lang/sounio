<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.haloperidol-sio562
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.haloperidol-sio562
-->

# Scenario dossier — haloperidol (SIO-562)

Contract: EPIC acceptance (15-drug). Computational coverage and predictive
evaluation reported separately; no scenario is called validated by parity alone.

NOTE (EPIC correction 31/10): haloperidol is a PBPK14 case with BBB/D2,
not PBPK28. It must not be grouped as PBPK28 validation.

## 1. Scientific question and sustained claim

Does the oral → PBPK14 → BBB (isf/icf) → D2 occupancy chain reproduce the
plasma-brain-occupancy link for a trapping basic drug? Sustained: the
numerical chain. NOT sustained: occupancy prediction in patients.

## 2. Route, formulation, dose, population, horizon

- Single oral 5 mg into gut depot with lag (ka 0.9 /h, F 0.65, tlag 0.25 h),
  72 h horizon, fixed-step RK4 dt=0.002.
- Population: reference adult physiology (Davies & Morris 1993 brain
  volumes). No covariates.

## 3. Partitioning, blood/plasma, free/total, units

- Kpuu_brain 3.0: lysosomal trapping beats P-gp efflux (opposite of
  rapamycin at 0.15). Source Loryan 2014 RAT — human use is assumed.
- fu_plasma 0.08 (Haase 1981). D2 Kd ~1.5 nM (Seeman 1976). Units mg/L;
  occupancy fraction.

## 4. Parameters — source, unit, class

| param | value | source | class |
| --- | --- | --- | --- |
| Kpuu_brain | 3.0 | Loryan 2014, rat | measured (rat) / assumed (human) |
| fu_plasma | 0.08 | Haase 1981 | measured |
| D2 Kd | 1.5 nM | Seeman 1976 radioligand | measured |
| ka/F/tlag | 0.9, 0.65, 0.25 h | scenario absorption | assumed (scenario convention) |
| BBB PS class | lipophilic | Gaohua 2016 | predicted/class |
| systemic Kp | Vss-based well-stirred | scenario convention | assumed — SIO-550 |

## 5. Fit versus evaluation data

No fit performed here. Evaluation: Node↔Sounio lockstep parity on
HALO endpoints (plasma, Kpuu, D2 occupancy); state minimum 0.000000
(never negative, SIO-504); dt vs dt/2 exact late agreement (SIO-504).
Therapeutic window 60-80% (Farde 1992 PET) cited, not run.

## 6. External comparison or explicit pendency

- Farde 1992 PET window and D2 occupancy: comparison NOT executed →
  PENDENCY.
- SIO-505: plasma t=0.5 h legend 0.008882 CONFIRMS current output.

## 7. Three columns

- Computational: VERIFIED (parity, positivity, convergence datum).
- External evaluation: PENDING.
- Permitted claim: verified plasma→brain→occupancy computation for
  haloperidol; no clinical occupancy claim.

## 8. Mechanism coverage

BBB isf/icf split + D2 occupancy, executed. Covered by executed model.

## 9. Figure/table link

HALO endpoints; SIO-504 addendum (RK4 row); SIO-505 (plasma t=0.5 h).

## 10. Versions

Self-contained ref
`tests/run-pass/dissertation_pbpk28_parity_ref_haloperidol.sio`;
production `scenarios/oral_haloperidol_bbb.sio`. Measured at 55c141ce.
