<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.rapamycin-sio562
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.rapamycin-sio562
-->

# Scenario dossier — rapamycin (SIO-562)

Contract: EPIC acceptance (15-drug). Computational coverage and predictive
evaluation reported separately; no scenario is called validated by parity alone.

## 1. Scientific question and sustained claim

Can the fully-coupled permeability-limited PBPK28 CN step carry rapamycin
from blood through liver/heart/gut FKBP12 binding to mTORC1 activity and
neointimal readout in one end-to-end run? Sustained: the numerical chain
is verified (parity, mass, convergence). NOT sustained: clinical prediction.

## 2. Route, formulation, dose, population, horizon

- Parity ref: IV bolus 0.05 mg into blood, 30 h horizon, dt=0.001.
- Clinical anchors (not run in the ref): oral (Ferron 1997) and Cypher
  DES stent with Higuchi release (Cordis 2003 IFU); `oral_rapamycin_bbb.sio`
  exists separately.
- Population: reference 70 kg adult (ICRP volumes, Brown 1997 vascular
  fractions). No covariates.

## 3. Partitioning, blood/plasma, free/total, units

- Kp vector in `pbpk28_params_rapamycin`: liver 5.40, kidney 4.20,
  brain 0.10 (P-gp efflux, Lampen 1998), heart 2.30, lung 3.10, gut 2.60.
  Per-organ provenance split (measured/predicted) is SIO-550, still open.
- fu_plasma 0.08 (Schreiber 1991, in vitro). Units mg/L throughout.

## 4. Parameters — source, unit, class

| param | value | source | class |
| --- | --- | --- | --- |
| CL_hep | 12.4 L/h, CV 58% | Ferron 1997, n=24 in vivo | measured |
| Vd | 12 L/kg | Kahan 2001 | measured |
| fu_plasma | 0.08 | Schreiber 1991 in vitro | measured |
| Kp_brain | 0.07-0.12 | Lampen 1998 (P-gp) | measured (in vitro/vivo mix) |
| ka oral | 1.1 /h | MacDonald 2000 | measured (not exercised in bolus ref) |
| Kp organ vector | see §3 | literature-anchored | pending SIO-550 split |

Partial-audit flags (2026-11-02, reused): clearance prior mixes IIV CL/F
with residual error and needs justification; GUM/MC history uses centres
12.4/12.7 L/h with different steps.

## 5. Fit versus evaluation data

Literature parameters were fit by their original authors, not by us.
Evaluation here: Node↔Sounio parity (RMSE < 1% per organ, gate case 1),
REF↔ALT dual-dt convergence (dt=0.001/0.0005), 12/12 mass monotonicity
(gate case 4), TMDD (case 5) and PD (case 6) parity. No independent
clinical dataset is used for evaluation (`obs_n() = 0`,
PENDING_OBSERVED; `validation/pbpk28_rapamycin_clinical.sio` awaits
observed data).

## 6. External comparison or explicit pendency

- Ferron 1997 CL and Cordis 2003 IFU are cited anchors; comparison not
  executed → PENDENCY (clinical validation).
- SIO-505: liver cv/cavg t=0.1 h legend values CONFIRM current output.

## 7. Three columns

- Computational: VERIFIED (parity, mass, convergence, 0 clamp fires per
  SIO-504 at 55c141ce).
- External evaluation: PENDING (no observed-data comparison run).
- Permitted claim: the CN chain is numerically verified for rapamycin;
  no predictive-validity claim.

## 8. Mechanism coverage

FKBP12 TMDD (liver/heart/gut) + mTORC1/neo-intima PD at heart. Covered
by executed model, not by name.

## 9. Figure/table link

Parity gate cases 1, 4, 5, 6; SIO-504 addendum (clamp 0 fires);
SIO-505 (liver t=0.1 h). Appendix B .tex outside repo.

## 10. Versions

Kernel: own clamped CN copy in
`tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio`;
stdlib clamped kernel `tsit5_pbpk28.sio::pbpk28_full_cn_step`.
Measured at main 55c141ce.
