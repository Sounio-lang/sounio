<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.venlafaxine-sio562
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.venlafaxine-sio562
-->

# Scenario dossier — venlafaxine XR (SIO-562)

Contract: EPIC acceptance (15-drug). Computational coverage and predictive
evaluation reported separately; no scenario is called validated by parity alone.

## 1. Scientific question and sustained claim

Does the XR matrix → gut → parent PBPK28 → CYP2D6 formation → ODV PBPK28
chain produce a steady-state ODV/parent readout with a closed-form oracle
and a certified per-run error bound? Sustained: the computation and its
certificate. NOT sustained: PGx prediction (phenotype ratios unsourced).

## 2. Route, formulation, dose, population, horizon

- Oral XR 75 mg q24h (DailyMed Effexor XR), Korsmeyer-Peppas matrix
  (n=0.65, Gohel 2008), ka 0.63 /h, F_XR 0.45 (Wang 2022 / Wyeth 2021 label).
- 10th dosing interval (steady state), dt=0.5 and 0.25 with per-dt
  certificates (cases 12-13).
- Phenotypes PM/IM/NM/UM via `pgx/cyp2d6_venlafaxine.sio`.

## 3. Partitioning, blood/plasma, free/total, units

- Parent Kp: R-R class estimates, liver/kidney elevated (metabolic sink).
- ODV Kp: lower (more hydrophilic). Provenance split is SIO-550.
- fu parent 0.27, ODV 0.30 (Wyeth 2021 label). Units mg/L.

## 4. Parameters — source, unit, class

| param | value | source | class |
| --- | --- | --- | --- |
| ka / F_XR | 0.63 /h, 0.45 | Wang 2022 joint PPK / Wyeth label | measured (popPK) |
| oral CL / metabolic CL | 100 / 43 L/h EM | Lessard 1999 (prev. mis-cited Klamerus) | measured |
| t-half parent/ODV | 5 / 11 h | Wyeth label | measured |
| ODV/V phenotype ratios | file values | UNSOURCED (Kirchheiner 2006 does not exist; forensics doc 2026-09-26) | assumed — explicit pendency |
| Kp/PS parent+ODV | file vectors | Rodgers-Rowland class estimates | predicted — SIO-550 |

## 5. Fit versus evaluation data

Wang 2022 popPK is the fit source (by its authors). Evaluation here:
Korsmeyer-Peppas release RMSE 0, parent/ODV max 0.000026%/0.000195%,
VSS certified=1 both engines both dt. Shams 2006 serum-ratio bands are
readout comparison only, not calibration.

## 6. External comparison or explicit pendency

- Shams 2006 bands: comparison only → PENDENCY (no validation claim).
- Fig B.3 regeneration IN PROGRESS under SIO-502 (Claude); legend value
  0.030624 STALE per SIO-505 (nucleus changed 26/09).

## 7. Three columns

- Computational: VERIFIED (cases 10-13; clamp injection booked in ODV,
  residual ~1e-12, SIO-504).
- External evaluation: PENDING (Shams comparison only; B.3 regen open).
- Permitted claim: certified computation of the XR→ODV chain; no PGx
  predictive claim while ratios are unsourced.

## 8. Mechanism coverage

Controlled-release matrix + CYP2D6 formation + PGx readout, all executed
in `scenarios/venlafaxine_xr.sio`. Covered by executed model.

## 9. Figure/table link

Gate cases 10-13; SIO-504 addendum; SIO-505 (legend stale); SIO-502 (B.3).

## 10. Versions

Kernel: `tsit5_pbpk28.sio::pbpk28_full_cn_step_sink_mut` (clamped) via
scenario; ref `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio`.
Re-run green at 55c141ce.
