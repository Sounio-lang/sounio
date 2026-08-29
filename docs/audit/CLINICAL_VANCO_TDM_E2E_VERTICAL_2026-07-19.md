<!-- docs:meta
topic_id: repo.docs.audit.clinical-vanco-tdm-e2e-vertical-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.clinical-vanco-tdm-e2e-vertical-2026-07-19
-->

# Vancomycin AUC/MIC TDM decision E2E — 2026-07-19

## Scope

| | |
|---|---|
| Branch | `feat/clinical-vanco-tdm-e2e` |
| Driver | `tests/stdlib/clinical/test_vanco_auc_tdm_e2e.sio` |
| Module | `stdlib/darwin_pbpk/pd/vancomycin_auc_gum.sio` (selftest ALL PASS) |
| Gate | `bash scripts/clinical_vanco_tdm_e2e_gate.sh` → `CLINICAL_VANCO_TDM_E2E_GATE_OK` |
| Engine | lean_single |

## Claim

GUM propagates seven uncertainty sources to AUC24/MIC; a Knightian p-box over ±1σ_CL returns **Recommend / Adjust / Refuse**. Same mg/kg dose is **Refuse** for CrCl=20 and not Refuse for CrCl=90. Compile-fail confidence witnesses remain in-tree.

## claims_not_made

bedside product · NONMEM FOCE · Madaros multi-module · MIMIC calibration · numpy/sklearn
