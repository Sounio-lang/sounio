<!-- docs:meta
topic_id: repo.docs.audit.clinical-midazolam-ddi-e2e-vertical-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.clinical-midazolam-ddi-e2e-vertical-2026-07-19
-->

# Midazolam CYP3A DDI E2E — 2026-07-19

## Scope

| | |
|---|---|
| Branch | `feat/clinical-midazolam-ddi-e2e` |
| Module | `stdlib/darwin_pbpk/validation/midazolam_ddi.sio` (6/6 PASS) |
| Driver | `tests/stdlib/clinical/test_midazolam_ddi_e2e.sio` |
| Gate | `bash scripts/clinical_midazolam_ddi_e2e_gate.sh` → `CLINICAL_MIDAZOLAM_DDI_E2E_GATE_OK` |
| Engine | **lean_single** multi-module |

## Claim

Mechanistic CYP3A well-stirred first-pass + competitive inhibition reproduces:

- oral **F ∈ (0.30, 0.45)** (Heizmann 1984)
- ketoconazole oral **AUCR ∈ (12, 18)** ≈15× (Olkkola 1994)
- gut contribution required for full AUCR (Thummel 1996)
- PGx ordering CYP3A5 expresser > non-expresser; CYP3A4\*22 slower

Complements vancomycin PK/TDM validation with a **DDI mechanism** vertical.

## claims_not_made

full C(t) digitization · bedside DDI product · Madaros multi-module · NONMEM FOCE · numpy/sklearn
