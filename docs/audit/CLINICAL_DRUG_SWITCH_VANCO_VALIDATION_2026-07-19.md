<!-- docs:meta
topic_id: repo.docs.audit.clinical-drug-switch-vanco-validation-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.clinical-drug-switch-vanco-validation-2026-07-19
-->

# Clinical model-validation drug switch — 2026-07-19

## Decision

| | |
|---|---|
| **Was** | PBPK28 rapamycin predicted-vs-observed (Ferron 1997 C(t)) |
| **Is** | **Vancomycin** 2-compartment IV + ASHP endpoint validation |

## Why switch

Web + MCP search for Ferron 1997 *digitizable whole-blood C(t)*:

- Correct paper is **Clin Pharmacol Ther 1997;61:416–428** (PMID 9129559), **not** 61:696–708 (wrong paper in scaffold comment).
- Abstract gives popPK parameters (CL/F, t½, ka) — **no free C(t) table**.
- Full text is paywalled (Wiley/Ovid). Firecrawl MCP unauthenticated in this environment.
- Secondary open sources do not reproduce Ferron concentration-time points.

Vancomycin has:

- Open literature bands (Matzke 1984 CL; Roberts 2011 Vc/Vp/Q; ASHP/SIDP/IDSA 2020 AUC/Cmax/trough).
- In-repo 2-comp IV model + selftest `vancomycin_icu_pbpk.sio` (6/6 PASS).
- TDM decision vertical already gated (`clinical_vanco_tdm_e2e_gate.sh`).

## Gates

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/clinical_vanco_model_validation_e2e_gate.sh
# → CLINICAL_VANCO_MODEL_VALIDATION_E2E_OK

bash scripts/clinical_vanco_tdm_e2e_gate.sh
# → CLINICAL_VANCO_TDM_E2E_GATE_OK  (decision / GUM budget)
```

## Rapamycin path

Keep `pbpk28_rapamycin_clinical.sio` as **PENDING** until PDF digitization exists. Do not fabricate C(t). Optionally fix citation to PMID 9129559 in a follow-up.
