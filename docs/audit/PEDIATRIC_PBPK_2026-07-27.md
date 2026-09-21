<!-- docs:meta
topic_id: repo.docs.audit.pediatric-pbpk-2026-07-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pediatric-pbpk-2026-07-27
-->

# Paediatric PBPK — functional maturation + AUC-guided vanco + gentamicin

**Date:** 2026-07-27  
**Status:** live (v6: noisy TDM ladder + neonate accumulation)  
**Module:** `stdlib/clinical/pediatric_pbpk.sio`  
**Receipt:** `tests/run-pass/pediatric_pbpk_receipt.sio` (18 self-tests)  
**Demo:** `examples/pediatric_pbpk_demo.sio` (16 base tests; examples/** may lag under parallel claims)  
**Gate:** `scripts/ci/pediatric_pbpk_gate.sh` → `PEDIATRIC_PBPK_GATE_OK`

## Scope

Educational / compiler-stdlib **paediatric PBPK spine** with:

1. Holford size allometry `(WT/70)^0.75` (CL) and `WT/70` (V)
2. Rhodin 2009 renal maturation on PMA (TM50=47.7 wk, Hill=3.4)
3. Hepatic CYP-like maturation (second organ axis)
4. Mosteller BSA + Schwartz bedside eGFR helper
5. Vancomycin 1-cmt SS Cmin + AUC24, GUM `Epistemic`, Knightian `PBox`
6. **Preterm GA28** cohort (PMA 30 wk) vs term neonate
7. **AUC-guided** dose for mid-target 500 mg·h/L + AUC p-box (↓CL only)
8. **Gentamicin** on the same renal spine (Vc 0.25 L/kg, CL∝GFR)
9. **Amikacin** (Vc 0.27 L/kg, CL∝GFR, trough screen 4–8 mg/L)
10. **Neonate interval compare**: fixed 30 mg/kg/day as q6h vs q12h
    (Cmin_q6 > Cmin_q12, Cmax_q6 < Cmax_q12, AUC24 equal)
11. **SS C(t) profile** within a dosing interval (≤16 points)
12. **IIV Monte-Carlo** N=32 lognormal CL/V (ω_CL=0.25, ω_V=0.20, seed=42)
13. **Literature anchors**: size(10 kg)≈0.2324; MF(PMA 40)∈(0.30,0.45)
14. **IIV grid cohort** (deterministic z∈[-2,2], not RNG — lean-safe)
15. **Fixed vs AUC-guided** under IIV — %AUC in 400–600 (perfect-CL upper bound)
16. **Multi-dose accumulation** — Cmin_n/Cmin_ss → 1; doses to 90% SS
17. **Noisy TDM** — fixed < noisy-CL guided < perfect-CL under IIV+assay error
18. **Neonate/preterm accumulation** — multi-dose build-up with r∈(0,1)

**Not medical guidance.**

## Expected physiology (15 mg/kg q12h vancomycin)

| Subject | GA | Weight | MF_renal | CL (order) |
|---|---:|---:|---:|---|
| Preterm 2 wk | 28 | 1.2 kg | **lowest** | **lowest** |
| Term neo 2 wk | 40 | 3.5 kg | low | low |
| Child 5 yr | — | 20 kg | ~1 | mid |
| Adult 70 kg | — | 70 kg | ~1 | Matzke@GFR120 |

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pediatric_pbpk_demo.sio
bash scripts/ci/pediatric_pbpk_gate.sh
```
