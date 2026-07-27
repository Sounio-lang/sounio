# Paediatric PBPK — functional maturation + AUC-guided vanco + gentamicin

**Date:** 2026-07-27  
**Status:** live (v2: preterm + AUC + gentamicin)  
**Module:** `stdlib/clinical/pediatric_pbpk.sio`  
**Demo:** `examples/pediatric_pbpk_demo.sio`  
**Gate:** `scripts/ci/pediatric_pbpk_gate.sh` → `PEDIATRIC_PBPK_GATE_OK` (9/9)

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
