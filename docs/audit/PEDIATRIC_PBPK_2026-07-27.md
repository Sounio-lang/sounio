# Paediatric PBPK — functional maturation + vancomycin receipt

**Date:** 2026-07-27  
**Status:** live  
**Module:** `stdlib/clinical/pediatric_pbpk.sio`  
**Demo:** `examples/pediatric_pbpk_demo.sio`  
**Gate:** `scripts/ci/pediatric_pbpk_gate.sh` → `PEDIATRIC_PBPK_GATE_OK`

## Scope

Educational / compiler-stdlib **paediatric PBPK spine** with:

1. Holford size allometry `(WT/70)^0.75` (CL) and `WT/70` (V)
2. Rhodin 2009 renal maturation on PMA (TM50=47.7 wk, Hill=3.4)
3. Hepatic CYP-like maturation (second organ axis)
4. Mosteller BSA + Schwartz bedside eGFR helper
5. Vancomycin 1-cmt SS Cmin + AUC24, GUM `Epistemic`, Knightian `PBox`

**Not medical guidance.** Adult trough window 10–20 mg/L is an educational screen only.

## Expected physiology (same 15 mg/kg q12h)

| Subject | Weight | MF_renal | Relative CL | Relative Cmin |
|---|---:|---:|---|---|
| Neonate ~2 wk | 3.5 kg | low | lowest | **highest** |
| Child 5 yr | 20 kg | near mature | mid | mid |
| Adult 70 kg | 70 kg | ~1 | Matzke@GFR120 | reference |

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pediatric_pbpk_demo.sio
bash scripts/ci/pediatric_pbpk_gate.sh
```
