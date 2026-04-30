# scripts/clinical/

M4 milestone — vancomycin TDM cohort processing.

## Files

- `process_tdm_cohort.sh` — driver shell script. Reads a CSV cohort and
  invokes the Sounio Knightian pipeline per patient.
- `data_synthetic/tdm_cohort_synthetic_v1.csv` — 20-patient synthetic
  cohort with realistic age/weight/CrCl/dose/Cmin distributions, used
  for skeleton testing while IRB / MIMIC-IV ETL is in flight.
- `runs/<timestamp>/` — output of each pipeline invocation.

## Status

**Skeleton (M4 stage).** The pipeline structure is in place but the
following pieces are deferred until real data lands:

1. **JSON-emit mode** in `stdlib/epistemic/knightian.sio` to replace
   the current concatenated-stdout output with parseable per-patient
   records. Tracked as M4 follow-up.
2. **MAE / coverage analyzer** (`compute_mae.py` or Sounio-native
   equivalent) over predicted vs measured Cmin. Cannot meaningfully
   run on synthetic data; deferred to real cohort.
3. **SOTA comparator** — wrap a NONMEM/pmetrics call from this
   script for paired-test analysis. Requires institutional licence;
   deferred to M5 with full cohort.
4. **Lean theorem closure** — the predictions emitted here will
   feed `formal/lean4/SounioVancomycinDosingSafety.lean`'s
   `cmin_within_implies_efficacy_and_safety` theorem at the
   instance level (one Lean obligation per patient-timepoint).

## Synthetic data caveat

`tdm_cohort_synthetic_v1.csv` is **not** real patient data. The 20
records were hand-crafted to exhibit:

- realistic adult ICU age/weight/sex distribution
- CrCl spread spanning AKI risk (18-135 mL/min)
- Cmin spread spanning subtherapeutic / therapeutic / toxic ranges
- realistic SOFA / nephrotoxic co-exposure / outcome correlations

Do not draw inferential conclusions from analyses on this file.
It exists to exercise the pipeline plumbing during IRB lead-time.
