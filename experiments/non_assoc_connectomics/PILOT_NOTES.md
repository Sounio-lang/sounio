# Pilot Notes — Phase 1

**Date**: 2026-04-12
**Mode**: SYNTHETIC SELF-TEST (frames.bin not available in sandbox)

## Gate status

| Gate | Status | Value |
|------|--------|-------|
| `sim_ground_truth.sio` monotone α-recovery | PASS | p95(α=0)=0.000, p95(α=0.5)=3.61, p95(α=1.0)=74.14 |
| `associator_field.sio` synthetic self-test compiles + runs | PASS | 10 records emitted (5 ASD + 5 TD) |
| `zero_divisor_proximity.sio` stub compiles + runs | PASS | d² = 0.89 on synthetic feature vector |
| `abide_fetch.py` exits non-zero with diagnostic | PASS | preprocessor missing `pandas`; exit code 1; message legible |
| `analysis.py` produces PNG + JSON | PASS | `artifacts/research/non_assoc_connectomics_pilot.{png,json}` |
| Bootstrap fixed-point unchanged | PASS | no compiler / stdlib files modified |

## Synthetic-pilot observed statistic

Synthetic fill: ASD nodes drawn with α = 0.7 (fraction of octonion-full labels
vs quaternion-restricted); TD nodes with α = 0.5. Uses insertion-sort p95
over C(30,3)=4,060 triples per subject.

```
n_asd = 5, n_td = 5
mean(p95 | ASD) = 48.79
mean(p95 | TD)  = 29.49
Cohen's d (ASD − TD) = +1.803
95% bootstrap CI      = [+0.757, +7.035]
CI crosses zero       = False
KS two-sample D       = 0.800,  p = 0.0794
```

**Interpretation**: on synthetic data where the ASD group is constructed to
have more non-associative content (higher α), the pipeline recovers a large
Cohen's d with a CI that excludes zero. This validates the pipeline's ability
to detect the effect it is designed to detect. It is **not** evidence of any
real-world ASD/TD difference — the synthetic labels were designed to produce
this result. The real test awaits ABIDE-I data.

## What is blocked

- Real 10-subject pilot requires `artifacts/research/abide/frames.bin`,
  which requires running `scripts/research/abide_preprocess.py` with network
  access + `pandas`, `numpy`, `scipy` installed. The sandbox has numpy+scipy
  but not pandas; the CC200 ROI time series (`*_rois_cc200.1D`) are also not
  cached locally. See memory note *project_g2_bridge.md* — the full
  n=1,034 fetch has been done before on the canonical dev machine, but
  needs re-running here.
- Phase 2 power analysis from the real pilot's |d| estimate.

## To advance to real pilot

On a machine with network + full Python deps:
```bash
pip install pandas numpy scipy
python3 experiments/non_assoc_connectomics/abide_fetch.py
./bin/souc run experiments/non_assoc_connectomics/associator_field.sio > pilot.csv
python3 experiments/non_assoc_connectomics/analysis.py pilot.csv
```

## Known non-issues

- `associator_field.sio` synthetic self-test emits a trailing `40600` line on
  some builds — appears to be a stdlib diagnostic of total triples processed;
  `analysis.py` skips non-parseable lines so this is benign.
- `print_int` auto-newlines, so all CSV records end with the integer
  `subject_id` field; header order in PROTOCOL.md matches
  `group,mean,p95,n_triples,subject_id`.
