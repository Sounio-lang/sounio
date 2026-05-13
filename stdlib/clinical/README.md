# `stdlib/clinical/`

Skeleton for the clinical / neurobiological extension of the Sounio
platform.  This directory is intentionally minimal: it only hosts
import-ready placeholders and documentation for the *parallel* plan
covering the depression-biomarker line of work (EEG + fMRI + Ollivier–
Ricci curvature).

## Intended layout

```
stdlib/clinical/
├── README.md                  # this file
├── eeg.sio                    # EEG ingestion + channel algebra (TBD)
├── fmri.sio                   # fMRI ROI volumes + motion correction (TBD)
├── orc.sio                    # Ollivier–Ricci curvature on connectomes (TBD)
└── biomarker.sio              # pipeline wiring EEG+fMRI+ORC into a single
                                # epistemic classifier (TBD)
```

## Why empty now

The surgical-interventions program (G3/G5/G7) and the biomarker program
are orthogonal contributions that share the same algebraic substrate
(sedenion ZD).  Keeping them in separate plans lets each program land
independently, without coupling compiler changes to clinical data
formats.

Nothing in this directory is imported or compiled in the current
release.  Touching these files should be the explicit scope of the
biomarker plan (see `docs/plans/` when it lands).
