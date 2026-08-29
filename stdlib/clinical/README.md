# `stdlib/clinical/`

Clinical digital-twin modules: pharmacokinetic/dosing-safety pipelines with
Knightian-uncertainty (p-box) propagation.

## Working modules

- `vancomycin_pbpk.sio` — vancomycin trough-screening twin (`predict_cmin_knightian`,
  `is_safe_dose`). Mirrors `formal/lean4/SounioVancomycinDosingSafety.lean`.
- `tacrolimus_oral_safety.sio` — tacrolimus C24h dosing-safety twin.
- `aminoglycoside_pbpk.sio` — aminoglycoside PBPK module.
- `biomarker.sio` — see `docs/plans/` for the parallel EEG+fMRI+ORC biomarker line.

## IMPORTANT: build engine

These modules import other modules (`use epistemic::knightian::*`), which hits an
open Madaros native-v2 bug (segfault in multi-module array/box lowering —
see `docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`).

**Always run clinical/PBPK code with:**

```bash
scripts/dev/run_clinical_twin.sh stdlib/clinical/vancomycin_pbpk.sio
```

Do **not** use the default `bin/souc run` (Madaros native-v2) for anything
under `stdlib/clinical/` until that bug is fixed — it will either fail to
compile or, for pure single-file f64 code, silently miscompile (see
`docs/audit/MADAROS_F64_NATIVE_V2_BUGREPORT.md`).
