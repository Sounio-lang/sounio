# PBPK-28 GPU parity: M12 serve hold bound + oral GUM arms

## Direction

Lane `pbpk-28-gpu-parity` runs the whole PBPK-28 series on GPU alongside the
LLM serve series (M5-M12). This slice adds the M12 starvation bound on the
serve side and the three oral GUM drug arms on the PBPK side, closing the
last PENDING arms of the dissertation parity gate.

## Slice

- Serve M12: absolute hold bound `SVB_MAX_HOLD` (32 passes) instead of linear
  aging; per-session `swait[8]` counters, M11 guard intact plus bound
  exception. `stdlib/llm/serve.sio` + 5 run-pass tests.
- Oral GUM arms on the existing `gut0` mold: venlafaxine XR 75 mg/168 h,
  midazolam IR 5 mg/168 h, haloperidol IR 5 mg/72 h with D2 receipt via the
  Kpuu bridge (no new struct types in the parity binary).
- Citation identity: `CITATION.cff` gains the verified v3.0.0-rc.2 Zenodo DOI;
  `README.md` links the persistent all-versions DOI.

## Receipts 2026-11-10

- `dissertation_pbpk28_gpu_parity_gate.sh`: 16/16, zero PENDING.
- `serve_prio` 4/4 modes PASS with causal signature; preempt/pump/batch no
  regressions; `souc check` OK.
- `cffconvert --validate`: citation metadata valid per schema 1.2.0.

## Notes

- Haloperidol readers stay inline literals until codegen issue #3495 is fixed
  (new struct types in the parity binary break bitwise parity; bisect on file).
- Live-5860 receipt still pending (same situation as M11).
- Final v3.0.0 DOI + SWHID are founder-side (no final tag yet; origin not
  archived in Software Heritage).
