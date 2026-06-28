# Immersive Dissertation PBPK Experience

This recovery-phase surface is a browser-based PBPK demonstration for modified-release venlafaxine XR plus O-desmethylvenlafaxine. It is designed to show how Sounio PBPK can connect release kinetics, CYP2D6 phenotype structure, numerical model traces, and molecular-scale narration without exposing raw observed clinical C(t) arrays in the browser.

## Evidence Boundary

- Current local proof: Canvas depth fallback, browser interaction, screenshot pixel nonblank check, static clinical firewall, and WebGPU contract checks.
- WebGPU target: WGSL compute shader and runtime harness are ready for a GPU host, but WebGPU is not promoted until the hard gate passes.
- Clinical boundary: The visible curves are model replay curves, not pointwise clinical calibration, bioequivalence, or patient-specific dosing advice. All displayed concentration-time profiles are illustrative replays of previously published population parameters; no new parameter estimation or clinical validation is performed or claimed.

## Local Fallback Bundle

```bash
python3 experiments/immersive-dissertation/scripts/run_validation_bundle.py \
  --browser firefox \
  --out-dir /tmp/sounio-immersive-recovery-phase1-validation

python3 experiments/immersive-dissertation/scripts/verify_validation_bundle_summary.py \
  /tmp/sounio-immersive-recovery-phase1-validation/validation-summary.json
```

## Hard WebGPU Promotion Gate

Run this only on a GPU/WebGPU-capable host.

```bash
python3 experiments/immersive-dissertation/scripts/run_validation_bundle.py \
  --require-webgpu \
  --browser chromium \
  --out-dir /tmp/sounio-immersive-validation-webgpu-hard

python3 experiments/immersive-dissertation/scripts/verify_validation_bundle_summary.py \
  --require-webgpu-proof \
  /tmp/sounio-immersive-validation-webgpu-hard/validation-summary.json
```

Promotion requires `WEBGPU_RUNTIME_PASS`, `WEBGPU_PBPK_KERNEL_RUNTIME_PASS`, and `VALIDATION_BUNDLE_SUMMARY_PASS` with no fallback markers.
