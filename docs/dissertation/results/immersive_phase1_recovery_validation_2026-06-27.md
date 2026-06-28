<!-- docs:meta
topic_id: repo.docs.dissertation.results.immersive-phase1-recovery-validation-2026-06-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.immersive-phase1-recovery-validation-2026-06-27
-->

# Immersive PBPK Dissertation Experience - Phase 1 Recovery Validation

Date: 2026-06-27

## Scope

This document records the first reconstructed phase after the original `/tmp` worktree was removed. The source surface now lives in the persistent worktree `/workspace/sounio-qual-recovery-20260627` under `experiments/immersive-dissertation/`.

## What Was Rebuilt

- Browser experience for venlafaxine XR plus O-desmethylvenlafaxine, including modified-release capsule, model C(t) curves, CYP2D6 conversion, and visible clinical firewall.
- WGSL compute shader for modified-release parent/metabolite replay.
- WebGPU kernel contract and render-quality contract with explicit claim boundaries.
- Validation bundle scripts for static checks, browser interaction, screenshot pixel proof, WebGPU runtime probe, PBPK compute-kernel runtime probe, and persisted summary verification.

## Claim Boundary

The local fallback bundle may prove that the demonstration is visible and interactive on a headless browser. It does not prove WebGPU execution, photorealism, observed clinical calibration, bioequivalence, or patient-specific dosing utility. All displayed concentration-time profiles are illustrative replays of previously published population parameters; no new parameter estimation or clinical validation is performed or claimed. WebGPU promotion requires the hard gate in `experiments/immersive-dissertation/README.md`.

## Required Evidence

The phase is acceptable only if the following pass locally:

- JSON and script syntax checks.
- `WEBGPU_PBPK_KERNEL_CONTRACT_PASS`.
- `GPU_PROMOTION_CONTRACT_PASS`.
- `IMMERSIVE_EXPERIENCE_STATIC_PASS`.
- `IMMERSIVE_BROWSER_INTERACTION_PASS`.
- `SCREENSHOT_PIXEL_PASS`.
- fallback-allowed WebGPU runtime probes.
- `IMMERSIVE_VALIDATION_BUNDLE_PASS`.
- `VALIDATION_BUNDLE_SUMMARY_PASS`.

The hard verifier must still fail on a non-WebGPU host if `--require-webgpu-proof` is used against fallback evidence.

## Local Run Result

Output directory:

```text
/tmp/sounio-immersive-recovery-phase1-validation-firefox
```

Commands:

```bash
python3 experiments/immersive-dissertation/scripts/run_validation_bundle.py \
  --browser firefox \
  --out-dir /tmp/sounio-immersive-recovery-phase1-validation-firefox

python3 experiments/immersive-dissertation/scripts/verify_validation_bundle_summary.py \
  /tmp/sounio-immersive-recovery-phase1-validation-firefox/validation-summary.json
```

Observed markers:

```text
IMMERSIVE_VALIDATION_BUNDLE_PASS
VALIDATION_BUNDLE_SUMMARY_PASS
```

The persisted summary reports eight checks and three artifacts:

- `/tmp/sounio-immersive-recovery-phase1-validation-firefox/immersive-screenshot.png`
- `/tmp/sounio-immersive-recovery-phase1-validation-firefox/webgpu-runtime.json`
- `/tmp/sounio-immersive-recovery-phase1-validation-firefox/webgpu-pbpk-kernel-runtime.json`

Machine-readable summary fields:

```text
webgpu_promotion_eligible=false
source_revision.short_head=bb970ea3a
claim_boundary includes no new parameter estimation or clinical validation
```

## Negative Promotion Check

Command:

```bash
python3 experiments/immersive-dissertation/scripts/verify_validation_bundle_summary.py \
  --require-webgpu-proof \
  /tmp/sounio-immersive-recovery-phase1-validation-firefox/validation-summary.json
```

Observed marker:

```text
HARD_PROOF_RC=1
```

This is the expected result for the current host. The summary contains fallback markers (`WEBGPU_RUNTIME_NOT_AVAILABLE` and `WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE`), so it cannot promote WebGPU execution, photorealism, or GPU-host claims.
