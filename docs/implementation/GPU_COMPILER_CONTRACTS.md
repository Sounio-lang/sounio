<!-- docs:meta
topic_id: repo.docs.implementation.gpu-compiler-contracts
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.gpu-compiler-contracts
-->

# GPU Compiler Contracts (Phase 0 Baseline)

This document defines the canonical machine-readable contracts for the 2026
GPU compiler rollout.

## Scope

Artifacts covered:

- `artifacts/omega/gpu_codegen_parity.v1.json`
- `artifacts/omega/gpu_binary_attestation.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/omega/gpu_public_contract.v1.json`
- `artifacts/omega/gpu_comprehensive_run.v1.json`

Primary scripts:

- `scripts/omega/omega_gpu_codegen_parity_gate.sh`
- `scripts/omega/omega_gpu_binary_attest_gate.sh`
- `scripts/omega/omega_gpu_runtime_attest_gate.sh`
- `scripts/omega/omega_gpu_public_contract_gate.sh`
- `scripts/omega/omega_gpu_comprehensive_run.sh`
- `scripts/gpu/gpu_surface_lowering_gate.sh`
- `scripts/gpu/gpu_compile_proof_gate.sh`
- `scripts/gpu/gpu_sim_runtime_gate.sh`
- `scripts/gpu/gpu_hardware_runtime_gate.sh`
- `scripts/gpu/gpu_capability_gate.sh`

Capability taxonomy reference:

- `docs/implementation/GPU_CAPABILITY_MODEL.md`

The repo now tracks GPU capability through explicit support classes:

- `gpu-surface-supported`
- `gpu-lowering-supported`
- `gpu-compile-proof`
- `gpu-sim-runtime-supported`
- `gpu-hardware-runtime-supported`
- `gpu-explicit-unsupported`

## Shared Status Semantics

- `status_summary` is one of `pass|fail|not_run`.
- `mode` is one of `auto|required|off`.
- All non-pass states include typed `blockers`.
- `required` mode is fail-closed.
- `auto` mode reports `not_run` for non-pass paths.

## Blocker Taxonomy

Canonical blocker classes:

- `target_unavailable`
- `isa_encode_unsupported`
- `binary_pack_fail`
- `driver_reject`
- `parity_fail`
- `perf_regression`
- `attestation_invalid`
- `ssh_unreachable`
- `remote_env_missing`
- `pinned_version_mismatch`
- `gpu_backend_unavailable`
- `runtime_test_fail`
- `public_contract_mismatch`
- `public_example_fail`
- `public_ptx_emit_fail`
- `native_lane_missing`
- `native_lane_compile_fail`
- `native_lane_runtime_fail`
- `native_lane_parity_fail`
- `native_lane_perf_regression`

## `gpu_codegen_parity.v1.json`

Purpose: compile the same fixture across CUDA + ROCm target lanes and record
per-target codegen output status.

Required top-level fields:

- `schema = "sounio.omega.gpu_codegen_parity.v1"`
- `generated_at_utc`
- `mode`
- `status_summary`
- `strict_parity` (bool)
- `targets[]` with per-lane status + output hash
- `targets[].compile_mode|provenance|packer_schema` for fallback provenance
- `native_lane_matrix_artifact` path to stdlib/native lane matrix source
- `native_lanes[]` with status for `onn|qnn|snn|spnn|quantnn|hyper_math|exceptional`
- `parity` summary object
- `blockers[]`

Bootstrap note:

- `packed_from_ptx` is an allowed transitional status when native binary
  emission is unavailable for a lane.
- CUDA fallback schema: `sounio.omega.cuda-packer.v1`
- ROCm fallback schema: `sounio.omega.rocm-packer.v1`

## `gpu_binary_attestation.v1.json`

Purpose: build cryptographic provenance from codegen outputs.

Required top-level fields:

- `schema = "sounio.omega.gpu_binary_attestation.v1"`
- `generated_at_utc`
- `mode`
- `status_summary`
- `source_parity_artifact`
- `binaries[]` (`path`, `size_bytes`, `sha256`, target metadata)
- `hash_chain` (`algorithm`, `entries[]`, `head`)
- `target_provenance[]`
- `native_lanes[]` mirrored from parity artifact for provenance continuity
- `blockers[]`

## `gpu_runtime_attest_gate.v1.json` Extensions

The runtime attestation contract now includes:

- `target_profiles[]` copied from codegen parity outputs
- `native_lanes[]` copied from codegen parity outputs
- `binary_provenance.entries[]` copied from binary attestation outputs
- `hash_chain` copied from binary attestation outputs

This keeps runtime proof tied to concrete codegen and binary materialization
artifacts.

## `gpu_public_contract.v1.json`

Purpose: prove the shipped GPU profile still matches the public docs and
examples instead of relying only on backend internals.

Required top-level fields:

- `schema = "sounio.omega.gpu_public_contract.v1"`
- `generated_at_utc`
- `mode`
- `status_summary`
- `reason`
- `profile` with `souc_path`, `souc_version`, and `public_surface`
- `checks[]` covering:
  - `souc info` reports GPU codegen enabled and JIT disabled
  - `build --help` advertises `--backend gpu`
  - public GPU examples pass `check`
  - public GPU examples emit PTX through `build --backend gpu`
  - `GPU.launch` / `GPU.sync` check successfully
  - unresolved `gpu.thread_id` / `gpu.alloc` remain classified as not-public in
    the checked artifact
- `support_evidence` linking the parity, binary-attestation, and runtime
  attestation artifacts
- `blockers[]`

## `gpu_comprehensive_run.v1.json`

Purpose: aggregate native-lane execution status, GPU codegen/attestation/runtime
gates, the public-contract gate, and real-L4 benchmark outcomes into one
optimization-ready run artifact.

Required top-level fields:

- `schema = "sounio.omega.gpu_comprehensive_run.v1"`
- `generated_at_utc`
- `mode`
- `status_summary`
- `reason`
- `environment` (`gpu_host`, `gpu_user`, `remote_dir`)
- `artifacts` map to source gate artifacts
- `steps[]` with per-step `status`, `reason`, `rc`, and typed `blockers[]`
- `hotspots[]` with optimization targets inferred from lane and benchmark signals
- `blockers[]`
