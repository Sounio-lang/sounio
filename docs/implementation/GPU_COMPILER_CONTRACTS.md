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
- `artifacts/omega/gpu_comprehensive_run.v1.json`

Primary scripts:

- `scripts/omega/omega_gpu_codegen_parity_gate.sh`
- `scripts/omega/omega_gpu_binary_attest_gate.sh`
- `scripts/omega/omega_gpu_runtime_attest_gate.sh`
- `scripts/omega/omega_gpu_comprehensive_run.sh`

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

## `gpu_comprehensive_run.v1.json`

Purpose: aggregate native-lane execution status, GPU codegen/attestation/runtime
gates, and real-L4 benchmark outcomes into one optimization-ready run artifact.

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
