# GPU Compiler Contracts (Phase 0 Baseline)

This document defines the canonical machine-readable contracts for the 2026
GPU compiler rollout.

## Scope

Artifacts covered:

- `artifacts/omega/gpu_codegen_parity.v1.json`
- `artifacts/omega/gpu_binary_attestation.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`

Primary scripts:

- `scripts/omega/omega_gpu_codegen_parity_gate.sh`
- `scripts/omega/omega_gpu_binary_attest_gate.sh`
- `scripts/omega/omega_gpu_runtime_attest_gate.sh`

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
- `parity` summary object
- `blockers[]`

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
- `blockers[]`

## `gpu_runtime_attest_gate.v1.json` Extensions

The runtime attestation contract now includes:

- `target_profiles[]` copied from codegen parity outputs
- `binary_provenance.entries[]` copied from binary attestation outputs
- `hash_chain` copied from binary attestation outputs

This keeps runtime proof tied to concrete codegen and binary materialization
artifacts.
