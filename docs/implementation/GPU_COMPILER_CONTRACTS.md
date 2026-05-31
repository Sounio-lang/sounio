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
- Kretikos local artifact bundles emitted by `bin/kretikos bundle`

Primary scripts:

- `scripts/omega/omega_gpu_codegen_parity_gate.sh`
- `scripts/omega/omega_gpu_binary_attest_gate.sh`
- `scripts/omega/omega_gpu_runtime_attest_gate.sh`
- `scripts/omega/omega_gpu_public_contract_gate.sh`
- `scripts/omega/omega_gpu_comprehensive_run.sh`
- `bin/kretikos`

## Shared Status Semantics

- `status_summary` is one of `pass|fail|not_run`.
- `mode` is one of `auto|required|off`.
- All non-pass states include typed `blockers`.
- `required` mode is fail-closed.
- `auto` mode reports `not_run` for non-pass paths.

## CUDA toolchain provenance (`toolchain` field)

GPU artifacts pin `souc_version`, but ptxas codegen and cuBLAS baselines change
across CUDA toolkit major versions — so a measured/attested result is only
comparable when the **toolchain that produced it** is recorded alongside it. As of
2026-05-30 **all GPU attestation artifacts carry it** — `gpu_codegen_parity.v1`,
`gpu_binary_attestation.v1`, `gpu_runtime_attest_gate.v1`, `gpu_public_contract.v1`,
`gpu_comprehensive_run.v1`, and the `l4-optimized-no-rust.v2` perf report — a
`toolchain` object captured on the host that ran ptxas / the kernels:

```json
"toolchain": {
  "captured_at_utc": "…Z", "captured_on": "user@host",
  "capture_status": "ok|partial|unavailable|not_captured",
  "cuda_release": "13.3", "nvcc_version": "…", "ptxas_version": "…",
  "driver_version": "610.43.02", "gpu_name": "Quadro RTX 8000",
  "gpu_compute_cap": "7.5", "cuda_version_json": "…", "probe_host": "…"
}
```

- Single source of truth: `scripts/omega/omega_capture_toolchain.sh`
  (`omega_capture_toolchain.sh user@host` over ssh; bare = local). Always emits a
  well-formed object; absent tools → `""` and a downgraded `capture_status`.
- `not_captured` means the host was unreachable / the gate short-circuited before
  capture — it never blocks a gate.
- Companion to the existing kretikos `toolchain_validation` block (which records
  *whether* ptxas/nvdisasm validated; this records *which versions*).
- Wiring: the producing gates (`codegen_parity`, `binary_attestation`,
  `runtime_attest`, perf bench) capture directly (env `OMEGA_GPU_TOOLCHAIN_JSON`
  overrides → local/remote capture → `not_captured`). The aggregators
  (`public_contract`, `comprehensive_run`) surface it from whichever sub-artifact
  captured a real value (runtime → parity → binary).

## `sounio.kretikos.bundle.v1`

Purpose: show the local Kretikos CLI can emit both GPU artifact families from
in-tree Sounio emitters, record hashes, and keep structural evidence separate
from runtime/assembler validation.

Produced by:

```bash
bin/kretikos bundle -o /tmp/kretikos-bundle
bin/kretikos bundle -o /tmp/kretikos-validated-bundle --validate-toolchain --validate-runtime
bin/kretikos run-source examples/kretikos/real_vec_add.sio -o /tmp/kretikos-source --validate-toolchain --validate-runtime
bin/kretikos run-manifest examples/kretikos/manifest.tsv -o /tmp/kretikos-corpus --validate-toolchain --validate-runtime
slurm-jobs/kretikos/submit-kretikos-source.sh examples/kretikos/real_vec_add.sio
slurm-jobs/kretikos/submit-kretikos-manifest.sh examples/kretikos/manifest.tsv
```

Required top-level fields:

- `schema = "sounio.kretikos.bundle.v1"`
- `status = "emitted"`
- `generated_at_utc`
- `compiler`
- `stdlib`
- `contract` with separate PTX, CUBIN, and runtime-boundary descriptions
- `structural_checks` with structural-check class, CUBIN ELF machine check
  (`e_machine = 190`), and unavailable-tool status
- `toolchain_validation` with `mode`, `ptxas`, and `nvdisasm` records
- `runtime_validation` with `mode`, CUDA Driver API status, selected runtime
  rung, kernel name, and log path
- `artifacts[]` containing one PTX record and one CUBIN record
- `artifacts[].sha256`
- `boundaries[]` listing non-claims

By default this bundle is structural. Optional validation modes are:

- `--validate-toolchain`: run `ptxas`/`nvdisasm` when available and record
  exact `not_run` reasons otherwise.
- `--require-toolchain`: fail closed unless `ptxas` and `nvdisasm` validate.
- `--validate-runtime`: build `scripts/gpu/nvidia_bare_driver_loader.c` and
  attempt the selected CUDA Driver API rung when a driver/device is available.
  Hosts without a C compiler may set `SOUNIO_KRETIKOS_RUNTIME_LOADER` to an
  executable prebuilt loader.
- `--require-runtime`: fail closed unless that runtime rung passes.

Local non-GPU hosts must report exact reasons such as `ptxas_missing`,
`nvdisasm_missing`, `c_compiler_missing`, or `cuda_driver_missing`; they must
not silently promote structural evidence to runtime proof. Toolchain and
runtime validation apply only to the predefined PTX/CUBIN artifact templates
selected by the bundle. They do not validate arbitrary user-written kernels.

## Real Source Profile Lane

`bin/kretikos run-source <source.sio>` is the first ready-to-run source path.
It checks the Sounio file first, classifies it into a supported Kretikos source
profile, emits the matching owned CUBIN bundle, and runs the selected CUDA
Driver API validation rung when runtime validation is enabled.

Supported profiles are intentionally explicit:

- `vec_add_f32`
- `vec_add_f64`
- `vec_sub_f64`
- `vec_mul_f64`
- `vec_div_f64`
- `fma_f64`
- `vec_sub_f32`
- `vec_mul_f32`
- `vec_div_f32`
- `fma_f32`
- `epistemic_elementwise_f32`
- `epistemic_dual_output_f32`
- `store_u32_const`

These thirteen profiles are runtime-backed: each maps to an owned CUBIN runtime
rung and may pass `--require-runtime` on a CUDA Driver API host.

Kretikos may later accept a wider structural-only source set. Structural-only
profiles must still typecheck as Sounio source and must emit owned PTX through
`bin/kretikos emit-ptx`, but they intentionally report
`runtime_validation.status = "not_run"` and
`runtime_validation.reason = "profile_not_runtime_backed"`. They fail closed
under `--require-runtime` until an owned CUBIN/runtime rung exists.

A source file may declare the profile with:

```sounio
// kretikos: profile=vec_add_f32
```

The Slurm entrypoint is:

```bash
slurm-jobs/kretikos/submit-kretikos-source.sh examples/kretikos/real_vec_add.sio
slurm-jobs/kretikos/submit-kretikos-manifest.sh examples/kretikos/manifest.tsv
```

That script embeds the source, Kretikos, stdlib slice, CUBIN emitters, and a
prebuilt CUDA Driver API loader into the sbatch payload. It runs from GPU-worker
local scratch and records the runtime verdict in the Slurm job comment. This is
ready for real source files that fit one of the profiles above; it is not yet a
claim of arbitrary Sounio GPU lowering.

`examples/kretikos/manifest.tsv` is the canonical corpus smoke. It covers every
runtime-backed source profile and is the promotion gate before claiming a new
profile is ready for user code.

`vec_sub_f32` currently uses the same owned affine elementwise kernel ABI as
`vec_add_f32`; the runtime oracle sets epsilon to `-2.0`, so the GPU computes
`x + y * (1 + epsilon) = x - y`. This is a runtime-backed CUBIN rung, not a
claim of arbitrary subtraction lowering.

`fma_f32` uses that same owned affine elementwise kernel ABI with a nonzero
epsilon lane. The runtime oracle interprets the inputs as `c + a * b` by
setting `c = x`, `a = y`, and `b = 1 + epsilon`. This is a runtime-backed
affine FMA rung, not a claim of general FMA lowering for arbitrary source
expressions.

`vec_mul_f32` uses that same owned affine elementwise kernel ABI with `x = 0`,
`y = a`, and `epsilon = b - 1`, so the GPU computes `a * b`. This is a
runtime-backed multiplication rung, not a claim of arbitrary source
multiplication lowering.

`vec_div_f32` uses that same owned affine elementwise kernel ABI with `x = 0`,
`y = numerator`, and `epsilon = (1 / denominator) - 1`, so the GPU computes
`numerator / denominator`. This is a runtime-backed division rung, not a claim
of arbitrary source division lowering.

The f64 vector rungs use owned SM80 CUBINs assembled from Kretikos PTX and
embedded in `self-hosted/gpu/nvidia_bare.sio` as canonical byte chunks.
`vec_add_f64`, `vec_sub_f64`, `vec_mul_f64`, and `vec_div_f64` use a
three-parameter double ABI: `x`, `y`, and `out`. Their PTX front doors emit
`ld.global.f64`, the selected f64 arithmetic opcode, and `st.global.f64`. The
CUDA Driver API oracle checks the selected operation against a double-precision
CPU oracle. These are runtime-backed vector rungs, not a claim of arbitrary f64
source lowering.

`fma_f64` uses the four-parameter double ABI: `a`, `b`, `c`, and `out`. Its PTX
front door emits `fma.rn.f64`, and the CUDA Driver API oracle checks
`out[i] = a[i] * b[i] + c[i]` with a double-precision tolerance. This is a
runtime-backed fused multiply-add rung, not a claim of general f64 expression
lowering.

`examples/kretikos/structural_manifest.tsv` covers any remaining
structural-only PTX profiles. It is currently empty beyond its header.

The executable f64 local gate is:

```bash
scripts/ci/kretikos_f64_runtime_gate.sh
```

That gate proves the Sounio source checks for every runtime-backed f64 profile,
the f64 PTX templates are emitted, the owned CUBINs are CUDA ELF64 with the
expected `sounio_bare_vec_*_f64_sm80` symbols, and the Kretikos bundle selects
the matching f64 runtime rung. The full runtime proof is the Slurm/L4 manifest
lane with `--require-runtime`.

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
- `profile_not_runtime_backed`
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
