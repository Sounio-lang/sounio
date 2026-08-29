<!-- docs:meta
topic_id: repo.docs.handoff.gpu-knowledge-vecmat-swarm-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.gpu-knowledge-vecmat-swarm-plan
-->

# GPU Knowledge Vec/Mat Swarm Plan

Status: open, not completion-ready.

This handoff keeps the GPU Knowledge Vec/Mat work moving without reserving a GPU.
The current local proof is a ptxas/toolchain and CUDA Driver API harness-shape
proof for a launchable Vec4 aggregate marker. It is not a CUDA device runtime
execution claim.

## Current Evidence

- Probe: `scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh`
- Backend pack/unpack probe: `scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh`
- Backend pack/unpack harness: `self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_harness.sio`
- Backend lean extract fallback: `self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_lean_extract.sio`
- Audit: `scripts/dev/gpu_knowledge_vecmat_evidence_audit.sh`
- Focused gate: `scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh`
- Package verifier: `scripts/dev/gpu_knowledge_vec4_package_verify.py artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json`
- Runtime receipt verifier: `scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass artifacts/gpu/dgx_spark_public_gpu_gate.v1.json`
- Runtime runbook: `scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh all-local`
- Slurm runtime probe: `scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh`
- Imported runtime probe: `scripts/dev/gpu_knowledge_vec4_imported_runtime_probe.sh`
- Completion auditor: `scripts/dev/gpu_knowledge_vecmat_completion_audit.py`
- Local marker package route: `DGX_SPARK_PACKAGE_ONLY=1 DGX_SPARK_PUBLIC_KERNELS=0 DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER=1 DGX_SPARK_RUNTIME=0 bash scripts/dev/dgx_spark_public_gpu_gate.sh`
- Optional DGX marker route: `DGX_SPARK_PUBLIC_KERNELS=0 DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER=1 bash scripts/dev/dgx_spark_public_gpu_gate.sh`
- Probe JSON: `artifacts/gpu/knowledge_vecmat_evidence_audit/ptxas_probe/gpu_knowledge_vec4_ptxas_probe.v1.json`
- Backend probe JSON: `artifacts/gpu/knowledge_vecmat_evidence_audit/backend_pack_unpack_probe/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json`
- Audit JSON: `artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_evidence_audit.v1.json`
- Queue JSON: `artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_swarm_queue.v1.json`
- Dispatch manifest: `artifacts/gpu/knowledge_vecmat_evidence_audit/swarm_dispatch/manifest.v1.json`
- Slurm runtime JSON: `artifacts/gpu/knowledge_vecmat_evidence_audit/slurm_runtime_probe/gpu_knowledge_vec4_slurm_runtime_probe.v1.json`
- Imported runtime JSON: `artifacts/gpu/knowledge_vecmat_evidence_audit/imported_runtime_probe/gpu_knowledge_vec4_imported_runtime_probe.v1.json`

The marker kernel is `gpu_knowledge_vec4_aggregate_marker`. It takes `out_ptr`,
stores f64 values `1,2,3,4` at byte offsets `0,32,64,96`, and emits a CUDA
Driver API runner that uses `cuLaunchKernel` and `cuMemcpyDtoH` to check those
copyback offsets.

The Slurm runtime probe launched the marker PTX on `gpu-orangefs` using a
CUDA Driver API runner with `dlopen(libcuda.so.1)` and PTX JIT, avoiding any
need for `nvcc`, `ptxas`, or CUDA headers on the compute node. Current receipt:
`PASS gpu_knowledge_vec4_aggregate_marker on NVIDIA RTX 4000 Ada Generation cc
8.9 copyback offsets=0,32,64,96 values=1,2,3,4`. This closes the CUDA device
runtime proof for the marker. It does not claim DGX Spark runtime authority.

The backend pack/unpack harness builds `gpu_vec4_pack_unpack` through a direct
Vec4 emitter inside `lower_to_ptx_wmma_lean.sio` and expects f64 loads/stores at
aggregate byte offsets `0,32,64,96`. The first full-module route was blocked
before PTX emission because importing `kernel_ir.sio` lowered large existing GPU
builder functions and reported `IR_MAX_INSTRS` for `gpu_build_gemm_shared_ir`,
`gpu_build_conv2d_ir`, and `gpu_build_epistemic_tiled_gemm_ir`. The lean
`GpuKernelIr.ops` route then exposed a modular runtime ABI problem around
by-value op access. The current direct backend Vec4 emitter avoids that ABI path,
emits valid PTX from Sounio code, validates the offset patterns, and assembles
with local `ptxas`. That closes the scoped automatic backend Vec4 pack/unpack
proof, but it is not a compiler-lowering or CUDA runtime claim.

The diagnostic stage probe `self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_stage_probe.sio`
narrows the blocker further: an empty kernel with no ops can lower/print, but
setting `op_count > 0` and reading `kernel.ops[0]` through the lean lowering path
causes either a runtime segfault or opcode corruption (`GpuRet` reaches the
small lowerer as an unsupported opcode). This points at the modular runtime ABI
for `GpuKernelIr.ops`, not the Vec4 PTX contract itself.

The same probe also runs `gpu_knowledge_vec4_pack_unpack_lean_extract.sio` as a
reference extract fallback. That fallback emits the same Vec4 PTX contract from
small Sounio code, strips compiler logs before `.version`, validates the offset
patterns, and assembles the PTX with local `ptxas`. This fallback is recorded as
`reference_extract_not_production_backend`; it is useful evidence that the PTX
contract and toolchain are sound, but the backend blocker is closed by the direct
emitter probe, not by this fallback.

The imported runtime fixture is now scoped and explicit:
`tests/run-pass/gpu_hlir_vec4_lane_plan_imported.sio` imports
`gpu_hlir_vec4_lane_plan_leaf`, receives a `[f64; 4]` lane plan through the
canonical imported/module path, validates values `1,2,3,4`, and validates the
logical aggregate offsets `0,32,64,96`. The probe log includes the modular
`lower_array` path and prints `PASS gpu_hlir_vec4_lane_plan_imported
offsets=0,32,64,96 values=1,2,3,4`. This closes the scoped imported runtime
fixture. It does not claim that all older imported-runtime known-failures are
fixed.

The DGX route is opt-in. With `DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER=1`, the gate
copies the generated marker PTX and runner to the DGX remote directory, assembles
the marker with the remote `ptxas`, and, when `DGX_SPARK_RUNTIME=1`, compiles and
runs the generated marker runner against the remote cubin. Without that flag, the
public GPU gate keeps its prior VecAdd/Slices behavior.

The local package route is also opt-in. With `DGX_SPARK_PACKAGE_ONLY=1` and
`DGX_SPARK_PUBLIC_KERNELS=0`, the gate writes a marker-only package under
`artifacts/gpu/dgx_spark_public_gpu_package` and exits before SSH. That mode is a
local ptxas/package proof only; it does not claim DGX toolchain authority or CUDA
device runtime.

The DGX Spark marker route also passed on `demetrios@192.168.3.43`
(`spark-8e54`, `aarch64`, NVIDIA GB10, CUDA 13.0, `sm_121`). The runtime receipt
is `artifacts/gpu/dgx_spark_public_gpu_gate.v1.json` with marker status
`runtime_pass` and output `PASS gpu_knowledge_vec4_aggregate_marker on NVIDIA
GB10 cc 12.1 copyback offsets=0,32,64,96 values=1,2,3,4`.

The same marker route also passed on `demetrios@192.168.3.24` (`spark-3c59`,
`aarch64`, NVIDIA GB10, CUDA 13.0, `sm_121`). That secondary receipt is
`artifacts/gpu/knowledge_vecmat_evidence_audit/dgx_runtime_attempt/dgx_spark_public_gpu_gate.192-168-3-24.runtime_pass.v1.json`.

The package includes `gpu_knowledge_vec4_package_manifest.v1.json`, which records
the PTX, runner, local-ptxas cubin, SHA256s, sizes, and launch contract. That
manifest is the portable handoff for a later DGX/device run.

Run `scripts/dev/gpu_knowledge_vec4_package_verify.py` against the package
manifest to verify file integrity and the non-runtime launch contract before
moving the package to a GPU host.

After a DGX/device run, use `scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py
--mode runtime-pass artifacts/gpu/dgx_spark_public_gpu_gate.v1.json` to require a
real remote receipt with `runtime_pass`, DGX identity fields, and the marker
runtime `PASS` output. Use `--mode not-run` only for the current package-only
receipt.

The runbook wrapper exposes the intended sequence:

- `all-local`: prepare package, verify package, verify current receipt as not-run.
- `run-dgx`: execute the opt-in DGX marker route with public kernels disabled
  unless explicitly overridden.
- `verify-runtime`: require a real `runtime_pass` DGX receipt.

The completion auditor writes
`artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_completion_audit.v1.json`.
It may only report `goal_status=complete` when backend pack/unpack, CUDA device
runtime, and the scoped imported runtime fixture all pass.

It also writes
`artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_open_blockers.v1.json`
with `.claude/PARALLEL_BLOCKER_CONTRACT.md`-shaped blocker records. Backend
pack/unpack, CUDA runtime, and imported runtime fixture records are retained as
`closed` when their probes pass.

The same auditor writes
`artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_operational_handoff.md`
with current SHA, branch, worktree, owned files, do-not-touch files, green gates,
failing gates, open blockers, artifacts, and next commands.

The DGX JSON separates marker states:

- `disabled`: marker route was not requested.
- `local_ptxas_only_not_remote_not_launched`: marker package was prepared locally
  and no DGX SSH/runtime was used.
- `ptxas_only_not_launched`: marker PTX assembled, but runtime launch was not run.
- `runtime_pass`: marker runner printed its explicit copyback `PASS` marker.

## Model Routing

- `current-codex`: integration edits, semantic boundary decisions, final
  evidence classification.
- `gpt-5.4-mini`: read-only mapping, bounded blocker reproduction, toolchain and
  runtime environment discovery, and cheap route classification. The latest
  explorer classified `gpu_knowledge_vec4_backend_pack_unpack_probe.sh` as the
  production-backend proof target if it passes, `gpu_knowledge_vec4_ptxas_probe.sh`
  as contract-only, and older PTX oracle tests as mirror/reference evidence.

## Open Lanes

- `gpu_backend_pack_unpack`: closed for the scoped direct backend Vec4 emitter.
  Current repro: `scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh`.
  The remaining `GpuKernelIr.ops` modular ABI issue is a separate hardening note,
  not the active Vec4 pack/unpack blocker.
- `cuda_runtime_vecmat_artifact`: adapt DGX/local CUDA runtime routes to load
  and launch the emitted marker cubin and verify copyback lanes on real device.
  Latest marker-only attempt reached the remote CUDA toolchain probe and failed
  at BatchMode SSH auth for `demetrios@192.168.3.43`; the saved receipt is
  `artifacts/gpu/knowledge_vecmat_evidence_audit/dgx_runtime_attempt/dgx_spark_public_gpu_gate.runtime_attempt.v1.json`.
  CUDA runtime itself is now closed by the Slurm runtime probe; DGX SSH is only
  an optional platform-specific follow-up.
- `imported_runtime_lower_array`: closed for the scoped imported Vec4 lane-plan
  fixture. Broader imported-runtime known-failures remain outside this GPU
  Knowledge Vec/Mat completion claim.

## Evidence Boundary

The audit may report `completion_ready=true` only for this scoped GPU Knowledge
Vec/Mat claim. A ptxas pass proves assembler acceptance and launch-contract
shape only; the Slurm runtime probe is the current CUDA device execution proof;
the imported runtime probe proves only the narrow imported Vec4 lane-plan
fixture. None of these proofs claims general GPU backend or general imported
runtime correctness.
