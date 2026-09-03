# GPU Knowledge Vec/Mat Lane: cuda_runtime_vecmat_artifact

- owner: gpu-runtime-owner
- model: gpt-5.4-mini
- status: cuda_device_runtime_pass
- required_action: adapt_cuda_runtime_route_for_vecmat_artifact
- gap: cuda_device_runtime_execution
- acceptance_gate: CUDA runtime loads and launches the emitted GPU Knowledge Vec4 marker artifact and verifies copyback offsets 0,32,64,96

## Runtime Routes
- slurm_gpu_runtime_probe: `scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh` (cuda_device_runtime_pass; CUDA device runtime proof on Slurm GPU; does not claim DGX Spark runtime; evidence: artifacts/gpu/knowledge_vecmat_evidence_audit/slurm_runtime_probe/gpu_knowledge_vec4_slurm_runtime_probe.v1.json; verify: scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh; receipt_verify: n/a; runbook: n/a)
- dgx_spark_marker_package_only: `scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh package-only` (local_package_pass; local ptxas/package proof only; no SSH, no DGX toolchain, no CUDA device runtime; evidence: artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json; verify: scripts/dev/gpu_knowledge_vec4_package_verify.py artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json; receipt_verify: scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode not-run artifacts/gpu/dgx_spark_public_gpu_gate.v1.json; runbook: scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh all-local)
- dgx_spark_public_gpu_gate: `scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx` (wired_opt_in_not_run; remote ptxas/runtime authority only when DGX SSH and CUDA runtime are available; evidence: n/a; verify: scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass artifacts/gpu/dgx_spark_public_gpu_gate.v1.json; receipt_verify: n/a; runbook: n/a)

## Evidence
- audit: artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_evidence_audit.v1.json
- queue: artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_swarm_queue.v1.json
- ptxas_probe: artifacts/gpu/knowledge_vecmat_evidence_audit/ptxas_probe/gpu_knowledge_vec4_ptxas_probe.v1.json

## Boundary
This handoff is not a completion claim. It preserves owner scope and the current evidence boundary.
