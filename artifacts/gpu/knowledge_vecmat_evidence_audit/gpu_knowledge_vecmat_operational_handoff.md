# GPU Knowledge Vec/Mat Operational Handoff

Current-SHA: ee61ec07c180c16108b48b52747ab5d42bfc13b7
Current-Branch: work/imported-module-elf-e2e-codex
Current-Worktree: /tmp/sounio-imported-module-elf-e2e
Dirty-Status: see `git status --short -- scripts/dev/gpu_knowledge_vec* scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh scripts/dev/dgx_spark_public_gpu_gate.sh artifacts/gpu/knowledge_vecmat_evidence_audit artifacts/gpu/dgx_spark_public_gpu_package artifacts/gpu/dgx_spark_public_gpu_gate.v1.json`
Current-Goal-Status: complete
Completion-Blockers: none

Owned-Files:
- scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh
- scripts/dev/gpu_knowledge_vec4_package_verify.py
- scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py
- scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh
- scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh
- scripts/dev/gpu_knowledge_vecmat_completion_audit.py
- scripts/dev/gpu_knowledge_vecmat_evidence_audit.sh
- scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh
- scripts/dev/dgx_spark_public_gpu_gate.sh
- self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_harness.sio
- self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_stage_probe.sio
- docs/handoff/gpu_knowledge_vecmat_swarm_plan.md
- artifacts/gpu/knowledge_vecmat_evidence_audit/*
- artifacts/gpu/dgx_spark_public_gpu_package/*
- artifacts/gpu/dgx_spark_public_gpu_gate.v1.json

Do-Not-Touch:
- self-hosted/compiler/module_frontend.sio
- self-hosted/ir/lower.sio
- unrelated neurodyn/ABIDE/dossier lanes

Last-Green-Gates:
- scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh
- scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh all-local
- scripts/dev/gpu_knowledge_vecmat_completion_audit.py
- git diff --check

Failing-Gates:
- bash scripts/dev/check_docs_registry.sh: stale docs/governance/topic-registry.v1.json and docs/governance/DOCS_ACCEPTANCE_REPORT.md
- tests/gpu/gate_public_gpu_cfg_build.sh: current branch public GPU frontend/PTX build failures when public kernels are enabled

Blocker-Records:
- BLK-20260706-gpu-knowledge-vecmat-dgx-runtime (B3, platform-resource, owner=gpu-runtime-owner, evidence=E3)
- BLK-20260706-gpu-knowledge-vecmat-backend-pack-unpack (B1, evidence-gap, owner=gpu-backend-owner, evidence=E3)
- BLK-20260706-gpu-knowledge-vecmat-imported-runtime (B1, compiler-semantics, owner=compiler-lowering-owner, evidence=E3)

Artifacts:
- artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_completion_audit.v1.json
- artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_open_blockers.v1.json
- artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_evidence_audit.v1.json
- artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_swarm_queue.v1.json
- artifacts/gpu/knowledge_vecmat_evidence_audit/backend_pack_unpack_probe/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json
- artifacts/gpu/knowledge_vecmat_evidence_audit/swarm_dispatch/manifest.v1.json
- artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json
- artifacts/gpu/dgx_spark_public_gpu_gate.v1.json

Next-Command:
- scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx
- scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh verify-runtime
- scripts/dev/gpu_knowledge_vecmat_completion_audit.py

Boundary:
- Current state includes local ptxas/package evidence plus Slurm CUDA device runtime evidence for the Vec4 marker; DGX Spark runtime remains optional and unproven.
- Do not mark complete until `goal_status=complete` in `artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_completion_audit.v1.json` and all non-closed blocker records are closed or waived.
