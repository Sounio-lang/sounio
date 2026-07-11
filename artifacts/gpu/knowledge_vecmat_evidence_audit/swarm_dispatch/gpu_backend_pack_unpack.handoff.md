# GPU Knowledge Vec/Mat Lane: gpu_backend_pack_unpack

- owner: gpu-backend-owner
- model: current-codex
- status: backend_ir_pack_unpack_ptxas_pass
- required_action: wire_automatic_backend_pack_unpack
- gap: automatic_backend_pack_unpack
- acceptance_gate: automatic Vec/Mat aggregate backend pack/unpack proof across relevant emitters without imported-lower fallback

## Runtime Routes

## Evidence
- audit: artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_evidence_audit.v1.json
- queue: artifacts/gpu/knowledge_vecmat_evidence_audit/gpu_knowledge_vecmat_swarm_queue.v1.json
- ptxas_probe: artifacts/gpu/knowledge_vecmat_evidence_audit/ptxas_probe/gpu_knowledge_vec4_ptxas_probe.v1.json

## Boundary
This handoff is not a completion claim. It preserves owner scope and the current evidence boundary.
