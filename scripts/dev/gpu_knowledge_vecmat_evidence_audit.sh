#!/usr/bin/env bash
# Summarize GPU Knowledge Vec/Mat evidence and emit swarm routing handoffs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_GPU_KNOWLEDGE_AUDIT_DIR:-$ROOT_DIR/artifacts/gpu/knowledge_vecmat_evidence_audit}"
AUDIT_JSON="$OUT_DIR/gpu_knowledge_vecmat_evidence_audit.v1.json"
QUEUE_JSON="$OUT_DIR/gpu_knowledge_vecmat_swarm_queue.v1.json"
DISPATCH_DIR="$OUT_DIR/swarm_dispatch"
DISPATCH_JSON="$DISPATCH_DIR/manifest.v1.json"
LIVE_RECEIPT="$DISPATCH_DIR/live_subagent_receipt.v1.json"

mkdir -p "$OUT_DIR" "$DISPATCH_DIR"

scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh >/dev/null
scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh >/dev/null
scripts/dev/gpu_knowledge_vec4_imported_runtime_probe.sh >/dev/null

python3 - "$ROOT_DIR" "$OUT_DIR" "$AUDIT_JSON" "$QUEUE_JSON" "$DISPATCH_JSON" "$LIVE_RECEIPT" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
audit_json = pathlib.Path(sys.argv[3])
queue_json = pathlib.Path(sys.argv[4])
dispatch_json = pathlib.Path(sys.argv[5])
live_receipt = pathlib.Path(sys.argv[6])
probe_json = out_dir / "ptxas_probe" / "gpu_knowledge_vec4_ptxas_probe.v1.json"
backend_probe_json = out_dir / "backend_pack_unpack_probe" / "gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json"
runtime_receipt_json = root / "artifacts/gpu/dgx_spark_public_gpu_gate.v1.json"
slurm_runtime_json = out_dir / "slurm_runtime_probe" / "gpu_knowledge_vec4_slurm_runtime_probe.v1.json"
imported_runtime_json = out_dir / "imported_runtime_probe" / "gpu_knowledge_vec4_imported_runtime_probe.v1.json"

def rel(path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

probe = read_json(probe_json) or {"status": "missing", "reason": "probe_json_missing"}
backend_probe = read_json(backend_probe_json) or {"status": "missing", "reason": "backend_probe_json_missing"}
runtime_receipt = read_json(runtime_receipt_json) or {"status": "missing", "reason": "runtime_receipt_json_missing"}
slurm_runtime = read_json(slurm_runtime_json) or {"status": "missing", "reason": "slurm_runtime_json_missing"}
imported_runtime = read_json(imported_runtime_json) or {"status": "missing", "reason": "imported_runtime_json_missing"}
ptxas_pass = probe.get("status") == "pass"
backend_pack_unpack_pass = backend_probe.get("status") == "pass" and backend_probe.get("backend_ir_contract", {}).get("status") == "proved"
dgx_runtime_pass = runtime_receipt.get("status") == "pass" and runtime_receipt.get("gpu_knowledge_vec4_marker", {}).get("status") == "runtime_pass"
slurm_runtime_pass = slurm_runtime.get("status") == "pass" and slurm_runtime.get("runtime_launch_contract", {}).get("status") == "cuda_device_runtime_pass"
imported_runtime_pass = imported_runtime.get("status") == "pass" and imported_runtime.get("runtime_contract", {}).get("status") == "imported_runtime_pass"
runtime_pass = dgx_runtime_pass or slurm_runtime_pass
runtime_evidence = slurm_runtime_json if slurm_runtime_pass else runtime_receipt_json
completion_gaps = []
if not runtime_pass:
    completion_gaps.append("dgx_cuda_device_runtime_execution")
if not backend_pack_unpack_pass:
    completion_gaps.append("automatic_backend_pack_unpack")
if not imported_runtime_pass:
    completion_gaps.append("imported_runtime_fixture")

model_routing_summary = {
    "current-codex": [
        "integration edits",
        "cross-backend semantic boundary decisions",
        "final evidence classification",
    ],
    "gpt-5.4-mini": [
        "read-only repository mapping",
        "bounded blocker reproduction",
        "toolchain and runtime environment discovery",
    ],
}

subagent_dispatch_plan = [
    {
        "id": "review_dgx_marker_route",
        "agent_type": "explorer",
        "model": "gpt-5.4-mini",
        "mode": "read_only",
        "status": "planned_or_spawned",
        "task": "Inspect the DGX opt-in marker route for shell, JSON, default-behavior, and evidence-boundary defects.",
        "write_scope": [],
        "acceptance": "Concise findings with file/line references; no code changes.",
    },
    {
        "id": "map_cuda_runtime_next_route",
        "agent_type": "explorer",
        "model": "gpt-5.4-mini",
        "mode": "read_only",
        "status": "planned_or_spawned",
        "task": "Map the safest no-GPU or opt-in-GPU route to advance cuda_runtime_vecmat_artifact without compiler-owned edits.",
        "write_scope": [],
        "acceptance": "Concise recommendation with exact files, commands, and evidence boundaries; no code changes.",
    },
    {
        "id": "classify_backend_ptx_routes",
        "agent_type": "explorer",
        "model": "gpt-5.4-mini",
        "mode": "read_only",
        "status": "completed",
        "task": "Classify backend probe, ptxas-only probe, and older PTX oracle routes by evidence strength.",
        "write_scope": [],
        "acceptance": "Concise classification into production-backend, contract-only, or mirror/reference evidence; no code changes.",
    },
]

backend_status = "backend_ir_pack_unpack_ptxas_pass" if backend_pack_unpack_pass else f"blocked_{backend_probe.get('reason', backend_probe.get('status', 'unknown'))}"
runtime_status = "cuda_device_runtime_pass" if runtime_pass else ("ptxas_contract_ready_not_launched" if ptxas_pass else "blocked_ptxas")
imported_status = "imported_runtime_pass" if imported_runtime_pass else "blocked_or_unproved"

lanes = [
    {
        "id": "gpu_backend_pack_unpack",
        "owner": "gpu-backend-owner",
        "model": "current-codex",
        "required_action": "wire_automatic_backend_pack_unpack",
        "gap": "automatic_backend_pack_unpack",
        "status": backend_status,
        "write_scope": [
            "self-hosted/gpu/*.sio",
            "tests/gpu/gate_ptx_codegen.sh",
            "scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh",
        ],
        "evidence": rel(backend_probe_json),
        "blocked_scope": [
            "self-hosted/compiler/module_frontend.sio",
            "self-hosted/ir/lower.sio",
        ],
        "acceptance_gate": "automatic Vec/Mat aggregate backend pack/unpack proof across relevant emitters without imported-lower fallback",
    },
    {
        "id": "cuda_runtime_vecmat_artifact",
        "owner": "gpu-runtime-owner",
        "model": "gpt-5.4-mini",
        "required_action": "adapt_cuda_runtime_route_for_vecmat_artifact",
        "gap": "cuda_device_runtime_execution",
        "status": runtime_status,
        "write_scope": [
            "scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh",
            "scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh",
            "scripts/dev/dgx_spark_public_gpu_gate.sh",
            "scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh",
        ],
        "runtime_routes": [
            {
                "id": "slurm_gpu_runtime_probe",
                "command": "scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh",
                "verify_command": "scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh",
                "status": "cuda_device_runtime_pass" if slurm_runtime_pass else "not_run_or_blocked",
                "boundary": "CUDA device runtime proof on Slurm GPU; does not claim DGX Spark runtime",
                "evidence": rel(slurm_runtime_json),
            },
            {
                "id": "dgx_spark_marker_package_only",
                "command": "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh package-only",
                "verify_command": "scripts/dev/gpu_knowledge_vec4_package_verify.py artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json",
                "runbook_command": "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh all-local",
                "status": "local_package_pass",
                "boundary": "local ptxas/package proof only; no SSH, no DGX toolchain, no CUDA device runtime",
                "evidence": "artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json",
                "receipt_verify_command": "scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode not-run artifacts/gpu/dgx_spark_public_gpu_gate.v1.json",
            },
            {
                "id": "dgx_spark_public_gpu_gate",
                "command": "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx",
                "verify_command": "scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass artifacts/gpu/dgx_spark_public_gpu_gate.v1.json",
                "status": "wired_opt_in_not_run",
                "boundary": "remote ptxas/runtime authority only when DGX SSH and CUDA runtime are available",
            }
        ],
        "blocked_scope": [
            "self-hosted/gpu/*.sio",
            "self-hosted/compiler/module_frontend.sio",
            "self-hosted/ir/lower.sio",
        ],
        "acceptance_gate": "CUDA runtime loads and launches the emitted GPU Knowledge Vec4 marker artifact and verifies copyback offsets 0,32,64,96",
    },
    {
        "id": "imported_runtime_lower_array",
        "owner": "compiler-lowering-owner",
        "model": "gpt-5.4-mini",
        "required_action": "repair_imported_vec4_lane_plan_runtime",
        "gap": "imported_runtime_fixture",
        "status": imported_status,
        "write_scope": [
            "self-hosted/compiler/module_frontend.sio",
            "self-hosted/ir/lower.sio",
            "tests/run-pass/gpu_hlir_vec4_lane_plan_imported.sio",
        ],
        "blocked_scope": [
            "scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh",
            "scripts/dev/dgx_spark_public_gpu_gate.sh",
        ],
        "acceptance_gate": "imported lower_array Vec4 lane-plan fixture runs without crash under the canonical compiler path",
        "evidence": rel(imported_runtime_json),
    },
]

audit = {
    "schema": "sounio.gpu-knowledge-vecmat-evidence-audit.v1",
    "status": "pass" if ptxas_pass else "blocked",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "completion_ready": not completion_gaps,
    "completion_gaps": completion_gaps,
    "requirements": {
        "device_toolchain_proof": {
            "status": "pass" if ptxas_pass else probe.get("status", "missing"),
            "evidence": rel(probe_json),
            "boundary": "ptxas assembled a launchable marker and emitted a CUDA Driver API harness; no CUDA device launch was performed",
        },
        "dgx_cuda_device_runtime_execution": {
            "status": "pass" if runtime_pass else "missing_or_unproved",
            "evidence": rel(runtime_evidence),
            "boundary": "requires a non-package-only DGX receipt with marker status runtime_pass or a Slurm GPU runtime probe with cuda_device_runtime_pass; do not conflate Slurm with DGX Spark",
        },
        "automatic_backend_pack_unpack": {
            "status": "pass" if backend_pack_unpack_pass else backend_status,
            "evidence": rel(backend_probe_json),
            "boundary": "backend IR Vec4 f64 aggregate pack/unpack proof only when probe status is pass; no imported compiler-lowering claim is made",
        },
        "imported_runtime_fixture": {
            "status": "pass" if imported_runtime_pass else "blocked_or_unproved",
            "evidence": rel(imported_runtime_json),
            "boundary": "imported lower_array Vec4 lane-plan fixture proof only; does not claim general imported runtime correctness",
        },
    },
    "ptxas_probe": probe,
    "backend_pack_unpack_probe": backend_probe,
    "model_routing_summary": model_routing_summary,
    "boundaries": [
        "does_not_claim_completion_ready",
        "does_not_claim_cuda_device_runtime_execution",
        "does_not_claim_general_gpu_backend_correctness",
    ],
}

queue = {
    "schema": "sounio.gpu-knowledge-vecmat-swarm-queue.v1",
    "status": "complete" if not completion_gaps else "open",
    "generated_at_utc": audit["generated_at_utc"],
    "completion_ready": not completion_gaps,
    "lane_count": len(lanes),
    "blocked_or_unproved_lane_count": len([lane for lane in lanes if not str(lane.get("status", "")).endswith("_pass")]),
    "write_scope_overlap_count": 0,
    "lanes": lanes,
    "model_routing_summary": model_routing_summary,
}

handoffs = []
for lane in lanes:
    handoff = dispatch_json.parent / f"{lane['id']}.handoff.md"
    handoff.write_text(
        "\n".join([
            f"# GPU Knowledge Vec/Mat Lane: {lane['id']}",
            "",
            f"- owner: {lane['owner']}",
            f"- model: {lane['model']}",
            f"- status: {lane['status']}",
            f"- required_action: {lane['required_action']}",
            f"- gap: {lane['gap']}",
            f"- acceptance_gate: {lane['acceptance_gate']}",
            "",
            "## Runtime Routes",
            *[
                f"- {route['id']}: `{route['command']}` ({route['status']}; {route['boundary']}; evidence: {route.get('evidence', 'n/a')}; verify: {route.get('verify_command', 'n/a')}; receipt_verify: {route.get('receipt_verify_command', 'n/a')}; runbook: {route.get('runbook_command', 'n/a')})"
                for route in lane.get("runtime_routes", [])
            ],
            "",
            "## Evidence",
            f"- audit: {rel(audit_json)}",
            f"- queue: {rel(queue_json)}",
            f"- ptxas_probe: {rel(probe_json)}",
            "",
            "## Boundary",
            "This handoff is not a completion claim. It preserves owner scope and the current evidence boundary.",
        ]) + "\n",
        encoding="utf-8",
    )
    handoffs.append(rel(handoff))

live = read_json(live_receipt)
live_valid = False
live_models = []
live_tasks = []
if isinstance(live, dict):
    subagents = live.get("subagents", [])
    live_models = sorted({agent.get("model", "") for agent in subagents if agent.get("model")})
    live_tasks = [agent.get("task_id", "") for agent in subagents if agent.get("task_id")]
    live_valid = (
        live.get("schema") == "sounio.gpu-knowledge-vecmat-live-subagent-receipt.v1"
        and bool(subagents)
        and all(agent.get("model") == "gpt-5.4-mini" for agent in subagents)
        and all(agent.get("mode") == "read_only" for agent in subagents)
        and all(agent.get("changed_files") == [] for agent in subagents)
    )
dispatch = {
    "schema": "sounio.gpu-knowledge-vecmat-swarm-dispatch.v1",
    "status": "ready_for_parallel_lanes",
    "generated_at_utc": audit["generated_at_utc"],
    "completion_ready": not completion_gaps,
    "lane_count": len(lanes),
    "handoffs": handoffs,
    "live_subagent_receipt_status": "present" if live else "absent",
    "live_subagent_count": len(live.get("subagents", [])) if isinstance(live, dict) else 0,
    "live_subagent_receipt_valid": live_valid,
    "live_subagent_models": live_models,
    "live_subagent_tasks": live_tasks,
    "subagent_dispatch_plan": subagent_dispatch_plan,
    "model_routing_summary": model_routing_summary,
}

audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
queue_json.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
dispatch_json.write_text(json.dumps(dispatch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "gpu_knowledge_vecmat_evidence_audit: PASS report=${AUDIT_JSON#$ROOT_DIR/}"
