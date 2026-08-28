#!/usr/bin/env python3
"""Requirement-by-requirement completion audit for GPU Knowledge Vec/Mat."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts/gpu/knowledge_vecmat_evidence_audit"
OUT_JSON = AUDIT_DIR / "gpu_knowledge_vecmat_completion_audit.v1.json"
BLOCKERS_JSON = AUDIT_DIR / "gpu_knowledge_vecmat_open_blockers.v1.json"
HANDOFF_MD = AUDIT_DIR / "gpu_knowledge_vecmat_operational_handoff.md"


def load(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip() or "<detached>"
    except Exception:
        return "<unknown>"


def main() -> int:
    audit_path = AUDIT_DIR / "gpu_knowledge_vecmat_evidence_audit.v1.json"
    queue_path = AUDIT_DIR / "gpu_knowledge_vecmat_swarm_queue.v1.json"
    dispatch_path = AUDIT_DIR / "swarm_dispatch/manifest.v1.json"
    probe_path = AUDIT_DIR / "ptxas_probe/gpu_knowledge_vec4_ptxas_probe.v1.json"
    backend_probe_path = AUDIT_DIR / "backend_pack_unpack_probe/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json"
    runtime_attempt_path = AUDIT_DIR / "dgx_runtime_attempt/dgx_spark_public_gpu_gate.runtime_attempt.v1.json"
    slurm_runtime_path = AUDIT_DIR / "slurm_runtime_probe/gpu_knowledge_vec4_slurm_runtime_probe.v1.json"
    imported_runtime_path = AUDIT_DIR / "imported_runtime_probe/gpu_knowledge_vec4_imported_runtime_probe.v1.json"
    package_manifest_path = ROOT / "artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json"
    receipt_path = ROOT / "artifacts/gpu/dgx_spark_public_gpu_gate.v1.json"

    audit = load(audit_path)
    queue = load(queue_path)
    dispatch = load(dispatch_path)
    probe = load(probe_path)
    backend_probe = load(backend_probe_path)
    runtime_attempt = load(runtime_attempt_path)
    slurm_runtime = load(slurm_runtime_path)
    imported_runtime = load(imported_runtime_path)
    package_manifest = load(package_manifest_path)
    receipt = load(receipt_path)
    backend_check_status = audit.get("requirements", {}).get("automatic_backend_pack_unpack", {}).get("status", "missing_or_unproved")
    backend_closed = backend_check_status == "pass"
    runtime_check_status = audit.get("requirements", {}).get("dgx_cuda_device_runtime_execution", {}).get("status", "missing_or_unproved")
    runtime_closed = runtime_check_status == "pass"
    imported_check_status = audit.get("requirements", {}).get("imported_runtime_fixture", {}).get("status", "missing_or_unproved")
    imported_closed = imported_check_status == "pass"
    runtime_attempt_reason = runtime_attempt.get("reason", "no_runtime_attempt_receipt")
    runtime_attempt_status = runtime_attempt.get("status", "missing")
    runtime_attempt_marker_status = runtime_attempt.get("gpu_knowledge_vec4_marker", {}).get("status", "missing")
    slurm_runtime_reason = slurm_runtime.get("reason", "no_slurm_runtime_receipt")
    slurm_runtime_status = slurm_runtime.get("status", "missing")
    runtime_evidence = slurm_runtime_path if slurm_runtime_path.exists() and slurm_runtime_status == "pass" else (runtime_attempt_path if runtime_attempt_path.exists() else receipt_path)

    runtime_routes = []
    lanes = queue.get("lanes", []) if isinstance(queue.get("lanes"), list) else []
    for lane in lanes:
        if lane.get("id") == "cuda_runtime_vecmat_artifact":
            runtime_routes = lane.get("runtime_routes", [])
            break

    checks = [
        {
            "id": "swarm_dispatch_and_model_routing",
            "status": "pass" if dispatch.get("live_subagent_receipt_valid") is True and dispatch.get("live_subagent_models") == ["gpt-5.4-mini"] else "missing_or_unproved",
            "evidence": rel(dispatch_path),
            "required_for_completion": True,
        },
        {
            "id": "ptxas_launch_contract",
            "status": "pass" if probe.get("status") == "pass" and probe.get("runtime_launch_contract", {}).get("status") == "ptxas_only_not_launched" else "missing_or_unproved",
            "evidence": rel(probe_path),
            "required_for_completion": True,
        },
        {
            "id": "portable_package_integrity",
            "status": "pass" if package_manifest.get("status") == "pass" and package_manifest.get("runtime_launch_contract", {}).get("status") == "local_package_only_not_remote_not_launched" else "missing_or_unproved",
            "evidence": rel(package_manifest_path),
            "required_for_completion": True,
        },
        {
            "id": "dgx_cuda_device_runtime_execution",
            "status": runtime_check_status,
            "evidence": rel(runtime_evidence),
            "required_for_completion": True,
            "acceptance": "Receipt must pass scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass or scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh must report cuda_device_runtime_pass",
        },
        {
            "id": "automatic_backend_pack_unpack",
            "status": backend_check_status,
            "evidence": rel(backend_probe_path),
            "required_for_completion": True,
        },
        {
            "id": "imported_runtime_fixture",
            "status": imported_check_status,
            "evidence": rel(imported_runtime_path),
            "required_for_completion": True,
        },
        {
            "id": "runtime_routes_declared",
            "status": "pass" if [route.get("id") for route in runtime_routes] == ["dgx_spark_marker_package_only", "dgx_spark_public_gpu_gate"] else "missing_or_unproved",
            "evidence": rel(queue_path),
            "required_for_completion": False,
        },
    ]

    completion_blockers = [
        check["id"]
        for check in checks
        if check["required_for_completion"] and check["status"] != "pass"
    ]

    payload = {
        "schema": "sounio.gpu-knowledge-vecmat-completion-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "goal_status": "complete" if not completion_blockers else "not_complete",
        "completion_ready": not completion_blockers,
        "completion_blockers": completion_blockers,
        "checks": checks,
        "boundaries": [
            "completion_requires_runtime_pass_receipt",
            "completion_requires_automatic_backend_pack_unpack",
            "completion_requires_imported_runtime_fixture",
            "do_not_mark_goal_complete_from_package_only_or_ptxas_only_evidence",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blockers = {
        "schema": "sounio.gpu-knowledge-vecmat-open-blockers.v1",
        "generated_at_utc": payload["generated_at_utc"],
        "source_completion_audit": rel(OUT_JSON),
        "blockers": [
            {
                "Blocker-ID": "BLK-20260706-gpu-knowledge-vecmat-dgx-runtime",
                "Status": "closed" if runtime_closed else "classified",
                "Severity": "B3",
                "Class": "platform-resource",
                "Owner": "gpu-runtime-owner",
                "Lane": "cuda_runtime_vecmat_artifact",
                "Worktree": str(ROOT),
                "Branch": "coord/lane-8c-dossier",
                "Files-Owned": "scripts/dev/dgx_spark_public_gpu_gate.sh; scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh; scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py; scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh",
                "Files-Read-Only": "artifacts/gpu/dgx_spark_public_gpu_package/*; artifacts/gpu/knowledge_vecmat_evidence_audit/*",
                "Do-Not-Touch": "self-hosted/compiler/module_frontend.sio; self-hosted/ir/lower.sio; self-hosted/gpu/*.sio",
                "Repro": "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx",
                "Observed": f"slurm runtime probe status {slurm_runtime_status} reason {slurm_runtime_reason}; latest saved DGX attempt status {runtime_attempt_status} reason {runtime_attempt_reason} marker_status {runtime_attempt_marker_status}; current local receipt is package_only with marker status {receipt.get('gpu_knowledge_vec4_marker', {}).get('status', 'missing')}",
                "Expected": "CUDA device runtime executes gpu_knowledge_vec4_aggregate_marker and prints PASS with copyback offsets 0,32,64,96",
                "Acceptance-Gate": "scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass artifacts/gpu/dgx_spark_public_gpu_gate.v1.json OR scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh reports status pass",
                "Evidence-Level": "E3" if runtime_closed else "E2",
                "Evidence": rel(runtime_evidence),
                "Fallback-Path": "Slurm GPU runtime probe via CUDA Driver API PTX JIT" if runtime_closed else "package-only marker package and not-run receipt verifier",
                "Legacy-Kept": "n/a",
                "LLM-Offload": "not-required",
                "Next-Action": "Closed for CUDA device runtime through Slurm; DGX Spark SSH remains an optional platform-access follow-up." if runtime_closed else "Establish SSH key or ControlMaster access to demetrios@192.168.3.43, rerun the marker-only DGX route, then run the runtime-pass verifier.",
            },
            {
                "Blocker-ID": "BLK-20260706-gpu-knowledge-vecmat-backend-pack-unpack",
                "Status": "closed" if backend_closed else "classified",
                "Severity": "B1",
                "Class": "evidence-gap",
                "Owner": "gpu-backend-owner",
                "Lane": "gpu_backend_pack_unpack",
                "Worktree": str(ROOT),
                "Branch": "coord/lane-8c-dossier",
                "Files-Owned": "self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_harness.sio; self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_stage_probe.sio; scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh; scripts/dev/gpu_knowledge_vecmat_evidence_audit.sh",
                "Files-Read-Only": "self-hosted/gpu/kernel_ir.sio; self-hosted/gpu/lower_to_ptx.sio; self-hosted/gpu/ptx.sio; scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh",
                "Do-Not-Touch": "self-hosted/compiler/module_frontend.sio; self-hosted/ir/lower.sio",
                "Repro": "scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh",
                "Observed": f"backend probe status {backend_probe.get('status', 'missing')} reason {backend_probe.get('reason', 'missing')}; check_exit={backend_probe.get('souc', {}).get('check_exit_code', 'n/a')} run_exit={backend_probe.get('souc', {}).get('run_exit_code', 'n/a')} ptxas_exit={backend_probe.get('ptxas', {}).get('exit_code', 'n/a')}; contract={backend_probe.get('backend_ir_contract', {}).get('status', 'n/a')}; lean_extract_fallback={backend_probe.get('lean_extract_fallback', {}).get('status', 'n/a')}:{backend_probe.get('lean_extract_fallback', {}).get('reason', 'n/a')}; minimal_import_runtime={backend_probe.get('minimal_import_runtime_probe', {}).get('reason', 'n/a')}",
                "Expected": "automatic Vec/Mat aggregate backend pack/unpack proof across relevant emitters without imported-lower fallback",
                "Acceptance-Gate": "scripts/dev/gpu_knowledge_vec4_backend_pack_unpack_probe.sh reports status pass plus scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh",
                "Evidence-Level": "E3" if backend_closed else "E2",
                "Evidence": "artifacts/gpu/knowledge_vecmat_evidence_audit/backend_pack_unpack_probe/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json",
                "Fallback-Path": "direct backend Vec4 emitter; lean extract retained as reference fallback" if backend_closed else "local ptxas/package witness only",
                "Legacy-Kept": "yes",
                "LLM-Offload": "not-required",
                "Next-Action": "Closed for the direct backend Vec4 emitter; keep the GpuKernelIr.ops modular ABI issue as a separate compiler/runtime hardening note." if backend_closed else "Reduce the remaining imported GPU dependency-lowering crash: a two-import kernel_ir_wmma_lean + ptx probe still segfaults while lowering ptx.sio as dependency, then rerun the Vec4 backend-IR/PTX probe until it emits PTX and passes ptxas.",
            },
            {
                "Blocker-ID": "BLK-20260706-gpu-knowledge-vecmat-imported-runtime",
                "Status": "closed" if imported_closed else "classified",
                "Severity": "B1",
                "Class": "compiler-semantics",
                "Owner": "compiler-lowering-owner",
                "Lane": "imported_runtime_lower_array",
                "Worktree": str(ROOT),
                "Branch": "coord/lane-8c-dossier",
                "Files-Owned": "tests/run-pass/gpu_hlir_vec4_lane_plan_leaf.sio; tests/run-pass/gpu_hlir_vec4_lane_plan_imported.sio; scripts/dev/gpu_knowledge_vec4_imported_runtime_probe.sh",
                "Files-Read-Only": "scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh; artifacts/gpu/knowledge_vecmat_evidence_audit/*",
                "Do-Not-Touch": "scripts/dev/dgx_spark_public_gpu_gate.sh; scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh",
                "Repro": "scripts/dev/gpu_knowledge_vecmat_completion_audit.py",
                "Observed": f"imported runtime probe status {imported_runtime.get('status', 'missing')} reason {imported_runtime.get('reason', 'missing')} check_exit={imported_runtime.get('souc', {}).get('check_exit_code', 'n/a')} run_exit={imported_runtime.get('souc', {}).get('run_exit_code', 'n/a')}",
                "Expected": "imported lower_array Vec4 lane-plan fixture runs under the canonical compiler path",
                "Acceptance-Gate": "scripts/dev/gpu_knowledge_vec4_imported_runtime_probe.sh plus scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh",
                "Evidence-Level": "E3" if imported_closed else "E2",
                "Evidence": rel(imported_runtime_path),
                "Fallback-Path": "none",
                "Legacy-Kept": "yes",
                "LLM-Offload": "not-required",
                "Next-Action": "Closed for the scoped imported Vec4 lane-plan fixture; keep broader imported-runtime known-failures out of this lane." if imported_closed else "Transfer to compiler-lowering owner before editing compiler-owned files.",
            },
        ],
    }
    BLOCKERS_JSON.write_text(json.dumps(blockers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blocker_lines = "\n".join(
        f"- {item['Blocker-ID']} ({item['Severity']}, {item['Class']}, owner={item['Owner']}, evidence={item['Evidence-Level']})"
        for item in blockers["blockers"]
    )
    completion_blocker_line = ", ".join(completion_blockers) if completion_blockers else "none"
    handoff = f"""# GPU Knowledge Vec/Mat Operational Handoff

Current-SHA: {git_text("rev-parse", "HEAD")}
Current-Branch: {git_text("branch", "--show-current")}
Current-Worktree: {ROOT}
Dirty-Status: see `git status --short -- scripts/dev/gpu_knowledge_vec* scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh scripts/dev/dgx_spark_public_gpu_gate.sh artifacts/gpu/knowledge_vecmat_evidence_audit artifacts/gpu/dgx_spark_public_gpu_package artifacts/gpu/dgx_spark_public_gpu_gate.v1.json`
Current-Goal-Status: {payload["goal_status"]}
Completion-Blockers: {completion_blocker_line}

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
{blocker_lines}

Artifacts:
- {rel(OUT_JSON)}
- {rel(BLOCKERS_JSON)}
- {rel(AUDIT_DIR / "gpu_knowledge_vecmat_evidence_audit.v1.json")}
- {rel(AUDIT_DIR / "gpu_knowledge_vecmat_swarm_queue.v1.json")}
- {rel(backend_probe_path)}
- {rel(AUDIT_DIR / "swarm_dispatch/manifest.v1.json")}
- artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json
- artifacts/gpu/dgx_spark_public_gpu_gate.v1.json

Next-Command:
- scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx
- scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh verify-runtime
- scripts/dev/gpu_knowledge_vecmat_completion_audit.py

Boundary:
- Current state includes local ptxas/package evidence plus Slurm CUDA device runtime evidence for the Vec4 marker; DGX Spark runtime remains optional and unproven.
- Do not mark complete until `goal_status=complete` in `{rel(OUT_JSON)}` and all non-closed blocker records are closed or waived.
"""
    HANDOFF_MD.write_text(handoff, encoding="utf-8")
    print(f"gpu_knowledge_vecmat_completion_audit: {payload['goal_status']} report={rel(OUT_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
