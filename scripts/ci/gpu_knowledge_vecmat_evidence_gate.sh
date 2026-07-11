#!/usr/bin/env bash
# Focused gate for GPU Knowledge Vec/Mat evidence routing.
#
# This gate validates the local ptxas marker probe, saved Slurm runtime receipt,
# imported runtime fixture, audit JSON, queue routing, and explicit boundaries.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

AUDIT_DIR="${SOUNIO_GPU_KNOWLEDGE_AUDIT_DIR:-$ROOT_DIR/artifacts/gpu/knowledge_vecmat_evidence_audit}"
AUDIT_JSON="$AUDIT_DIR/gpu_knowledge_vecmat_evidence_audit.v1.json"
QUEUE_JSON="$AUDIT_DIR/gpu_knowledge_vecmat_swarm_queue.v1.json"
DISPATCH_JSON="$AUDIT_DIR/swarm_dispatch/manifest.v1.json"
PROBE_JSON="$AUDIT_DIR/ptxas_probe/gpu_knowledge_vec4_ptxas_probe.v1.json"
PROBE_PTX="$AUDIT_DIR/ptxas_probe/gpu_knowledge_vec4_aggregate_marker.ptx"
PROBE_RUNNER="$AUDIT_DIR/ptxas_probe/gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu"
BACKEND_PROBE_JSON="$AUDIT_DIR/backend_pack_unpack_probe/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json"
BACKEND_HARNESS="$ROOT_DIR/self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_harness.sio"
SLURM_RUNTIME_PROBE="$ROOT_DIR/scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh"
SLURM_RUNTIME_JSON="$AUDIT_DIR/slurm_runtime_probe/gpu_knowledge_vec4_slurm_runtime_probe.v1.json"
IMPORTED_RUNTIME_PROBE="$ROOT_DIR/scripts/dev/gpu_knowledge_vec4_imported_runtime_probe.sh"
IMPORTED_RUNTIME_JSON="$AUDIT_DIR/imported_runtime_probe/gpu_knowledge_vec4_imported_runtime_probe.v1.json"
DGX_GATE="$ROOT_DIR/scripts/dev/dgx_spark_public_gpu_gate.sh"
DGX_JSON="$ROOT_DIR/artifacts/gpu/dgx_spark_public_gpu_gate.v1.json"
DGX_PACKAGE_DIR="$ROOT_DIR/artifacts/gpu/dgx_spark_public_gpu_package"
PACKAGE_VERIFY="$ROOT_DIR/scripts/dev/gpu_knowledge_vec4_package_verify.py"
RUNTIME_RECEIPT_VERIFY="$ROOT_DIR/scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py"
RUNTIME_RUNBOOK="$ROOT_DIR/scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh"
COMPLETION_AUDIT="$ROOT_DIR/scripts/dev/gpu_knowledge_vecmat_completion_audit.py"
COMPLETION_JSON="$AUDIT_DIR/gpu_knowledge_vecmat_completion_audit.v1.json"
BLOCKERS_JSON="$AUDIT_DIR/gpu_knowledge_vecmat_open_blockers.v1.json"
OPERATIONAL_HANDOFF="$AUDIT_DIR/gpu_knowledge_vecmat_operational_handoff.md"

scripts/dev/gpu_knowledge_vecmat_evidence_audit.sh >/dev/null
if python3 - "$DGX_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
receipt = json.loads(path.read_text(encoding="utf-8"))
marker = receipt.get("gpu_knowledge_vec4_marker", {})
raise SystemExit(0 if receipt.get("status") == "pass" and marker.get("status") == "runtime_pass" else 1)
PY
then
  "$PACKAGE_VERIFY" "$DGX_PACKAGE_DIR/gpu_knowledge_vec4_package_manifest.v1.json" >/dev/null
  "$RUNTIME_RECEIPT_VERIFY" --mode runtime-pass "$DGX_JSON" >/dev/null
else
  "$RUNTIME_RUNBOOK" all-local >/dev/null
  "$PACKAGE_VERIFY" "$DGX_PACKAGE_DIR/gpu_knowledge_vec4_package_manifest.v1.json" >/dev/null
  "$RUNTIME_RECEIPT_VERIFY" --mode not-run "$DGX_JSON" >/dev/null
fi
"$COMPLETION_AUDIT" >/dev/null

python3 - "$AUDIT_JSON" "$QUEUE_JSON" "$DISPATCH_JSON" "$PROBE_JSON" "$PROBE_PTX" "$PROBE_RUNNER" "$BACKEND_PROBE_JSON" "$BACKEND_HARNESS" "$SLURM_RUNTIME_PROBE" "$SLURM_RUNTIME_JSON" "$IMPORTED_RUNTIME_PROBE" "$IMPORTED_RUNTIME_JSON" "$DGX_GATE" "$DGX_JSON" "$DGX_PACKAGE_DIR" "$PACKAGE_VERIFY" "$RUNTIME_RECEIPT_VERIFY" "$RUNTIME_RUNBOOK" "$COMPLETION_AUDIT" "$COMPLETION_JSON" "$BLOCKERS_JSON" "$OPERATIONAL_HANDOFF" <<'PY'
import json
import pathlib
import sys

audit_path, queue_path, dispatch_path, probe_path, ptx_path, runner_path, backend_probe_path, backend_harness_path, slurm_runtime_probe_path, slurm_runtime_json_path, imported_runtime_probe_path, imported_runtime_json_path, dgx_gate_path, dgx_json_path, dgx_package_dir, package_verify_path, runtime_receipt_verify_path, runtime_runbook_path, completion_audit_path, completion_json_path, blockers_json_path, operational_handoff_path = map(pathlib.Path, sys.argv[1:])

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def load(path):
    require(path.exists() and path.stat().st_size > 0, f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def load_optional(path):
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "missing", "reason": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))

audit = load(audit_path)
queue = load(queue_path)
dispatch = load(dispatch_path)
probe = load(probe_path)
backend_probe = load(backend_probe_path)
slurm_runtime = load_optional(slurm_runtime_json_path)
imported_runtime = load_optional(imported_runtime_json_path)
dgx_json = load(dgx_json_path)
ptx = ptx_path.read_text(encoding="utf-8")
runner = runner_path.read_text(encoding="utf-8")
dgx_gate = dgx_gate_path.read_text(encoding="utf-8")
package_verify = package_verify_path.read_text(encoding="utf-8")
runtime_receipt_verify = runtime_receipt_verify_path.read_text(encoding="utf-8")
runtime_runbook = runtime_runbook_path.read_text(encoding="utf-8")
slurm_runtime_probe = slurm_runtime_probe_path.read_text(encoding="utf-8")
imported_runtime_probe = imported_runtime_probe_path.read_text(encoding="utf-8")
completion_audit = completion_audit_path.read_text(encoding="utf-8")
operational_handoff = operational_handoff_path.read_text(encoding="utf-8")
backend_harness = backend_harness_path.read_text(encoding="utf-8")
package_manifest_path = dgx_package_dir / "gpu_knowledge_vec4_package_manifest.v1.json"
package_manifest = load(package_manifest_path)
completion = load(completion_json_path)
open_blockers = load(blockers_json_path)

require(audit.get("schema") == "sounio.gpu-knowledge-vecmat-evidence-audit.v1", "bad audit schema")
require(audit.get("status") == "pass", "audit did not pass")
dgx_runtime_pass = dgx_json.get("status") == "pass" and dgx_json.get("gpu_knowledge_vec4_marker", {}).get("status") == "runtime_pass"
slurm_runtime_pass = slurm_runtime.get("status") == "pass" and slurm_runtime.get("runtime_launch_contract", {}).get("status") == "cuda_device_runtime_pass"
runtime_pass = dgx_runtime_pass or slurm_runtime_pass
imported_runtime_pass = imported_runtime.get("status") == "pass" and imported_runtime.get("runtime_contract", {}).get("status") == "imported_runtime_pass"
expected_completion_ready = runtime_pass and backend_probe.get("status") == "pass" and imported_runtime_pass
expected_completion_gaps = []
if not runtime_pass:
    expected_completion_gaps.append("dgx_cuda_device_runtime_execution")
if backend_probe.get("status") != "pass":
    expected_completion_gaps.append("automatic_backend_pack_unpack")
if not imported_runtime_pass:
    expected_completion_gaps.append("imported_runtime_fixture")
require(audit.get("completion_gaps") == expected_completion_gaps, "unexpected completion gaps")
require(audit.get("completion_ready") is expected_completion_ready, "audit completion readiness mismatch")

require(probe.get("status") == "pass", "ptxas probe did not pass")
contract = probe.get("runtime_launch_contract", {})
require(contract.get("status") == "ptxas_only_not_launched", "runtime contract overclaimed launch")
require(contract.get("kernel") == "gpu_knowledge_vec4_aggregate_marker", "bad marker kernel")
require(contract.get("params") == ["out_ptr"], "bad marker params")
require(contract.get("copyback_offsets_bytes") == [0, 32, 64, 96], "bad copyback offsets")
require(contract.get("expected_value_lanes") == [1.0, 2.0, 3.0, 4.0], "bad expected lanes")

require(backend_probe.get("schema") == "sounio.gpu-knowledge-vec4-backend-pack-unpack-probe.v1", "bad backend probe schema")
require(backend_probe.get("status") in {"pass", "blocked"}, "backend probe must be pass or classified blocked")
backend_contract = backend_probe.get("backend_ir_contract", {})
require(backend_contract.get("kernel") == "gpu_vec4_pack_unpack", "backend probe kernel mismatch")
require(backend_contract.get("lane_offsets_bytes") == [0, 32, 64, 96], "backend probe lane offsets mismatch")
require(backend_contract.get("lane_type") == "f64", "backend probe lane type mismatch")
require("does_not_claim_imported_compiler_lowering" in backend_probe.get("boundaries", []), "backend probe missing imported-lowering nonclaim")
require("does_not_claim_cuda_device_runtime_execution" in backend_probe.get("boundaries", []), "backend probe missing device-runtime nonclaim")
lean_extract = backend_probe.get("lean_extract_fallback", {})
require(lean_extract.get("classification") == "reference_extract_not_production_backend", "lean extract fallback classification mismatch")
require("does_not_claim_canonical_gpu_kernel_ir_backend" in lean_extract.get("boundaries", []), "lean extract fallback missing canonical-backend nonclaim")
if lean_extract.get("status") == "pass":
    require(lean_extract.get("check_exit_code") == 0, "lean extract check must pass")
    require(lean_extract.get("run_exit_code") == 0, "lean extract run must pass")
    require(lean_extract.get("ptxas_exit_code") == 0, "lean extract ptxas must pass")
    require(lean_extract.get("reason") == "lean_extract_ptxas_pass", "lean extract pass reason mismatch")
    for key in ["ptx", "cubin", "raw", "log"]:
        require(lean_extract.get("artifacts", {}).get(key, {}).get("present") is True, f"lean extract missing artifact {key}")
if backend_probe.get("status") == "blocked":
    backend_reason = backend_probe.get("reason")
    require(backend_reason in {"souc_ir_max_instrs_before_ptx", "souc_runtime_segfault_after_compile", "souc_check_failed", "souc_harness_failed"}, "backend blocked reason mismatch")
    souc = backend_probe.get("souc", {})
    if backend_reason == "souc_ir_max_instrs_before_ptx":
        require("IR_MAX_INSTRS" in souc.get("run_log_tail", ""), "backend blocked log missing IR_MAX_INSTRS evidence")
    if backend_reason == "souc_runtime_segfault_after_compile":
        require(souc.get("check_exit_code") == 0, "backend segfault blocker must have check pass")
        require(souc.get("run_exit_code") == 139, "backend segfault blocker must have run exit 139")
        run_log_tail = souc.get("run_log_tail", "")
        require("Segmentation fault" in run_log_tail, "backend blocked log missing segfault evidence")
        compiled_before_failure = souc.get("compiled_before_runtime_failure")
        require(compiled_before_failure in {True, False}, "backend segfault blocker missing compile-before-failure classification")
        if compiled_before_failure is False:
            require("imported_compile:" in run_log_tail or "Native compilation failed" in run_log_tail, "backend compile-path segfault missing imported-lower evidence")
        min_import = backend_probe.get("minimal_import_runtime_probe", {})
        require(min_import.get("classification") == "diagnostic_not_backend_contract", "backend segfault missing minimal import diagnostic classification")
        require("diagnoses_minimal_modular_import_runtime_only" in min_import.get("boundaries", []), "minimal import diagnostic missing boundary")
        require(min_import.get("reason") in {"minimal_imported_elf_segfault_after_compile", "minimal_imported_elf_pass"}, "minimal import diagnostic reason mismatch")
        require(min_import.get("no_import", {}).get("check_exit_code") == 0, "minimal no-import check must pass")
        require(min_import.get("no_import", {}).get("run_exit_code") == 0, "minimal no-import run must pass")
        require(min_import.get("imported", {}).get("check_exit_code") == 0, "minimal imported check must pass")
        if min_import.get("reason") == "minimal_imported_elf_segfault_after_compile":
            require(min_import.get("imported", {}).get("run_exit_code") == 139, "minimal imported run must segfault")
        else:
            require(min_import.get("imported", {}).get("run_exit_code") == 0, "minimal imported run must pass")
        require("Compilation successful!" in min_import.get("imported", {}).get("run_log_tail", ""), "minimal imported diagnostic missing compile-success evidence")
else:
    require(backend_contract.get("status") == "proved", "backend pass did not prove contract")
    require(backend_probe.get("artifacts", {}).get("ptx", {}).get("present") is True, "backend pass missing PTX")
    require(backend_probe.get("artifacts", {}).get("cubin", {}).get("present") is True, "backend pass missing cubin")
require("fn main()" in backend_harness, "backend harness missing main")
require("gpu_lower_vec4_pack_unpack_ops_to_ptx" in backend_harness, "backend harness missing direct backend Vec4 lower call")
require("load_param(1, 0)" in backend_harness and "store_f64(8, 4)" in backend_harness, "backend harness missing Vec4 op sequence")
for offset in ["32", "64", "96"]:
    require("add_offset" in backend_harness and offset in backend_harness, f"backend harness missing offset {offset}")

require("SOUNIO_AGG_RUNTIME_CONTRACT" in ptx, "PTX missing aggregate runtime contract marker")
require(".param .u64 out_ptr" in ptx, "PTX missing out_ptr param")
require("st.global.f64 [%rd1+96]" in ptx, "PTX missing final lane store")
require("cuLaunchKernel" in runner, "runner missing cuLaunchKernel")
require("cuMemcpyDtoH" in runner, "runner missing cuMemcpyDtoH")

require(queue.get("lane_count") == 3, "queue lane_count mismatch")
require(queue.get("write_scope_overlap_count") == 0, "queue write scope overlap")
require(queue.get("completion_ready") is expected_completion_ready, "queue completion readiness mismatch")
require(queue.get("status") == ("complete" if expected_completion_ready else "open"), "queue status mismatch")
lane_ids = [lane.get("id") for lane in queue.get("lanes", [])]
require(lane_ids == ["gpu_backend_pack_unpack", "cuda_runtime_vecmat_artifact", "imported_runtime_lower_array"], "queue lane order mismatch")
require(queue["lanes"][0].get("status") in {"backend_ir_pack_unpack_ptxas_pass", "blocked_souc_ir_max_instrs_before_ptx", "blocked_souc_runtime_segfault_after_compile", "blocked_souc_check_failed", "blocked_souc_harness_failed"}, "backend lane status mismatch")
require(queue["lanes"][0].get("evidence") == "artifacts/gpu/knowledge_vecmat_evidence_audit/backend_pack_unpack_probe/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json", "backend lane evidence mismatch")
expected_runtime_lane_status = "cuda_device_runtime_pass" if runtime_pass else "ptxas_contract_ready_not_launched"
require(queue["lanes"][1].get("status") == expected_runtime_lane_status, "runtime lane status mismatch")
runtime_routes = queue["lanes"][1].get("runtime_routes", [])
require([route.get("id") for route in runtime_routes] == ["slurm_gpu_runtime_probe", "dgx_spark_marker_package_only", "dgx_spark_public_gpu_gate"], "runtime lane route ids mismatch")
require(runtime_routes[0].get("status") == ("cuda_device_runtime_pass" if slurm_runtime_pass else "not_run_or_blocked"), "slurm route status mismatch")
require(runtime_routes[0].get("command") == "scripts/dev/gpu_knowledge_vec4_slurm_runtime_probe.sh", "slurm route command mismatch")
require(runtime_routes[0].get("evidence") == "artifacts/gpu/knowledge_vecmat_evidence_audit/slurm_runtime_probe/gpu_knowledge_vec4_slurm_runtime_probe.v1.json", "slurm route evidence mismatch")
require(runtime_routes[1].get("status") == "local_package_pass", "package-only route status mismatch")
require(runtime_routes[1].get("command") == "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh package-only", "package-only route command mismatch")
require(runtime_routes[1].get("evidence") == "artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json", "package-only route evidence mismatch")
require(runtime_routes[1].get("verify_command") == "scripts/dev/gpu_knowledge_vec4_package_verify.py artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json", "package-only route verify command mismatch")
require(runtime_routes[1].get("receipt_verify_command") == "scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode not-run artifacts/gpu/dgx_spark_public_gpu_gate.v1.json", "package-only receipt verify command mismatch")
require(runtime_routes[1].get("runbook_command") == "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh all-local", "package-only runbook command mismatch")
require(runtime_routes[2].get("status") == "wired_opt_in_not_run", "DGX route status mismatch")
require(runtime_routes[2].get("command") == "scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx", "DGX route command mismatch")
require(runtime_routes[2].get("verify_command") == "scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass artifacts/gpu/dgx_spark_public_gpu_gate.v1.json", "DGX route runtime verify command mismatch")
require(queue["lanes"][2].get("status") == ("imported_runtime_pass" if imported_runtime_pass else "blocked_or_unproved"), "imported lane status mismatch")
require(queue["lanes"][2].get("evidence") == "artifacts/gpu/knowledge_vecmat_evidence_audit/imported_runtime_probe/gpu_knowledge_vec4_imported_runtime_probe.v1.json", "imported lane evidence mismatch")
require("gpt-5.4-mini" in queue.get("model_routing_summary", {}), "missing mini model route")
require("current-codex" in queue.get("model_routing_summary", {}), "missing current-codex route")

require(dispatch.get("lane_count") == 3, "dispatch lane_count mismatch")
require(len(dispatch.get("handoffs", [])) == 3, "dispatch handoff count mismatch")
require(dispatch.get("completion_ready") is expected_completion_ready, "dispatch completion readiness mismatch")
subagent_plan = dispatch.get("subagent_dispatch_plan", [])
require(len(subagent_plan) == 3, "dispatch subagent plan count mismatch")
require([agent.get("id") for agent in subagent_plan] == ["review_dgx_marker_route", "map_cuda_runtime_next_route", "classify_backend_ptx_routes"], "dispatch subagent ids mismatch")
require(all(agent.get("model") == "gpt-5.4-mini" for agent in subagent_plan), "dispatch subagent model mismatch")
require(all(agent.get("mode") == "read_only" for agent in subagent_plan), "dispatch subagent mode mismatch")
require(all(agent.get("write_scope") == [] for agent in subagent_plan), "dispatch subagent write scope must be empty")
if dispatch.get("live_subagent_receipt_status") == "present":
    require(dispatch.get("live_subagent_receipt_valid") is True, "live subagent receipt invalid")
    require(dispatch.get("live_subagent_models") == ["gpt-5.4-mini"], "live subagent models mismatch")
    require(sorted(dispatch.get("live_subagent_tasks", [])) == ["classify_backend_ptx_routes", "map_cuda_runtime_next_route", "review_dgx_marker_route"], "live subagent tasks mismatch")

require("DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" in dgx_gate, "DGX gate missing marker opt-in flag")
require("DGX_SPARK_PACKAGE_ONLY" in dgx_gate, "DGX gate missing package-only flag")
require("DGX_SPARK_PUBLIC_KERNELS" in dgx_gate, "DGX gate missing public-kernels selector")
require("write_package_manifest" in dgx_gate, "DGX gate missing package manifest writer")
require("gpu_knowledge_vec4_aggregate_marker.ptx" in dgx_gate, "DGX gate missing marker PTX path")
require("gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu" in dgx_gate, "DGX gate missing marker runner path")
require("gpu_knowledge_vec4_aggregate_marker.cubin" in dgx_gate, "DGX gate missing marker cubin path")
require("run_gpu_knowledge_vec4_marker" in dgx_gate, "DGX gate missing marker runtime binary")
require('"status": marker_status' in dgx_gate, "DGX gate missing marker status JSON field")
require('GPU_KNOWLEDGE_MARKER_STATUS="ptxas_only_not_launched"' in dgx_gate, "DGX gate missing ptxas-only marker status")
require('GPU_KNOWLEDGE_MARKER_STATUS="runtime_pass"' in dgx_gate, "DGX gate missing runtime_pass marker status")
require("PASS gpu_knowledge_vec4_aggregate_marker" in dgx_gate, "DGX gate missing runtime PASS marker enforcement")
require("package_only_no_remote_ssh" in dgx_gate, "DGX gate missing package-only no-ssh boundary")
require("package_only_does_not_claim_dgx_toolchain_or_runtime" in dgx_gate, "DGX gate missing package-only nonclaim")
require("does_not_claim_automatic_backend_pack_unpack" in dgx_gate, "DGX gate missing backend boundary")
require("does_not_claim_imported_runtime_fixture" in dgx_gate, "DGX gate missing imported fixture boundary")
require("gpu_knowledge_vec4_package_verify" in package_verify, "package verifier missing identity string")
require("local_package_only_not_remote_not_launched" in package_verify, "package verifier missing runtime nonclaim status")
require("does_not_claim_cuda_device_runtime_execution" in package_verify, "package verifier missing device runtime nonclaim")
require("gpu_knowledge_vec4_runtime_receipt_verify" in runtime_receipt_verify, "runtime receipt verifier missing identity string")
require("runtime-pass" in runtime_receipt_verify, "runtime receipt verifier missing runtime-pass mode")
require("not-run" in runtime_receipt_verify, "runtime receipt verifier missing not-run mode")
require("PASS gpu_knowledge_vec4_aggregate_marker" in runtime_receipt_verify, "runtime receipt verifier missing PASS marker")
require("gpu_knowledge_vec4_dgx_runtime_runbook" in runtime_runbook, "runtime runbook missing identity string")
for mode in ["package-only", "verify-package", "run-dgx", "verify-runtime", "all-local"]:
    require(mode in runtime_runbook, f"runtime runbook missing mode {mode}")
require("DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER=1" in runtime_runbook, "runtime runbook missing DGX marker flag")
require('DGX_SPARK_PUBLIC_KERNELS="${DGX_SPARK_PUBLIC_KERNELS:-0}"' in runtime_runbook, "runtime runbook must default run-dgx to marker-only public-kernels-off")
require("DGX_SPARK_PACKAGE_ONLY=1" in runtime_runbook, "runtime runbook missing package-only flag")
require("--mode runtime-pass" in runtime_runbook, "runtime runbook missing runtime-pass verifier")
require("--mode not-run" in runtime_runbook, "runtime runbook missing not-run verifier")
require("gpu_knowledge_vec4_slurm_runtime_probe" in slurm_runtime_probe, "slurm runtime probe missing identity string")
require("dlopen(\"libcuda.so.1\"" in slurm_runtime_probe, "slurm runtime probe must use CUDA Driver API dlopen")
require("cuModuleLoadDataEx" in slurm_runtime_probe, "slurm runtime probe must load PTX through CUDA driver JIT")
if slurm_runtime_pass:
    require("PASS gpu_knowledge_vec4_aggregate_marker" in slurm_runtime.get("stdout_tail", ""), "slurm runtime pass missing PASS marker")
    require(slurm_runtime.get("runtime_launch_contract", {}).get("copyback_offsets_bytes") == [0, 32, 64, 96], "slurm runtime copyback offsets mismatch")
    require("does_not_claim_dgx_spark_runtime" in slurm_runtime.get("boundaries", []), "slurm runtime must not claim DGX Spark")
require("gpu_knowledge_vec4_imported_runtime_probe" in imported_runtime_probe, "imported runtime probe missing identity string")
require("gpu_hlir_vec4_lane_plan_imported.sio" in imported_runtime_probe, "imported runtime probe missing harness")
if imported_runtime_pass:
    require("PASS gpu_hlir_vec4_lane_plan_imported" in imported_runtime.get("stdout_tail", ""), "imported runtime pass missing PASS marker")
    require(imported_runtime.get("souc", {}).get("check_exit_code") == 0, "imported runtime check did not pass")
    require(imported_runtime.get("souc", {}).get("run_exit_code") == 0, "imported runtime run did not pass")
    require(imported_runtime.get("runtime_contract", {}).get("copyback_offsets_bytes") == [0, 32, 64, 96], "imported runtime offsets mismatch")
    require("does_not_claim_general_imported_runtime_correctness" in imported_runtime.get("boundaries", []), "imported runtime missing general nonclaim")
require("gpu_knowledge_vecmat_completion_audit" in completion_audit, "completion auditor missing identity string")
require("do_not_mark_goal_complete_from_package_only_or_ptxas_only_evidence" in completion_audit, "completion auditor missing package-only noncompletion boundary")

require(dgx_json.get("schema") == "sounio.dgx-spark-public-gpu-gate.v1", "bad DGX JSON schema")
require(dgx_json.get("status") == "pass", "DGX report did not pass")
require(dgx_json.get("public_kernels_enabled") is False, "DGX marker test must disable public kernels")
marker_status = dgx_json.get("gpu_knowledge_vec4_marker", {}).get("status")
if marker_status == "runtime_pass":
    require(dgx_json.get("reason") == "dgx_spark_public_gpu_validated", "DGX runtime reason mismatch")
    require(dgx_json.get("package_only") is False, "runtime DGX report cannot be package-only")
    require("PASS gpu_knowledge_vec4_aggregate_marker" in dgx_json.get("gpu_knowledge_vec4_marker", {}).get("runtime_output", ""), "DGX runtime missing PASS marker")
    for key in ("hostname", "uname_m", "ptxas_version", "nvcc_version"):
        require(dgx_json.get("remote", {}).get(key), f"DGX runtime missing remote {key}")
    require("dgx_spark_is_cuda_toolchain_and_runtime_authority" in dgx_json.get("boundaries", []), "DGX runtime JSON missing authority boundary")
    require("package_only_does_not_claim_dgx_toolchain_or_runtime" not in dgx_json.get("boundaries", []), "DGX runtime JSON still has package-only nonclaim")
else:
    require(dgx_json.get("reason") == "dgx_spark_package_only_prepared", "DGX package-only reason mismatch")
    require(dgx_json.get("package_only") is True, "DGX package-only report must be package-only")
    require(dgx_json.get("package_manifest", "").endswith("gpu_knowledge_vec4_package_manifest.v1.json"), "DGX JSON missing package manifest path")
    require(marker_status == "local_ptxas_only_not_remote_not_launched", "DGX marker package status mismatch")
    require("package_only_no_remote_ssh" in dgx_json.get("boundaries", []), "DGX JSON missing package-only no-ssh boundary")
    require("package_only_does_not_claim_dgx_toolchain_or_runtime" in dgx_json.get("boundaries", []), "DGX JSON missing package-only nonclaim")
    require("dgx_spark_is_cuda_toolchain_and_runtime_authority" not in dgx_json.get("boundaries", []), "DGX package-only JSON overclaims remote authority")
require((dgx_package_dir / "gpu_knowledge_vec4_aggregate_marker.ptx").exists(), "DGX package missing marker PTX")
require((dgx_package_dir / "gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu").exists(), "DGX package missing marker runner")
require((dgx_package_dir / "gpu_knowledge_vec4_aggregate_marker.local-ptxas.cubin").exists(), "DGX package missing local ptxas cubin")
require(package_manifest.get("schema") == "sounio.gpu-knowledge-vec4-dgx-package.v1", "bad package manifest schema")
require(package_manifest.get("status") == "pass", "package manifest did not pass")
require(package_manifest.get("runtime_launch_contract", {}).get("status") == "local_package_only_not_remote_not_launched", "package manifest overclaimed runtime")
require(package_manifest.get("runtime_launch_contract", {}).get("copyback_offsets_bytes") == [0, 32, 64, 96], "package manifest copyback offsets mismatch")
for key in ["ptx", "runtime_harness", "local_ptxas_cubin"]:
    entry = package_manifest.get("files", {}).get(key, {})
    require(entry.get("bytes", 0) > 0, f"package manifest file {key} has no bytes")
    require(len(entry.get("sha256", "")) == 64, f"package manifest file {key} sha256 invalid")
require("package_only_no_remote_ssh" in package_manifest.get("boundaries", []), "package manifest missing no-ssh boundary")
require("does_not_claim_cuda_device_runtime_execution" in package_manifest.get("boundaries", []), "package manifest missing device runtime nonclaim")
require(completion.get("schema") == "sounio.gpu-knowledge-vecmat-completion-audit.v1", "bad completion audit schema")
require(completion.get("goal_status") == ("complete" if expected_completion_ready else "not_complete"), "completion audit goal status mismatch")
require(completion.get("completion_ready") is expected_completion_ready, "completion audit readiness mismatch")
expected_completion_blockers = []
if not runtime_pass:
    expected_completion_blockers.append("dgx_cuda_device_runtime_execution")
if backend_probe.get("status") != "pass":
    expected_completion_blockers.append("automatic_backend_pack_unpack")
if not imported_runtime_pass:
    expected_completion_blockers.append("imported_runtime_fixture")
require(completion.get("completion_blockers", []) == expected_completion_blockers, "completion blockers mismatch")
if backend_probe.get("status") == "pass":
    require("automatic_backend_pack_unpack" not in completion.get("completion_blockers", []), "completion audit kept closed backend blocker open")
if runtime_pass:
    require("dgx_cuda_device_runtime_execution" not in completion.get("completion_blockers", []), "completion audit kept closed runtime blocker open")
if imported_runtime_pass:
    require("imported_runtime_fixture" not in completion.get("completion_blockers", []), "completion audit kept closed imported blocker open")
require("do_not_mark_goal_complete_from_package_only_or_ptxas_only_evidence" in completion.get("boundaries", []), "completion audit missing noncompletion boundary")
require(open_blockers.get("schema") == "sounio.gpu-knowledge-vecmat-open-blockers.v1", "bad open blockers schema")
blockers = open_blockers.get("blockers", [])
require(len(blockers) == 3, "open blocker count mismatch")
blocker_ids = {item.get("Blocker-ID") for item in blockers}
require(blocker_ids == {
    "BLK-20260706-gpu-knowledge-vecmat-dgx-runtime",
    "BLK-20260706-gpu-knowledge-vecmat-backend-pack-unpack",
    "BLK-20260706-gpu-knowledge-vecmat-imported-runtime",
}, "open blocker ids mismatch")
required_blocker_fields = [
    "Blocker-ID", "Status", "Severity", "Class", "Owner", "Lane", "Worktree",
    "Branch", "Files-Owned", "Do-Not-Touch", "Repro", "Observed", "Expected",
    "Acceptance-Gate", "Evidence-Level", "Evidence", "Fallback-Path",
    "Legacy-Kept", "LLM-Offload", "Next-Action",
]
for blocker in blockers:
    for field in required_blocker_fields:
        require(blocker.get(field), f"open blocker missing {field}")
    require(blocker.get("Evidence-Level") in {"E2", "E3", "E4"}, "open blocker evidence level too weak")
    require(blocker.get("Status") in {"classified", "owned", "fixing", "review-ready", "merge-ready", "closed", "waived"}, "open blocker status invalid")
if backend_probe.get("status") == "pass":
    backend_blockers = [item for item in blockers if item.get("Blocker-ID") == "BLK-20260706-gpu-knowledge-vecmat-backend-pack-unpack"]
    require(len(backend_blockers) == 1 and backend_blockers[0].get("Status") == "closed", "backend blocker record must be closed when probe passes")
if runtime_pass:
    runtime_blockers = [item for item in blockers if item.get("Blocker-ID") == "BLK-20260706-gpu-knowledge-vecmat-dgx-runtime"]
    require(len(runtime_blockers) == 1 and runtime_blockers[0].get("Status") == "closed", "runtime blocker record must be closed when runtime probe passes")
if imported_runtime_pass:
    imported_blockers = [item for item in blockers if item.get("Blocker-ID") == "BLK-20260706-gpu-knowledge-vecmat-imported-runtime"]
    require(len(imported_blockers) == 1 and imported_blockers[0].get("Status") == "closed", "imported blocker record must be closed when imported probe passes")
for field in [
    "Current-SHA:", "Current-Branch:", "Current-Worktree:", "Dirty-Status:",
    "Current-Goal-Status:", "Completion-Blockers:", "Owned-Files:", "Do-Not-Touch:", "Last-Green-Gates:", "Failing-Gates:",
    "Blocker-Records:", "Artifacts:", "Next-Command:",
]:
    require(field in operational_handoff, f"operational handoff missing {field}")
for blocker_id in blocker_ids:
    require(blocker_id in operational_handoff, f"operational handoff missing {blocker_id}")
require("scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh run-dgx" in operational_handoff, "operational handoff missing run-dgx next command")
require(f"Current-Goal-Status: {'complete' if expected_completion_ready else 'not_complete'}" in operational_handoff, "operational handoff goal status mismatch")
require("goal_status=complete" in operational_handoff, "operational handoff missing completion boundary")
PY

echo "gpu_knowledge_vecmat_evidence_gate: PASS report=${AUDIT_JSON#$ROOT_DIR/}"
