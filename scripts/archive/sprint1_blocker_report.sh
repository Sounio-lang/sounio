#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_MD="${1:-$ROOT_DIR/artifacts/sprint1/sprint1_blocker_report.md}"
OUT_JSON="${SOUNIO_SPRINT1_BLOCKER_JSON:-$ROOT_DIR/artifacts/sprint1/sprint1_blocker_report.v1.json}"
CRITICAL_JSON="${SOUNIO_SPRINT1_CRITICAL_JSON:-$ROOT_DIR/artifacts/sprint1/critical_bug_fixes_gate.v1.json}"
PERF_JSON="${SOUNIO_SPRINT1_PERF_JSON:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_gate.v1.json}"
RUN_LANE_JSON="${SOUNIO_SPRINT1_RUN_LANE_JSON:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_run_lane.v1.json}"
JIT_DEBUG_JSON="${SOUNIO_SPRINT1_JIT_DEBUG_JSON:-$ROOT_DIR/artifacts/sprint1/jit_runtime_debug.v1.json}"
JIT_IMPORT_STRESS_JSON="${SOUNIO_SPRINT1_JIT_IMPORT_STRESS_JSON:-$ROOT_DIR/artifacts/sprint1/jit_import_stress_debug.v1.json}"
JIT_WASM_STACK_JSON="${SOUNIO_SPRINT1_JIT_WASM_STACK_JSON:-$ROOT_DIR/artifacts/sprint1/jit_wasm_stack_depth_debug.v1.json}"
JIT_IR_TIMEOUT_JSON="${SOUNIO_SPRINT1_JIT_IR_TIMEOUT_JSON:-$ROOT_DIR/artifacts/sprint1/jit_ir_timeout_repro.v1.json}"
JIT_BIGPUSH_JSON="${SOUNIO_SPRINT1_JIT_BIGPUSH_JSON:-$ROOT_DIR/artifacts/sprint1/jit_root_cause_bigpush.v1.json}"

mkdir -p "$(dirname "$OUT_MD")"
mkdir -p "$(dirname "$OUT_JSON")"

python3 - "$OUT_MD" "$OUT_JSON" "$CRITICAL_JSON" "$PERF_JSON" "$RUN_LANE_JSON" "$JIT_DEBUG_JSON" "$JIT_IMPORT_STRESS_JSON" "$JIT_WASM_STACK_JSON" "$JIT_IR_TIMEOUT_JSON" "$JIT_BIGPUSH_JSON" <<'PY'
import datetime as dt
import json
import os
import sys

out_md, out_json, critical_path, perf_path, run_lane_path, jit_debug_path, jit_import_path, jit_wasm_path, jit_ir_path, jit_bigpush_path = sys.argv[1:11]

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

critical = load_json(critical_path)
perf = load_json(perf_path)
run_lane = load_json(run_lane_path)
jit_debug = load_json(jit_debug_path)
jit_import = load_json(jit_import_path)
jit_wasm = load_json(jit_wasm_path)
jit_ir = load_json(jit_ir_path)
jit_bigpush = load_json(jit_bigpush_path)

critical_steps = critical.get("steps", []) if critical else []
non_perf_steps = [
    s for s in critical_steps
    if s.get("name") not in {"int_to_string_benchmark", "int_to_string_benchmark_run_lane"}
]
correctness_closed = bool(non_perf_steps) and all(s.get("status") == "pass" for s in non_perf_steps)

perf_status = (perf or {}).get("status", "not_run")
perf_reason = (perf or {}).get("reason", "missing")
perf_mode = (perf or {}).get("mode", "none")
perf_net = ((perf or {}).get("metrics") or {}).get("net_seconds")
perf_runner = (perf or {}).get("runner", "")

run_status = (run_lane or {}).get("status", "not_run")
run_reason = (run_lane or {}).get("reason", "missing")
run_net = ((run_lane or {}).get("metrics") or {}).get("net_seconds")
run_runner = (run_lane or {}).get("runner", "")

jit_import_status = (jit_import or {}).get("status", "not_run")
jit_import_reason = (jit_import or {}).get("reason", "missing")
jit_import_runner = (jit_import or {}).get("runner", "")
jit_import_summary = (jit_import or {}).get("summary") or {}
jit_import_stack_count = int(jit_import_summary.get("stack_overflow_count") or 0)
jit_import_timeout_count = int(jit_import_summary.get("timeout_count") or 0)
jit_import_panic_count = int(jit_import_summary.get("jit_panic_count") or 0)

jit_wasm_status = (jit_wasm or {}).get("status", "not_run")
jit_wasm_reason = (jit_wasm or {}).get("reason", "missing")
jit_wasm_stage = (jit_wasm or {}).get("first_failure_stage", "")
jit_wasm_summary = (jit_wasm or {}).get("summary") or {}
jit_wasm_stack_count = int(jit_wasm_summary.get("stack_overflow_count") or 0)
jit_wasm_timeout_count = int(jit_wasm_summary.get("timeout_count") or 0)
jit_wasm_compile_count = int(jit_wasm_summary.get("compile_error_count") or 0)

jit_ir_status = (jit_ir or {}).get("status", "not_run")
jit_ir_reason = (jit_ir or {}).get("reason", "missing")
jit_ir_lane_split = (jit_ir or {}).get("lane_split", "unknown")
jit_ir_summary = (jit_ir or {}).get("summary") or {}
jit_ir_timeout_count = int(jit_ir_summary.get("timeout_count") or 0)
jit_ir_compile_count = int(jit_ir_summary.get("compile_error_count") or 0)

jit_bigpush_status = (jit_bigpush or {}).get("status", "not_run")
jit_bigpush_reason = (jit_bigpush or {}).get("reason", "missing")
jit_bigpush_hypotheses = (jit_bigpush or {}).get("hypotheses") or []
jit_bigpush_families = (jit_bigpush or {}).get("families") or {}
jit_bigpush_wasm_first = ((jit_bigpush_families.get("wasm") or {}).get("first_failure_probe")) or ""
jit_bigpush_ir_first = ((jit_bigpush_families.get("ir") or {}).get("first_failure_probe")) or ""

jit_summary = (jit_debug or {}).get("summary") or {}
jit_blocked = bool(jit_summary.get("jit_runtime_blocked", False)) if jit_debug else None
jit_usable_count = jit_summary.get("usable_candidate_count") if jit_debug else None
jit_blockers = (jit_debug or {}).get("blockers") or []
jit_has_string_issue = any(
    str(b).startswith("string:segfault:")
    or str(b).startswith("string:timeout:")
    or str(b).startswith("string:missing_marker:")
    for b in jit_blockers
)
jit_has_source_runtime_timeout = any(str(b).startswith("source_runtime:timeout:") for b in jit_blockers)
jit_has_source_runtime_stack_overflow = any(str(b).startswith("source_runtime:stack_overflow:") for b in jit_blockers)
jit_has_main_source_timeout = any(str(b).startswith("source_main:timeout:") for b in jit_blockers)
jit_has_main_source_stack_overflow = any(str(b).startswith("source_main:stack_overflow:") for b in jit_blockers)

blocker_classes = []
if perf_reason in {"jit_string_runtime_unavailable", "jit_runner_unusable", "jit_runner_required_missing"}:
    blocker_classes.append("jit_runtime_blocked")
if perf_status == "fail" and perf_reason == "target_not_met" and perf_mode == "run":
    blocker_classes.append("run_mode_perf_target_miss")
if perf_status == "fail" and perf_reason == "target_not_met" and perf_mode == "jit":
    blocker_classes.append("jit_mode_perf_target_miss")
if jit_blocked and jit_has_source_runtime_timeout:
    blocker_classes.append("jit_source_probe_timeout")
if jit_blocked and jit_has_source_runtime_stack_overflow:
    blocker_classes.append("jit_source_probe_stack_overflow")
if jit_blocked and jit_has_string_issue:
    blocker_classes.append("jit_string_probe_issue")
if jit_has_main_source_timeout:
    blocker_classes.append("jit_main_source_probe_timeout_non_gating")
if jit_has_main_source_stack_overflow:
    blocker_classes.append("jit_main_source_probe_stack_overflow_non_gating")
if jit_import_status == "fail" and jit_import_reason == "stack_overflow_detected":
    blocker_classes.append("jit_import_probe_stack_overflow_non_gating")
if jit_import_status == "fail" and jit_import_reason == "timeout_detected":
    blocker_classes.append("jit_import_probe_timeout_non_gating")
if jit_import_status == "fail" and jit_import_reason == "jit_panic_detected":
    blocker_classes.append("jit_import_probe_jit_panic_non_gating")
if jit_import_status == "not_run":
    blocker_classes.append("jit_import_probe_not_run_non_gating")
if jit_wasm_status == "fail" and jit_wasm_reason == "stack_overflow_detected":
    blocker_classes.append("jit_wasm_depth_stack_overflow_non_gating")
if jit_wasm_status == "fail" and jit_wasm_reason == "timeout_detected":
    blocker_classes.append("jit_wasm_depth_timeout_non_gating")
if jit_wasm_status == "fail" and jit_wasm_reason == "compile_error_detected":
    blocker_classes.append("jit_wasm_depth_compile_error_non_gating")
if jit_wasm_status == "not_run":
    blocker_classes.append("jit_wasm_depth_not_run_non_gating")
if jit_ir_status == "fail" and jit_ir_reason == "ir_module_timeout_detected":
    blocker_classes.append("jit_ir_timeout_lane_non_gating")
if jit_ir_status == "fail" and jit_ir_reason == "parser_dependency_probe_failure":
    blocker_classes.append("jit_ir_parser_dependency_lane_non_gating")
if jit_ir_status == "fail" and jit_ir_reason == "compile_error_detected":
    blocker_classes.append("jit_ir_compile_error_lane_non_gating")
if jit_ir_status == "not_run":
    blocker_classes.append("jit_ir_timeout_repro_not_run_non_gating")
if jit_bigpush_status == "fail":
    blocker_classes.append("jit_bigpush_failure_non_gating")
if jit_bigpush_status == "not_run":
    blocker_classes.append("jit_bigpush_not_run_non_gating")
if "wasm_real_only_failure" in jit_bigpush_hypotheses:
    blocker_classes.append("jit_bigpush_wasm_real_only_non_gating")
if "ir_real_only_failure" in jit_bigpush_hypotheses:
    blocker_classes.append("jit_bigpush_ir_real_only_non_gating")
if "wasm_shadow_failure_reproduced" in jit_bigpush_hypotheses:
    blocker_classes.append("jit_bigpush_wasm_shadow_repro_non_gating")
if "ir_shadow_failure_reproduced" in jit_bigpush_hypotheses:
    blocker_classes.append("jit_bigpush_ir_shadow_repro_non_gating")
if perf_status == "not_run":
    blocker_classes.append("perf_not_run")

diagnostic_classes = [c for c in blocker_classes if c.endswith("_non_gating")]
blocking_classes = [c for c in blocker_classes if not c.endswith("_non_gating")]
if not blocking_classes and perf_status != "pass":
    blocking_classes.append("unclassified")

next_actions = []
if "jit_string_probe_issue" in blocker_classes:
    next_actions.append("Fix JIT string runtime behavior (str_slice/str_from_bytes path) in the JIT-capable souc binary.")
if "jit_source_probe_timeout" in blocker_classes:
    next_actions.append("Unblock JIT execution of self-hosted/compiler/main.sio -- --bench-int-to-string 0 (current source probe timeout).")
if "jit_source_probe_stack_overflow" in blocker_classes:
    next_actions.append("Unblock JIT stack overflow when running self-hosted/compiler/main.sio in jit mode.")
if "jit_main_source_probe_timeout_non_gating" in blocker_classes:
    next_actions.append("Investigate timeout in diagnostic main.sio JIT probe (non-gating).")
if "jit_main_source_probe_stack_overflow_non_gating" in blocker_classes:
    next_actions.append("Investigate stack overflow in diagnostic main.sio JIT probe (non-gating, likely module_loader import depth).")
if "jit_import_probe_stack_overflow_non_gating" in blocker_classes:
    next_actions.append("Investigate stack overflows in jit import stress probes and isolate recursive import/runtime paths (non-gating).")
if "jit_import_probe_timeout_non_gating" in blocker_classes:
    next_actions.append("Investigate slow/hung modules in jit import stress probes and add per-module timeout telemetry (non-gating).")
if "jit_import_probe_jit_panic_non_gating" in blocker_classes:
    next_actions.append("Investigate jit panic lanes surfaced by jit import stress probes (non-gating).")
if "jit_import_probe_not_run_non_gating" in blocker_classes:
    next_actions.append("Re-run scripts/sprint1_jit_import_stress_debug.sh to refresh module-level jit diagnostics (non-gating).")
if "jit_wasm_depth_stack_overflow_non_gating" in blocker_classes:
    next_actions.append("Investigate wasm stack-depth lane and reduce recursion/import pressure starting from first_failure_stage in jit_wasm_stack_depth_debug.")
if "jit_wasm_depth_timeout_non_gating" in blocker_classes:
    next_actions.append("Investigate wasm timeout lane and split wasm module probes further by submodule to isolate load bottlenecks.")
if "jit_wasm_depth_compile_error_non_gating" in blocker_classes:
    next_actions.append("Fix wasm probe fixture compatibility issues so wasm stack-depth diagnostics stay signal-rich.")
if "jit_wasm_depth_not_run_non_gating" in blocker_classes:
    next_actions.append("Re-run scripts/sprint1_jit_wasm_stack_depth_debug.sh to refresh focused wasm diagnostics (non-gating).")
if "jit_ir_timeout_lane_non_gating" in blocker_classes:
    next_actions.append("Investigate ir::ir timeout lane using parser_ast vs ir_ir fixture split from jit_ir_timeout_repro.")
if "jit_ir_parser_dependency_lane_non_gating" in blocker_classes:
    next_actions.append("Investigate parser::ast dependency probe failures in jit_ir_timeout_repro before attributing issues to ir::ir.")
if "jit_ir_compile_error_lane_non_gating" in blocker_classes:
    next_actions.append("Fix compile errors in ir timeout repro fixtures to preserve parser_ast vs ir_ir signal split.")
if "jit_ir_timeout_repro_not_run_non_gating" in blocker_classes:
    next_actions.append("Re-run scripts/sprint1_jit_ir_timeout_repro.sh to refresh ir timeout diagnostics (non-gating).")
if "jit_bigpush_failure_non_gating" in blocker_classes:
    next_actions.append("Inspect jit_root_cause_bigpush family-first-failure probes and promote strongest hypothesis into focused fix lane.")
if "jit_bigpush_not_run_non_gating" in blocker_classes:
    next_actions.append("Re-run scripts/sprint1_jit_root_cause_bigpush.sh to refresh deep root-cause diagnostics (non-gating).")
if "jit_bigpush_wasm_real_only_non_gating" in blocker_classes:
    next_actions.append("Treat wasm failure as real-module-specific and compare real wasm::mod against shadow stages in jit_probe modules.")
if "jit_bigpush_ir_real_only_non_gating" in blocker_classes:
    next_actions.append("Treat ir timeout as real-module-specific and compare real ir::ir against ir shadow stages in jit_probe modules.")
if "jit_bigpush_wasm_shadow_repro_non_gating" in blocker_classes:
    next_actions.append("Treat wasm failures as reproducible in shadow stages and focus on smallest failing shadow stage first.")
if "jit_bigpush_ir_shadow_repro_non_gating" in blocker_classes:
    next_actions.append("Treat ir failures as reproducible in shadow stages and focus on smallest failing shadow stage first.")
if "jit_mode_perf_target_miss" in blocker_classes or "run_mode_perf_target_miss" in blocker_classes:
    next_actions.append("Continue optimizing int_to_string benchmark lane until net_seconds < 1.0 for 1,000,000 iterations.")
if jit_debug is not None:
    next_actions.append("Re-run scripts/sprint1_jit_runtime_debug.sh and require usable_candidate_count >= 1.")
next_actions.append("Re-run scripts/sprint1_critical_bug_fixes_gate.sh and expect int_to_string_benchmark to pass in jit mode.")

payload = {
    "schema": "sounio.sprint1.blocker_report.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": {
        "correctness_closed": correctness_closed,
        "sprint1_overall_status": (critical or {}).get("overall_status", "unknown"),
        "perf_gate": {
            "status": perf_status,
            "reason": perf_reason,
            "mode": perf_mode,
            "net_seconds": perf_net,
            "runner": perf_runner,
        },
        "run_lane": {
            "status": run_status,
            "reason": run_reason,
            "net_seconds": run_net,
            "runner": run_runner,
            "non_gating": True,
        },
        "jit_debug": {
            "available": jit_debug is not None,
            "jit_runtime_blocked": jit_blocked,
            "usable_candidate_count": jit_usable_count,
        },
        "jit_import_stress": {
            "available": jit_import is not None,
            "status": jit_import_status,
            "reason": jit_import_reason,
            "runner": jit_import_runner,
            "stack_overflow_count": jit_import_stack_count,
            "timeout_count": jit_import_timeout_count,
            "jit_panic_count": jit_import_panic_count,
            "non_gating": True,
        },
        "jit_wasm_stack_depth": {
            "available": jit_wasm is not None,
            "status": jit_wasm_status,
            "reason": jit_wasm_reason,
            "first_failure_stage": jit_wasm_stage,
            "stack_overflow_count": jit_wasm_stack_count,
            "timeout_count": jit_wasm_timeout_count,
            "compile_error_count": jit_wasm_compile_count,
            "non_gating": True,
        },
        "jit_ir_timeout_repro": {
            "available": jit_ir is not None,
            "status": jit_ir_status,
            "reason": jit_ir_reason,
            "lane_split": jit_ir_lane_split,
            "timeout_count": jit_ir_timeout_count,
            "compile_error_count": jit_ir_compile_count,
            "non_gating": True,
        },
        "jit_root_cause_bigpush": {
            "available": jit_bigpush is not None,
            "status": jit_bigpush_status,
            "reason": jit_bigpush_reason,
            "hypotheses": jit_bigpush_hypotheses,
            "wasm_first_failure_probe": jit_bigpush_wasm_first,
            "ir_first_failure_probe": jit_bigpush_ir_first,
            "non_gating": True,
        },
    },
    "blocker_classes": blocking_classes,
    "diagnostic_classes": diagnostic_classes,
    "evidence_paths": {
        "critical_gate": critical_path,
        "perf_gate": perf_path,
        "run_lane": run_lane_path,
        "jit_debug": jit_debug_path,
        "jit_import_stress": jit_import_path,
        "jit_wasm_stack_depth": jit_wasm_path,
        "jit_ir_timeout_repro": jit_ir_path,
        "jit_root_cause_bigpush": jit_bigpush_path,
    },
    "next_actions": next_actions,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

md = []
md.append("# Sprint 1 Blocker Report")
md.append("")
md.append(f"Generated: {payload['generated_at']}")
md.append("")
md.append("## Status")
md.append(f"- correctness_closed: `{correctness_closed}`")
md.append(f"- sprint1_overall_status: `{payload['status']['sprint1_overall_status']}`")
md.append(f"- perf_gate: `{perf_status}` reason=`{perf_reason}` mode=`{perf_mode}` net_seconds=`{perf_net}`")
md.append(f"- run_lane (non-gating): `{run_status}` reason=`{run_reason}` net_seconds=`{run_net}`")
if jit_debug is None:
    md.append("- jit_debug: `missing`")
else:
    md.append(f"- jit_debug: `available` jit_runtime_blocked=`{jit_blocked}` usable_candidate_count=`{jit_usable_count}`")
if jit_import is None:
    md.append("- jit_import_stress (non-gating): `missing`")
else:
    md.append(
        f"- jit_import_stress (non-gating): `{jit_import_status}` reason=`{jit_import_reason}` "
        f"stack_overflow_count=`{jit_import_stack_count}` timeout_count=`{jit_import_timeout_count}` jit_panic_count=`{jit_import_panic_count}`"
    )
if jit_wasm is None:
    md.append("- jit_wasm_stack_depth (non-gating): `missing`")
else:
    md.append(
        f"- jit_wasm_stack_depth (non-gating): `{jit_wasm_status}` reason=`{jit_wasm_reason}` "
        f"first_failure_stage=`{jit_wasm_stage}` stack_overflow_count=`{jit_wasm_stack_count}` timeout_count=`{jit_wasm_timeout_count}`"
    )
if jit_ir is None:
    md.append("- jit_ir_timeout_repro (non-gating): `missing`")
else:
    md.append(
        f"- jit_ir_timeout_repro (non-gating): `{jit_ir_status}` reason=`{jit_ir_reason}` "
        f"lane_split=`{jit_ir_lane_split}` timeout_count=`{jit_ir_timeout_count}` compile_error_count=`{jit_ir_compile_count}`"
    )
if jit_bigpush is None:
    md.append("- jit_root_cause_bigpush (non-gating): `missing`")
else:
    md.append(
        f"- jit_root_cause_bigpush (non-gating): `{jit_bigpush_status}` reason=`{jit_bigpush_reason}` "
        f"hypotheses=`{jit_bigpush_hypotheses}` wasm_first_failure_probe=`{jit_bigpush_wasm_first}` ir_first_failure_probe=`{jit_bigpush_ir_first}`"
    )
md.append("")
md.append("## Blocker Classes")
for item in blocking_classes:
    md.append(f"- `{item}`")
if not blocking_classes:
    md.append("- `(none)`")
if diagnostic_classes:
    md.append("")
    md.append("## Diagnostic Classes")
    for item in diagnostic_classes:
        md.append(f"- `{item}`")
md.append("")
md.append("## Evidence")
for key, value in payload["evidence_paths"].items():
    md.append(f"- {key}: `{value}`")
md.append("")
md.append("## Next Actions")
for action in payload["next_actions"]:
    md.append(f"- {action}")
md.append("")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"wrote: {out_json}")
print(f"wrote: {out_md}")
PY
