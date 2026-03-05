#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_MD="${1:-$ROOT_DIR/artifacts/sprint1/sprint1_blocker_report.md}"
OUT_JSON="${SOUNIO_SPRINT1_BLOCKER_JSON:-$ROOT_DIR/artifacts/sprint1/sprint1_blocker_report.v1.json}"
CRITICAL_JSON="${SOUNIO_SPRINT1_CRITICAL_JSON:-$ROOT_DIR/artifacts/sprint1/critical_bug_fixes_gate.v1.json}"
PERF_JSON="${SOUNIO_SPRINT1_PERF_JSON:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_gate.v1.json}"
RUN_LANE_JSON="${SOUNIO_SPRINT1_RUN_LANE_JSON:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_run_lane.v1.json}"
JIT_DEBUG_JSON="${SOUNIO_SPRINT1_JIT_DEBUG_JSON:-$ROOT_DIR/artifacts/sprint1/jit_runtime_debug.v1.json}"

mkdir -p "$(dirname "$OUT_MD")"
mkdir -p "$(dirname "$OUT_JSON")"

python3 - "$OUT_MD" "$OUT_JSON" "$CRITICAL_JSON" "$PERF_JSON" "$RUN_LANE_JSON" "$JIT_DEBUG_JSON" <<'PY'
import datetime as dt
import json
import os
import sys

out_md, out_json, critical_path, perf_path, run_lane_path, jit_debug_path = sys.argv[1:7]

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

critical = load_json(critical_path)
perf = load_json(perf_path)
run_lane = load_json(run_lane_path)
jit_debug = load_json(jit_debug_path)

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

jit_summary = (jit_debug or {}).get("summary") or {}
jit_blocked = bool(jit_summary.get("jit_runtime_blocked", False)) if jit_debug else None
jit_usable_count = jit_summary.get("usable_candidate_count") if jit_debug else None

blocker_classes = []
if perf_reason in {"jit_string_runtime_unavailable", "jit_runner_unusable", "jit_runner_required_missing"}:
    blocker_classes.append("jit_runtime_blocked")
if perf_status == "fail" and perf_reason == "target_not_met" and perf_mode == "run":
    blocker_classes.append("run_mode_perf_target_miss")
if perf_status == "not_run":
    blocker_classes.append("perf_not_run")
if not blocker_classes:
    blocker_classes.append("unclassified")

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
    },
    "blocker_classes": blocker_classes,
    "evidence_paths": {
        "critical_gate": critical_path,
        "perf_gate": perf_path,
        "run_lane": run_lane_path,
        "jit_debug": jit_debug_path,
    },
    "next_actions": [
        "Fix JIT string runtime behavior (str_slice/str_from_bytes path) in the JIT-capable souc binary.",
        "Re-run scripts/sprint1_jit_runtime_debug.sh and require usable_candidate_count >= 1.",
        "Re-run scripts/sprint1_critical_bug_fixes_gate.sh and expect int_to_string_benchmark to pass in jit mode.",
    ],
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
md.append("")
md.append("## Blocker Classes")
for item in blocker_classes:
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
