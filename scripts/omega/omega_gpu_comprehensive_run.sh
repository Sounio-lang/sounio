#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: required dependency missing: python3" >&2
  exit 2
fi

MODE="${OMEGA_GPU_COMPREHENSIVE_MODE:-auto}"
OUT_JSON="${OMEGA_GPU_COMPREHENSIVE_OUT:-$ROOT_DIR/artifacts/omega/gpu_comprehensive_run.v1.json}"
LOG_PATH="${OMEGA_GPU_COMPREHENSIVE_LOG:-$ROOT_DIR/artifacts/omega/gpu_comprehensive_run.log}"

RUN_STDLIB_NATIVE_GATE="${OMEGA_GPU_RUN_STDLIB_NATIVE_GATE:-1}"
RUN_OMEGA_GATES="${OMEGA_GPU_RUN_OMEGA_GATES:-1}"
RUN_PUBLIC_CONTRACT_GATE="${OMEGA_GPU_RUN_PUBLIC_CONTRACT_GATE:-1}"
RUN_L4_GATE="${OMEGA_GPU_RUN_L4_GATE:-1}"
L4_ENABLE_NCU="${OMEGA_GPU_COMPREHENSIVE_ENABLE_NCU:-0}"

NATIVE_LANE_MATRIX_JSON="${OMEGA_NATIVE_LANE_MATRIX_JSON:-$ROOT_DIR/artifacts/stdlib/native_lane_matrix.v1.json}"
STDLIB_STATUS_JSON="${OMEGA_STDLIB_HYPER_STATUS_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_hyper_execution_status.v1.json}"
PARITY_JSON="${OMEGA_GPU_CODEGEN_PARITY_JSON:-$ROOT_DIR/artifacts/omega/gpu_codegen_parity.v1.json}"
BINARY_JSON="${OMEGA_GPU_BINARY_ATTEST_JSON:-$ROOT_DIR/artifacts/omega/gpu_binary_attestation.v1.json}"
RUNTIME_JSON="${OMEGA_GPU_RUNTIME_GATE_JSON:-$ROOT_DIR/artifacts/omega/gpu_runtime_attest_gate.v1.json}"
PUBLIC_CONTRACT_JSON="${OMEGA_GPU_PUBLIC_CONTRACT_JSON:-$ROOT_DIR/artifacts/omega/gpu_public_contract.v1.json}"
L4_JSON="${OMEGA_L4_NATIVE_GATE_JSON:-$ROOT_DIR/artifacts/omega/l4_native_shadow_gate.v1.json}"

GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"

mode_for_stage() {
  local raw="$1"
  if [[ "$raw" == "required" || "$raw" == "auto" || "$raw" == "off" ]]; then
    printf '%s' "$raw"
    return
  fi
  if [[ "$MODE" == "required" ]]; then
    printf 'required'
  elif [[ "$MODE" == "off" ]]; then
    printf 'off'
  else
    printf 'auto'
  fi
}

STDLIB_MODE="$(mode_for_stage "${OMEGA_GPU_COMPREHENSIVE_STDLIB_MODE:-}")"
PARITY_MODE="$(mode_for_stage "${OMEGA_GPU_COMPREHENSIVE_PARITY_MODE:-}")"
BINARY_MODE="$(mode_for_stage "${OMEGA_GPU_COMPREHENSIVE_BINARY_MODE:-}")"
RUNTIME_MODE="$(mode_for_stage "${OMEGA_GPU_COMPREHENSIVE_RUNTIME_MODE:-}")"
PUBLIC_CONTRACT_MODE="$(mode_for_stage "${OMEGA_GPU_COMPREHENSIVE_PUBLIC_CONTRACT_MODE:-}")"
L4_MODE="$(mode_for_stage "${OMEGA_GPU_COMPREHENSIVE_L4_MODE:-}")"

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_GPU_COMPREHENSIVE_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

normalize_bool() {
  local raw="$1"
  case "$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) printf '1' ;;
    0|false|no|off) printf '0' ;;
    *)
      echo "error: invalid boolean value: $raw" >&2
      exit 2
      ;;
  esac
}

RUN_STDLIB_NATIVE_GATE="$(normalize_bool "$RUN_STDLIB_NATIVE_GATE")"
RUN_OMEGA_GATES="$(normalize_bool "$RUN_OMEGA_GATES")"
RUN_PUBLIC_CONTRACT_GATE="$(normalize_bool "$RUN_PUBLIC_CONTRACT_GATE")"
RUN_L4_GATE="$(normalize_bool "$RUN_L4_GATE")"
L4_ENABLE_NCU="$(normalize_bool "$L4_ENABLE_NCU")"

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")" "$(dirname "$NATIVE_LANE_MATRIX_JSON")" "$(dirname "$STDLIB_STATUS_JSON")"
: >"$LOG_PATH"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
STEPS_JSONL="$TMP_DIR/steps.jsonl"
touch "$STEPS_JSONL"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

normalize_status_token() {
  local raw="$1"
  local lowered
  lowered="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr '-' '_')"
  case "$lowered" in
    pass) printf 'pass' ;;
    fail|failed|blocked|error|not_found) printf 'fail' ;;
    not_run|notrun|skipped|"") printf 'not_run' ;;
    *)
      printf 'not_run'
      ;;
  esac
}

append_step_record() {
  local step="$1"
  local executed="$2"
  local raw_status="$3"
  local reason="$4"
  local rc="$5"
  local artifact="$6"
  local blockers_json="$7"
  local details_json="${8:-{}}"
  local norm_status
  norm_status="$(normalize_status_token "$raw_status")"
  jq -cn \
    --arg step "$step" \
    --argjson executed "$executed" \
    --arg raw_status "$raw_status" \
    --arg status "$norm_status" \
    --arg reason "$reason" \
    --argjson rc "$rc" \
    --arg artifact "$(to_rel "$artifact")" \
    --arg blockers_raw "$blockers_json" \
    --arg details_raw "$details_json" \
    '{
      step: $step,
      executed: $executed,
      status: $status,
      raw_status: $raw_status,
      reason: $reason,
      rc: $rc,
      artifact: $artifact,
      blockers: ($blockers_raw | fromjson? // []),
      details: ($details_raw | fromjson? // {})
    }' >>"$STEPS_JSONL"
}

append_step_from_artifact() {
  local step="$1"
  local artifact="$2"
  local rc="$3"

  local raw_status="not_run"
  local reason=""
  local blockers_json='[]'
  local details_json='{}'

  if [[ -f "$artifact" ]]; then
    raw_status="$(jq -r '.status_summary // .status // .overall_status // "not_run"' "$artifact" 2>/dev/null || echo "not_run")"
    reason="$(jq -r '.reason // .benchmark.reason // .ncu_metrics.reason // empty' "$artifact" 2>/dev/null || true)"
    blockers_json="$(jq -c '.blockers // .blocked_reasons // .benchmark.blocked_reasons // []' "$artifact" 2>/dev/null || echo '[]')"
    details_json="$(jq -c '{
      mode: (.mode // null),
      schema: (.schema // null),
      generated_at_utc: (.generated_at_utc // null)
    }' "$artifact" 2>/dev/null || echo '{}')"
  else
    raw_status="not_run"
    reason="artifact_missing"
    blockers_json='["target_unavailable"]'
  fi

  append_step_record "$step" true "$raw_status" "$reason" "$rc" "$artifact" "$blockers_json" "$details_json"
}

run_step_script() {
  local step="$1"
  local artifact="$2"
  shift 2
  local rc=0
  set +e
  "$@" >>"$LOG_PATH" 2>&1
  rc=$?
  set -e
  append_step_from_artifact "$step" "$artifact" "$rc"
}

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o StrictHostKeyChecking=no
)

if [[ "$MODE" == "off" ]]; then
  append_step_record "gpu_host_preflight" false "not_run" "gate_disabled" 0 "$LOG_PATH" '[]' '{}'
else
  gpu_info=""
  set +e
  gpu_info="$(ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" "cd ${REMOTE_DIR} >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1" 2>/dev/null)"
  preflight_rc=$?
  set -e
  if [[ $preflight_rc -eq 0 && -n "$gpu_info" ]]; then
    append_step_record "gpu_host_preflight" true "pass" "" 0 "$LOG_PATH" '[]' "$(jq -cn --arg gpu "$gpu_info" '{gpu: $gpu}')"
  else
    preflight_reason="ssh_unreachable"
    append_step_record "gpu_host_preflight" true "$( [[ "$MODE" == "required" ]] && echo "fail" || echo "not_run" )" "$preflight_reason" "$preflight_rc" "$LOG_PATH" '["ssh_unreachable"]' '{}'
  fi
fi

if [[ "$MODE" != "off" && "$RUN_STDLIB_NATIVE_GATE" == "1" ]]; then
  run_step_script \
    "stdlib_native_lanes" \
    "$STDLIB_STATUS_JSON" \
    env \
      STDLIB_HYPER_GATE_MODE="$STDLIB_MODE" \
      STDLIB_HYPER_STATUS_OUT="$STDLIB_STATUS_JSON" \
      STDLIB_NATIVE_LANE_MATRIX_OUT="$NATIVE_LANE_MATRIX_JSON" \
      bash "$ROOT_DIR/scripts/stdlib/stdlib_hyper_execution_gate.sh"
else
  append_step_record "stdlib_native_lanes" false "not_run" "disabled" 0 "$STDLIB_STATUS_JSON" '[]' '{}'
fi

if [[ "$MODE" != "off" && "$RUN_OMEGA_GATES" == "1" ]]; then
  run_step_script \
    "gpu_codegen_parity" \
    "$PARITY_JSON" \
    env \
      OMEGA_GPU_CODEGEN_PARITY_MODE="$PARITY_MODE" \
      OMEGA_GPU_CODEGEN_PARITY_OUT="$PARITY_JSON" \
      OMEGA_NATIVE_LANE_MATRIX_JSON="$NATIVE_LANE_MATRIX_JSON" \
      bash "$ROOT_DIR/scripts/omega/omega_gpu_codegen_parity_gate.sh"

  run_step_script \
    "gpu_binary_attestation" \
    "$BINARY_JSON" \
    env \
      OMEGA_GPU_BINARY_ATTEST_MODE="$BINARY_MODE" \
      OMEGA_GPU_BINARY_ATTEST_OUT="$BINARY_JSON" \
      OMEGA_GPU_CODEGEN_PARITY_JSON="$PARITY_JSON" \
      bash "$ROOT_DIR/scripts/omega/omega_gpu_binary_attest_gate.sh"

  run_step_script \
    "gpu_runtime_attestation" \
    "$RUNTIME_JSON" \
    env \
      OMEGA_GPU_RUNTIME_GATE_MODE="$RUNTIME_MODE" \
      OMEGA_GPU_RUNTIME_GATE_OUT="$RUNTIME_JSON" \
      OMEGA_GPU_CODEGEN_PARITY_JSON="$PARITY_JSON" \
      OMEGA_GPU_BINARY_ATTEST_JSON="$BINARY_JSON" \
      GPU_HOST="$GPU_HOST" \
      GPU_USER="$GPU_USER" \
      REMOTE_DIR="$REMOTE_DIR" \
      bash "$ROOT_DIR/scripts/omega/omega_gpu_runtime_attest_gate.sh"
else
  append_step_record "gpu_codegen_parity" false "not_run" "disabled" 0 "$PARITY_JSON" '[]' '{}'
  append_step_record "gpu_binary_attestation" false "not_run" "disabled" 0 "$BINARY_JSON" '[]' '{}'
  append_step_record "gpu_runtime_attestation" false "not_run" "disabled" 0 "$RUNTIME_JSON" '[]' '{}'
fi

if [[ "$MODE" != "off" && "$RUN_PUBLIC_CONTRACT_GATE" == "1" ]]; then
  run_step_script \
    "gpu_public_contract" \
    "$PUBLIC_CONTRACT_JSON" \
    env \
      OMEGA_GPU_PUBLIC_CONTRACT_MODE="$PUBLIC_CONTRACT_MODE" \
      OMEGA_GPU_PUBLIC_CONTRACT_OUT="$PUBLIC_CONTRACT_JSON" \
      OMEGA_GPU_CODEGEN_PARITY_JSON="$PARITY_JSON" \
      OMEGA_GPU_BINARY_ATTEST_JSON="$BINARY_JSON" \
      OMEGA_GPU_RUNTIME_GATE_JSON="$RUNTIME_JSON" \
      bash "$ROOT_DIR/scripts/omega/omega_gpu_public_contract_gate.sh"
else
  append_step_record "gpu_public_contract" false "not_run" "disabled" 0 "$PUBLIC_CONTRACT_JSON" '[]' '{}'
fi

if [[ "$MODE" != "off" && "$RUN_L4_GATE" == "1" ]]; then
  run_step_script \
    "l4_native_shadow_benchmark" \
    "$L4_JSON" \
    env \
      OMEGA_L4_NATIVE_GATE_MODE="$L4_MODE" \
      OMEGA_L4_ENABLE_NCU="$L4_ENABLE_NCU" \
      OMEGA_L4_NATIVE_GATE_OUT="$L4_JSON" \
      GPU_HOST="$GPU_HOST" \
      GPU_USER="$GPU_USER" \
      REMOTE_DIR="$REMOTE_DIR" \
      bash "$ROOT_DIR/scripts/omega/omega_l4_native_shadow_gate.sh"
else
  append_step_record "l4_native_shadow_benchmark" false "not_run" "disabled" 0 "$L4_JSON" '[]' '{}'
fi

python3 - "$ROOT_DIR" "$MODE" "$OUT_JSON" "$LOG_PATH" "$STEPS_JSONL" "$STDLIB_STATUS_JSON" "$NATIVE_LANE_MATRIX_JSON" "$PARITY_JSON" "$BINARY_JSON" "$RUNTIME_JSON" "$PUBLIC_CONTRACT_JSON" "$L4_JSON" "$GPU_HOST" "$GPU_USER" "$REMOTE_DIR" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
out_json = Path(sys.argv[3])
log_path = Path(sys.argv[4])
steps_path = Path(sys.argv[5])
stdlib_status_path = Path(sys.argv[6])
lane_matrix_path = Path(sys.argv[7])
parity_path = Path(sys.argv[8])
binary_path = Path(sys.argv[9])
runtime_path = Path(sys.argv[10])
public_contract_path = Path(sys.argv[11])
l4_gate_path = Path(sys.argv[12])
gpu_host = sys.argv[13]
gpu_user = sys.argv[14]
remote_dir = sys.argv[15]


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root))
    except Exception:
        return str(p)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


steps = []
for line in steps_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        steps.append(json.loads(line))
    except Exception:
        pass

all_blockers = []
for step in steps:
    all_blockers.extend([str(x) for x in step.get("blockers", []) if str(x)])

stdlib_status = load_json(stdlib_status_path) or {}
lane_matrix = load_json(lane_matrix_path) or {}
parity = load_json(parity_path) or {}
binary = load_json(binary_path) or {}
runtime = load_json(runtime_path) or {}
public_contract = load_json(public_contract_path) or {}
l4_gate = load_json(l4_gate_path) or {}

all_blockers.extend([str(x) for x in stdlib_status.get("blockers", []) if str(x)])
all_blockers.extend([str(x) for x in parity.get("blockers", []) if str(x)])
all_blockers.extend([str(x) for x in binary.get("blockers", []) if str(x)])
all_blockers.extend([str(x) for x in runtime.get("blockers", []) if str(x)])
all_blockers.extend([str(x) for x in public_contract.get("blockers", []) if str(x)])
all_blockers.extend([str(x) for x in l4_gate.get("benchmark", {}).get("blocked_reasons", []) if str(x)])

executed_steps = [s for s in steps if s.get("executed")]
executed_statuses = [str(s.get("status", "not_run")) for s in executed_steps]
has_fail = any(s == "fail" for s in executed_statuses)
has_not_run = any(s == "not_run" for s in executed_statuses)
all_pass = bool(executed_steps) and all(s == "pass" for s in executed_statuses)

if mode == "off":
    status_summary = "not_run"
    reason = "gate_disabled"
elif has_fail:
    status_summary = "fail"
    reason = "one_or_more_steps_failed"
elif mode == "required" and has_not_run:
    status_summary = "fail"
    reason = "required_step_not_run"
elif all_pass:
    status_summary = "pass"
    reason = "all_steps_passed"
else:
    status_summary = "not_run"
    reason = "partial_execution"

hotspots = []
benchmark_summary = {}

for step in executed_steps:
    if str(step.get("status", "not_run")) != "fail":
        continue
    hotspots.append({
        "priority": "high",
        "area": "pipeline_gate",
        "lane": str(step.get("step", "unknown")),
        "signal": str(step.get("reason", "")) or "step_failed",
        "recommended_action": "resolve this gate failure before tuning downstream performance"
    })

native_lanes = parity.get("native_lanes")
if not isinstance(native_lanes, list):
    native_lanes = lane_matrix.get("lanes", [])

for lane in native_lanes:
    if not isinstance(lane, dict):
        continue
    lane_name = str(lane.get("lane", lane.get("name", "unknown")))
    lane_status = str(lane.get("status", "not_run")).lower()
    lane_reason = str(lane.get("reason", ""))
    if lane_status != "pass":
        hotspots.append({
            "priority": "high",
            "area": "native_lane",
            "lane": lane_name,
            "signal": lane_status,
            "reason": lane_reason or "lane_not_pass",
            "recommended_action": "stabilize compile/runtime parity for this native lane before release"
        })

for target in parity.get("targets", []) if isinstance(parity.get("targets"), list) else []:
    if not isinstance(target, dict):
        continue
    if str(target.get("status", "")).lower() == "pass":
        continue
    hotspots.append({
        "priority": "high",
        "area": "backend_codegen",
        "lane": str(target.get("lane", "unknown")),
        "signal": str(target.get("reason", "target_unavailable")),
        "target": str(target.get("target", "")),
        "recommended_action": "enable GPU backend build and target toolchain for this lane"
    })

for test in runtime.get("tests", []) if isinstance(runtime.get("tests"), list) else []:
    if not isinstance(test, dict):
        continue
    if str(test.get("status", "")).lower() == "pass":
        continue
    hotspots.append({
        "priority": "high",
        "area": "runtime_attestation",
        "lane": str(test.get("name", "unknown")),
        "signal": str(test.get("status", "fail")).lower(),
        "reason": str(test.get("error_excerpt", "")),
        "recommended_action": "fix runtime smoke status classification and backend runtime readiness"
    })

l4_report_rel = l4_gate.get("benchmark", {}).get("report_path") or l4_gate.get("benchmark", {}).get("latest_path")
l4_report = None
if isinstance(l4_report_rel, str) and l4_report_rel:
    l4_report = load_json(root / l4_report_rel)

if isinstance(l4_report, dict):
    targets = l4_report.get("targets", {})
    target_dim = int(targets.get("target_dimension", 4096))
    strict_max = targets.get("strict_overhead_max")
    fast_min = targets.get("fast_speedup_min")
    drift_max = targets.get("fast_drift_max")
    by_dim = l4_report.get("results", {}).get("by_dimension", {})
    target = by_dim.get(str(target_dim), {})
    ratios = target.get("ratios", {})
    drift = target.get("drift", {})
    strict_overhead = ratios.get("overhead_x_shadow_strict_vs_cublas")
    value_overhead = ratios.get("overhead_x_value_only_vs_cublas")
    fast_speedup = ratios.get("speedup_x_shadow_fast_vs_shadow_strict")
    fast_drift = drift.get("numerical_drift_vs_strict_median")
    benchmark_summary = {
        "report_path": l4_report_rel,
        "target_dimension": target_dim,
        "strict_overhead_x_vs_cublas": strict_overhead,
        "value_only_overhead_x_vs_cublas": value_overhead,
        "fast_speedup_x_vs_strict": fast_speedup,
        "fast_drift_vs_strict": fast_drift,
    }

    if strict_overhead is None or (strict_max is not None and strict_overhead > strict_max):
        hotspots.append({
            "priority": "high",
            "area": "kernel_shadow_strict",
            "lane": "exceptional",
            "signal": "strict_overhead_above_target",
            "actual": strict_overhead,
            "target": strict_max,
            "recommended_action": "optimize strict shadow kernel pipeline and tensor-core utilization"
        })
    if fast_speedup is None or (fast_min is not None and fast_speedup < fast_min):
        hotspots.append({
            "priority": "high",
            "area": "kernel_shadow_fast",
            "lane": "exceptional",
            "signal": "fast_speedup_below_target",
            "actual": fast_speedup,
            "target": fast_min,
            "recommended_action": "increase fast-vs-strict speedup via scheduling, memory, and launch tuning"
        })
    if fast_drift is None or (drift_max is not None and fast_drift > drift_max):
        hotspots.append({
            "priority": "high",
            "area": "kernel_shadow_fast",
            "lane": "exceptional",
            "signal": "fast_drift_above_target",
            "actual": fast_drift,
            "target": drift_max,
            "recommended_action": "reduce numerical drift in fast lane or tighten validity/provenance controls"
        })

    if strict_overhead is not None and strict_overhead > 2.5:
        hotspots.append({
            "priority": "medium",
            "area": "kernel_shadow_strict",
            "lane": "exceptional",
            "signal": "high_overhead_vs_cublas",
            "actual": strict_overhead,
            "reference": 2.5,
            "recommended_action": "reduce strict shadow overhead by improving memory movement and issue efficiency"
        })
    if value_overhead is not None and value_overhead > 2.3:
        hotspots.append({
            "priority": "medium",
            "area": "kernel_value_only",
            "lane": "exceptional",
            "signal": "value_only_gap_vs_cublas",
            "actual": value_overhead,
            "reference": 2.3,
            "recommended_action": "close baseline kernel gap versus cublas before shadow-path micro-optimizations"
        })
    if fast_speedup is not None and fast_speedup < 1.12:
        hotspots.append({
            "priority": "medium",
            "area": "kernel_shadow_fast",
            "lane": "exceptional",
            "signal": "fast_speedup_headroom",
            "actual": fast_speedup,
            "reference": 1.12,
            "recommended_action": "raise fast lane speedup headroom with register pressure and launch tuning"
        })

    for dim, payload in by_dim.items():
        if not isinstance(payload, dict):
            continue
        for variant in ("cublas_baseline", "value_only_baseline", "shadow_strict", "shadow_fast"):
            status = str(payload.get(variant, {}).get("status", "NOT RUN")).lower()
            if status not in {"pass"}:
                hotspots.append({
                    "priority": "medium",
                    "area": "benchmark_variant",
                    "lane": "exceptional",
                    "dimension": dim,
                    "variant": variant,
                    "signal": status,
                    "recommended_action": "restore runnable benchmark variant for this dimension before comparing optimizations"
                })

hotspots = unique([json.dumps(x, sort_keys=True) for x in hotspots])
hotspots = [json.loads(x) for x in hotspots]

def _tc(obj):
    t = obj.get("toolchain") if isinstance(obj, dict) else None
    return t if isinstance(t, dict) and t.get("capture_status") not in (None, "not_captured") else None
toolchain = _tc(runtime) or _tc(parity) or _tc(binary) or _tc(public_contract.get("support_evidence", {})) or {"capture_status": "not_captured"}

payload = {
    "schema": "sounio.omega.gpu_comprehensive_run.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "status_summary": status_summary,
    "reason": reason,
    "environment": {
        "gpu_host": gpu_host,
        "gpu_user": gpu_user,
        "remote_dir": remote_dir,
        "toolchain": toolchain,
    },
    "artifacts": {
        "stdlib_status": rel(stdlib_status_path),
        "native_lane_matrix": rel(lane_matrix_path),
        "gpu_codegen_parity": rel(parity_path),
        "gpu_binary_attestation": rel(binary_path),
        "gpu_runtime_attestation": rel(runtime_path),
        "gpu_public_contract": rel(public_contract_path),
        "l4_native_shadow_gate": rel(l4_gate_path),
    },
    "steps": steps,
    "benchmark_summary": benchmark_summary,
    "hotspots": hotspots,
    "blockers": unique(all_blockers),
    "log_path": rel(log_path),
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

final_status="$(jq -r '.status_summary // "not_run"' "$OUT_JSON")"
final_reason="$(jq -r '.reason // ""' "$OUT_JSON")"

if [[ "$final_status" == "pass" ]]; then
  echo "omega_gpu_comprehensive_run: status=pass report=$OUT_JSON"
  exit 0
fi
if [[ "$final_status" == "not_run" ]]; then
  echo "omega_gpu_comprehensive_run: status=not_run reason=$final_reason report=$OUT_JSON"
  exit 0
fi
echo "omega_gpu_comprehensive_run: status=fail reason=$final_reason report=$OUT_JSON" >&2
exit 2
