#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
  exit 2
fi

MODE="${OMEGA_L4_NATIVE_GATE_MODE:-auto}"
OUT_JSON="${OMEGA_L4_NATIVE_GATE_OUT:-$ROOT_DIR/artifacts/omega/l4_native_shadow_gate.v1.json}"
LOG_PATH="${OMEGA_L4_NATIVE_GATE_LOG:-$ROOT_DIR/artifacts/omega/l4_native_shadow_gate.log}"
REPORT_DIR="${OMEGA_L4_REPORT_DIR:-$ROOT_DIR/artifacts/omega/l4_runs}"

GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"

ENABLE_NCU="${OMEGA_L4_ENABLE_NCU:-0}"
FORCE_REGENERATE_SHADOW_VARIANTS="${OMEGA_L4_FORCE_REGENERATE_SHADOW_VARIANTS:-1}"
BENCH_SCRIPT="${OMEGA_L4_BENCH_SCRIPT:-}"

if [[ -z "$BENCH_SCRIPT" ]]; then
  if [[ -x "$ROOT_DIR/scripts/benchmarks/l4_optimized_no_rust_bench.sh" ]]; then
    BENCH_SCRIPT="$ROOT_DIR/scripts/benchmarks/l4_optimized_no_rust_bench.sh"
  else
    BENCH_SCRIPT="$ROOT_DIR/scripts/l4_optimized_no_rust_bench.sh"
  fi
fi

TARGET_DIM="${TARGET_DIM:-4096}"
TARGET_STRICT_OVERHEAD_MAX="${TARGET_STRICT_OVERHEAD_MAX:-3.0}"
TARGET_FAST_SPEEDUP_MIN="${TARGET_FAST_SPEEDUP_MIN:-1.05}"
TARGET_FAST_DRIFT_MAX="${TARGET_FAST_DRIFT_MAX:-0.02}"

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_L4_NATIVE_GATE_MODE must be one of: auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")" "$REPORT_DIR"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

emit_status_json() {
  local status="$1"
  local reason="$2"
  local benchmark_status="$3"
  local report_path="$4"
  local latest_path="$5"
  local strict_overhead="$6"
  local fast_speedup="$7"
  local fast_drift="$8"
  local bench_rc="$9"
  local blocked_reasons="${10}"

  local report_rel latest_rel log_rel
  report_rel="$(to_rel "$report_path")"
  latest_rel="$(to_rel "$latest_path")"
  log_rel="$(to_rel "$LOG_PATH")"

  jq -cn \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "$status" \
    --arg reason "$reason" \
    --arg mode "$MODE" \
    --arg gpu_host "$GPU_HOST" \
    --arg gpu_user "$GPU_USER" \
    --arg remote_dir "$REMOTE_DIR" \
    --arg benchmark_status "$benchmark_status" \
    --arg report_path "$report_rel" \
    --arg latest_path "$latest_rel" \
    --arg log_path "$log_rel" \
    --argjson target_dim "$TARGET_DIM" \
    --argjson strict_target_max "$TARGET_STRICT_OVERHEAD_MAX" \
    --argjson fast_target_min "$TARGET_FAST_SPEEDUP_MIN" \
    --argjson drift_target_max "$TARGET_FAST_DRIFT_MAX" \
    --argjson strict_overhead "${strict_overhead:-null}" \
    --argjson fast_speedup "${fast_speedup:-null}" \
    --argjson fast_drift "${fast_drift:-null}" \
    --argjson enable_ncu "$ENABLE_NCU" \
    --argjson force_regen "$FORCE_REGENERATE_SHADOW_VARIANTS" \
    --argjson bench_rc "$bench_rc" \
    --argjson blocked_reasons "$blocked_reasons" \
    '{
      schema: "sounio.omega.l4-native-shadow-gate.v1",
      generated_at_utc: $generated_at_utc,
      status: $status,
      reason: $reason,
      mode: $mode,
      benchmark: {
        status: $benchmark_status,
        report_path: $report_path,
        latest_path: $latest_path,
        bench_rc: $bench_rc,
        blocked_reasons: $blocked_reasons,
        metrics: {
          target_dim: $target_dim,
          strict_overhead_x_vs_cublas: $strict_overhead,
          fast_speedup_x_vs_strict: $fast_speedup,
          fast_drift_vs_strict: $fast_drift
        },
        targets: {
          strict_overhead_max: $strict_target_max,
          fast_speedup_min: $fast_target_min,
          fast_drift_max: $drift_target_max
        }
      },
      launch: {
        gpu_host: $gpu_host,
        gpu_user: $gpu_user,
        remote_dir: $remote_dir,
        enable_ncu: ($enable_ncu == 1),
        force_regenerate_shadow_variants: ($force_regen == 1),
        require_native_shadow_variants: true,
        shadow_variant_generator_preference: "native",
        allow_legacy_shadow_augment_fallback: false
      },
      log_path: $log_path
    }' > "$OUT_JSON"
}

if [[ "$MODE" == "off" ]]; then
  emit_status_json "not_run" "gate_disabled" "NOT RUN" "" "" "" "" "" 0 "[]"
  echo "omega_l4_native_shadow_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

if [[ ! -x "$BENCH_SCRIPT" ]]; then
  if [[ "$MODE" == "required" ]]; then
    emit_status_json "fail" "benchmark_script_missing" "NOT RUN" "" "" "" "" "" 1 "[\"target_unavailable\"]"
    echo "omega_l4_native_shadow_gate: status=fail reason=benchmark_script_missing script=$BENCH_SCRIPT report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "benchmark_script_missing" "NOT RUN" "" "" "" "" "" 0 "[\"target_unavailable\"]"
  echo "omega_l4_native_shadow_gate: status=not_run reason=benchmark_script_missing script=$BENCH_SCRIPT report=$OUT_JSON"
  exit 0
fi

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o StrictHostKeyChecking=no
)

if ! ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" "cd ${REMOTE_DIR} && echo ok" >/dev/null 2>&1; then
  if [[ "$MODE" == "required" ]]; then
    emit_status_json "fail" "ssh_unreachable" "NOT RUN" "" "" "" "" "" 1 "[]"
    echo "omega_l4_native_shadow_gate: status=fail reason=ssh_unreachable report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "ssh_unreachable" "NOT RUN" "" "" "" "" "" 0 "[]"
  echo "omega_l4_native_shadow_gate: status=not_run reason=ssh_unreachable report=$OUT_JSON"
  exit 0
fi

timestamp_utc="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
report_path="$REPORT_DIR/l4_optimized_no_rust_report.${timestamp_utc}.v2.json"
latest_path="$REPORT_DIR/l4_optimized_no_rust_report.latest.v2.json"
run_started_epoch="$(date +%s)"

set +e
SHADOW_VARIANT_GENERATOR_PREFERENCE="native" \
FORCE_REGENERATE_SHADOW_VARIANTS="$FORCE_REGENERATE_SHADOW_VARIANTS" \
ALLOW_LEGACY_SHADOW_AUGMENT_FALLBACK="0" \
REQUIRE_NATIVE_SHADOW_VARIANTS="1" \
ENABLE_NCU="$ENABLE_NCU" \
GPU_HOST="$GPU_HOST" \
GPU_USER="$GPU_USER" \
REMOTE_DIR="$REMOTE_DIR" \
REPORT_DIR="$REPORT_DIR" \
REPORT_PATH="$report_path" \
LATEST_PATH="$latest_path" \
TARGET_DIM="$TARGET_DIM" \
TARGET_STRICT_OVERHEAD_MAX="$TARGET_STRICT_OVERHEAD_MAX" \
TARGET_FAST_SPEEDUP_MIN="$TARGET_FAST_SPEEDUP_MIN" \
TARGET_FAST_DRIFT_MAX="$TARGET_FAST_DRIFT_MAX" \
bash "$BENCH_SCRIPT" >"$LOG_PATH" 2>&1
bench_rc=$?
set -e

report_read_path=""
if [[ -f "$report_path" ]]; then
  report_read_path="$report_path"
elif [[ -f "$latest_path" ]]; then
  latest_mtime="$(stat -c %Y "$latest_path" 2>/dev/null || echo 0)"
  if [[ "$latest_mtime" -ge "$run_started_epoch" ]]; then
    report_read_path="$latest_path"
  fi
fi

benchmark_status="NOT RUN"
strict_overhead=""
fast_speedup=""
fast_drift=""
blocked_reasons="[]"
if [[ -n "$report_read_path" && -f "$report_read_path" ]]; then
  benchmark_status="$(jq -r '.overall_status // "NOT RUN"' "$report_read_path")"
  strict_overhead="$(jq -r --arg dim "$TARGET_DIM" '.results.by_dimension[$dim].ratios.overhead_x_shadow_strict_vs_cublas // empty' "$report_read_path")"
  fast_speedup="$(jq -r --arg dim "$TARGET_DIM" '.results.by_dimension[$dim].ratios.speedup_x_shadow_fast_vs_shadow_strict // empty' "$report_read_path")"
  fast_drift="$(jq -r --arg dim "$TARGET_DIM" '.results.by_dimension[$dim].drift.numerical_drift_vs_strict_median // empty' "$report_read_path")"
  blocked_reasons="$(jq -c '.blocked_reasons // []' "$report_read_path")"
elif [[ "$bench_rc" -ne 0 ]]; then
  benchmark_status="FAIL"
  blocked_reasons='["target_unavailable"]'
fi

if [[ "$bench_rc" -eq 0 && "$benchmark_status" == "PASS" ]]; then
  emit_status_json "pass" "benchmark_pass" "$benchmark_status" "$report_read_path" "$latest_path" "$strict_overhead" "$fast_speedup" "$fast_drift" "$bench_rc" "$blocked_reasons"
  echo "omega_l4_native_shadow_gate: status=pass benchmark_status=$benchmark_status report=$OUT_JSON"
  exit 0
fi

emit_status_json "fail" "benchmark_failed" "$benchmark_status" "$report_read_path" "$latest_path" "$strict_overhead" "$fast_speedup" "$fast_drift" "$bench_rc" "$blocked_reasons"
echo "omega_l4_native_shadow_gate: status=fail benchmark_status=$benchmark_status bench_rc=$bench_rc report=$OUT_JSON" >&2
exit 2
