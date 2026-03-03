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

MODE="${OMEGA_L4_TUNING_SWEEP_MODE:-auto}"
OUT_JSON="${OMEGA_L4_TUNING_SWEEP_OUT:-$ROOT_DIR/artifacts/omega/l4_tuning_sweep.v1.json}"
LOG_PATH="${OMEGA_L4_TUNING_SWEEP_LOG:-$ROOT_DIR/artifacts/omega/l4_tuning_sweep.log}"
RUN_DIR="${OMEGA_L4_TUNING_SWEEP_RUN_DIR:-$ROOT_DIR/artifacts/omega/l4_tuning_profiles}"

GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"
GEMM_DIMS="${OMEGA_L4_TUNING_SWEEP_GEMM_DIMS:-4096}"
TARGET_DIM="${OMEGA_L4_TUNING_SWEEP_TARGET_DIM:-4096}"

TARGET_STRICT_OVERHEAD_MAX="${OMEGA_L4_TUNING_STRICT_OVERHEAD_MAX:-3.0}"
TARGET_FAST_SPEEDUP_MIN="${OMEGA_L4_TUNING_FAST_SPEEDUP_MIN:-1.12}"
TARGET_FAST_DRIFT_MAX="${OMEGA_L4_TUNING_FAST_DRIFT_MAX:-0.02}"

PROFILES_JSON_DEFAULT='[
  {"name":"baseline_tuned","strict_sqrt_interval":2,"strict_prov_interval":1,"strict_feedback_interval":32,"fast_prov_interval":64,"fast_feedback_interval":0,"strict_slot_cap":16,"fast_slot_cap":8},
  {"name":"fast_prov_96","strict_sqrt_interval":2,"strict_prov_interval":1,"strict_feedback_interval":32,"fast_prov_interval":96,"fast_feedback_interval":0,"strict_slot_cap":16,"fast_slot_cap":8},
  {"name":"fast_prov_128","strict_sqrt_interval":2,"strict_prov_interval":1,"strict_feedback_interval":32,"fast_prov_interval":128,"fast_feedback_interval":0,"strict_slot_cap":16,"fast_slot_cap":8}
]'
PROFILES_JSON="${OMEGA_L4_TUNING_SWEEP_PROFILES_JSON:-$PROFILES_JSON_DEFAULT}"

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_L4_TUNING_SWEEP_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")" "$RUN_DIR"
: >"$LOG_PATH"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

emit_not_run() {
  jq -cn \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg mode "$MODE" \
    --arg log_path "$(to_rel "$LOG_PATH")" \
    '{
      schema: "sounio.omega.l4-tuning-sweep.v1",
      generated_at_utc: $generated_at_utc,
      mode: $mode,
      status_summary: "not_run",
      reason: "gate_disabled",
      target_dim: 0,
      targets: {},
      profiles: [],
      recommended_profile: null,
      blockers: [],
      log_path: $log_path
    }' >"$OUT_JSON"
}

if [[ "$MODE" == "off" ]]; then
  emit_not_run
  echo "omega_l4_tuning_sweep_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ROWS_JSONL="$TMP_DIR/rows.jsonl"
touch "$ROWS_JSONL"

python3 - "$PROFILES_JSON" <<'PY' >"$TMP_DIR/profiles.tsv"
import json
import sys

profiles = json.loads(sys.argv[1])
for row in profiles:
    print(
        "\t".join(
            [
                str(row.get("name", "")),
                str(row.get("strict_sqrt_interval", "")),
                str(row.get("strict_prov_interval", "")),
                str(row.get("strict_feedback_interval", "")),
                str(row.get("fast_prov_interval", "")),
                str(row.get("fast_feedback_interval", "")),
                str(row.get("strict_slot_cap", "")),
                str(row.get("fast_slot_cap", "")),
            ]
        )
    )
PY

while IFS=$'\t' read -r profile_name strict_sqrt strict_prov strict_feedback fast_prov fast_feedback strict_slots fast_slots <&3; do
  [[ -z "$profile_name" ]] && continue

  profile_gate_json="$RUN_DIR/${profile_name}.l4_gate.v1.json"
  profile_gate_log="$RUN_DIR/${profile_name}.l4_gate.log"

  set +e
  # Keep loop input isolated from ssh/scp subprocesses invoked by the gate.
  env \
    GEMM_DIMS="$GEMM_DIMS" \
    TARGET_DIM="$TARGET_DIM" \
    OMEGA_L4_NATIVE_GATE_MODE="required" \
    OMEGA_L4_ENABLE_NCU="0" \
    OMEGA_L4_NATIVE_GATE_OUT="$profile_gate_json" \
    OMEGA_L4_NATIVE_GATE_LOG="$profile_gate_log" \
    SHADOW_STRICT_SQRT_INTERVAL="$strict_sqrt" \
    SHADOW_STRICT_PROV_INTERVAL="$strict_prov" \
    SHADOW_STRICT_FEEDBACK_INTERVAL="$strict_feedback" \
    SHADOW_FAST_PROV_INTERVAL="$fast_prov" \
    SHADOW_FAST_FEEDBACK_INTERVAL="$fast_feedback" \
    SHADOW_STRICT_SLOT_CAP="$strict_slots" \
    SHADOW_FAST_SLOT_CAP="$fast_slots" \
    GPU_HOST="$GPU_HOST" \
    GPU_USER="$GPU_USER" \
    REMOTE_DIR="$REMOTE_DIR" \
    bash "$ROOT_DIR/scripts/omega/omega_l4_native_shadow_gate.sh" </dev/null >>"$LOG_PATH" 2>&1
  profile_rc=$?
  set -e

  profile_status="not_run"
  profile_reason="artifact_missing"
  strict_overhead=""
  fast_speedup=""
  fast_drift=""
  report_path=""
  benchmark_status="NOT RUN"
  if [[ -f "$profile_gate_json" ]]; then
    profile_status="$(jq -r '.status // "not_run"' "$profile_gate_json" 2>/dev/null || echo "not_run")"
    profile_reason="$(jq -r '.reason // "unknown"' "$profile_gate_json" 2>/dev/null || echo "unknown")"
    benchmark_status="$(jq -r '.benchmark.status // "NOT RUN"' "$profile_gate_json" 2>/dev/null || echo "NOT RUN")"
    strict_overhead="$(jq -r '.benchmark.metrics.strict_overhead_x_vs_cublas // empty' "$profile_gate_json" 2>/dev/null || true)"
    fast_speedup="$(jq -r '.benchmark.metrics.fast_speedup_x_vs_strict // empty' "$profile_gate_json" 2>/dev/null || true)"
    fast_drift="$(jq -r '.benchmark.metrics.fast_drift_vs_strict // empty' "$profile_gate_json" 2>/dev/null || true)"
    report_path="$(jq -r '.benchmark.report_path // empty' "$profile_gate_json" 2>/dev/null || true)"
  fi

  if [[ "$strict_overhead" == "null" ]]; then strict_overhead=""; fi
  if [[ "$fast_speedup" == "null" ]]; then fast_speedup=""; fi
  if [[ "$fast_drift" == "null" ]]; then fast_drift=""; fi

  jq -cn \
    --arg profile "$profile_name" \
    --arg status "$profile_status" \
    --arg reason "$profile_reason" \
    --arg benchmark_status "$benchmark_status" \
    --arg report_path "$report_path" \
    --arg gate_json "$(to_rel "$profile_gate_json")" \
    --arg gate_log "$(to_rel "$profile_gate_log")" \
    --argjson gate_rc "$profile_rc" \
    --arg strict_overhead "$strict_overhead" \
    --arg fast_speedup "$fast_speedup" \
    --arg fast_drift "$fast_drift" \
    --argjson strict_sqrt "$strict_sqrt" \
    --argjson strict_prov "$strict_prov" \
    --argjson strict_feedback "$strict_feedback" \
    --argjson fast_prov "$fast_prov" \
    --argjson fast_feedback "$fast_feedback" \
    --argjson strict_slots "$strict_slots" \
    --argjson fast_slots "$fast_slots" \
    '{
      profile: $profile,
      status: $status,
      reason: $reason,
      benchmark_status: $benchmark_status,
      gate_rc: $gate_rc,
      gate_artifact: $gate_json,
      gate_log: $gate_log,
      report_path: $report_path,
      metrics: {
        strict_overhead_x_vs_cublas: (if $strict_overhead == "" then null else ($strict_overhead | tonumber) end),
        fast_speedup_x_vs_strict: (if $fast_speedup == "" then null else ($fast_speedup | tonumber) end),
        fast_drift_vs_strict: (if $fast_drift == "" then null else ($fast_drift | tonumber) end)
      },
      tuning: {
        strict_sqrt_interval: $strict_sqrt,
        strict_prov_interval: $strict_prov,
        strict_feedback_interval: $strict_feedback,
        fast_prov_interval: $fast_prov,
        fast_feedback_interval: $fast_feedback,
        strict_slot_cap: $strict_slots,
        fast_slot_cap: $fast_slots
      }
    }' >>"$ROWS_JSONL"
done 3<"$TMP_DIR/profiles.tsv"

analysis_json="$(python3 - "$ROWS_JSONL" "$TARGET_STRICT_OVERHEAD_MAX" "$TARGET_FAST_SPEEDUP_MIN" "$TARGET_FAST_DRIFT_MAX" <<'PY'
import json
import sys
from pathlib import Path

rows_path = Path(sys.argv[1])
target_strict = float(sys.argv[2])
target_speedup = float(sys.argv[3])
target_drift = float(sys.argv[4])

rows = []
for line in rows_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    rows.append(json.loads(line))

for row in rows:
    metrics = row.get("metrics", {})
    so = metrics.get("strict_overhead_x_vs_cublas")
    fs = metrics.get("fast_speedup_x_vs_strict")
    fd = metrics.get("fast_drift_vs_strict")
    has_metrics = so is not None and fs is not None and fd is not None
    meets = (
        row.get("status") == "pass"
        and has_metrics
        and so <= target_strict
        and fs >= target_speedup
        and fd <= target_drift
    )
    row["meets_targets"] = bool(meets)
    if has_metrics:
        score = fs - (0.15 * so) - (0.5 * fd)
    else:
        score = -1e9
    row["score"] = score

ranked = sorted(
    rows,
    key=lambda r: (
        0 if r.get("meets_targets") else 1,
        0 if r.get("status") == "pass" else 1,
        -float(r.get("metrics", {}).get("fast_speedup_x_vs_strict") or -1e9),
        float(r.get("metrics", {}).get("strict_overhead_x_vs_cublas") or 1e9),
        float(r.get("metrics", {}).get("fast_drift_vs_strict") or 1e9),
        -float(r.get("score", -1e9)),
    ),
)

best = ranked[0] if ranked else None
status = "pass" if best and best.get("meets_targets") else "fail"
reason = "targets_met" if status == "pass" else "no_profile_meets_targets"
blockers = [] if status == "pass" else ["perf_regression"]

print(
    json.dumps(
        {
            "profiles": ranked,
            "best": best,
            "status": status,
            "reason": reason,
            "blockers": blockers,
        },
        separators=(",", ":"),
    )
)
PY
)"

status_summary="$(python3 - "$analysis_json" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("status", "fail"))
PY
)"
reason="$(python3 - "$analysis_json" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("reason", "no_profile_meets_targets"))
PY
)"
profiles_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
print(json.dumps(json.loads(sys.argv[1]).get("profiles", []), separators=(",", ":")))
PY
)"
recommended_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
print(json.dumps(json.loads(sys.argv[1]).get("best"), separators=(",", ":")))
PY
)"
blockers_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
print(json.dumps(json.loads(sys.argv[1]).get("blockers", []), separators=(",", ":")))
PY
)"

if [[ "$status_summary" != "pass" && "$MODE" != "required" ]]; then
  status_summary="not_run"
  reason="no_profile_meets_targets_auto"
fi

jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg mode "$MODE" \
  --arg status_summary "$status_summary" \
  --arg reason "$reason" \
  --argjson target_dim "$TARGET_DIM" \
  --argjson target_strict "$TARGET_STRICT_OVERHEAD_MAX" \
  --argjson target_speedup "$TARGET_FAST_SPEEDUP_MIN" \
  --argjson target_drift "$TARGET_FAST_DRIFT_MAX" \
  --argjson profiles "$profiles_json" \
  --argjson recommended_profile "$recommended_json" \
  --argjson blockers "$blockers_json" \
  --arg log_path "$(to_rel "$LOG_PATH")" \
  --arg run_dir "$(to_rel "$RUN_DIR")" \
  '{
    schema: "sounio.omega.l4-tuning-sweep.v1",
    generated_at_utc: $generated_at_utc,
    mode: $mode,
    status_summary: $status_summary,
    reason: $reason,
    target_dim: $target_dim,
    targets: {
      strict_overhead_x_vs_cublas_max: $target_strict,
      fast_speedup_x_vs_strict_min: $target_speedup,
      fast_drift_vs_strict_max: $target_drift
    },
    profiles: $profiles,
    recommended_profile: $recommended_profile,
    blockers: $blockers,
    run_dir: $run_dir,
    log_path: $log_path
  }' >"$OUT_JSON"

if [[ "$status_summary" == "pass" ]]; then
  echo "omega_l4_tuning_sweep_gate: status=pass report=$OUT_JSON"
  exit 0
fi
if [[ "$status_summary" == "not_run" ]]; then
  echo "omega_l4_tuning_sweep_gate: status=not_run reason=$reason report=$OUT_JSON"
  exit 0
fi

echo "omega_l4_tuning_sweep_gate: status=fail reason=$reason report=$OUT_JSON" >&2
exit 2
