#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${PLAN_BIG_OVERNIGHT_HOURLY_REPORT_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_hourly_report.v1.json}"
OUT_JSONL="${PLAN_BIG_OVERNIGHT_HOURLY_REPORT_JSONL:-$ROOT_DIR/artifacts/omega/overnight_plan_big/hourly_report.v1.jsonl}"
TAIL_LINES="${PLAN_BIG_OVERNIGHT_HOURLY_REPORT_TAIL_LINES:-20}"
STRICT_CANARY_ENABLE="${PLAN_BIG_STRICT_CANARY_ENABLE:-1}"
RETENTION_ENABLE="${PLAN_BIG_OVERNIGHT_RETENTION_ENABLE:-1}"

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_JSONL")"

status_json="$(bash scripts/overnight_plan_big_status.sh --json --tail-lines "$TAIL_LINES")"
health_json='{}'
gate_json='{}'
burnin_json='{}'

if [[ -f "$ROOT_DIR/artifacts/omega/overnight_plan_big_health.v1.json" ]]; then
  health_json="$(cat "$ROOT_DIR/artifacts/omega/overnight_plan_big_health.v1.json")"
fi
if [[ -f "$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json" ]]; then
  gate_json="$(cat "$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json")"
fi
if [[ -f "$ROOT_DIR/artifacts/omega/overnight_plan_big_burnin.v1.json" ]]; then
  burnin_json="$(cat "$ROOT_DIR/artifacts/omega/overnight_plan_big_burnin.v1.json")"
fi

strict_canary_json='{}'
strict_canary_rc=0
if [[ "$STRICT_CANARY_ENABLE" == "1" ]]; then
  set +e
  bash scripts/plan_big_strict_canary.sh >/dev/null 2>&1
  strict_canary_rc=$?
  set -e
  if [[ -f "$ROOT_DIR/artifacts/omega/plan_big_strict_canary/latest.v1.json" ]]; then
    strict_canary_json="$(cat "$ROOT_DIR/artifacts/omega/plan_big_strict_canary/latest.v1.json")"
  fi
fi

retention_json='{}'
retention_rc=0
if [[ "$RETENTION_ENABLE" == "1" ]]; then
  set +e
  bash scripts/rotate_overnight_plan_big_artifacts.sh >/dev/null 2>&1
  retention_rc=$?
  set -e
  if [[ -f "$ROOT_DIR/artifacts/omega/overnight_plan_big_retention.v1.json" ]]; then
    retention_json="$(cat "$ROOT_DIR/artifacts/omega/overnight_plan_big_retention.v1.json")"
  fi
fi

report_json="$(jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson status "$status_json" \
  --argjson health "$health_json" \
  --argjson gate "$gate_json" \
  --argjson burnin "$burnin_json" \
  --argjson strict_canary "$strict_canary_json" \
  --argjson strict_canary_rc "$strict_canary_rc" \
  --argjson retention "$retention_json" \
  --argjson retention_rc "$retention_rc" \
  '{
    schema:"sounio.plan.big.overnight.hourly-report.v1",
    generated_at_utc:$generated_at_utc,
    snapshot:{
      state: ($status.state // "unknown"),
      runner_pid: ($status.runner.pid // ""),
      heartbeat_fresh: ($status.heartbeat.fresh // false),
      latest_status: ($status.latest.status // ""),
      latest_rc: ($status.latest.rc // 0),
      health_healthy: ($health.healthy // null),
      health_action: ($health.action // ""),
      gate_status: ($gate.status // "unknown"),
      gate_pass_marker: ($gate.pass_marker // null),
      burnin_status: ($burnin.status // "unknown"),
      burnin_checks_total: ($burnin.checks_total // null),
      burnin_checks_passed: ($burnin.checks_passed // null),
      strict_canary_status: ($strict_canary.status // "unknown"),
      strict_canary_rc: $strict_canary_rc,
      retention_status: (if $retention_rc == 0 then "pass" else "fail" end),
      retention_rc: $retention_rc
    },
    sources:{
      status:"artifacts/omega/overnight_plan_big/heartbeat.v1.json + latest.v1.json",
      health:"artifacts/omega/overnight_plan_big_health.v1.json",
      gate:"artifacts/omega/plan_big_gate_status.v1.json",
      burnin:"artifacts/omega/overnight_plan_big_burnin.v1.json",
      strict_canary:"artifacts/omega/plan_big_strict_canary/latest.v1.json",
      retention:"artifacts/omega/overnight_plan_big_retention.v1.json"
    },
    status:$status,
    health:$health,
    gate:$gate,
    burnin:$burnin,
    strict_canary:$strict_canary,
    strict_canary_rc:$strict_canary_rc,
    retention:$retention,
    retention_rc:$retention_rc
  }')"

printf '%s\n' "$report_json" > "$OUT_JSON"
printf '%s\n' "$report_json" >> "$OUT_JSONL"

echo "OVERNIGHT_PLAN_BIG_HOURLY_REPORT_PASS"
echo "JSON: $OUT_JSON"
echo "JSONL: $OUT_JSONL"
