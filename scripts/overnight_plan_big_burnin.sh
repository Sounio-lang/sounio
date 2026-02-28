#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${PLAN_BIG_OVERNIGHT_BURNIN_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_burnin.v1.json}"
HEALTH_JSON="${PLAN_BIG_OVERNIGHT_HEALTH_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_health.v1.json}"
GATE_JSON="${PLAN_BIG_GATE_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json}"

DURATION_SEC="${PLAN_BIG_OVERNIGHT_BURNIN_DURATION_SEC:-86400}"
CHECK_INTERVAL_SEC="${PLAN_BIG_OVERNIGHT_BURNIN_CHECK_INTERVAL_SEC:-60}"
TAIL_LINES="${PLAN_BIG_OVERNIGHT_BURNIN_TAIL_LINES:-20}"
AUTO_HEAL="${PLAN_BIG_OVERNIGHT_BURNIN_AUTO_HEAL:-1}"
FAIL_FAST="${PLAN_BIG_OVERNIGHT_BURNIN_FAIL_FAST:-1}"

usage() {
  cat <<USAGE
Usage: scripts/overnight_plan_big_burnin.sh [options]

Options:
  --duration-sec N         Total soak duration in seconds (default: 86400)
  --check-interval-sec N   Seconds between checks (default: 60)
  --tail-lines N           Tail lines to include in status/health checks (default: 20)
  --auto-heal 0|1          Use health check auto-heal behavior (default: 1)
  --fail-fast 0|1          Stop on first failed check (default: 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration-sec)
      DURATION_SEC="$2"
      shift 2
      ;;
    --check-interval-sec)
      CHECK_INTERVAL_SEC="$2"
      shift 2
      ;;
    --tail-lines)
      TAIL_LINES="$2"
      shift 2
      ;;
    --auto-heal)
      AUTO_HEAL="$2"
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$DURATION_SEC" =~ ^[0-9]+$ ]] || [[ "$DURATION_SEC" -eq 0 ]]; then
  echo "error: --duration-sec must be positive integer" >&2
  exit 2
fi
if ! [[ "$CHECK_INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$CHECK_INTERVAL_SEC" -eq 0 ]]; then
  echo "error: --check-interval-sec must be positive integer" >&2
  exit 2
fi
if ! [[ "$TAIL_LINES" =~ ^[0-9]+$ ]] || [[ "$TAIL_LINES" -eq 0 ]]; then
  echo "error: --tail-lines must be positive integer" >&2
  exit 2
fi
if [[ "$AUTO_HEAL" != "0" && "$AUTO_HEAL" != "1" ]]; then
  echo "error: --auto-heal must be 0 or 1" >&2
  exit 2
fi
if [[ "$FAIL_FAST" != "0" && "$FAIL_FAST" != "1" ]]; then
  echo "error: --fail-fast must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_JSON")"

tmp_cycles_jsonl="$(mktemp)"
trap 'rm -f "$tmp_cycles_jsonl"' EXIT

start_epoch="$(date -u +%s)"
started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
checks_total=0
checks_passed=0
first_failure_at_utc=""
first_failure_reason=""

while true; do
  now_epoch="$(date -u +%s)"
  elapsed_sec=$((now_epoch - start_epoch))
  if [[ "$elapsed_sec" -ge "$DURATION_SEC" ]]; then
    break
  fi

  checks_total=$((checks_total + 1))
  check_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  status_json="$(scripts/overnight_plan_big_status.sh --json --tail-lines "$TAIL_LINES")"
  state="$(jq -r '.state // ""' <<<"$status_json")"
  stale_state=false
  if [[ "$state" == "stale_lock" || "$state" == "stale_runner_pid" || "$state" == "stale_pair" ]]; then
    stale_state=true
  fi

  health_rc=0
  if [[ "$AUTO_HEAL" -eq 1 ]]; then
    scripts/check_overnight_plan_big_health.sh --auto-heal --tail-lines "$TAIL_LINES" >/dev/null 2>&1 || health_rc=$?
  else
    scripts/check_overnight_plan_big_health.sh --tail-lines "$TAIL_LINES" >/dev/null 2>&1 || health_rc=$?
  fi
  health_healthy=false
  health_action=""
  health_failed_checks='[]'
  if [[ -f "$HEALTH_JSON" ]]; then
    health_healthy="$(jq -r '.healthy // false' "$HEALTH_JSON" 2>/dev/null || echo false)"
    health_action="$(jq -r '.action // ""' "$HEALTH_JSON" 2>/dev/null || echo "")"
    health_failed_checks="$(jq -c '.failed_checks // []' "$HEALTH_JSON" 2>/dev/null || echo '[]')"
  fi

  gate_present=false
  gate_valid=false
  gate_continuity=false
  gate_status=""
  gate_pass_marker=""
  if [[ -f "$GATE_JSON" ]]; then
    gate_present=true
    gate_schema="$(jq -r '.schema // ""' "$GATE_JSON" 2>/dev/null || true)"
    gate_status="$(jq -r '.status // ""' "$GATE_JSON" 2>/dev/null || true)"
    gate_pass_marker="$(jq -r '.pass_marker // false' "$GATE_JSON" 2>/dev/null || true)"
    if [[ "$gate_schema" == "sounio.plan.big.gate-status.v1" ]]; then
      gate_valid=true
    fi
    if [[ "$gate_valid" == "true" ]]; then
      if [[ "$gate_status" == "pass" && "$gate_pass_marker" == "true" ]]; then
        gate_continuity=true
      elif [[ "$gate_status" == "fail" && "$gate_pass_marker" == "false" ]]; then
        gate_continuity=true
      fi
    fi
  fi

  cycle_pass=false
  cycle_reason="ok"
  if [[ "$stale_state" == "true" ]]; then
    cycle_reason="stale_state"
  elif [[ "$health_healthy" != "true" || "$health_rc" -ne 0 ]]; then
    cycle_reason="health_contract_failed"
  elif [[ "$gate_continuity" != "true" ]]; then
    cycle_reason="gate_marker_continuity_failed"
  else
    cycle_pass=true
    checks_passed=$((checks_passed + 1))
  fi

  jq -cn \
    --arg check_at_utc "$check_at_utc" \
    --arg state "$state" \
    --argjson stale_state "$stale_state" \
    --argjson health_healthy "$health_healthy" \
    --argjson health_rc "$health_rc" \
    --arg health_action "$health_action" \
    --argjson health_failed_checks "$health_failed_checks" \
    --argjson gate_present "$gate_present" \
    --argjson gate_valid "$gate_valid" \
    --arg gate_status "$gate_status" \
    --arg gate_pass_marker "$gate_pass_marker" \
    --argjson gate_continuity "$gate_continuity" \
    --argjson cycle_pass "$cycle_pass" \
    --arg cycle_reason "$cycle_reason" \
    '{
      check_at_utc:$check_at_utc,
      state:$state,
      stale_state:$stale_state,
      health:{healthy:$health_healthy,rc:$health_rc,action:$health_action,failed_checks:$health_failed_checks},
      gate:{present:$gate_present,valid:$gate_valid,status:$gate_status,pass_marker:$gate_pass_marker,continuity:$gate_continuity},
      cycle_pass:$cycle_pass,
      cycle_reason:$cycle_reason
    }' >>"$tmp_cycles_jsonl"

  if [[ "$cycle_pass" != "true" && -z "$first_failure_at_utc" ]]; then
    first_failure_at_utc="$check_at_utc"
    first_failure_reason="$cycle_reason"
  fi

  if [[ "$cycle_pass" != "true" && "$FAIL_FAST" -eq 1 ]]; then
    break
  fi

  sleep "$CHECK_INTERVAL_SEC"
done

finished_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
finish_epoch="$(date -u +%s)"
duration_actual_sec=$((finish_epoch - start_epoch))
if [[ "$duration_actual_sec" -lt 0 ]]; then
  duration_actual_sec=0
fi

burnin_status="fail"
if [[ "$checks_total" -gt 0 && "$checks_total" -eq "$checks_passed" ]]; then
  burnin_status="pass"
fi

jq -s \
  --arg generated_at_utc "$finished_at_utc" \
  --arg started_at_utc "$started_at_utc" \
  --arg finished_at_utc "$finished_at_utc" \
  --arg status "$burnin_status" \
  --arg first_failure_at_utc "$first_failure_at_utc" \
  --arg first_failure_reason "$first_failure_reason" \
  --argjson duration_target_sec "$DURATION_SEC" \
  --argjson duration_actual_sec "$duration_actual_sec" \
  --argjson check_interval_sec "$CHECK_INTERVAL_SEC" \
  --argjson checks_total "$checks_total" \
  --argjson checks_passed "$checks_passed" \
  --argjson auto_heal "$AUTO_HEAL" \
  --argjson fail_fast "$FAIL_FAST" \
  '{
    schema:"sounio.plan.big.overnight.burnin.v1",
    generated_at_utc:$generated_at_utc,
    started_at_utc:$started_at_utc,
    finished_at_utc:$finished_at_utc,
    duration_target_sec:$duration_target_sec,
    duration_actual_sec:$duration_actual_sec,
    check_interval_sec:$check_interval_sec,
    checks_total:$checks_total,
    checks_passed:$checks_passed,
    first_failure_at_utc:(if $first_failure_at_utc == "" then null else $first_failure_at_utc end),
    first_failure_reason:(if $first_failure_reason == "" then null else $first_failure_reason end),
    auto_heal:($auto_heal == 1),
    fail_fast:($fail_fast == 1),
    status:$status,
    checks: .
  }' "$tmp_cycles_jsonl" >"$OUT_JSON"

if [[ "$burnin_status" == "pass" ]]; then
  echo "OVERNIGHT_PLAN_BIG_BURNIN_PASS"
  echo "JSON: $OUT_JSON"
  exit 0
fi

echo "error: overnight burn-in failed" >&2
echo "JSON: $OUT_JSON" >&2
exit 1
