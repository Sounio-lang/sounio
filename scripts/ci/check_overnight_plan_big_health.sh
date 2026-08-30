#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${PLAN_BIG_OVERNIGHT_HEALTH_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_health.v1.json}"
AUTO_HEAL=0
TAIL_LINES=20
REVALIDATE_SEC="${PLAN_BIG_OVERNIGHT_HEALTH_REVALIDATE_SEC:-120}"

usage() {
  cat <<USAGE
Usage: scripts/ci/check_overnight_plan_big_health.sh [--auto-heal] [--tail-lines N] [--revalidate-sec N]

Options:
  --auto-heal     Attempt restart when state is stale/stopped
  --tail-lines N  Log tail lines for embedded status snapshot (default: 20)
  --revalidate-sec N  Seconds to wait for healthy contract after remediation (default: 120)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-heal)
      AUTO_HEAL=1
      shift
      ;;
    --tail-lines)
      TAIL_LINES="$2"
      shift 2
      ;;
    --revalidate-sec)
      REVALIDATE_SEC="$2"
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

if ! [[ "$TAIL_LINES" =~ ^[0-9]+$ ]] || [[ "$TAIL_LINES" -eq 0 ]]; then
  echo "error: --tail-lines must be positive integer" >&2
  exit 2
fi
if ! [[ "$REVALIDATE_SEC" =~ ^[0-9]+$ ]] || [[ "$REVALIDATE_SEC" -eq 0 ]]; then
  echo "error: --revalidate-sec must be positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_JSON")"

evaluate_contract() {
  local status_json="$1"
  jq -cn \
    --argjson status "$status_json" \
    '{
      checks: {
        state_running: ($status.state == "running"),
        runner_lock_consistent: (
          $status.runner.live and
          $status.lock.live and
          ($status.runner.pid != "") and
          ($status.runner.pid == $status.lock.pid)
        ),
        heartbeat_valid_non_stale: (
          $status.heartbeat.valid and
          ($status.heartbeat.stale | not) and
          $status.heartbeat.pid_live
        ),
        latest_valid: $status.latest.valid,
        latest_pass: (
          $status.latest.status == "pass" and
          ($status.latest.rc == 0) and
          ($status.latest.pass_marker == true)
        )
      }
    }
    | . + {
        healthy: (.checks | to_entries | all(.value == true)),
        failed_checks: (.checks | to_entries | map(select(.value != true)) | map(.key))
      }'
}

pre_status_json="$(scripts/overnight_plan_big_status.sh --json --tail-lines "$TAIL_LINES")"
pre_state="$(jq -r '.state' <<<"$pre_status_json")"
pre_contract="$(evaluate_contract "$pre_status_json")"
healthy="$(jq -r '.healthy' <<<"$pre_contract")"
failed_checks="$(jq -c '.failed_checks' <<<"$pre_contract")"

post_status_json="$pre_status_json"
post_contract="$pre_contract"
post_state="$pre_state"
action="none"
notes=""
restart_pid=""
restart_log="$(mktemp)"
trap 'rm -f "$restart_log"' EXIT

if [[ "$healthy" != "true" && "$AUTO_HEAL" -eq 1 ]]; then
  action="restart_attempted"
  {
    echo "[health] remediation start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash scripts/stop_overnight_plan_big.sh --force || true
    bash scripts/start_overnight_plan_big.sh --interval-sec 900 --max-runs 0 --stop-on-pass 0 || true
  } >"$restart_log" 2>&1

  healthy="false"
  for _ in $(seq 1 "$REVALIDATE_SEC"); do
    post_status_json="$(scripts/overnight_plan_big_status.sh --json --tail-lines "$TAIL_LINES")"
    post_contract="$(evaluate_contract "$post_status_json")"
    healthy="$(jq -r '.healthy' <<<"$post_contract")"
    if [[ "$healthy" == "true" ]]; then
      break
    fi
    sleep 1
  done

  post_state="$(jq -r '.state' <<<"$post_status_json")"
  restart_pid="$(jq -r '.runner.pid // ""' <<<"$post_status_json")"
  failed_checks="$(jq -c '.failed_checks' <<<"$post_contract")"
  if [[ "$healthy" == "true" ]]; then
    action="restart_succeeded"
  else
    action="restart_failed"
    notes="restart_attempted_but_contract_unhealthy"
  fi
else
  post_status_json="$pre_status_json"
  post_contract="$pre_contract"
  post_state="$pre_state"
  healthy="$(jq -r '.healthy' <<<"$post_contract")"
  failed_checks="$(jq -c '.failed_checks' <<<"$post_contract")"
fi

jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg pre_state "$pre_state" \
  --arg post_state "$post_state" \
  --arg action "$action" \
  --arg notes "$notes" \
  --arg restart_pid "$restart_pid" \
  --argjson healthy "$healthy" \
  --argjson auto_heal "$AUTO_HEAL" \
  --argjson failed_checks "$failed_checks" \
  --arg restart_log_path "${restart_log#$ROOT_DIR/}" \
  --argjson pre_contract "$pre_contract" \
  --argjson post_contract "$post_contract" \
  --argjson pre_status "$pre_status_json" \
  --argjson post_status "$post_status_json" \
  '{
    schema:"sounio.plan.big.overnight.health.v1",
    generated_at_utc:$generated_at_utc,
    healthy:$healthy,
    state:$post_state,
    pre_state:$pre_state,
    post_state:$post_state,
    auto_heal:($auto_heal == 1),
    action:$action,
    restart_pid:$restart_pid,
    notes:$notes,
    failed_checks:$failed_checks,
    restart_log_path:$restart_log_path,
    pre_contract:$pre_contract,
    post_contract:$post_contract,
    pre_status:$pre_status,
    status:$post_status
  }' > "$OUT_JSON"

if [[ "$healthy" == "true" ]]; then
  echo "OVERNIGHT_PLAN_BIG_HEALTH_PASS"
  echo "JSON: $OUT_JSON"
  exit 0
fi

echo "error: overnight plan-big health check failed (state=$post_state checks=$failed_checks action=$action)" >&2
echo "JSON: $OUT_JSON" >&2
exit 1
