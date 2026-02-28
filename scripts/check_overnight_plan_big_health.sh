#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${PLAN_BIG_OVERNIGHT_HEALTH_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_health.v1.json}"
AUTO_HEAL=0
TAIL_LINES=20

usage() {
  cat <<USAGE
Usage: scripts/check_overnight_plan_big_health.sh [--auto-heal] [--tail-lines N]

Options:
  --auto-heal     Attempt restart when state is stale/stopped
  --tail-lines N  Log tail lines for embedded status snapshot (default: 20)
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

mkdir -p "$(dirname "$OUT_JSON")"

status_json="$(scripts/overnight_plan_big_status.sh --json --tail-lines "$TAIL_LINES")"
state="$(jq -r '.state' <<<"$status_json")"
healthy=false
action="none"
notes=""

case "$state" in
  running)
    healthy=true
    ;;
  stale_lock|stale_runner_pid|stale_pair|stopped)
    healthy=false
    ;;
  *)
    healthy=false
    notes="unknown_state"
    ;;
esac

if [[ "$healthy" == "false" && "$AUTO_HEAL" -eq 1 ]]; then
  action="restart"
  bash scripts/start_overnight_plan_big.sh --interval-sec 900 --max-runs 0 --stop-on-pass 0 >/tmp/plan_big_health_restart.log 2>&1 || true
  healthy=false
  for _ in $(seq 1 10); do
    status_json="$(scripts/overnight_plan_big_status.sh --json --tail-lines "$TAIL_LINES")"
    state="$(jq -r '.state' <<<"$status_json")"
    if [[ "$state" == "running" ]]; then
      healthy=true
      break
    fi
    sleep 1
  done
  if [[ "$healthy" != "true" ]]; then
    notes="restart_attempted_but_not_running"
  fi
fi

jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg state "$state" \
  --arg action "$action" \
  --arg notes "$notes" \
  --argjson healthy "$healthy" \
  --argjson auto_heal "$AUTO_HEAL" \
  --argjson status "$status_json" \
  '{
    schema:"sounio.plan.big.overnight.health.v1",
    generated_at_utc:$generated_at_utc,
    healthy:$healthy,
    state:$state,
    auto_heal:($auto_heal == 1),
    action:$action,
    notes:$notes,
    status:$status
  }' > "$OUT_JSON"

if [[ "$healthy" == "true" ]]; then
  echo "OVERNIGHT_PLAN_BIG_HEALTH_PASS"
  echo "JSON: $OUT_JSON"
  exit 0
fi

echo "error: overnight plan-big health check failed (state=$state)" >&2
echo "JSON: $OUT_JSON" >&2
exit 1
