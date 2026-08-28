#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BOARD_JSON="${PLAN_BIG_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_status_board.v1.json}"
BOARD_MD="${PLAN_BIG_STATUS_MD:-$ROOT_DIR/artifacts/omega/plan_big_status_board.md}"
OUT_JSON="${PLAN_BIG_GATE_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json}"
OVERNIGHT_HEALTH_REQUIRED="${PLAN_BIG_OVERNIGHT_HEALTH_REQUIRED:-0}"
OVERNIGHT_AUTO_HEAL="${PLAN_BIG_OVERNIGHT_HEALTH_AUTO_HEAL:-1}"
OVERNIGHT_BURNIN_REQUIRED="${PLAN_BIG_OVERNIGHT_BURNIN_REQUIRED:-0}"
OVERNIGHT_BURNIN_JSON="${PLAN_BIG_OVERNIGHT_BURNIN_JSON:-$ROOT_DIR/artifacts/omega/overnight_plan_big_burnin.v1.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-overnight-health)
      OVERNIGHT_HEALTH_REQUIRED=1
      shift
      ;;
    --without-overnight-health)
      OVERNIGHT_HEALTH_REQUIRED=0
      shift
      ;;
    --overnight-auto-heal)
      OVERNIGHT_AUTO_HEAL=1
      shift
      ;;
    --overnight-no-auto-heal)
      OVERNIGHT_AUTO_HEAL=0
      shift
      ;;
    --require-overnight-burnin)
      OVERNIGHT_BURNIN_REQUIRED=1
      shift
      ;;
    --no-require-overnight-burnin)
      OVERNIGHT_BURNIN_REQUIRED=0
      shift
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

if [[ "$OVERNIGHT_HEALTH_REQUIRED" != "0" && "$OVERNIGHT_HEALTH_REQUIRED" != "1" ]]; then
  echo "error: PLAN_BIG_OVERNIGHT_HEALTH_REQUIRED must be 0 or 1" >&2
  exit 2
fi
if [[ "$OVERNIGHT_AUTO_HEAL" != "0" && "$OVERNIGHT_AUTO_HEAL" != "1" ]]; then
  echo "error: PLAN_BIG_OVERNIGHT_HEALTH_AUTO_HEAL must be 0 or 1" >&2
  exit 2
fi
if [[ "$OVERNIGHT_BURNIN_REQUIRED" != "0" && "$OVERNIGHT_BURNIN_REQUIRED" != "1" ]]; then
  echo "error: PLAN_BIG_OVERNIGHT_BURNIN_REQUIRED must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$(dirname "$BOARD_JSON")" "$(dirname "$BOARD_MD")" "$(dirname "$OUT_JSON")"

bash "$ROOT_DIR/scripts/overnight/plan_big_status_board.sh" --run-gates

missing=0
for path in "$BOARD_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "error: missing required board artifact: $path" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

gates_pass="false"
critical_pass="false"
overnight_health_pass="null"
overnight_health_ran=false
overnight_health_rc=0
overnight_health_action="skipped"
overnight_burnin_checked=false
overnight_burnin_pass="null"
overnight_burnin_reason="skipped"

if jq -e '.gates | length > 0 and all(.status == "pass")' "$BOARD_JSON" >/dev/null 2>&1; then
  gates_pass="true"
fi

if [[ "$OVERNIGHT_HEALTH_REQUIRED" -eq 1 ]]; then
  overnight_health_ran=true
  overnight_health_action="checked"
  if [[ "$OVERNIGHT_AUTO_HEAL" -eq 1 ]]; then
    if bash "$ROOT_DIR/scripts/check_overnight_plan_big_health.sh" --auto-heal; then
      overnight_health_pass="true"
      overnight_health_rc=0
    else
      overnight_health_pass="false"
      overnight_health_rc=$?
    fi
  else
    if bash "$ROOT_DIR/scripts/check_overnight_plan_big_health.sh"; then
      overnight_health_pass="true"
      overnight_health_rc=0
    else
      overnight_health_pass="false"
      overnight_health_rc=$?
    fi
  fi
fi

if jq -e '
  .summary.parallel_cutover_status == "pass" and
  .summary.track_a_status == "pass" and
  .summary.track_b_status == "pass" and
  .summary.track_b_order_status == "pass" and
  .summary.claude_operational_contract_status == "pass" and
  .summary.lsp_smoke_status == "pass" and
  .summary.ui_type.backlog_quality_status == "pass" and
  .summary.ui_type.reclassify_candidates == 0
' "$BOARD_JSON" >/dev/null 2>&1; then
  critical_pass="true"
fi

if [[ "$OVERNIGHT_HEALTH_REQUIRED" -eq 1 && "$overnight_health_pass" != "true" ]]; then
  critical_pass="false"
fi
if [[ "$OVERNIGHT_BURNIN_REQUIRED" -eq 1 ]]; then
  overnight_burnin_checked=true
  if [[ ! -f "$OVERNIGHT_BURNIN_JSON" ]]; then
    overnight_burnin_pass="false"
    overnight_burnin_reason="missing_burnin_artifact"
  else
    burnin_schema="$(jq -r '.schema // ""' "$OVERNIGHT_BURNIN_JSON" 2>/dev/null || true)"
    burnin_status="$(jq -r '.status // ""' "$OVERNIGHT_BURNIN_JSON" 2>/dev/null || true)"
    if [[ "$burnin_schema" == "sounio.plan.big.overnight.burnin.v1" && "$burnin_status" == "pass" ]]; then
      overnight_burnin_pass="true"
      overnight_burnin_reason="burnin_pass"
    else
      overnight_burnin_pass="false"
      overnight_burnin_reason="burnin_not_pass"
    fi
  fi
fi
if [[ "$OVERNIGHT_BURNIN_REQUIRED" -eq 1 && "$overnight_burnin_pass" != "true" ]]; then
  critical_pass="false"
fi

status="fail"
if [[ "$gates_pass" == "true" && "$critical_pass" == "true" ]]; then
  status="pass"
fi

jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg status "$status" \
  --arg board_json "${BOARD_JSON#$ROOT_DIR/}" \
  --arg board_md "${BOARD_MD#$ROOT_DIR/}" \
  --argjson gates_pass "$gates_pass" \
  --argjson critical_pass "$critical_pass" \
  --argjson overnight_health_required "$OVERNIGHT_HEALTH_REQUIRED" \
  --argjson overnight_health_ran "$overnight_health_ran" \
  --argjson overnight_health_pass "$overnight_health_pass" \
  --argjson overnight_health_rc "$overnight_health_rc" \
  --arg overnight_health_action "$overnight_health_action" \
  --argjson overnight_burnin_required "$OVERNIGHT_BURNIN_REQUIRED" \
  --argjson overnight_burnin_checked "$overnight_burnin_checked" \
  --argjson overnight_burnin_pass "$overnight_burnin_pass" \
  --arg overnight_burnin_reason "$overnight_burnin_reason" \
  --arg overnight_burnin_json "${OVERNIGHT_BURNIN_JSON#$ROOT_DIR/}" \
  '{
    schema: "sounio.plan.big.gate-status.v1",
    generated_at_utc: $generated_at_utc,
    status: $status,
    board_json: $board_json,
    board_md: $board_md,
    checks: {
      gates_pass: $gates_pass,
      critical_pass: $critical_pass,
      overnight_health_required: ($overnight_health_required == 1),
      overnight_health_ran: $overnight_health_ran,
      overnight_health_pass: $overnight_health_pass,
      overnight_health_rc: $overnight_health_rc,
      overnight_health_action: $overnight_health_action,
      overnight_burnin_required: ($overnight_burnin_required == 1),
      overnight_burnin_checked: $overnight_burnin_checked,
      overnight_burnin_pass: $overnight_burnin_pass,
      overnight_burnin_reason: $overnight_burnin_reason,
      overnight_burnin_json: $overnight_burnin_json
    },
    pass_marker: ($status == "pass")
  }' > "$OUT_JSON"

if [[ "$status" != "pass" ]]; then
  echo "error: PLAN_BIG gate failed" >&2
  cat "$OUT_JSON" >&2
  exit 1
fi

echo "PLAN_BIG_GATE_PASS"
echo "JSON: $OUT_JSON"
