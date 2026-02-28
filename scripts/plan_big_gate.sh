#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BOARD_JSON="${PLAN_BIG_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_status_board.v1.json}"
BOARD_MD="${PLAN_BIG_STATUS_MD:-$ROOT_DIR/artifacts/omega/plan_big_status_board.md}"
OUT_JSON="${PLAN_BIG_GATE_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_gate_status.v1.json}"

mkdir -p "$(dirname "$BOARD_JSON")" "$(dirname "$BOARD_MD")" "$(dirname "$OUT_JSON")"

bash "$ROOT_DIR/scripts/plan_big_status_board.sh" --run-gates

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

if jq -e '.gates | length > 0 and all(.status == "pass")' "$BOARD_JSON" >/dev/null 2>&1; then
  gates_pass="true"
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
  '{
    schema: "sounio.plan.big.gate-status.v1",
    generated_at_utc: $generated_at_utc,
    status: $status,
    board_json: $board_json,
    board_md: $board_md,
    checks: {
      gates_pass: $gates_pass,
      critical_pass: $critical_pass
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
