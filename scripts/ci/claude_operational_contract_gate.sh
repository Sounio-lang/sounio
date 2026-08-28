#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${SOUNIO_CLAUDE_CONTRACT_OUT:-$ROOT_DIR/artifacts/omega/claude_operational_contract_status.v1.json}"
CANONICAL_DOC="${SOUNIO_CLAUDE_CANONICAL_DOC:-$ROOT_DIR/.claude/PLAN_CANONICAL_EXECUTION.md}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_PATH")"

extract_marked_list_json() {
  local file="$1"
  local start_marker="$2"
  local end_marker="$3"
  awk -v start="$start_marker" -v end="$end_marker" '
    $0 ~ start { in_section=1; next }
    $0 ~ end { in_section=0; exit }
    in_section { print }
  ' "$file" \
    | sed -n 's/.*`\([^`]*\)`.*/\1/p' \
    | jq -R -s 'split("\n") | map(select(length > 0))'
}

run_check() {
  local name="$1"
  shift
  local log="$TMP_DIR/${name}.log"
  if "$@" >"$log" 2>&1; then
    printf '{"name":"%s","status":"pass","log":"%s"}' "$name" "$log"
    return 0
  fi
  printf '{"name":"%s","status":"fail","log":"%s"}' "$name" "$log"
  return 1
}

CHECKS=()

check_json="$(run_check prompt_execution_contract bash scripts/ci/check_prompt_execution_contract.sh || true)"
CHECKS+=("$check_json")

check_json="$(run_check plan_consistency bash scripts/ci/check_claude_plan_consistency.sh || true)"
CHECKS+=("$check_json")

check_json="$(run_check check_sio_window bash scripts/ci/check_check_sio_integration_window.sh || true)"
CHECKS+=("$check_json")

check_json="$(run_check parallel_blocker_contract bash scripts/ci/check_parallel_blocker_contract.sh || true)"
CHECKS+=("$check_json")

check_json="$(run_check madaros_blocker_contract bash scripts/ci/madaros_blocker_contract_gate.sh || true)"
CHECKS+=("$check_json")

checks_joined=""
for item in "${CHECKS[@]}"; do
  if [[ -n "$checks_joined" ]]; then
    checks_joined+=","
  fi
  checks_joined+="$item"
done

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
canonical_precedence_json="$(extract_marked_list_json "$CANONICAL_DOC" "CANONICAL_PRECEDENCE_START" "CANONICAL_PRECEDENCE_END")"
locked_track_b_order_json="$(extract_marked_list_json "$CANONICAL_DOC" "LOCKED_TRACK_B_ORDER_START" "LOCKED_TRACK_B_ORDER_END")"

jq -cn \
  --arg generated_at_utc "$generated_at_utc" \
  --argjson checks "[$checks_joined]" \
  --argjson canonical_precedence "$canonical_precedence_json" \
  --argjson locked_track_b_order "$locked_track_b_order_json" \
  '{
    schema: "sounio.claude.operational.contract.v1",
    generated_at_utc: $generated_at_utc,
    canonical_precedence: $canonical_precedence,
    locked_track_b_order: $locked_track_b_order,
    checks: $checks,
    status: (if ($checks | all(.status == "pass")) then "pass" else "fail" end)
  }' >"$OUT_PATH"

if ! jq -e '.status == "pass"' "$OUT_PATH" >/dev/null; then
  echo "error: claude operational contract gate failed" >&2
  cat "$OUT_PATH" >&2
  exit 1
fi

echo "CLAUDE_OPERATIONAL_CONTRACT_PASS out=$OUT_PATH"
