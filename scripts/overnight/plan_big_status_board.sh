#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_GATES=0
if [[ "${1:-}" == "--run-gates" ]]; then
  RUN_GATES=1
fi

OUT_JSON="${PLAN_BIG_STATUS_JSON:-$ROOT_DIR/artifacts/omega/plan_big_status_board.v1.json}"
OUT_MD="${PLAN_BIG_STATUS_MD:-$ROOT_DIR/artifacts/omega/plan_big_status_board.md}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_MD")"

run_gate() {
  local name="$1"
  shift
  local log="$TMP_DIR/${name}.log"
  if "$@" >"$log" 2>&1; then
    jq -cn --arg name "$name" --arg status "pass" --arg log "$log" '{name:$name,status:$status,log:$log}'
    return 0
  fi
  jq -cn --arg name "$name" --arg status "fail" --arg log "$log" '{name:$name,status:$status,log:$log}'
  return 1
}

copy_log() {
  local src="$1"
  local dst_rel="$2"
  local dst="$ROOT_DIR/$dst_rel"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

check_file_status() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "present"
  else
    echo "missing"
  fi
}

extract_json_field_or() {
  local file="$1"
  local query="$2"
  local fallback="$3"
  if [[ -f "$file" ]]; then
    jq -r "$query // \"$fallback\"" "$file" 2>/dev/null || echo "$fallback"
  else
    echo "$fallback"
  fi
}

declare -a GATE_RESULTS

if [[ "$RUN_GATES" -eq 1 ]]; then
  gate_json="$(run_gate claude_plan_consistency bash scripts/ci/check_claude_plan_consistency.sh || true)"
  GATE_RESULTS+=("$gate_json")
  copy_log "$(jq -r '.log' <<<"$gate_json")" "artifacts/omega/plan_big_gate_claude_plan_consistency.log"

  gate_json="$(run_gate claude_operational_contract bash scripts/ci/claude_operational_contract_gate.sh || true)"
  GATE_RESULTS+=("$gate_json")
  copy_log "$(jq -r '.log' <<<"$gate_json")" "artifacts/omega/plan_big_gate_claude_operational_contract.log"

  gate_json="$(run_gate lsp_smoke bash scripts/ci/lsp_smoke_gate.sh || true)"
  GATE_RESULTS+=("$gate_json")
  copy_log "$(jq -r '.log' <<<"$gate_json")" "artifacts/omega/plan_big_gate_lsp_smoke.log"

  gate_json="$(run_gate ui_type_deignore bash scripts/ci/ui_type_deignore_audit.sh || true)"
  GATE_RESULTS+=("$gate_json")
  copy_log "$(jq -r '.log' <<<"$gate_json")" "artifacts/omega/plan_big_gate_ui_type_deignore.log"

  gate_json="$(run_gate ui_type_backlog_quality bash scripts/ci/ui_type_backlog_quality_gate.sh || true)"
  GATE_RESULTS+=("$gate_json")
  copy_log "$(jq -r '.log' <<<"$gate_json")" "artifacts/omega/plan_big_gate_ui_type_backlog_quality.log"
fi

gates_json='[]'
if [[ "${#GATE_RESULTS[@]}" -gt 0 ]]; then
  joined=""
  for item in "${GATE_RESULTS[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$item"
  done
  gates_json="[$joined]"
fi

parallel_status_file="artifacts/omega/parallel_cutover_status.v1.json"
selfhost_progress_file="artifacts/omega/selfhost_compiler_progress.v1.json"
claude_contract_file="artifacts/omega/claude_operational_contract_status.v1.json"
lsp_status_file="artifacts/omega/lsp_smoke_status.v1.json"
ui_type_file="artifacts/omega/ui_type_deignore_audit.v1.json"
ui_type_bucket_file="artifacts/omega/ui_type_deignore_backlog_buckets.v1.json"
ui_type_batch_plan_file="artifacts/omega/ui_type_deignore_batch_plan.v1.json"
ui_type_backlog_quality_file="artifacts/omega/ui_type_backlog_quality.v1.json"

parallel_status="$(extract_json_field_or "$parallel_status_file" '.status' 'unknown')"
track_a_status="$(extract_json_field_or "$parallel_status_file" '.track_a_status' 'unknown')"
track_b_status="$(extract_json_field_or "$parallel_status_file" '.track_b_status' 'unknown')"
track_b_order_status="$(extract_json_field_or "$parallel_status_file" '.track_b_order_status' 'unknown')"

selfhost_status="$(extract_json_field_or "$selfhost_progress_file" '.status' 'unknown')"
claude_contract_status="$(extract_json_field_or "$claude_contract_file" '.status' 'unknown')"
lsp_smoke_status="$(extract_json_field_or "$lsp_status_file" '.status' 'unknown')"

ui_ignored="$(extract_json_field_or "$ui_type_file" '.totals.ignored_files' '0')"
ui_ready="$(extract_json_field_or "$ui_type_file" '.totals.ready_count' '0')"
ui_needs_fix="$(extract_json_field_or "$ui_type_file" '.totals.needs_fix_count' '0')"
ui_still_blocked="$(extract_json_field_or "$ui_type_file" '.totals.still_blocked_count' '0')"
ui_reclassify_candidates="$(extract_json_field_or "$ui_type_batch_plan_file" '.totals.reclassify_candidates' '0')"
ui_backlog_quality_status="$(extract_json_field_or "$ui_type_backlog_quality_file" '.status' 'unknown')"

git_dirty_count="$(git status --porcelain | wc -l | tr -d ' ')"

overall="pass"
for s in "$parallel_status" "$track_a_status" "$track_b_status" "$track_b_order_status" "$claude_contract_status" "$lsp_smoke_status"; do
  if [[ "$s" != "pass" ]]; then
    overall="attention"
  fi
done
if [[ "$ui_backlog_quality_status" != "pass" ]]; then
  overall="attention"
fi
if [[ "$ui_ready" == "0" ]]; then
  overall="attention"
fi
if [[ "$RUN_GATES" -eq 1 ]]; then
  if ! jq -e 'all(.status == "pass")' <<<"$gates_json" >/dev/null 2>&1; then
    overall="attention"
  fi
fi

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -cn \
  --arg generated_at_utc "$generated_at_utc" \
  --arg overall "$overall" \
  --arg parallel_status_file "$parallel_status_file" \
  --arg parallel_status "$parallel_status" \
  --arg track_a_status "$track_a_status" \
  --arg track_b_status "$track_b_status" \
  --arg track_b_order_status "$track_b_order_status" \
  --arg selfhost_progress_file "$selfhost_progress_file" \
  --arg selfhost_status "$selfhost_status" \
  --arg claude_contract_file "$claude_contract_file" \
  --arg claude_contract_status "$claude_contract_status" \
  --arg lsp_status_file "$lsp_status_file" \
  --arg lsp_smoke_status "$lsp_smoke_status" \
  --arg ui_type_file "$ui_type_file" \
  --arg ui_type_bucket_file "$ui_type_bucket_file" \
  --arg ui_type_batch_plan_file "$ui_type_batch_plan_file" \
  --arg ui_type_backlog_quality_file "$ui_type_backlog_quality_file" \
  --argjson ui_ignored "$ui_ignored" \
  --argjson ui_ready "$ui_ready" \
  --argjson ui_needs_fix "$ui_needs_fix" \
  --argjson ui_still_blocked "$ui_still_blocked" \
  --argjson ui_reclassify_candidates "$ui_reclassify_candidates" \
  --arg ui_backlog_quality_status "$ui_backlog_quality_status" \
  --argjson run_gates "$RUN_GATES" \
  --argjson gates "$gates_json" \
  --argjson git_dirty_count "$git_dirty_count" \
  '{
    schema: "sounio.plan.big.status-board.v1",
    generated_at_utc: $generated_at_utc,
    run_gates: ($run_gates == 1),
    overall: $overall,
    summary: {
      parallel_cutover_status: $parallel_status,
      track_a_status: $track_a_status,
      track_b_status: $track_b_status,
      track_b_order_status: $track_b_order_status,
      selfhost_progress_status: $selfhost_status,
      claude_operational_contract_status: $claude_contract_status,
      lsp_smoke_status: $lsp_smoke_status,
      ui_type: {
        ignored_files: $ui_ignored,
        ready_count: $ui_ready,
        needs_fix_count: $ui_needs_fix,
        still_blocked_count: $ui_still_blocked,
        reclassify_candidates: $ui_reclassify_candidates,
        backlog_quality_status: $ui_backlog_quality_status
      }
    },
    sources: {
      parallel_cutover_status_file: $parallel_status_file,
      selfhost_progress_file: $selfhost_progress_file,
      claude_operational_contract_file: $claude_contract_file,
      lsp_smoke_status_file: $lsp_status_file,
      ui_type_audit_file: $ui_type_file,
      ui_type_bucket_file: $ui_type_bucket_file,
      ui_type_batch_plan_file: $ui_type_batch_plan_file,
      ui_type_backlog_quality_file: $ui_type_backlog_quality_file
    },
    gates: $gates,
    workspace: {
      git_dirty_count: $git_dirty_count
    }
  }' > "$OUT_JSON"

{
  echo "# Plan BIG Status Board"
  echo
  echo "- generated_at_utc: $(jq -r '.generated_at_utc' "$OUT_JSON")"
  echo "- overall: $(jq -r '.overall' "$OUT_JSON")"
  echo "- run_gates: $(jq -r '.run_gates' "$OUT_JSON")"
  echo
  echo "## Core Status"
  echo "- parallel_cutover: $(jq -r '.summary.parallel_cutover_status' "$OUT_JSON")"
  echo "- track_a: $(jq -r '.summary.track_a_status' "$OUT_JSON")"
  echo "- track_b: $(jq -r '.summary.track_b_status' "$OUT_JSON")"
  echo "- track_b_order: $(jq -r '.summary.track_b_order_status' "$OUT_JSON")"
  echo "- claude_operational_contract: $(jq -r '.summary.claude_operational_contract_status' "$OUT_JSON")"
  echo "- lsp_smoke: $(jq -r '.summary.lsp_smoke_status' "$OUT_JSON")"
  echo
  echo "## UI Type Backlog"
  echo "- ignored_files: $(jq -r '.summary.ui_type.ignored_files' "$OUT_JSON")"
  echo "- ready_count: $(jq -r '.summary.ui_type.ready_count' "$OUT_JSON")"
  echo "- needs_fix_count: $(jq -r '.summary.ui_type.needs_fix_count' "$OUT_JSON")"
  echo "- still_blocked_count: $(jq -r '.summary.ui_type.still_blocked_count' "$OUT_JSON")"
  echo "- reclassify_candidates: $(jq -r '.summary.ui_type.reclassify_candidates' "$OUT_JSON")"
  echo "- backlog_quality_status: $(jq -r '.summary.ui_type.backlog_quality_status' "$OUT_JSON")"
  echo
  echo "## Gate Runs"
  if jq -e '.gates | length > 0' "$OUT_JSON" >/dev/null; then
    jq -r '.gates[] | "- " + .name + ": " + .status' "$OUT_JSON"
  else
    echo "- (not executed in this run)"
  fi
  echo
  echo "## Workspace"
  echo "- git_dirty_count: $(jq -r '.workspace.git_dirty_count' "$OUT_JSON")"
} > "$OUT_MD"

echo "PLAN_BIG_STATUS_BOARD_PASS"
echo "JSON: $OUT_JSON"
echo "MD:   $OUT_MD"
