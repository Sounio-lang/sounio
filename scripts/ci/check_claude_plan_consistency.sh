#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
CANONICAL_DOC="${SOUNIO_CLAUDE_CANONICAL_DOC:-.claude/PLAN_CANONICAL_EXECUTION.md}"
PLAN_ORIGINAL_DOC="${SOUNIO_CLAUDE_PLAN_ORIGINAL_DOC:-docs/archived/PLAN_ORIGINAL.md}"

search_fixed() {
  local file="$1"
  local literal="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q "$literal" "$file"
  else
    grep -F -q -- "$literal" "$file"
  fi
}

search_regex() {
  local file="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q "$pattern" "$file"
  else
    grep -E -q -- "$pattern" "$file"
  fi
}

first_match_line() {
  local file="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n "$pattern" "$file" | head -n1 | cut -d: -f1 || true
  else
    grep -n -E -- "$pattern" "$file" | head -n1 | cut -d: -f1 || true
  fi
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "error: required file missing: $p" >&2
    exit 1
  fi
}

assert_contains_literal() {
  local file="$1"
  local literal="$2"
  if ! search_fixed "$file" "$literal"; then
    echo "error: missing required literal in $file: $literal" >&2
    exit 1
  fi
}

assert_order_in_file() {
  local file="$1"
  local a b c d e
  a="$(first_match_line "$file" "data_structures\\.md")"
  b="$(first_match_line "$file" "gpu_ir_expansion\\.md")"
  c="$(first_match_line "$file" "hlir_lowering\\.md")"
  d="$(first_match_line "$file" "metal_msl_codegen\\.md")"
  e="$(first_match_line "$file" "ptx_regalloc_expansion\\.md")"

  if [[ -z "$a" || -z "$b" || -z "$c" || -z "$d" || -z "$e" ]]; then
    echo "error: could not locate all Track B stages in $file" >&2
    exit 1
  fi

  if ! [[ "$a" -lt "$b" && "$b" -lt "$c" && "$c" -lt "$d" && "$d" -lt "$e" ]]; then
    echo "error: canonical Track B order violated in $file" >&2
    exit 1
  fi
}

assert_has_marker() {
  local file="$1"
  local marker="$2"
  if ! search_regex "$file" "$marker"; then
    echo "error: missing marker '$marker' in $file" >&2
    exit 1
  fi
}

assert_historical_redirect() {
  local file="$1"
  if ! search_regex "$file" "^# Historical Context Only \\(Non-Canonical\\)" ; then
    echo "error: missing historical-context banner in $file" >&2
    exit 1
  fi
  assert_contains_literal "$file" ".claude/PLAN_CANONICAL_EXECUTION.md"
  assert_contains_literal "$file" ".claude/OPERATIONAL_CANONICAL_INDEX.md"
  assert_contains_literal "$file" ".claude/PROMPT_EXECUTION_CONTRACT.md"
}

require_file "$PLAN_ORIGINAL_DOC"
require_file "$CANONICAL_DOC"
require_file ".claude/OPERATIONAL_CANONICAL_INDEX.md"
require_file ".claude/PROMPT_EXECUTION_CONTRACT.md"

assert_order_in_file "$PLAN_ORIGINAL_DOC"
assert_order_in_file "$CANONICAL_DOC"

assert_has_marker "$CANONICAL_DOC" "CANONICAL_PRECEDENCE_START"
assert_has_marker "$CANONICAL_DOC" "CANONICAL_PRECEDENCE_END"
assert_has_marker "$CANONICAL_DOC" "LOCKED_TRACK_B_ORDER_START"
assert_has_marker "$CANONICAL_DOC" "LOCKED_TRACK_B_ORDER_END"

assert_contains_literal "$CANONICAL_DOC" ".claude/session-context.md"
assert_contains_literal "$CANONICAL_DOC" ".claude/OPERATIONAL_CANONICAL_INDEX.md"
assert_contains_literal "$CANONICAL_DOC" ".claude/PROMPT_EXECUTION_CONTRACT.md"
assert_contains_literal "$CANONICAL_DOC" "artifacts/omega/claude_operational_contract_status.v1.json"

assert_contains_literal ".claude/OPERATIONAL_CANONICAL_INDEX.md" ".claude/PLAN_CANONICAL_EXECUTION.md"
assert_contains_literal ".claude/OPERATIONAL_CANONICAL_INDEX.md" ".claude/plan.md"
assert_contains_literal ".claude/OPERATIONAL_CANONICAL_INDEX.md" ".claude/pending.md"
assert_contains_literal ".claude/OPERATIONAL_CANONICAL_INDEX.md" ".claude/session-context.md"

assert_contains_literal ".claude/PROMPT_EXECUTION_CONTRACT.md" ".claude/PLAN_CANONICAL_EXECUTION.md"

if [[ -x "scripts/ci/check_prompt_execution_contract.sh" ]]; then
  bash scripts/ci/check_prompt_execution_contract.sh
fi

if [[ -f ".claude/plan.md" ]]; then
  assert_historical_redirect ".claude/plan.md"
fi
if [[ -f ".claude/pending.md" ]]; then
  assert_historical_redirect ".claude/pending.md"
fi
if [[ -f ".claude/session-context.md" ]]; then
  assert_historical_redirect ".claude/session-context.md"
fi

echo "CLAUDE_PLAN_CONSISTENCY_PASS"
