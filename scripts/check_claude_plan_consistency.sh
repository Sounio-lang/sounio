#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "error: required file missing: $p" >&2
    exit 1
  fi
}

assert_order_in_file() {
  local file="$1"
  local a b c d e
  a="$(rg -n "data_structures\\.md" "$file" | head -n1 | cut -d: -f1 || true)"
  b="$(rg -n "gpu_ir_expansion\\.md" "$file" | head -n1 | cut -d: -f1 || true)"
  c="$(rg -n "hlir_lowering\\.md" "$file" | head -n1 | cut -d: -f1 || true)"
  d="$(rg -n "metal_msl_codegen\\.md" "$file" | head -n1 | cut -d: -f1 || true)"
  e="$(rg -n "ptx_regalloc_expansion\\.md" "$file" | head -n1 | cut -d: -f1 || true)"

  if [[ -z "$a" || -z "$b" || -z "$c" || -z "$d" || -z "$e" ]]; then
    echo "error: could not locate all Track B stages in $file" >&2
    exit 1
  fi

  if ! [[ "$a" -lt "$b" && "$b" -lt "$c" && "$c" -lt "$d" && "$d" -lt "$e" ]]; then
    echo "error: canonical Track B order violated in $file" >&2
    exit 1
  fi
}

assert_historical_banner() {
  local file="$1"
  if ! rg -q "^# Historical Context Only \(Non-Canonical\)" "$file"; then
    echo "error: missing historical-context banner in $file" >&2
    exit 1
  fi
}

require_file "PLAN_ORIGINAL.md"
require_file ".claude/PLAN_CANONICAL_EXECUTION.md"

assert_order_in_file "PLAN_ORIGINAL.md"
assert_order_in_file ".claude/PLAN_CANONICAL_EXECUTION.md"

if [[ -f ".claude/OPERATIONAL_CANONICAL_INDEX.md" ]]; then
  assert_order_in_file ".claude/OPERATIONAL_CANONICAL_INDEX.md"
fi

if [[ -f ".claude/plan.md" ]]; then
  assert_historical_banner ".claude/plan.md"
fi
if [[ -f ".claude/pending.md" ]]; then
  assert_historical_banner ".claude/pending.md"
fi
if [[ -f ".claude/session-context.md" ]]; then
  assert_historical_banner ".claude/session-context.md"
fi

echo "CLAUDE_PLAN_CONSISTENCY_PASS"
