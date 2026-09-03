#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FILE=".claude/PROMPT_EXECUTION_CONTRACT.md"
WINDOW_FILE=".claude/check_sio_integration_window.v1.json"

if [[ ! -f "$FILE" ]]; then
  echo "error: required file missing: $FILE" >&2
  exit 1
fi
if [[ ! -f "$WINDOW_FILE" ]]; then
  echo "error: required file missing: $WINDOW_FILE" >&2
  exit 1
fi

assert_literal() {
  local literal="$1"
  if command -v rg >/dev/null 2>&1; then
    if ! rg -F -q "$literal" "$FILE"; then
      echo "error: missing required literal in $FILE: $literal" >&2
      exit 1
    fi
  else
    if ! grep -F -q -- "$literal" "$FILE"; then
      echo "error: missing required literal in $FILE: $literal" >&2
      exit 1
    fi
  fi
}

# Matrix schema
assert_literal "| Prompt | Owner model | Phase | Depends on | Target files | Required gates | Evidence artifacts |"

# Required prompt rows
assert_literal '`codex_epistemic_expansion.md`'
assert_literal '`codex_borrow_integration.md`'
assert_literal '`kimi_diagnostic_hardening.md`'
assert_literal '`codex_closure_effects.md`'
assert_literal '`glm_package_manager.md`'
assert_literal '`minimax_error_messages.md`'
assert_literal '`glm_test_deignore.md`'
assert_literal '`deepseek_epistemic_algebra.md`'
assert_literal '`kimi_lsp_server.md`'

# Policy blocks
assert_literal "## Merge Policy (mandatory sequence per lane)"
assert_literal "1. feature-complete"
assert_literal "2. gate-complete"
assert_literal "3. artifact refresh"
assert_literal "## Conflict Protocol"
assert_literal "self-hosted/check/check.sio"
assert_literal '## Active Serialized `check.sio` Window'
assert_literal ".claude/check_sio_integration_window.v1.json"
assert_literal "scripts/check_check_sio_integration_window.sh"
assert_literal "## Done Criteria Per Prompt"

echo "PROMPT_EXECUTION_CONTRACT_PASS"
