#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CODEX_HOOKS="$ROOT_DIR/.codex/hooks.json"
CLAUDE_HOOKS="$ROOT_DIR/.claude/settings.json"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-hook-config.XXXXXX")"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-hook-config-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

native_count() {
  local file="$1" agent="$2"
  grep -Fc "bin/sounio-loom\\\" agent-hook --agent $agent" "$file" || true
}

validate_no_legacy_bridge() {
  local file="$1"
  ! grep -Eiq 'python|pypy|rustc|cargo|sounio_coord_agent_hook\.py' "$file"
}

run_native_roundtrip() {
  local agent="$1"
  local session_id="hook-config-$agent-$$"
  local receipt="$TEST_ROOT/$agent-receipt.tsv"
  export SOUNIO_COORD_DIR="$TEST_ROOT/coord"
  export SOUNIO_LOOM_HOOK_TEST_MODE=1
  export SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$receipt"
  unset SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME
  unset TMUX TMUX_PANE SOUNIO_AGENTD_SOCKET SOUNIO_AGENTD_TOKEN_FILE
  unset SOUNIO_AGENTD_WORKTREE

  printf '{"hook_event_name":"SessionStart","session_id":"%s","cwd":"%s"}\n' \
    "$session_id" "$ROOT_DIR" |
    "$ROOT_DIR/bin/sounio-loom" agent-hook --agent "$agent" >/dev/null
  printf '{"hook_event_name":"SessionEnd","session_id":"%s","cwd":"%s"}\n' \
    "$session_id" "$ROOT_DIR" |
    "$ROOT_DIR/bin/sounio-loom" agent-hook --agent "$agent" >/dev/null

  grep -Fq 'producing_language=OCaml' "$receipt" ||
    fail "$agent native round trip omitted its operational language"
  grep -Fq 'semantic_authority_language=Sounio' "$receipt" ||
    fail "$agent native round trip omitted Sounio authority"
}

[[ "$(native_count "$CODEX_HOOKS" codex)" -eq 5 ]] ||
  fail "Codex lifecycle/write hooks are not all native"
[[ "$(native_count "$CLAUDE_HOOKS" claude)" -eq 6 ]] ||
  fail "Claude lifecycle/write hooks are not all native"
validate_no_legacy_bridge "$CODEX_HOOKS" ||
  fail "Codex hook config retains a prohibited Python/Rust bridge"
validate_no_legacy_bridge "$CLAUDE_HOOKS" ||
  fail "Claude hook config retains a prohibited Python/Rust bridge"

grep -Fq '"matcher": "apply_patch|Edit|Write"' "$CODEX_HOOKS" ||
  fail "Codex structured-write matcher drifted"
grep -Fq '"matcher": "Write|Edit|MultiEdit|NotebookEdit"' "$CLAUDE_HOOKS" ||
  fail "Claude structured-write matcher drifted"
! grep -Eq '"matcher": "[^"]*(Bash|Exec)' "$CODEX_HOOKS" ||
  fail "Codex Bash/Exec was attached before capability-custody and outcome gates"
! grep -Eq '"matcher": "[^"]*(Bash|Exec)' "$CLAUDE_HOOKS" ||
  fail "Claude Bash/Exec was attached before capability-custody and outcome gates"

runtime_info="$($ROOT_DIR/bin/sounio-loom runtime-info)"
grep -Fq 'selection=shared' <<< "$runtime_info" ||
  fail "hook launcher did not select the shared runtime"
grep -Fq 'runtime_version=2026.08.27.35' <<< "$runtime_info" ||
  fail "hook launcher selected the wrong runtime version"

run_native_roundtrip codex
run_native_roundtrip claude

cp "$CODEX_HOOKS" "$TEST_ROOT/codex-sabotaged.json"
sed -i 's#bin/sounio-loom\\" agent-hook#scripts/dev/sounio_coord_agent_hook.py\\"#' \
  "$TEST_ROOT/codex-sabotaged.json"
if validate_no_legacy_bridge "$TEST_ROOT/codex-sabotaged.json"; then
  fail "causal sabotage did not trip the native-bridge rule"
fi

printf '%s\n' \
  'sounio-loom-hook-config-selftest: PASS codex_hooks=5 claude_hooks=6 roundtrip=codex+claude bridge=OCaml semantic_authority=Sounio python=absent rust=absent exec_policy=frozen-v2 exec_attachment=blocked-same-uid-custody-and-outcome sabotage=refused runtime=2026.08.27.35'
