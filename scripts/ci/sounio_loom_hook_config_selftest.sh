#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CODEX_HOOKS="$ROOT_DIR/.codex/hooks.json"
CLAUDE_HOOKS="$ROOT_DIR/.claude/settings.json"
CURSOR_HOOKS="$ROOT_DIR/.cursor/hooks.json"
GROK_HOOKS="$ROOT_DIR/.grok/hooks/loom-native.json"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook.config.XXXXXX")"
LOOM_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"

cleanup() {
  local supervisor_state="$TEST_ROOT/coord/obligation-supervisor.state"
  local supervisor_pid=""
  local supervisor_argv=""

  if [[ -r "$supervisor_state" ]]; then
    supervisor_pid="$(sed -n 's/^pid=//p' "$supervisor_state" | head -1)"
  fi
  if [[ "$supervisor_pid" =~ ^[1-9][0-9]*$ && -r "/proc/$supervisor_pid/cmdline" ]]; then
    supervisor_argv="$(tr '\0' '\n' < "/proc/$supervisor_pid/cmdline")"
    if grep -Fxq 'obligation-supervise' <<< "$supervisor_argv" &&
       grep -Fxq "$TEST_ROOT/coord" <<< "$supervisor_argv"; then
      kill "$supervisor_pid"
      for _ in {1..100}; do
        [[ ! -e "/proc/$supervisor_pid" ]] && break
        sleep 0.01
      done
      [[ ! -e "/proc/$supervisor_pid" ]] || kill -KILL "$supervisor_pid"
    fi
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-hook-config-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

native_count() {
  local file="$1" agent="$2"
  grep -Fc "bin/sounio-loom-runtime\\\" agent-hook --agent $agent" "$file" || true
}

validate_no_legacy_bridge() {
  local file="$1"
  ! grep -Eiq 'python|pypy|rustc|cargo|sounio_coord_agent_hook\.py' "$file"
}

run_native_roundtrip() {
  local agent="$1"
  local session_id="hook-config-$agent-$$"
  local receipt="$TEST_ROOT/$agent-receipt.tsv"
  local start_event end_event config capability
  export SOUNIO_COORD_DIR="$TEST_ROOT/coord"
  export SOUNIO_COORD_RUNTIME_MODE=local
  export SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1
  export SOUNIO_LOOM_HOOK_TEST_MODE=1
  export SOUNIO_LOOM_COORD_AUTO=0
  export SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$receipt"
  export SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$ROOT_DIR"
  export SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT="$ROOT_DIR"
  export SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-language-authority-runtime"
  export SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-cutover"
  unset TMUX TMUX_PANE SOUNIO_AGENTD_SOCKET SOUNIO_AGENTD_TOKEN_FILE
  unset SOUNIO_AGENTD_WORKTREE

  case "$agent" in
    codex)
      config="$CODEX_HOOKS"
      printf -v start_event '{"hook_event_name":"SessionStart","session_id":"%s","cwd":"%s"}' "$session_id" "$ROOT_DIR"
      printf -v end_event '{"hook_event_name":"Stop","session_id":"%s","cwd":"%s"}' "$session_id" "$ROOT_DIR"
      ;;
    claude)
      config="$CLAUDE_HOOKS"
      printf -v start_event '{"hook_event_name":"SessionStart","session_id":"%s","cwd":"%s"}' "$session_id" "$ROOT_DIR"
      printf -v end_event '{"hook_event_name":"SessionEnd","session_id":"%s","cwd":"%s"}' "$session_id" "$ROOT_DIR"
      ;;
    cursor)
      config="$CURSOR_HOOKS"
      printf -v start_event '{"hook_event_name":"sessionStart","session_id":"%s","cwd":"%s"}' "$session_id" "$ROOT_DIR"
      printf -v end_event '{"hook_event_name":"sessionEnd","session_id":"%s","cwd":"%s"}' "$session_id" "$ROOT_DIR"
      ;;
    grok)
      config="$GROK_HOOKS"
      printf -v start_event '{"hookEventName":"session_start","sessionId":"%s","cwd":"%s","workspaceRoot":"%s"}' "$session_id" "$ROOT_DIR" "$ROOT_DIR"
      printf -v end_event '{"hookEventName":"session_end","sessionId":"%s","cwd":"%s","workspaceRoot":"%s"}' "$session_id" "$ROOT_DIR" "$ROOT_DIR"
      ;;
    *) fail "unsupported provider: $agent" ;;
  esac
  export SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$config"
  printf '%s\n' "$start_event" | "$LOOM_RUNTIME" agent-hook --agent "$agent" >/dev/null
  printf '%s\n' "$end_event" | "$LOOM_RUNTIME" agent-hook --agent "$agent" >/dev/null

  grep -Fq 'producing_language=OCaml' "$receipt" ||
    fail "$agent native round trip omitted its operational language"
  grep -Fq 'semantic_authority_language=Sounio' "$receipt" ||
    fail "$agent native round trip omitted Sounio authority"
  if [[ "$agent" != codex ]]; then
    capability="$TEST_ROOT/coord/hook-capabilities/${agent}--session-${session_id:0:24}.capability"
    [[ ! -e "$capability" ]] ||
      fail "$agent SessionEnd left its local native capability active"
  fi
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
[[ -x "$LOOM_RUNTIME" ]] || fail "source-fresh Loom runtime was not built"

[[ "$(native_count "$CODEX_HOOKS" codex)" -eq 5 ]] ||
  fail "Codex lifecycle/write hooks are not all native"
[[ "$(native_count "$CLAUDE_HOOKS" claude)" -eq 6 ]] ||
  fail "Claude lifecycle/write hooks are not all native"
[[ "$(native_count "$CURSOR_HOOKS" cursor)" -eq 6 ]] ||
  fail "Cursor lifecycle/write hooks are not all native"
[[ "$(native_count "$GROK_HOOKS" grok)" -eq 6 ]] ||
  fail "Grok lifecycle/write hooks are not all native"
[[ "$(grep -Fc 'SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT=\"$runtime_dir/policy/language-authority\"' "$CODEX_HOOKS")" -eq 5 ]] ||
  fail "Codex hooks do not pin the frozen authority capsule to the selected runtime"
[[ "$(grep -Fc 'SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT=\"$runtime_dir/policy/language-authority\"' "$CLAUDE_HOOKS")" -eq 6 ]] ||
  fail "Claude hooks do not pin the frozen authority capsule to the selected runtime"
[[ "$(grep -Fc 'readlink -f' "$CODEX_HOOKS")" -eq 5 ]] ||
  fail "Codex hooks do not resolve the active runtime exactly once"
[[ "$(grep -Fc 'readlink -f' "$CLAUDE_HOOKS")" -eq 6 ]] ||
  fail "Claude hooks do not resolve the active runtime exactly once"
[[ "$(grep -Fc 'exec env SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT=' "$CODEX_HOOKS")" -eq 5 ]] ||
  fail "Codex hooks do not replace the command shell with the native runtime"
[[ "$(grep -Fc 'exec env SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT=' "$CLAUDE_HOOKS")" -eq 6 ]] ||
  fail "Claude hooks do not replace the command shell with the native runtime"
validate_no_legacy_bridge "$CODEX_HOOKS" ||
  fail "Codex hook config retains a prohibited Python/Rust bridge"
validate_no_legacy_bridge "$CLAUDE_HOOKS" ||
  fail "Claude hook config retains a prohibited Python/Rust bridge"
validate_no_legacy_bridge "$CURSOR_HOOKS" ||
  fail "Cursor hook config retains a prohibited Python/Rust bridge"
validate_no_legacy_bridge "$GROK_HOOKS" ||
  fail "Grok hook config retains a prohibited Python/Rust bridge"

grep -Fq '"matcher": "apply_patch|Edit|Write"' "$CODEX_HOOKS" ||
  fail "Codex structured-write matcher drifted"
grep -Fq '"matcher": "Write|Edit|MultiEdit|NotebookEdit"' "$CLAUDE_HOOKS" ||
  fail "Claude structured-write matcher drifted"
! grep -Eq '"matcher": "[^"]*(Bash|Exec)' "$CODEX_HOOKS" ||
  fail "Codex Bash/Exec was attached before capability-custody and outcome gates"
! grep -Eq '"matcher": "[^"]*(Bash|Exec)' "$CLAUDE_HOOKS" ||
  fail "Claude Bash/Exec was attached before capability-custody and outcome gates"

runtime_info="$("$LOOM_RUNTIME" 2>&1 || true)"
runtime_selection=source-fresh
grep -Fq 'Sounio Loom 2026.08.31.0' <<< "$runtime_info" ||
  fail "source-fresh hook runtime has the wrong version"
grep -Fq 'agent-hook --agent codex|claude|cursor|grok' <<< "$runtime_info" ||
  fail "source-fresh hook runtime omits the four-provider native ingress"

run_native_roundtrip codex
run_native_roundtrip claude
run_native_roundtrip cursor
run_native_roundtrip grok

cp "$CODEX_HOOKS" "$TEST_ROOT/codex-sabotaged.json"
printf '\npython3 prohibited.py\n' >> "$TEST_ROOT/codex-sabotaged.json"
if validate_no_legacy_bridge "$TEST_ROOT/codex-sabotaged.json"; then
  fail "causal sabotage did not trip the native-bridge rule"
fi

printf '%s\n' \
  "sounio-loom-hook-config-selftest: PASS codex_hooks=5 claude_hooks=6 cursor_hooks=6 grok_hooks=6 launcher_runtime=$runtime_selection:2026.08.31.0 hook_runtime=source-fresh authority=Sounio-9045 local_roundtrip=codex+claude+cursor+grok production_wake_eligible=no bridge=OCaml semantic_authority=Sounio python=absent rust=absent exec_policy=frozen-v2 exec_attachment=blocked-general-bash-closure python_sabotage=refused"
