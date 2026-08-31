#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook.XXXXXX")"
AUTHORITY_RUNTIME="$TEST_ROOT/sounio-loom-language-authority-runtime"
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
CUTOVER_RUNTIME="$TEST_ROOT/sounio-loom-native-hook-cutover"
CUTOVER_MANIFEST="$ROOT_DIR/tools/loom/native_hook_cutover.freeze.v1"
CUTOVER_TOOLCHAIN_ROOT="$TEST_ROOT/cutover-toolchain"
COORD_DIR="$TEST_ROOT/coord"
DECISION_LOG="$TEST_ROOT/agent-hook.tsv"
SENTINEL_DIR="$TEST_ROOT/sentinel-bin"
SENTINEL_MARKER="$TEST_ROOT/prohibited-runtime-executed"
SIBLING_ROOT="$TEST_ROOT/sibling-worktree"
SIBLING_ADDED=0
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SESSION_ID="native-hook-selftest-$$"
SESSION_LANE="session-${SESSION_ID:0:24}"
TMUX_SOCKET="$TEST_ROOT/native-hook-tmux.sock"
TMUX_SESSION="native-hook-$$"
TMUX_SESSION_ID="native-tmux-selftest-$$"
TMUX_SESSION_LANE="session-${TMUX_SESSION_ID:0:24}"
TMUX_HARNESS="$TEST_ROOT/codex"
TMUX_HARNESS_SCRIPT="$TEST_ROOT/native-hook-harness.sh"
TMUX_READY="$TEST_ROOT/native-hook.ready"
TMUX_LOG="$TEST_ROOT/native-hook-tmux.log"
WRONG_CWD_SESSION="native-hook-wrong-cwd-$$"
WRONG_CWD_ID="native-tmux-wrong-cwd-$$"
WRONG_CWD_LANE="session-${WRONG_CWD_ID:0:24}"
WRONG_CWD_READY="$TEST_ROOT/native-hook-wrong-cwd.ready"
WRONG_CWD_LOG="$TEST_ROOT/native-hook-wrong-cwd.log"

cleanup() {
  tmux -S "$TMUX_SOCKET" kill-server >/dev/null 2>&1 || true
  SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
    "$ROOT_DIR/bin/sounio-coord" obligation-supervisor-stop \
    --timeout-seconds 5 >/dev/null 2>&1 || true
  if [[ "$SIBLING_ADDED" -eq 1 ]]; then
    git -C "$ROOT_DIR" worktree remove --force "$SIBLING_ROOT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-native-hook-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_hook() {
  local event="$1"
  local provider="${2:-codex}"
  set +e
  HOOK_OUTPUT="$(printf '%s\n' "$event" | "$LOOM" agent-hook --agent "$provider" 2>&1)"
  HOOK_RC=$?
  set -e
}

wait_for_file() {
  local path="$1" attempt
  for attempt in $(seq 1 300); do
    [[ -f "$path" ]] && return 0
    sleep 0.1
  done
  return 1
}

wait_for_endpoint_absence() {
  local lane="$1" attempt
  for attempt in $(seq 1 100); do
    if ! SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
      "$ROOT_DIR/bin/sounio-coord" endpoint-status \
      --agent codex --lane "$lane" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

mkdir -p "$SENTINEL_DIR"
for forbidden in python python3 pypy pypy3 cargo rustc; do
  printf '#!/usr/bin/env bash\nprintf prohibited >%q\nexit 97\n' "$SENTINEL_MARKER" \
    >"$SENTINEL_DIR/$forbidden"
  chmod 0755 "$SENTINEL_DIR/$forbidden"
done

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
frozen_executable_commit="$(sed -n 's/^sounio_executable_commit=//p' "$AUTHORITY_MANIFEST")"
[[ -n "$frozen_executable_commit" ]] || fail 'language-authority manifest omitted its executable commit'
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$frozen_executable_commit" \
  bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$TOOLCHAIN_ROOT"
SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC="$TOOLCHAIN_ROOT/bin/souc" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$AUTHORITY_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
cutover_executable_commit="$(sed -n 's/^sounio_executable_commit=//p' "$CUTOVER_MANIFEST")"
[[ -n "$cutover_executable_commit" ]] || fail 'native-hook cutover manifest omitted its executable commit'
mkdir -p "$CUTOVER_TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$cutover_executable_commit" \
  bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$CUTOVER_TOOLCHAIN_ROOT"
SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_SOUC="$CUTOVER_TOOLCHAIN_ROOT/bin/souc" \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT="$CUTOVER_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_cutover.sh" >/dev/null
git -C "$ROOT_DIR" worktree add --detach --no-checkout "$SIBLING_ROOT" HEAD >/dev/null
SIBLING_ADDED=1

export PATH="$SENTINEL_DIR:$PATH"
export SOUNIO_COORD_DIR="$COORD_DIR"
export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$AUTHORITY_RUNTIME"
export SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$CUTOVER_RUNTIME"
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$DECISION_LOG"
export SOUNIO_LOOM_HOOK_TEST_MODE=1
export SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1
export SOUNIO_COORD_NATIVE_HOOK_WAKE_SELFTEST=1
unset TMUX TMUX_PANE SOUNIO_AGENTD_SOCKET SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_WORKTREE

session_start="{\"hook_event_name\":\"SessionStart\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\"}"
run_hook "$session_start"
[[ "$HOOK_RC" -eq 0 && "$HOOK_OUTPUT" == *'Sounio coordination joined:'* ]] ||
  fail "native SessionStart failed: rc=$HOOK_RC output=$HOOK_OUTPUT"
capability_file="$COORD_DIR/hook-capabilities/codex--$SESSION_LANE.capability"
[[ -f "$capability_file" ]] || fail 'local hook omitted its native attestation'
grep -qx 'state=NATIVE_HOOK_ATTESTED' "$capability_file" ||
  fail 'local hook wrote the wrong attestation state'
grep -qx 'wake_eligible=0' "$capability_file" ||
  fail 'local hook selftest minted production wake eligibility'
grep -Eq '^producer_sha256=[0-9a-f]{64}$' "$capability_file" ||
  fail 'local hook omitted the OCaml producer hash'
grep -Eq '^coord_sha256=[0-9a-f]{64}$' "$capability_file" ||
  fail 'local hook omitted the coordination runtime hash'

capability_backup="$TEST_ROOT/native-hook-capability.backup"
cp "$capability_file" "$capability_backup"
sed -i 's/^source_sha=.*/source_sha=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$capability_file"
set +e
source_tamper_output="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" hook-capability-status \
  --agent codex --lane "$SESSION_LANE" 2>&1)"
source_tamper_rc=$?
set -e
[[ "$source_tamper_rc" -ne 0 && \
  "$source_tamper_output" == *'reason=source-binding-drift'* ]] ||
  fail "tampered capability source binding remained current: rc=$source_tamper_rc output=$source_tamper_output"
mv "$capability_backup" "$capability_file"

set +e
direct_output="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" hook-capability-register --agent codex \
  --lane "$SESSION_LANE" --session-id "$SESSION_ID" 2>&1)"
direct_rc=$?
set -e
[[ "$direct_rc" -ne 0 && \
  "$direct_output" == *'requires the matching OCaml runtime parent'* ]] ||
  fail "direct shell registration was not refused: rc=$direct_rc output=$direct_output"

exec_session="exec-shell-selftest-$$"
set +e
exec_output="$(printf '%s\n' \
  "{\"hook_event_name\":\"SessionStart\",\"session_id\":\"$exec_session\",\"cwd\":\"$ROOT_DIR\"}" | \
  env -u SOUNIO_COORD_NATIVE_HOOK_SELFTEST bash -c \
    'exec "$1" agent-hook --agent codex' _ "$LOOM" 2>&1)"
exec_rc=$?
set -e
[[ "$exec_rc" -eq 2 && \
  "$exec_output" == *'matching OCaml runtime parent'* ]] ||
  fail "exec-from-shell native mint was not refused: rc=$exec_rc output=$exec_output"
supervisor_status="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" obligation-supervisor-status)"
[[ "$supervisor_status" == *'state=live'* ]] || \
  fail "SessionStart did not ensure the native retry supervisor: $supervisor_status"

message_output="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_DURABLE_OBLIGATIONS=0 "$ROOT_DIR/bin/sounio-coord" send \
  --agent sender --lane native-hook-test --to-agent codex --to-lane "$SESSION_LANE" \
  --kind info --message 'native prompt boundary start witness')"
message_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$message_output")"
[[ -n "$message_id" ]] || fail 'native hook witness message was not persisted'
prompt_event="{\"hook_event_name\":\"UserPromptSubmit\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"prompt\":\"native hook witness\"}"
run_hook "$prompt_event"
[[ "$HOOK_RC" -eq 0 && "$HOOK_OUTPUT" == *"MESSAGE id=$message_id "* ]] ||
  fail "UserPromptSubmit did not consume the durable inbox: rc=$HOOK_RC output=$HOOK_OUTPUT"
injection_receipt="$COORD_DIR/message-injections/$message_id--codex--$SESSION_LANE.injected"
[[ -f "$injection_receipt" ]] || fail 'UserPromptSubmit omitted its injection receipt'

write_event="{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"tools/loom/native-hook-probe.txt\"}}"
run_hook "$write_event"
[[ "$HOOK_RC" -eq 0 ]] || fail "native write authorization failed: rc=$HOOK_RC output=$HOOK_OUTPUT"

patch_event="{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"apply_patch\",\"tool_input\":{\"patch\":\"*** Add File: tools/loom/native-hook-patch-probe.txt\\n+probe\\n\"}}"
run_hook "$patch_event"
[[ "$HOOK_RC" -eq 0 ]] || fail "native patch authorization failed: rc=$HOOK_RC output=$HOOK_OUTPUT"

outside_event="{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$TEST_ROOT/outside.txt\"}}"
run_hook "$outside_event"
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'write-path-outside-current-repository'* ]] ||
  fail "outside write did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

sibling_event="{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$SIBLING_ROOT/probe.txt\"}}"
run_hook "$sibling_event"
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'write-path-outside-session-worktree'* ]] ||
  fail "sibling-worktree write did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

run_hook '{"hook_event_name":'
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'invalid-json'* ]] ||
  fail "malformed event did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

run_hook '{"hook_event_name":"Stop",}'
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'trailing-object-comma'* ]] ||
  fail "trailing object comma did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

run_hook '{"hook_event_name":"Stop","extra":[1,]}'
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'trailing-array-comma'* ]] ||
  fail "trailing array comma did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

missing_path_event="{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Write\",\"tool_input\":{}}"
run_hook "$missing_path_event"
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'write-path-missing'* ]] ||
  fail "pathless write did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

duplicate_event="{\"hook_event_name\":\"Stop\",\"session_id\":\"$SESSION_ID\",\"session_id\":\"shadow\",\"cwd\":\"$ROOT_DIR\"}"
run_hook "$duplicate_event"
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'duplicate-key-session_id'* ]] ||
  fail "duplicate JSON key did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

set +e
LOG_OVERRIDE_OUTPUT="$(printf '%s\n' "$write_event" | \
  env -u SOUNIO_LOOM_HOOK_TEST_MODE \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$TEST_ROOT/unauthorized-log.tsv" \
  "$LOOM" agent-hook --agent codex 2>&1)"
LOG_OVERRIDE_RC=$?
set -e
[[ "$LOG_OVERRIDE_RC" -eq 2 && \
  "$LOG_OVERRIDE_OUTPUT" == *'decision-log-override-requires-test-mode'* ]] ||
  fail "production log override did not fail closed: rc=$LOG_OVERRIDE_RC output=$LOG_OVERRIDE_OUTPUT"
[[ ! -e "$TEST_ROOT/unauthorized-log.tsv" ]] || fail "unauthorized decision log was created"

set +e
MISSING_OUTPUT="$(printf '%s\n' "$write_event" | \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_MANIFEST="$TEST_ROOT/missing.freeze" \
  "$LOOM" agent-hook --agent codex 2>&1)"
MISSING_RC=$?
set -e
[[ "$MISSING_RC" -eq 2 && "$MISSING_OUTPUT" == *'Sounio-authority-policy-missing'* ]] ||
  fail "missing policy did not fail closed: rc=$MISSING_RC output=$MISSING_OUTPUT"

cp "$ROOT_DIR/tools/loom/language_authority.freeze.v1" "$TEST_ROOT/tampered.freeze"
printf '\n' >>"$TEST_ROOT/tampered.freeze"
set +e
TAMPER_OUTPUT="$(printf '%s\n' "$write_event" | \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_MANIFEST="$TEST_ROOT/tampered.freeze" \
  "$LOOM" agent-hook --agent codex 2>&1)"
TAMPER_RC=$?
set -e
[[ "$TAMPER_RC" -eq 2 && "$TAMPER_OUTPUT" == *'Sounio-authority-policy-hash-mismatch'* ]] ||
  fail "tampered policy did not fail closed: rc=$TAMPER_RC output=$TAMPER_OUTPUT"

cp "$AUTHORITY_RUNTIME" "$TEST_ROOT/tampered-authority"
printf x >>"$TEST_ROOT/tampered-authority"
chmod 0755 "$TEST_ROOT/tampered-authority"
set +e
BINARY_OUTPUT="$(printf '%s\n' "$write_event" | \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$TEST_ROOT/tampered-authority" \
  "$LOOM" agent-hook --agent codex 2>&1)"
BINARY_RC=$?
set -e
[[ "$BINARY_RC" -eq 2 && "$BINARY_OUTPUT" == *'Sounio-authority-runtime-hash-mismatch'* ]] ||
  fail "tampered Sounio runtime did not fail closed: rc=$BINARY_RC output=$BINARY_OUTPUT"

post_event="{\"hook_event_name\":\"PostToolUse\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"tools/loom/native-hook-probe.txt\"}}"
run_hook "$post_event"
[[ "$HOOK_RC" -eq 0 ]] || fail "native PostToolUse failed: rc=$HOOK_RC output=$HOOK_OUTPUT"

session_end="{\"hook_event_name\":\"SessionEnd\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\"}"
run_hook "$session_end"
[[ "$HOOK_RC" -eq 0 ]] || fail "native SessionEnd failed: rc=$HOOK_RC output=$HOOK_OUTPUT"
if SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" hook-capability-status \
  --agent codex --lane "$SESSION_LANE" >/dev/null 2>&1; then
  fail 'SessionEnd left a native hook capability behind'
fi

[[ -f "$DECISION_LOG" ]] || fail "native hook omitted its decision log"
grep -Fq $'decision=ALLOW\treason=SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045' \
  "$DECISION_LOG" || fail "decision log omitted the Sounio ALLOW"
grep -Fq $'decision=DENY\treason=write-path-outside-current-repository' "$DECISION_LOG" ||
  fail "decision log omitted the outside-write DENY"
grep -Fq $'decision=DENY\treason=write-path-outside-session-worktree' "$DECISION_LOG" ||
  fail "decision log omitted the sibling-worktree DENY"
grep -Fq $'decision=DENY\treason=Sounio-authority-policy-missing' "$DECISION_LOG" ||
  fail "decision log omitted the missing-policy DENY"
grep -Fq $'decision=DENY\treason=Sounio-authority-policy-hash-mismatch' "$DECISION_LOG" ||
  fail "decision log omitted the policy-tamper DENY"
grep -Fq $'decision=DENY\treason=Sounio-authority-runtime-hash-mismatch' "$DECISION_LOG" ||
  fail "decision log omitted the runtime-tamper DENY"
grep -Fq $'decision=DENY\treason=write-path-missing' "$DECISION_LOG" ||
  fail "decision log omitted the pathless-write DENY"
grep -Fq $'sounio_source_sha256=545b0ae24fa78344aa96186eacaff4f9dc24ed7155adbed758cb4c85d1b3cd82\tsemantics_sha256=27c5fd758d161026c5c41d0cd0be0f1aa90bd4e3f4287da3c60fb748d1334882\tproducing_language=OCaml\tlanguage_role=OPERATIONAL_REALIZATION\tsemantic_authority_language=Sounio\tsemantic_authority_role=SEMANTIC_AUTHORITY\tsemantic_authority_origin=worktree' \
  "$DECISION_LOG" || fail "decision receipt omitted the authority chain"
grep -Fq $'\ttoolchain=OCaml ' "$DECISION_LOG" ||
  fail "decision receipt omitted the toolchain"
grep -Fq $'\thardware=os_type=' "$DECISION_LOG" ||
  fail "decision receipt omitted the hardware"
grep -Fq $'\tcommand=sounio-loom agent-hook --agent codex event=SessionStart tool=none\tcommand_sha256=' \
  "$DECISION_LOG" || fail "decision receipt omitted the command"
grep -Fq $'\tparent_authority_result=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$DECISION_LOG" || fail "decision receipt omitted the parent authority result"
grep -Fq $'\tresult=SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045' \
  "$DECISION_LOG" || fail "decision receipt omitted the result"

cursor_event="{\"hook_event_name\":\"sessionStart\",\"session_id\":\"cursor-native-hook-$$\",\"workspace_roots\":[\"$ROOT_DIR\"]}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_event" cursor
[[ "$HOOK_RC" -eq 0 ]] || fail "Cursor native dialect was refused: rc=$HOOK_RC output=$HOOK_OUTPUT"
cursor_legacy_cwd_event="{\"hook_event_name\":\"sessionStart\",\"session_id\":\"cursor-native-cwd-hook-$$\",\"cwd\":\"$ROOT_DIR\"}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_legacy_cwd_event" cursor
[[ "$HOOK_RC" -eq 0 ]] || fail "Cursor cwd dialect was refused: rc=$HOOK_RC output=$HOOK_OUTPUT"
grok_event="{\"hookEventName\":\"session_start\",\"sessionId\":\"grok-native-hook-$$\",\"cwd\":\"$ROOT_DIR\",\"workspaceRoot\":\"$ROOT_DIR\"}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$grok_event" grok
[[ "$HOOK_RC" -eq 0 ]] || fail "Grok native dialect was refused: rc=$HOOK_RC output=$HOOK_OUTPUT"
grep -Fq $'provider=cursor\tdialect=cursor-camel' "$DECISION_LOG" ||
  fail 'decision receipt omitted the Cursor provider binding'
grep -Fq $'provider=grok\tdialect=grok-camel' "$DECISION_LOG" ||
  fail 'decision receipt omitted the Grok provider binding'

cursor_wrong_dialect="{\"hookEventName\":\"sessionStart\",\"sessionId\":\"cursor-wrong-$$\",\"cwd\":\"$ROOT_DIR\"}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_wrong_dialect" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'provider-hook-dialect-mismatch:field=hookEventName'* ]] ||
  fail "Cursor accepted the Grok field dialect: rc=$HOOK_RC output=$HOOK_OUTPUT"
cursor_empty_roots="{\"hook_event_name\":\"sessionStart\",\"session_id\":\"cursor-empty-roots-$$\",\"workspace_roots\":[]}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_empty_roots" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'hook-workspace-roots-empty'* ]] ||
  fail "Cursor accepted an empty workspace root inventory: rc=$HOOK_RC output=$HOOK_OUTPUT"
cursor_ambiguous_roots="{\"hook_event_name\":\"sessionStart\",\"session_id\":\"cursor-ambiguous-roots-$$\",\"workspace_roots\":[\"$ROOT_DIR\",\"$SIBLING_ROOT\"]}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_ambiguous_roots" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'hook-workspace-roots-ambiguous'* ]] ||
  fail "Cursor accepted ambiguous workspace roots: rc=$HOOK_RC output=$HOOK_OUTPUT"
cursor_invalid_roots="{\"hook_event_name\":\"sessionStart\",\"session_id\":\"cursor-invalid-roots-$$\",\"workspace_roots\":[7]}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_invalid_roots" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'invalid-json:workspace_roots-items-must-be-strings'* ]] ||
  fail "Cursor accepted a non-string workspace root: rc=$HOOK_RC output=$HOOK_OUTPUT"
cursor_conflicting_root="{\"hook_event_name\":\"sessionStart\",\"session_id\":\"cursor-conflicting-root-$$\",\"cwd\":\"$ROOT_DIR\",\"workspace_roots\":[\"$SIBLING_ROOT\"]}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$cursor_conflicting_root" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'hook-workspace-root-conflict'* ]] ||
  fail "Cursor accepted conflicting workspace roots: rc=$HOOK_RC output=$HOOK_OUTPUT"
grok_wrong_dialect="{\"hook_event_name\":\"SessionStart\",\"session_id\":\"grok-wrong-$$\",\"cwd\":\"$ROOT_DIR\"}"
SOUNIO_LOOM_COORD_AUTO=0 run_hook "$grok_wrong_dialect" grok
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'provider-hook-dialect-mismatch:field=hook_event_name'* ]] ||
  fail "Grok accepted the snake field dialect: rc=$HOOK_RC output=$HOOK_OUTPUT"

SOUNIO_LOOM_COORD_AUTO=0 SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$TEST_ROOT/missing-provider.json" \
  run_hook "$cursor_event" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'native-hook-provider-config-missing'* ]] ||
  fail "missing provider config did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"
printf '{"hooks":{"sessionStart":[]}}\n' >"$TEST_ROOT/tampered-provider.json"
SOUNIO_LOOM_COORD_AUTO=0 SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$TEST_ROOT/tampered-provider.json" \
  run_hook "$cursor_event" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'native-hook-provider-config-not-direct:cursor'* ]] ||
  fail "non-native provider config did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

SOUNIO_LOOM_COORD_AUTO=0 \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_MANIFEST="$TEST_ROOT/missing-cutover.freeze" \
  run_hook "$cursor_event" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'Sounio-native-hook-cutover-policy-missing'* ]] ||
  fail "missing cutover policy did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"
cp "$CUTOVER_MANIFEST" "$TEST_ROOT/tampered-cutover.freeze"
printf '\n' >>"$TEST_ROOT/tampered-cutover.freeze"
SOUNIO_LOOM_COORD_AUTO=0 \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_MANIFEST="$TEST_ROOT/tampered-cutover.freeze" \
  run_hook "$cursor_event" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'Sounio-native-hook-cutover-policy-hash-mismatch'* ]] ||
  fail "tampered cutover policy did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"
cp "$CUTOVER_RUNTIME" "$TEST_ROOT/tampered-cutover-runtime"
printf x >>"$TEST_ROOT/tampered-cutover-runtime"
chmod 0755 "$TEST_ROOT/tampered-cutover-runtime"
SOUNIO_LOOM_COORD_AUTO=0 \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$TEST_ROOT/tampered-cutover-runtime" \
  run_hook "$cursor_event" cursor
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'Sounio-native-hook-cutover-runtime-hash-mismatch'* ]] ||
  fail "tampered cutover runtime did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

[[ ! -e "$SENTINEL_MARKER" ]] || fail "Python or Rust was executed by the native hook"

status="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" status)"
[[ "$status" != *"session-$SESSION_ID"* ]] || fail "SessionEnd left an active native-hook claim"

command -v tmux >/dev/null 2>&1 || fail 'tmux is required for the native endpoint fixture'
cp "$(command -v bash)" "$TMUX_HARNESS"
chmod 0755 "$TMUX_HARNESS"
cat >"$TMUX_HARNESS_SCRIPT" <<'HARNESS'
#!/usr/bin/env bash
set -euo pipefail

loom="$1"
root="$2"
session_id="$3"
ready="$4"
log="$5"
session_start="{\"hook_event_name\":\"SessionStart\",\"session_id\":\"$session_id\",\"cwd\":\"$root\"}"
printf '%s\n' "$session_start" | "$loom" agent-hook --agent codex >>"$log" 2>&1
: >"$ready"
while IFS= read -r input; do
  if [[ "$input" == __END__ ]]; then
    event="{\"hook_event_name\":\"SessionEnd\",\"session_id\":\"$session_id\",\"cwd\":\"$root\"}"
    printf '%s\n' "$event" | "$loom" agent-hook --agent codex >>"$log" 2>&1
    break
  fi
  event="{\"hook_event_name\":\"UserPromptSubmit\",\"session_id\":\"$session_id\",\"cwd\":\"$root\"}"
  printf '%s\n' "$event" | "$loom" agent-hook --agent codex >>"$log" 2>&1
done
HARNESS
chmod 0755 "$TMUX_HARNESS_SCRIPT"

mkdir -p "$TEST_ROOT/not-a-repository"
tmux -S "$TMUX_SOCKET" new-session -d -s "$WRONG_CWD_SESSION" \
  -c "$TEST_ROOT/not-a-repository" \
  "$TMUX_HARNESS '$TMUX_HARNESS_SCRIPT' '$LOOM' '$ROOT_DIR' '$WRONG_CWD_ID' '$WRONG_CWD_READY' '$WRONG_CWD_LOG'"
if ! wait_for_file "$WRONG_CWD_READY"; then
  wrong_cwd_pane="$(tmux -S "$TMUX_SOCKET" list-panes -a \
    -F 'session=#{session_name} pane=#{pane_id} dead=#{pane_dead} status=#{pane_dead_status} command=#{pane_current_command}' \
    2>&1 || true)"
  wrong_cwd_screen="$(tmux -S "$TMUX_SOCKET" capture-pane -p -S -200 \
    -t "$WRONG_CWD_SESSION" 2>&1 || true)"
  wrong_cwd_pane_pid="$(tmux -S "$TMUX_SOCKET" display-message -p \
    -t "$WRONG_CWD_SESSION" '#{pane_pid}' 2>/dev/null || true)"
  wrong_cwd_tree="$(
    ps -o pid=,ppid=,stat=,wchan=,comm=,args= --ppid "$wrong_cwd_pane_pid" 2>&1 || true
    for child in $(ps -o pid= --ppid "$wrong_cwd_pane_pid" 2>/dev/null); do
      ps -o pid=,ppid=,stat=,wchan=,comm=,args= --ppid "$child" 2>&1 || true
    done
  )"
  wrong_cwd_log="$(cat "$WRONG_CWD_LOG" 2>/dev/null || true)"
  fail "wrong-cwd native harness did not start: panes=$wrong_cwd_pane screen=$wrong_cwd_screen tree=$wrong_cwd_tree log=$wrong_cwd_log"
fi
wrong_cwd_capability="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" hook-capability-status \
  --agent codex --lane "$WRONG_CWD_LANE")"
[[ "$wrong_cwd_capability" == *'state=NATIVE_HOOK_ATTESTED wake_eligible=0 '* ]] ||
  fail "wrong-cwd harness omitted its local native attestation: $wrong_cwd_capability"
if SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" endpoint-status \
  --agent codex --lane "$WRONG_CWD_LANE" >/dev/null 2>&1; then
  fail 'native hook registered a tmux pane whose current path was not the session repository'
fi
tmux -S "$TMUX_SOCKET" kill-session -t "$WRONG_CWD_SESSION"

missing_pane_id="native-tmux-missing-pane-$$"
missing_pane_lane="session-${missing_pane_id:0:24}"
missing_pane_event="{\"hook_event_name\":\"SessionStart\",\"session_id\":\"$missing_pane_id\",\"cwd\":\"$ROOT_DIR\"}"
missing_pane_output="$(printf '%s\n' "$missing_pane_event" | env \
  TMUX="$TMUX_SOCKET,1,0" TMUX_PANE='%99999' \
  "$TMUX_HARNESS" -c 'read -r event; printf "%s\n" "$event" | "$1" agent-hook --agent codex' \
  _ "$LOOM" 2>&1)"
[[ "$missing_pane_output" == *'Sounio coordination joined:'* ]] ||
  fail "missing-pane native hook did not complete lifecycle registration: $missing_pane_output"
if SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" endpoint-status \
  --agent codex --lane "$missing_pane_lane" >/dev/null 2>&1; then
  fail 'native hook registered a nonexistent tmux pane'
fi

tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" -c "$ROOT_DIR" \
  "$TMUX_HARNESS '$TMUX_HARNESS_SCRIPT' '$LOOM' '$ROOT_DIR' '$TMUX_SESSION_ID' '$TMUX_READY' '$TMUX_LOG'"
wait_for_file "$TMUX_READY" || fail 'native tmux harness did not start'
tmux_pane="$(tmux -S "$TMUX_SOCKET" display-message -p -t "$TMUX_SESSION" '#{pane_id}')"
endpoint_status="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" endpoint-status \
  --agent codex --lane "$TMUX_SESSION_LANE")"
[[ "$endpoint_status" == *" state=active "* && \
  "$endpoint_status" == *" transport=tmux address=$tmux_pane "* ]] ||
  fail "native hook omitted its verified tmux endpoint: $endpoint_status"

SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" claim --agent sender --lane native-tmux-fixture \
  --intent 'native tmux wake fixture sender' --files native-tmux-fixture.test >/dev/null
wake_output="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_DURABLE_OBLIGATIONS=0 SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS=5000 \
  "$ROOT_DIR/bin/sounio-coord" send --agent sender --lane native-tmux-fixture \
  --to-agent codex --to-lane "$TMUX_SESSION_LANE" --kind request \
  --message 'native OCaml tmux wake fixture')"
wake_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<<"$wake_output")"
[[ -n "$wake_message" ]] || fail 'native tmux wake returned no message id'
grep -q "^WAKE_STARTED message_id=$wake_message .*address=$tmux_pane .*generation=" \
  <<<"$wake_output" || fail "native hook did not confirm the tmux wake: $wake_output"
message_status="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" message-status --agent sender \
  --lane native-tmux-fixture --message "$wake_message")"
grep -q 'injected=1 .*wakes=1 wake_pending=0$' <<<"$message_status" ||
  fail "native tmux wake did not close its durable handshake: $message_status"
grep -q "MESSAGE id=$wake_message " "$TMUX_LOG" ||
  fail 'native prompt hook did not read the durable message body after the metadata wake'

tmux -S "$TMUX_SOCKET" send-keys -l -t "$tmux_pane" '__END__'
tmux -S "$TMUX_SOCKET" send-keys -t "$tmux_pane" Enter
wait_for_endpoint_absence "$TMUX_SESSION_LANE" ||
  fail 'native SessionEnd left its tmux endpoint active'
tmux -S "$TMUX_SOCKET" kill-session -t "$TMUX_SESSION" >/dev/null 2>&1 || true
SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" release --agent sender --lane native-tmux-fixture \
  --reason 'native tmux fixture complete' >/dev/null

printf '%s\n' \
  'sounio-loom-native-hook-selftest: PASS language=OCaml semantic_authority=Sounio action=9045 session=roundtrip hook_state=NATIVE_HOOK_ATTESTED production_wake_eligible=no source_binding_tamper=refused direct_shell_mint=refused exec_shell_mint=refused prompt_boundary=injected retry_supervisor=live tmux_endpoint=native tmux_wake=started missing_pane=refused wrong_cwd_pane=refused writes=authorized outside_write=refused sibling_worktree=refused pathless_write=refused malformed=refused strict_json=refused duplicate_json=refused policy_missing=refused policy_tamper=refused runtime_tamper=refused cutover_policy_missing=refused cutover_policy_tamper=refused cutover_runtime_tamper=refused log_redirect=refused providers=codex,claude,cursor,grok dialect_mismatch=refused config_missing=refused config_non_native=refused decision_receipt=complete python=not-executed rust=not-executed'
