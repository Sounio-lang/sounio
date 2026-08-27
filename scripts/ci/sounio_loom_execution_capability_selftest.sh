#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-capability.XXXXXX")"
CAPABILITY_DIR="$TEST_ROOT/capabilities"
COORD_DIR="$TEST_ROOT/coord"
LANGUAGE_RUNTIME="$TEST_ROOT/sounio-language-authority"
EXECUTION_RUNTIME="$TEST_ROOT/sounio-execution-authority"
LANGUAGE_LOG="$TEST_ROOT/language.tsv"
EXECUTION_LOG="$TEST_ROOT/execution.tsv"
SENTINEL_DIR="$TEST_ROOT/sentinel-bin"
SENTINEL_MARKER="$TEST_ROOT/prohibited-runtime-executed"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SESSION_ID="execution-capability-selftest-$$"

cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-execution-capability-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_hook() {
  local event="$1" agent="${2:-codex}"
  set +e
  HOOK_OUTPUT="$(printf '%s\n' "$event" | "$LOOM" agent-hook --agent "$agent" \
    --test-file-capability-fixture 2>&1)"
  HOOK_RC=$?
  set -e
}

execution_event() {
  local command="$1" cwd="${2:-$ROOT_DIR}"
  jq -cn --arg session "$SESSION_ID" --arg root "$ROOT_DIR" \
    --arg cwd "$cwd" --arg command "$command" \
    '{hook_event_name:"PreToolUse",session_id:$session,cwd:$root,
      tool_name:"exec_command",tool_input:{cmd:$command,workdir:$cwd}}'
}

claude_execution_event() {
  local command="$1"
  jq -cn --arg session "$SESSION_ID" --arg root "$ROOT_DIR" \
    --arg command "$command" \
    '{hook_event_name:"PreToolUse",session_id:$session,cwd:$root,
      tool_name:"Bash",tool_input:{command:$command,description:"capability contract"}}'
}

assert_output_envelope() {
  local field="$1"
  printf '%s\n' "$HOOK_OUTPUT" | jq -e --arg field "$field" '
    (keys == ["hookSpecificOutput"]) and
    (.hookSpecificOutput.hookEventName == "PreToolUse") and
    (.hookSpecificOutput.permissionDecision == "allow") and
    (.hookSpecificOutput.permissionDecisionReason |
      startswith("Sounio 9021 authorized")) and
    (.hookSpecificOutput.updatedInput | type == "object") and
    (.hookSpecificOutput.updatedInput[$field] | type == "string") and
    (has("updatedInput") | not)
  ' >/dev/null || fail "hook output does not match the Codex/Claude PreToolUse envelope"
}

issue() {
  local command="$1" cwd="${2:-$ROOT_DIR}"
  run_hook "$(execution_event "$command" "$cwd")"
  [[ "$HOOK_RC" -eq 0 ]] ||
    fail "capability issue failed for $command: rc=$HOOK_RC output=$HOOK_OUTPUT"
  assert_output_envelope cmd
  REPLACEMENT="$(printf '%s\n' "$HOOK_OUTPUT" | jq -er \
    '.hookSpecificOutput.updatedInput.cmd')" ||
    fail "hook output omitted the replacement command"
  [[ "$HOOK_OUTPUT" != *"$command"* ]] ||
    fail "hook output leaked the original command"
  [[ "$REPLACEMENT" == *' exec-capability --test-file-capability-fixture --token '* ]] ||
    fail "replacement command has an unexpected shape: $REPLACEMENT"
  TOKEN="${REPLACEMENT##*--token }"
  [[ "$TOKEN" =~ ^[0-9a-f]{64}$ ]] || fail "replacement token is not 256-bit hex"
  CAPABILITY_PATH="$CAPABILITY_DIR/$TOKEN.cap"
  [[ -f "$CAPABILITY_PATH" ]] || fail "capability record was not written"
  [[ "$(stat -c '%a' "$CAPABILITY_PATH")" == 600 ]] ||
    fail "capability record is not mode 600"
}

run_replacement() {
  local replacement="$1" cwd="${2:-$ROOT_DIR}"
  set +e
  EXEC_OUTPUT="$(cd "$cwd" && /bin/sh -c "$replacement" 2>&1)"
  EXEC_RC=$?
  set -e
}

capability_count() {
  find "$CAPABILITY_DIR" -maxdepth 1 -type f | wc -l | tr -d ' '
}

decode_hex() {
  local value="$1" index
  [[ "$value" =~ ^([0-9a-f][0-9a-f])*$ ]] || fail "invalid hex fixture"
  for ((index = 0; index < ${#value}; index += 2)); do
    printf '%b' "\\x${value:index:2}"
  done
}

expect_issue_denied() {
  local command="$1" code="$2" reason="$3"
  run_hook "$(execution_event "$command")"
  [[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *"rc=$code:"* \
     && "$HOOK_OUTPUT" == *"reason=$reason"* ]] ||
    fail "command was not denied as $code/$reason: $command rc=$HOOK_RC output=$HOOK_OUTPUT"
  [[ "$(capability_count)" == 0 ]] ||
    fail "denied command left a capability: $command"
}

mkdir -m 700 "$CAPABILITY_DIR" "$SENTINEL_DIR"
for forbidden in python python3 pypy pypy3 cargo rustc; do
  printf '#!/usr/bin/env bash\nprintf prohibited >%q\nexit 97\n' "$SENTINEL_MARKER" \
    >"$SENTINEL_DIR/$forbidden"
  chmod 0755 "$SENTINEL_DIR/$forbidden"
done
ln -s "$SENTINEL_DIR/python" "$SENTINEL_DIR/python-alias"
ln -s "$SENTINEL_DIR/rustc" "$SENTINEL_DIR/rust-alias"
printf '#!/usr/bin/env python\nprintf prohibited >%q\nexit 97\n' "$SENTINEL_MARKER" \
  >"$SENTINEL_DIR/python-shebang"
chmod 0755 "$SENTINEL_DIR/python-shebang"

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$LANGUAGE_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
SOUNIO_LOOM_EXECUTION_AUTHORITY_OUTPUT="$EXECUTION_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_execution_authority.sh" >/dev/null

export PATH="$SENTINEL_DIR:$PATH"
export SOUNIO_COORD_DIR="$COORD_DIR"
export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_LOOM_HOOK_TEST_MODE=1
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$LANGUAGE_RUNTIME"
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$LANGUAGE_LOG"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_RUNTIME="$EXECUTION_RUNTIME"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_LOG="$EXECUTION_LOG"
export SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR="$CAPABILITY_DIR"
unset TMUX TMUX_PANE SOUNIO_AGENTD_SOCKET SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_WORKTREE

session_start="$(jq -cn --arg session "$SESSION_ID" --arg root "$ROOT_DIR" \
  '{hook_event_name:"SessionStart",session_id:$session,cwd:$root}')"
run_hook "$session_start"
[[ "$HOOK_RC" -eq 0 && "$HOOK_OUTPUT" == *'Sounio coordination joined:'* ]] ||
  fail "SessionStart failed: rc=$HOOK_RC output=$HOOK_OUTPUT"

run_hook "$(claude_execution_event '/usr/bin/true')" claude
[[ "$HOOK_RC" -eq 0 ]] ||
  fail "Claude Bash capability issue failed: rc=$HOOK_RC output=$HOOK_OUTPUT"
assert_output_envelope command
claude_replacement="$(printf '%s\n' "$HOOK_OUTPUT" | jq -er \
  '.hookSpecificOutput.updatedInput.command')" ||
  fail "Claude hook output omitted the replacement command"
[[ "$HOOK_OUTPUT" != *'/usr/bin/true'* ]] ||
  fail "Claude hook output leaked the original command"
run_replacement "$claude_replacement"
[[ "$EXEC_RC" -eq 0 ]] ||
  fail "Claude Bash replacement did not execute: rc=$EXEC_RC output=$EXEC_OUTPUT"

issue '/usr/bin/printf hello'
[[ "$(grep -c '^schema=loom-execution-capability-v1$' "$CAPABILITY_PATH")" -eq 1 ]] ||
  fail "capability schema is missing or duplicated"
for field in token issued_us expires_us uid root_hex cwd_hex command_hex \
  command_sha256 environment_record_hex environment_sha256 executable_hex \
  executable_sha256 broker_sha256 manifest_sha256 source_sha256 semantics_sha256 \
  hardware_record_hex hardware_sha256 producing_language language_role language \
  purpose surface execution_class closure_attested argv_count frame_hex decision_hex \
  record_sha256; do
  [[ "$(grep -c "^${field}=" "$CAPABILITY_PATH")" -eq 1 ]] ||
    fail "capability field $field is missing or duplicated"
done
[[ "$(grep '^producing_language=' "$CAPABILITY_PATH")" == \
   'producing_language=NativeTool' ]] || fail "unexpected producing language"
[[ "$(grep '^language_role=' "$CAPABILITY_PATH")" == \
   'language_role=NATIVE_MECHANICAL' ]] || fail "unexpected language role"
[[ "$(grep '^closure_attested=' "$CAPABILITY_PATH")" == \
   'closure_attested=1' ]] || fail "audited leaf closure was not attested"

environment_line="$(grep '^environment_record_hex=' "$CAPABILITY_PATH")"
environment_hex="${environment_line#*=}"
decode_hex "$environment_hex" >"$TEST_ROOT/environment.record"
environment_hash_line="$(grep '^environment_sha256=' "$CAPABILITY_PATH")"
environment_hash="${environment_hash_line#*=}"
[[ "$(sha256sum "$TEST_ROOT/environment.record" | cut -d' ' -f1)" == \
   "$environment_hash" ]] || fail "environment record hash does not match"
grep -Fqx 'schema=loom-execution-environment-v1' "$TEST_ROOT/environment.record" ||
  fail "environment record schema is missing"
grep -Fq 'PATH=hex:' "$TEST_ROOT/environment.record" ||
  fail "environment record omitted PATH"
grep -Fqx 'BASH_ENV=absent' "$TEST_ROOT/environment.record" ||
  fail "environment record omitted the absent shell startup vector"
grep -Fq 'SOUNIO_COORD_DIR=hex:' "$TEST_ROOT/environment.record" ||
  fail "environment record omitted the coordination boundary"

first_token="$TOKEN"
first_replacement="$REPLACEMENT"
run_replacement "$REPLACEMENT"
[[ "$EXEC_RC" -eq 0 && "$EXEC_OUTPUT" == hello ]] ||
  fail "authorized shell bridge did not execute printf: rc=$EXEC_RC output=$EXEC_OUTPUT"
[[ ! -e "$CAPABILITY_PATH" ]] || fail "consumed capability remained on disk"

run_replacement "$first_replacement"
[[ "$EXEC_RC" -eq 126 && "$EXEC_OUTPUT" == *'capability-missing-or-replayed'* ]] ||
  fail "capability replay was not refused: rc=$EXEC_RC output=$EXEC_OUTPUT"

expect_issue_denied 'python --version' 210 forbidden-language
expect_issue_denied 'cargo --version' 210 forbidden-language
expect_issue_denied "$SENTINEL_DIR/python-alias --version" 210 forbidden-language
expect_issue_denied "$SENTINEL_DIR/rust-alias --version" 210 forbidden-language
expect_issue_denied '/usr/bin/env python --version' 210 forbidden-language
expect_issue_denied "$SENTINEL_DIR/python-shebang" 210 forbidden-language
expect_issue_denied '/usr/bin/timeout 1 python --version' 226 dynamic-execution-unclassified
expect_issue_denied '/usr/bin/printf nope | python --version' 226 dynamic-execution-unclassified
expect_issue_denied 'sounio-command-that-does-not-exist' 226 dynamic-execution-unclassified
expect_issue_denied '/usr/bin/head --version' 227 execution-closure-unattested
expect_issue_denied 'git commit -m probe' 227 execution-closure-unattested

export BASH_ENV="$TEST_ROOT/untrusted-startup"
run_hook "$(execution_event '/usr/bin/true')"
unset BASH_ENV
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'unsafe-shell-bridge-environment:BASH_ENV'* ]] ||
  fail "BASH_ENV startup vector was not refused: rc=$HOOK_RC output=$HOOK_OUTPUT"
[[ "$(capability_count)" == 0 ]] || fail "BASH_ENV denial left a capability"

export LD_LIBRARY_PATH="$TEST_ROOT"
run_hook "$(execution_event '/usr/bin/true')"
unset LD_LIBRARY_PATH
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'unsafe-shell-bridge-environment:LD_LIBRARY_PATH'* ]] ||
  fail "dynamic-loader vector was not refused: rc=$HOOK_RC output=$HOOK_OUTPUT"
[[ "$(capability_count)" == 0 ]] || fail "loader denial left a capability"

export SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST="$TEST_ROOT/missing.freeze"
run_hook "$(execution_event '/usr/bin/true')"
unset SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'execution-authority-policy-missing'* ]] ||
  fail "missing policy did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

cp "$ROOT_DIR/tools/loom/execution_authority.freeze.v2" "$TEST_ROOT/unreadable.freeze"
chmod 000 "$TEST_ROOT/unreadable.freeze"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST="$TEST_ROOT/unreadable.freeze"
run_hook "$(execution_event '/usr/bin/true')"
unset SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST
chmod 0600 "$TEST_ROOT/unreadable.freeze"
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'unreadable.freeze'* ]] ||
  fail "unreadable policy did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"
grep -Eq $'phase=ISSUE\tdecision=DENY\treason=.*unreadable\.freeze' "$EXECUTION_LOG" ||
  fail "unreadable policy refusal was not recorded in the execution journal"

cp "$ROOT_DIR/tools/loom/execution_authority.freeze.v2" "$TEST_ROOT/tampered.freeze"
printf '\n' >>"$TEST_ROOT/tampered.freeze"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST="$TEST_ROOT/tampered.freeze"
run_hook "$(execution_event '/usr/bin/true')"
unset SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'execution-authority-policy-hash-mismatch'* ]] ||
  fail "tampered policy did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

cp "$EXECUTION_RUNTIME" "$TEST_ROOT/tampered-authority"
printf x >>"$TEST_ROOT/tampered-authority"
chmod 0755 "$TEST_ROOT/tampered-authority"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_RUNTIME="$TEST_ROOT/tampered-authority"
run_hook "$(execution_event '/usr/bin/true')"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_RUNTIME="$EXECUTION_RUNTIME"
[[ "$HOOK_RC" -eq 2 && "$HOOK_OUTPUT" == *'execution-authority-runtime-hash-mismatch'* ]] ||
  fail "tampered authority runtime did not fail closed: rc=$HOOK_RC output=$HOOK_OUTPUT"

issue '/usr/bin/true'
sed -i 's/schema=loom-execution-capability-v1/schema=loom-execution-capability-v0/' \
  "$CAPABILITY_PATH"
run_replacement "$REPLACEMENT"
[[ "$EXEC_RC" -eq 126 && "$EXEC_OUTPUT" == *'capability-record-digest-mismatch'* ]] ||
  fail "tampered capability was not refused: rc=$EXEC_RC output=$EXEC_OUTPUT"
[[ ! -e "$CAPABILITY_PATH" ]] || fail "tampered capability was not consumed"

export SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS=1
issue '/usr/bin/true'
sleep 2
run_replacement "$REPLACEMENT"
unset SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
[[ "$EXEC_RC" -eq 126 && "$EXEC_OUTPUT" == *'capability-expired'* ]] ||
  fail "expired capability was not refused: rc=$EXEC_RC output=$EXEC_OUTPUT"
[[ ! -e "$CAPABILITY_PATH" ]] || fail "expired capability was not consumed"

issue '/usr/bin/true'
run_replacement "$REPLACEMENT" "$ROOT_DIR/tools/loom"
[[ "$EXEC_RC" -eq 126 && "$EXEC_OUTPUT" == *'capability-cwd-mismatch'* ]] ||
  fail "cwd drift was not refused: rc=$EXEC_RC output=$EXEC_OUTPUT"
[[ ! -e "$CAPABILITY_PATH" ]] || fail "cwd-mismatched capability was not consumed"

issue '/usr/bin/true'
cp "$LOOM" "$TEST_ROOT/loom-tampered"
chmod 0700 "$TEST_ROOT/loom-tampered"
printf x >>"$TEST_ROOT/loom-tampered" || fail "could not tamper broker fixture"
chmod 0755 "$TEST_ROOT/loom-tampered"
set +e
EXEC_OUTPUT="$("$TEST_ROOT/loom-tampered" exec-capability \
  --test-file-capability-fixture --token "$TOKEN" 2>&1)"
EXEC_RC=$?
set -e
[[ "$EXEC_RC" -eq 126 && "$EXEC_OUTPUT" == *'capability-broker-hash-mismatch'* ]] ||
  fail "copied/tampered broker was not refused: rc=$EXEC_RC output=$EXEC_OUTPUT"
[[ ! -e "$CAPABILITY_PATH" ]] || fail "broker-mismatched capability was not consumed"

export LOOM_EXECUTION_TEST_TAG=issued
issue '/usr/bin/true'
export LOOM_EXECUTION_TEST_TAG=changed
run_replacement "$REPLACEMENT"
unset LOOM_EXECUTION_TEST_TAG
[[ "$EXEC_RC" -eq 126 && "$EXEC_OUTPUT" == *'capability-environment-mismatch'* ]] ||
  fail "environment drift was not refused: rc=$EXEC_RC output=$EXEC_OUTPUT"
[[ ! -e "$CAPABILITY_PATH" ]] || fail "environment-mismatched capability was not consumed"

[[ -f "$EXECUTION_LOG" ]] || fail "execution decision log is missing"
grep -Fq $'phase=ISSUE\tdecision=ALLOW\treason=audited-leaf' "$EXECUTION_LOG" ||
  fail "decision log omitted ISSUE ALLOW"
grep -Fq $'phase=CONSUME\tdecision=ALLOW\treason=single-use-capability' \
  "$EXECUTION_LOG" || fail "decision log omitted CONSUME ALLOW"
grep -Fq $'phase=CONSUME\tdecision=DENY\treason=capability-missing-or-replayed' \
  "$EXECUTION_LOG" || fail "decision log omitted replay DENY"
grep -Fq 'SOUNIO_EXECUTION_AUTHORITY_DENY code=210 reason=forbidden-language' \
  "$EXECUTION_LOG" || fail "decision log omitted the Sounio Python/Rust refusal"
grep -Eq $'manifest_sha256=[0-9a-f]{64}\tsounio_source_sha256=[0-9a-f]{64}\tsemantics_sha256=[0-9a-f]{64}' \
  "$EXECUTION_LOG" || fail "decision log omitted the frozen Sounio chain"
grep -Eq $'hardware_sha256=[0-9a-f]{64}\tenvironment_sha256=[0-9a-f]{64}' \
  "$EXECUTION_LOG" || fail "decision log omitted hardware/environment hashes"
grep -Fq $'producing_language=NativeTool\tlanguage_role=NATIVE_MECHANICAL' \
  "$EXECUTION_LOG" || fail "decision log omitted producer role"
grep -Fq $'execution_result=pending' "$EXECUTION_LOG" ||
  fail "decision log promoted a pre-exec receipt into an outcome"

[[ ! -e "$SENTINEL_MARKER" ]] || fail "Python or Rust executed during the broker gate"
[[ "$(capability_count)" == 0 ]] || fail "capability files remain after the gate"
[[ -z "$(find "$CAPABILITY_DIR" -maxdepth 1 -name '*.consuming.*' -print -quit)" ]] ||
  fail "consuming capability files remain after the gate"

session_end="$(jq -cn --arg session "$SESSION_ID" --arg root "$ROOT_DIR" \
  '{hook_event_name:"SessionEnd",session_id:$session,cwd:$root}')"
run_hook "$session_end"
[[ "$HOOK_RC" -eq 0 ]] || fail "SessionEnd failed: rc=$HOOK_RC output=$HOOK_OUTPUT"

printf '%s\n' \
  "sounio-loom-execution-capability-selftest: PASS language=Sounio broker=OCaml codex_pretool_output=accepted claude_pretool_output=accepted allowed_leaf=executed shell_bridge=roundtrip environment=bound replay=refused python=refused rust=refused aliases=refused shebangs=refused env_forwarding=refused wrappers=refused dynamic=refused missing=refused closure=refused commit=refused startup_injection=refused loader_injection=refused policy_missing=refused policy_unreadable=refused+logged policy_tamper=refused runtime_tamper=refused record_tamper=refused expiry=refused cwd_drift=refused broker_drift=refused environment_drift=refused prohibited_executed=false execution_result=pending same_uid_peer_isolation=false first_token_sha256=$(printf '%s' "$first_token" | sha256sum | cut -d' ' -f1)"
