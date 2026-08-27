#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook.XXXXXX")"
AUTHORITY_RUNTIME="$TEST_ROOT/sounio-loom-language-authority-runtime"
COORD_DIR="$TEST_ROOT/coord"
DECISION_LOG="$TEST_ROOT/agent-hook.tsv"
SENTINEL_DIR="$TEST_ROOT/sentinel-bin"
SENTINEL_MARKER="$TEST_ROOT/prohibited-runtime-executed"
SIBLING_ROOT="$TEST_ROOT/sibling-worktree"
SIBLING_ADDED=0
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SESSION_ID="native-hook-selftest-$$"

cleanup() {
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
  set +e
  HOOK_OUTPUT="$(printf '%s\n' "$event" | "$LOOM" agent-hook --agent codex 2>&1)"
  HOOK_RC=$?
  set -e
}

mkdir -p "$SENTINEL_DIR"
for forbidden in python python3 pypy pypy3 cargo rustc; do
  printf '#!/usr/bin/env bash\nprintf prohibited >%q\nexit 97\n' "$SENTINEL_MARKER" \
    >"$SENTINEL_DIR/$forbidden"
  chmod 0755 "$SENTINEL_DIR/$forbidden"
done

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$AUTHORITY_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
git -C "$ROOT_DIR" worktree add --detach --no-checkout "$SIBLING_ROOT" HEAD >/dev/null
SIBLING_ADDED=1

export PATH="$SENTINEL_DIR:$PATH"
export SOUNIO_COORD_DIR="$COORD_DIR"
export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$AUTHORITY_RUNTIME"
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$DECISION_LOG"
export SOUNIO_LOOM_HOOK_TEST_MODE=1
unset TMUX TMUX_PANE SOUNIO_AGENTD_SOCKET SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_WORKTREE

session_start="{\"hook_event_name\":\"SessionStart\",\"session_id\":\"$SESSION_ID\",\"cwd\":\"$ROOT_DIR\"}"
run_hook "$session_start"
[[ "$HOOK_RC" -eq 0 && "$HOOK_OUTPUT" == *'Sounio coordination joined:'* ]] ||
  fail "native SessionStart failed: rc=$HOOK_RC output=$HOOK_OUTPUT"

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

[[ -f "$DECISION_LOG" ]] || fail "native hook omitted its decision log"
grep -Fq $'decision=ALLOW\treason=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
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
grep -Fq $'sounio_source_sha256=64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da\tsemantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff\tproducing_language=OCaml\tlanguage_role=OPERATIONAL_REALIZATION\tsemantic_authority_language=Sounio\tsemantic_authority_role=SEMANTIC_AUTHORITY' \
  "$DECISION_LOG" || fail "decision receipt omitted the authority chain"
grep -Fq $'\ttoolchain=OCaml ' "$DECISION_LOG" ||
  fail "decision receipt omitted the toolchain"
grep -Fq $'\thardware=os_type=' "$DECISION_LOG" ||
  fail "decision receipt omitted the hardware"
grep -Fq $'\tcommand=sounio-loom agent-hook --agent codex event=SessionStart tool=none\tcommand_sha256=' \
  "$DECISION_LOG" || fail "decision receipt omitted the command"
grep -Fq $'\tresult=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$DECISION_LOG" || fail "decision receipt omitted the result"

[[ ! -e "$SENTINEL_MARKER" ]] || fail "Python or Rust was executed by the native hook"

status="$(SOUNIO_COORD_DIR="$COORD_DIR" SOUNIO_COORD_RUNTIME_MODE=local \
  "$ROOT_DIR/bin/sounio-coord" status)"
[[ "$status" != *"session-$SESSION_ID"* ]] || fail "SessionEnd left an active native-hook claim"

printf '%s\n' \
  'sounio-loom-native-hook-selftest: PASS language=OCaml semantic_authority=Sounio session=roundtrip writes=authorized outside_write=refused sibling_worktree=refused pathless_write=refused malformed=refused strict_json=refused duplicate_json=refused policy_missing=refused policy_tamper=refused runtime_tamper=refused log_redirect=refused decision_receipt=complete python=not-executed rust=not-executed'
