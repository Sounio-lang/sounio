#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
STATE="$TEST_ROOT/coord"
SOURCE_LOOM_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
LOOM_RUNTIME="$REPO/tools/loom/_build/default/src/loom.exe"

cleanup() {
  if [[ -x "$REPO/bin/sounio-coord" ]]; then
    SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_RUNTIME_MODE=local \
    SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
      "$REPO/bin/sounio-coord" obligation-supervisor-stop \
      --timeout-seconds 5 >/dev/null 2>&1 || true
  fi
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-agent-hook-selftest: FAIL: $*" >&2
  exit 1
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
[[ -x "$SOURCE_LOOM_RUNTIME" ]] || fail "Loom runtime was not built: $SOURCE_LOOM_RUNTIME"

mkdir -p "$REPO/bin" "$REPO/scripts/dev" "$REPO/self-hosted/parser" \
  "$REPO/tools/loom/_build/default/src"
cp "$SOURCE_LOOM_RUNTIME" "$LOOM_RUNTIME"
cp "$ROOT_DIR/bin/sounio-coord" "$REPO/bin/sounio-coord"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$REPO/scripts/dev/"
chmod +x "$REPO/bin/sounio-coord" "$REPO/scripts/dev/sounio_coord_runtime.sh"
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Hook Selftest'
git -C "$REPO" config user.email 'coord-hook-selftest@sounio.local'
printf 'seed\n' > "$REPO/README.md"
printf 'parser\n' > "$REPO/self-hosted/parser/ast.sio"
printf 'items\n' > "$REPO/self-hosted/parser/items.sio"
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b second-lane "$SECOND"
rm "$SECOND/tools/loom/_build/default/src/loom.exe"
ln -s "$LOOM_RUNTIME" "$SECOND/tools/loom/_build/default/src/loom.exe"

run_hook() {
  local agent="$1" cwd="$2" payload="$3"
  local config="$ROOT_DIR/.${agent}/hooks.json"
  [[ "$agent" != claude ]] || config="$ROOT_DIR/.claude/settings.json"
  printf '%s\n' "$payload" | \
    SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_RUNTIME_MODE=local \
    SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    SOUNIO_LOOM_HOOK_TEST_MODE=1 SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
    SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-language-authority-runtime" \
    SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-cutover" \
    SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$config" \
    "$cwd/tools/loom/_build/default/src/loom.exe" agent-hook --agent "$agent"
}

run_coord() {
  local cwd="$1"
  shift
  (cd "$cwd" && SOUNIO_COORD_DIR="$STATE" \
    SOUNIO_COORD_DURABLE_OBLIGATIONS=0 bin/sounio-coord "$@")
}

output="$(run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"SessionStart\"}")"
grep -q 'agent=codex lane=session-codex-a' <<< "$output" || \
  fail 'Codex session identity was not injected'
output="$(run_coord "$REPO" brief --max-rows 4)"
grep -q 'ACTIVE claim_id=codex--session-codex-a' <<< "$output" || \
  fail 'Codex session presence was not registered'

run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"apply_patch\",\"tool_input\":{\"patch\":\"*** Update File: self-hosted/parser/ast.sio\"}}"
run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"apply_patch\",\"tool_input\":{\"patch\":\"*** Update File: self-hosted/parser/items.sio\"}}"
output="$(run_coord "$REPO" brief --max-rows 4)"
grep -q 'files=self-hosted/parser/ast.sio,self-hosted/parser/items.sio' <<< "$output" || \
  fail 'automatic write scope did not accumulate files'

run_coord "$SECOND" claim --agent codex --lane cross-worktree --ttl-seconds 600 \
  --intent 'cross-worktree target owned explicitly' \
  --files self-hosted/parser/cross-new.sio self-hosted/parser/own-new.sio >/dev/null
set +e
claimed_cross_output="$(run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$SECOND/self-hosted/parser/cross-new.sio\"}}" 2>&1)"
claimed_cross_rc=$?
set -e
[[ "$claimed_cross_rc" -eq 2 ]] ||
  fail "claimed cross-worktree write returned $claimed_cross_rc instead of 2"
grep -q 'write-path-outside-session-worktree' <<< "$claimed_cross_output" ||
  fail 'claimed cross-worktree write escaped the native session boundary'
printf '%s\n' \
  "{\"session_id\":\"codex-target\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$SECOND/self-hosted/parser/own-new.sio\"}}" | \
  SOUNIO_COORD_DIR="$STATE" \
  SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-language-authority-runtime" \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-cutover" \
  SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$ROOT_DIR/.codex/hooks.json" \
  "$SECOND/tools/loom/_build/default/src/loom.exe" agent-hook --agent codex
output="$(run_coord "$SECOND" brief --max-rows 6)"
grep -Fq "ACTIVE claim_id=codex--cross-worktree" <<< "$output" || \
  fail 'cross-worktree claim disappeared during target authorization'
grep -Fq "worktree=$SECOND" <<< "$output" || \
  fail 'cross-worktree claim was not retained on the target worktree'

set +e
cross_log="$TEST_ROOT/unclaimed-cross-write.log"
run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$SECOND/self-hosted/parser/unclaimed-new.sio\"}}" \
  >"$cross_log" 2>&1
cross_rc=$?
set -e
cross_output="$(<"$cross_log")"
[[ "$cross_rc" -eq 2 ]] || fail "unclaimed cross-worktree write returned $cross_rc instead of 2"
grep -q 'write-path-outside-session-worktree' <<< "$cross_output" || \
  fail 'unclaimed cross-worktree write escaped the native session boundary'

output="$(run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionStart\"}")"
grep -q 'agent=claude lane=session-claude-b' <<< "$output" || \
  fail 'Claude session identity was not injected'
run_coord "$REPO" send --agent observer --lane announcements --kind info \
  --message 'broadcast hook exclusion marker' >/dev/null
send_output="$(run_coord "$REPO" send --agent codex --lane session-codex-a \
  --to-agent claude --to-lane session-claude-b --kind request \
  --message 'Please review the parser ownership boundary')"
message_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$send_output")"
[[ -n "$message_id" ]] || fail 'message id was not returned'

output="$(run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"PostToolUse\",\"tool_name\":\"Read\",\"tool_input\":{}}")"
grep -q "MESSAGE id=$message_id" <<< "$output" || fail 'message was not delivered to Claude'
grep -q 'broadcast hook exclusion marker' <<< "$output" && \
  fail 'hook injected a broadcast alongside directed work'
grep -q "ack --agent claude --lane session-claude-b --message <id>" <<< "$output" || \
  fail 'hook did not show the explicit acknowledgement command'
output="$(run_coord "$REPO" message-status --agent codex --lane session-codex-a \
  --message "$message_id")"
grep -q 'request_state=open injected=1 acknowledged=0 responses=0' <<< "$output" || \
  fail 'hook delivery did not create an injection receipt'
run_coord "$SECOND" ack --agent claude --lane session-claude-b --message "$message_id" >/dev/null
output="$(run_coord "$SECOND" inbox --agent claude --lane session-claude-b --directed-only)"
grep -q '^inbox_messages=0$' <<< "$output" || fail 'acknowledged message remained unread'
output="$(run_coord "$REPO" message-status --agent codex --lane session-codex-a \
  --message "$message_id")"
grep -q 'request_state=open injected=1 acknowledged=1 responses=0' <<< "$output" || \
  fail 'explicit acknowledgement was not distinct from injection'
reply_output="$(run_coord "$SECOND" send --agent claude --lane session-claude-b \
  --kind reply --reply-to "$message_id" --message 'Parser boundary reviewed')"
reply_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$reply_output")"
output="$(run_coord "$REPO" wait --agent codex --lane session-codex-a \
  --message "$message_id" --timeout-seconds 0)"
grep -q "^WAIT_RESPONSE request_id=$message_id request_state=answered$" <<< "$output" || \
  fail 'native wait did not observe the cross-worktree reply'
grep -q "MESSAGE id=$reply_id" <<< "$output" || fail 'native wait returned the wrong reply'
output="$(run_coord "$REPO" message-status --agent codex --lane session-codex-a \
  --message "$message_id")"
grep -q "request_state=answered injected=1 acknowledged=1 responses=1 latest_response=$reply_id" <<< "$output" || \
  fail 'cross-worktree request lifecycle did not close as answered'

set +e
conflict_output="$(run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"self-hosted/parser/ast.sio\"}}" 2>&1)"
conflict_rc=$?
set -e
[[ "$conflict_rc" -eq 2 ]] || fail "conflicting write returned $conflict_rc instead of 2"
grep -q 'requested file set overlaps an active claim' <<< "$conflict_output" || \
  fail 'conflicting write did not explain the ownership collision'
output="$(run_coord "$REPO" inbox --agent codex --lane session-codex-a)"
grep -q 'kind=request text=Write conflict requested by claude/session-claude-b' <<< "$output" || \
  fail 'conflict owner did not receive an automatic message'

run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionEnd\"}"
output="$(run_coord "$SECOND" brief --max-rows 4)"
grep -q 'ACTIVE claim_id=claude--session-claude-b' <<< "$output" && \
  fail 'Claude session claim survived SessionEnd'

run_coord "$SECOND" release --agent codex --lane cross-worktree \
  --reason 'cross-worktree selftest complete' >/dev/null

ephemeral_output="$(run_coord "$REPO" send --agent codex --lane session-codex-a \
  --to-agent claude --to-lane session-claude-b --kind info \
  --ttl-seconds 1 --message 'ephemeral selftest message')"
ephemeral_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$ephemeral_output")"
run_coord "$SECOND" injected --agent claude --lane session-claude-b \
  --messages "$ephemeral_id" >/dev/null
run_coord "$SECOND" ack --agent claude --lane session-claude-b \
  --message "$ephemeral_id" >/dev/null
sleep 2
output="$(run_coord "$REPO" prune)"
grep -q 'pruned_messages=1$' <<< "$output" || fail 'expired message was not pruned'
[[ -z "$(find "$STATE/message-injections" -name "$ephemeral_id--*" -print -quit)" ]] || \
  fail 'prune left an orphan injection receipt'
[[ -z "$(find "$STATE/message-acks" -name "$ephemeral_id--*" -print -quit)" ]] || \
  fail 'prune left an orphan acknowledgement receipt'

echo 'sounio-coord-agent-hook-selftest: PASS'
