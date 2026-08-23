#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-wake-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
STATE="$TEST_ROOT/state"
SOCKET="$TEST_ROOT/tmux.sock"
RECEIVER="$TEST_ROOT/receiver.js"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
RUNTIME="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh"

cleanup() {
  tmux -S "$SOCKET" kill-server >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-wake-selftest: FAIL: $*" >&2
  exit 1
}

command -v tmux >/dev/null 2>&1 || fail 'tmux is required'
command -v node >/dev/null 2>&1 || fail 'node is required'

mkdir -p "$REPO/bin" "$REPO/scripts/dev"
cp "$ROOT_DIR/bin/sounio-coord" "$REPO/bin/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$REPO/scripts/dev/"
chmod +x "$REPO/bin/sounio-coord" "$REPO/scripts/dev/"*
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Wake Selftest'
git -C "$REPO" config user.email 'coord-wake-selftest@sounio.local'
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b recipient-lane "$SECOND"

cat > "$RECEIVER" <<'JS'
const fs = require("fs");
const log = process.argv[2];
process.stdin.on("data", (chunk) => fs.appendFileSync(log, chunk));
process.stdin.resume();
setInterval(() => {}, 1000);
JS

coord() {
  local worktree="$1"
  shift
  (cd "$worktree" && SOUNIO_COORD_DIR="$STATE" "$RUNTIME" "$@")
}

wait_for_text() {
  local file="$1" pattern="$2" attempt
  for attempt in $(seq 1 50); do
    [[ -f "$file" ]] && grep -q "$pattern" "$file" && return 0
    sleep 0.1
  done
  return 1
}

coord "$REPO" claim --agent sender --lane origin --intent 'wake sender' \
  --files sender.test >/dev/null
coord "$SECOND" claim --agent codex --lane recipient --intent 'wake recipient' \
  --files recipient.test >/dev/null

tmux -S "$SOCKET" new-session -d -s recipient -c "$SECOND" \
  "node '$RECEIVER' '$RECEIVER_LOG'"
sleep 0.2
pane="$(tmux -S "$SOCKET" display-message -p -t recipient '#{pane_id}')"
[[ -n "$pane" ]] || fail 'tmux pane was not created'

output="$(coord "$SECOND" endpoint-register --agent codex --lane recipient \
  --harness codex --transport tmux --address "$pane" --socket "$SOCKET" \
  --ttl-seconds 300)"
grep -q '^ENDPOINT_REGISTERED ' <<< "$output" || fail 'endpoint was not registered'

coord "$SECOND" claim --agent codex --lane duplicate --intent 'duplicate endpoint sabotage' \
  --files duplicate.test >/dev/null
if coord "$SECOND" endpoint-register --agent codex --lane duplicate --harness codex \
  --transport tmux --address "$pane" --socket "$SOCKET" >/dev/null 2>&1; then
  fail 'one tmux pane was assigned to two active lanes'
fi
coord "$SECOND" release --agent codex --lane duplicate --reason 'sabotage complete' >/dev/null

secret='raw-message-text-must-not-be-injected'
output="$(coord "$REPO" send --agent sender --lane origin --to-agent codex \
  --to-lane recipient --thread "$secret" --kind request --message "$secret" 2>&1)"
message_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$message_id" ]] || fail 'directed send did not return a message id'
grep -q "^WAKE_DELIVERED message_id=$message_id " <<< "$output" || \
  fail 'directed send did not wake the verified endpoint'
wait_for_text "$RECEIVER_LOG" "$message_id" || fail 'wake prompt did not reach recipient stdin'
grep -q 'Run bin/sounio-coord inbox' "$RECEIVER_LOG" || fail 'wake omitted the inbox command'
if grep -q "$secret" "$RECEIVER_LOG"; then
  fail 'wake injected raw message content into the recipient harness'
fi

bytes_before_retry="$(wc -c < "$RECEIVER_LOG" | tr -d ' ')"
output="$(coord "$REPO" wake --agent sender --lane origin --message "$message_id")"
grep -q "^WAKE_SKIPPED message_id=$message_id .*reason=already-delivered$" <<< "$output" || \
  fail 'wake retry was not deduplicated'
sleep 0.2
bytes_after_retry="$(wc -c < "$RECEIVER_LOG" | tr -d ' ')"
[[ "$bytes_before_retry" == "$bytes_after_retry" ]] || \
  fail 'deduplicated wake reached recipient more than once'
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$message_id")"
grep -q 'request_state=open injected=0 acknowledged=0 responses=0 .*wakes=1$' <<< "$output" || \
  fail 'message status did not expose the wake receipt independently'

SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$SECOND" send --agent codex \
  --lane legacy-recipient --to-agent sender --to-lane origin --kind info \
  --message 'establish a historical endpoint without registration' >/dev/null
legacy_secret='legacy-raw-message-must-not-be-injected'
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane legacy-recipient --kind request \
  --message "$legacy_secret")"
legacy_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$legacy_message" ]] || fail 'history-discovered send did not return a message id'
grep -q "^WAKE_DELIVERED message_id=$legacy_message .*discovery=history$" <<< "$output" || \
  fail 'stale client was not woken through its unique message history'
wait_for_text "$RECEIVER_LOG" "$legacy_message" || \
  fail 'history-discovered wake did not reach the recipient'
if grep -q "$legacy_secret" "$RECEIVER_LOG"; then
  fail 'history-discovered wake injected raw message content'
fi

SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$SECOND" send --agent codex \
  --lane moved-recipient --to-agent sender --to-lane origin --kind info \
  --message 'record branch before drift' >/dev/null
git -C "$SECOND" switch -qc moved-after-history
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane moved-recipient --kind info \
  --message 'must fail closed after branch drift')"
grep -q '^WAKE_UNAVAILABLE .*status=unavailable$' <<< "$output" || \
  fail 'history discovery followed a worktree after branch drift'
git -C "$SECOND" switch -q recipient-lane

if coord "$REPO" endpoint-register --agent sender --lane origin --harness codex \
  --transport tmux --address "$pane" --socket "$SOCKET" >/dev/null 2>&1; then
  fail 'endpoint registration accepted a pane from another worktree'
fi

tmux -S "$SOCKET" respawn-pane -k -t "$pane" -c "$SECOND" \
  "node '$RECEIVER' '$TEST_ROOT/drift.log'"
sleep 0.2
output="$(coord "$REPO" send --agent sender --lane origin --to-agent codex \
  --to-lane recipient --kind info --message 'endpoint drift sabotage' 2>&1)"
drift_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$drift_message" ]] || fail 'drift sabotage message was not persisted'
grep -q "WAKE_REFUSED message_id=$drift_message .*reason=endpoint-drift" <<< "$output" || \
  fail 'changed pane process did not fail closed'
grep -q "WAKE_UNAVAILABLE message_id=$drift_message status=failed-closed" <<< "$output" || \
  fail 'sender did not receive the failed-closed delivery status'
output="$(coord "$SECOND" inbox --agent codex --lane recipient --all)"
grep -q "MESSAGE id=$drift_message " <<< "$output" || \
  fail 'failed wake removed the durable bus message'

coord "$SECOND" endpoint-register --agent codex --lane recipient --harness codex \
  --transport tmux --address "$pane" --socket "$SOCKET" --ttl-seconds 1 >/dev/null
sleep 2
output="$(coord "$REPO" send --agent sender --lane origin --to-agent codex \
  --to-lane recipient --kind info --message 'expired endpoint sabotage' 2>&1)"
expired_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
grep -q "WAKE_UNAVAILABLE message_id=$expired_message status=stale" <<< "$output" || \
  fail 'expired endpoint remained deliverable'
output="$(coord "$REPO" prune)"
grep -q '^pruned_endpoints=1$' <<< "$output" || fail 'expired endpoint was not pruned'

coord "$SECOND" endpoint-register --agent codex --lane recipient --harness codex \
  --transport tmux --address "$pane" --socket "$SOCKET" --ttl-seconds 300 >/dev/null
coord "$SECOND" release --agent codex --lane recipient --reason 'lifecycle sabotage' >/dev/null
if coord "$SECOND" endpoint-status --agent codex --lane recipient >/dev/null 2>&1; then
  fail 'claim release left a delivery endpoint active'
fi

hook_event="{\"session_id\":\"hook-wake\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionStart\"}"
printf '%s\n' "$hook_event" | env \
  SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_DIR="$STATE" \
  TMUX="$SOCKET,1,0" TMUX_PANE="$pane" \
  python3 "$SECOND/scripts/dev/sounio_coord_agent_hook.py" --agent codex >/dev/null
output="$(coord "$SECOND" endpoint-status --agent codex --lane session-hook-wake)"
grep -q '^ENDPOINT_STATUS .* state=active .* harness=codex transport=tmux ' <<< "$output" || \
  fail 'session hook did not auto-register its verified tmux endpoint'

hook_event="{\"session_id\":\"hook-wake\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionEnd\"}"
printf '%s\n' "$hook_event" | env \
  SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_DIR="$STATE" \
  TMUX="$SOCKET,1,0" TMUX_PANE="$pane" \
  python3 "$SECOND/scripts/dev/sounio_coord_agent_hook.py" --agent codex >/dev/null
if coord "$SECOND" endpoint-status --agent codex --lane session-hook-wake >/dev/null 2>&1; then
  fail 'session end left its delivery endpoint active'
fi

echo 'sounio-coord-wake-selftest: PASS'
