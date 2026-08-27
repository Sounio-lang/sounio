#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-wake-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
BACKGROUND="$TEST_ROOT/background-worktree"
GROK_HOME="$TEST_ROOT/grok-cli2"
HISTORY_HOME="$TEST_ROOT/history-home"
STATE="$TEST_ROOT/state"
SOCKET="$TEST_ROOT/tmux.sock"
RECEIVER="$TEST_ROOT/receiver.js"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
GROK_BIN="$TEST_ROOT/grok"
CLAUDE_BIN="$TEST_ROOT/claude"
CLAUDE_LOG="$TEST_ROOT/claude-receiver.log"
STALLED_LOG="$TEST_ROOT/stalled-receiver.log"
INSERT_CRASH_LOG="$TEST_ROOT/insert-crash-receiver.log"
RETRY_LOG="$TEST_ROOT/retry-receiver.log"
RUNTIME="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh"

cleanup() {
  tmux -S "$SOCKET" kill-server >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$GROK_HOME" >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$BACKGROUND" >/dev/null 2>&1 || true
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
git -C "$REPO" worktree add -q -b background-lane "$BACKGROUND"
git -C "$REPO" worktree add -q -b grok-session "$GROK_HOME"

cat > "$RECEIVER" <<'JS'
const fs = require("fs");
const { spawnSync } = require("child_process");
const log = process.argv[2];
const runtime = process.argv[3];
const state = process.argv[4];
const history = process.argv[5];
const mode = process.argv[6] || "no-auto";
let input = "";
let submissions = 0;
let pendingMessage = null;
let pendingRecipient = null;
process.stdin.on("data", (chunk) => {
  fs.appendFileSync(log, chunk);
  input += chunk.toString("utf8");
  if (!input.includes("\n")) return;
  for (const line of input.split(/\r?\n/)) {
    const message = line.match(/Sounio coordination wake: \S+ (msg-[^ ]+)/);
    const recipient = line.match(/inbox --agent ([^ ]+) --lane ([^ ]+)/);
    if (message && recipient) {
      pendingMessage = message[1];
      pendingRecipient = [recipient[1], recipient[2]];
    }
  }
  submissions += 1;
  const autoInject = mode === "auto" || (mode === "auto-after-two" && submissions >= 2);
  if (autoInject && pendingMessage && pendingRecipient) {
    const result = spawnSync(runtime,
      ["injected", "--agent", pendingRecipient[0], "--lane", pendingRecipient[1],
       "--messages", pendingMessage],
      { cwd: process.cwd(), env: { ...process.env,
          SOUNIO_COORD_DIR: state, SOUNIO_COORD_HISTORY_HOME: history },
        encoding: "utf8" });
    fs.appendFileSync(`${log}.inject`, `${result.status}\n${result.stdout}${result.stderr}`);
  }
  input = "";
});
process.stdin.resume();
setInterval(() => {}, 1000);
JS

coord() {
  local worktree="$1"
  shift
  (cd "$worktree" && SOUNIO_COORD_DIR="$STATE" \
    SOUNIO_COORD_HISTORY_HOME="$HISTORY_HOME" SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    "$RUNTIME" "$@")
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
  "node '$RECEIVER' '$RECEIVER_LOG' '$RUNTIME' '$STATE' '$HISTORY_HOME' auto"
sleep 0.2
pane="$(tmux -S "$SOCKET" display-message -p -t recipient '#{pane_id}')"
[[ -n "$pane" ]] || fail 'tmux pane was not created'
cp "$(command -v node)" "$GROK_BIN"
tmux -S "$SOCKET" new-window -d -t recipient -n grok -c "$GROK_HOME" \
  "$GROK_BIN '$RECEIVER' '$TEST_ROOT/grok-receiver.log' '$RUNTIME' '$STATE' '$HISTORY_HOME' auto"
grok_pane="$(tmux -S "$SOCKET" display-message -p -t recipient:grok '#{pane_id}')"
[[ -n "$grok_pane" ]] || fail 'grok compatibility pane was not created'
cp "$(command -v node)" "$CLAUDE_BIN"
tmux -S "$SOCKET" new-window -d -t recipient -n claude-session -c "$SECOND" \
  "$CLAUDE_BIN '$RECEIVER' '$CLAUDE_LOG' '$RUNTIME' '$STATE' '$HISTORY_HOME' auto"
claude_pane="$(tmux -S "$SOCKET" display-message -p -t recipient:claude-session '#{pane_id}')"
[[ -n "$claude_pane" ]] || fail 'Claude session-history pane was not created'

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
grep -q "^WAKE_STARTED message_id=$message_id " <<< "$output" || \
  fail 'directed send did not receive a generation-bound start acknowledgement'
wait_for_text "$RECEIVER_LOG" "$message_id" || fail 'wake prompt did not reach recipient stdin'
grep -q 'Run bin/sounio-coord inbox' "$RECEIVER_LOG" || fail 'wake omitted the inbox command'
if grep -q "$secret" "$RECEIVER_LOG"; then
  fail 'wake injected raw message content into the recipient harness'
fi

bytes_before_retry="$(wc -c < "$RECEIVER_LOG" | tr -d ' ')"
output="$(coord "$REPO" wake --agent sender --lane origin --message "$message_id")"
grep -q "^WAKE_SKIPPED message_id=$message_id .*reason=already-started" <<< "$output" || \
  fail 'wake retry was not deduplicated'
sleep 0.2
bytes_after_retry="$(wc -c < "$RECEIVER_LOG" | tr -d ' ')"
[[ "$bytes_before_retry" == "$bytes_after_retry" ]] || \
  fail 'deduplicated wake reached recipient more than once'
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$message_id")"
grep -q 'request_state=open injected=1 acknowledged=0 responses=0 .*wakes=1 wake_pending=0$' <<< "$output" || \
  fail 'message status did not separate injection, start, and pending receipts'

# Insertion and Enter are transport attempts, not proof that the agent started.
coord "$SECOND" claim --agent codex --lane stalled --intent 'start acknowledgement sabotage' \
  --files stalled.test >/dev/null
tmux -S "$SOCKET" new-window -d -t recipient -n stalled -c "$SECOND" \
  "node '$RECEIVER' '$STALLED_LOG' '$RUNTIME' '$STATE' '$HISTORY_HOME' no-auto"
stalled_pane="$(tmux -S "$SOCKET" display-message -p -t recipient:stalled '#{pane_id}')"
coord "$SECOND" endpoint-register --agent codex --lane stalled --harness codex \
  --transport tmux --address "$stalled_pane" --socket "$SOCKET" --ttl-seconds 300 >/dev/null
output="$(SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS=150 coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane stalled --kind request \
  --message 'insertion without turn start must remain pending')"
stalled_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$stalled_message" ]] || fail 'stalled wake did not persist a message'
grep -q "^WAKE_PENDING message_id=$stalled_message .*state=awaiting-start" <<< "$output" || \
  fail 'insertion-only sabotage was promoted to delivery'
grep -q "^WAKE_UNAVAILABLE message_id=$stalled_message status=pending-start$" <<< "$output" || \
  fail 'insertion-only sabotage did not fail closed'
wait_for_text "$STALLED_LOG" "$stalled_message" || fail 'stalled prompt was not inserted'
prompt_count_before="$(grep -o "$stalled_message" "$STALLED_LOG" | wc -l | tr -d ' ')"
output="$(SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS=100 coord "$REPO" wake --agent sender \
  --lane origin --message "$stalled_message" 2>&1 || true)"
grep -q "WAKE_PENDING message_id=$stalled_message .*state=awaiting-start" <<< "$output" || \
  fail 'pending retry did not remain fail closed'
prompt_count_after="$(grep -o "$stalled_message" "$STALLED_LOG" | wc -l | tr -d ' ')"
[[ "$prompt_count_before" -ge 1 && "$prompt_count_after" == "$prompt_count_before" ]] || \
  fail 'pending retry reinserted the prompt'
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$stalled_message")"
grep -q 'injected=0 .*wakes=0 wake_pending=1$' <<< "$output" || \
  fail 'pending status fabricated a start receipt'
output="$(coord "$SECOND" injected --agent codex --lane stalled --messages "$stalled_message")"
grep -q "^WAKE_STARTED message_id=$stalled_message .*generation=" <<< "$output" || \
  fail 'real hook injection did not promote the matching generation'
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$stalled_message")"
grep -q 'injected=1 .*wakes=1 wake_pending=0$' <<< "$output" || \
  fail 'generation-bound start did not close the pending submission'
coord "$SECOND" release --agent codex --lane stalled --reason 'start sabotage complete' >/dev/null
tmux -S "$SOCKET" kill-window -t recipient:stalled

# Crash/failure after the terminal write but before its confirmation must leave
# an uncertain record. Recovery may observe the exact id and submit it, but it
# may not blindly write the full prompt a second time.
coord "$SECOND" claim --agent codex --lane insert-crash \
  --intent 'external insertion crash sabotage' --files insert-crash.test >/dev/null
tmux -S "$SOCKET" new-window -d -t recipient -n insert-crash -c "$SECOND" \
  "node '$RECEIVER' '$INSERT_CRASH_LOG' '$RUNTIME' '$STATE' '$HISTORY_HOME' no-auto"
sleep 0.2
insert_crash_pane="$(tmux -S "$SOCKET" display-message -p -t recipient:insert-crash '#{pane_id}')"
coord "$SECOND" endpoint-register --agent codex --lane insert-crash --harness codex \
  --transport tmux --address "$insert_crash_pane" --socket "$SOCKET" \
  --ttl-seconds 300 >/dev/null
output="$(SOUNIO_COORD_TEST_FAIL_AFTER_WAKE_INSERT=1 \
  SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS=100 coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane insert-crash --kind request \
  --message 'insertion crash must never duplicate the prompt' 2>&1 || true)"
insert_crash_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$insert_crash_message" ]] || fail 'insertion crash did not persist a message'
grep -q "WAKE_PENDING message_id=$insert_crash_message .*state=insertion-uncertain .*sabotage=after-external-insert" \
  <<< "$output" || fail 'insertion crash did not preserve its uncertain boundary'
insert_crash_submissions=("$STATE/message-wake-submissions/$insert_crash_message"--*.submitted)
[[ "${#insert_crash_submissions[@]}" -eq 1 ]] || \
  fail 'insertion crash did not leave exactly one durable submission'
insert_crash_submission="${insert_crash_submissions[0]}"
grep -q '^state=prepared$' "$insert_crash_submission" || \
  fail 'insertion crash was mislabeled as submitted'
grep -q '^insertion_state=uncertain$' "$insert_crash_submission" || \
  fail 'insertion crash omitted its uncertain state'
grep -q '^submitted_utc=$' "$insert_crash_submission" || \
  fail 'insertion crash fabricated a submission timestamp'
insert_crash_capture="$(tmux -S "$SOCKET" capture-pane -p -J -t "$insert_crash_pane")"
grep -q "$insert_crash_message" <<< "$insert_crash_capture" || \
  fail 'insertion crash did not reach the external-effect boundary'
insert_crash_count_before="$(grep -o 'Sounio coordination wake:' \
  <<< "$insert_crash_capture" | wc -l | tr -d ' ')"
sleep 1
output="$(SOUNIO_COORD_WAKE_RETRY_WAIT_MILLIS=100 coord "$REPO" wake-reconcile)"
grep -q "WAKE_PENDING message_id=$insert_crash_message .*state=awaiting-start" <<< "$output" || \
  fail 'automatic exact-id recovery did not advance to submit-only pending state'
grep -q '^WAKE_RECONCILE attempted=1 started=0 pending=1 ' <<< "$output" || \
  fail 'retry supervisor did not recover the uncertain insertion'
grep -q '^state=submitted$' "$insert_crash_submission" || \
  fail 'successful Enter did not persist submitted state'
inserted_utc="$(sed -n 's/^inserted_utc=//p' "$insert_crash_submission")"
submitted_utc="$(sed -n 's/^submitted_utc=//p' "$insert_crash_submission")"
[[ -n "$inserted_utc" && -n "$submitted_utc" && \
  ( "$submitted_utc" == "$inserted_utc" || "$submitted_utc" > "$inserted_utc" ) ]] || \
  fail "receipt order violated: inserted=$inserted_utc submitted=$submitted_utc"
wait_for_text "$INSERT_CRASH_LOG" "$insert_crash_message" || \
  fail 'submit-only recovery did not release the existing terminal input'
insert_crash_capture="$(tmux -S "$SOCKET" capture-pane -p -J -t "$insert_crash_pane")"
insert_crash_count_after="$(grep -o 'Sounio coordination wake:' \
  <<< "$insert_crash_capture" | wc -l | tr -d ' ')"
[[ "$insert_crash_count_before" -eq 1 && \
  "$insert_crash_count_after" == "$insert_crash_count_before" ]] || \
  fail 'uncertain insertion recovery duplicated the prompt'
[[ "$(grep -o 'Sounio coordination wake:' "$INSERT_CRASH_LOG" | wc -l | tr -d ' ')" -eq 1 ]] || \
  fail 'receiver observed more than one prompt after uncertain recovery'
output="$(coord "$REPO" message-status --agent sender --lane origin \
  --message "$insert_crash_message")"
grep -q 'injected=0 .*wakes=0 wake_pending=1$' <<< "$output" || \
  fail 'insertion crash fabricated a start receipt'
output="$(coord "$SECOND" injected --agent codex --lane insert-crash \
  --messages "$insert_crash_message")"
grep -q "^WAKE_STARTED message_id=$insert_crash_message .*generation=" <<< "$output" || \
  fail 'confirmed insert-crash recovery did not promote at the real hook boundary'
coord "$SECOND" release --agent codex --lane insert-crash \
  --reason 'external insertion crash sabotage complete' >/dev/null
tmux -S "$SOCKET" kill-window -t recipient:insert-crash

# The control service retries submit only. The second submit starts the turn
# without a human Enter and without duplicating the prompt text.
coord "$SECOND" claim --agent codex --lane retry-auto --intent 'automatic submit retry' \
  --files retry-auto.test >/dev/null
tmux -S "$SOCKET" new-window -d -t recipient -n retry-auto -c "$SECOND" \
  "node '$RECEIVER' '$RETRY_LOG' '$RUNTIME' '$STATE' '$HISTORY_HOME' auto-after-two"
retry_pane="$(tmux -S "$SOCKET" display-message -p -t recipient:retry-auto '#{pane_id}')"
coord "$SECOND" endpoint-register --agent codex --lane retry-auto --harness codex \
  --transport tmux --address "$retry_pane" --socket "$SOCKET" --ttl-seconds 300 >/dev/null
output="$(SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS=100 coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane retry-auto --kind request \
  --message 'control service must submit without human enter')"
retry_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
grep -q "WAKE_PENDING message_id=$retry_message .*state=awaiting-start" <<< "$output" || \
  fail 'first ignored submit did not remain pending'
wait_for_text "$RETRY_LOG" "$retry_message" || fail 'retry prompt was not inserted'
retry_prompt_count_before="$(grep -o "$retry_message" "$RETRY_LOG" | wc -l | tr -d ' ')"
sleep 1
output="$(SOUNIO_COORD_WAKE_RETRY_WAIT_MILLIS=800 coord "$REPO" wake-reconcile)"
grep -q '^WAKE_RECONCILE attempted=1 started=1 pending=0 ' <<< "$output" || \
  fail 'control-service reconciliation did not start the pending turn'
retry_prompt_count_after="$(grep -o "$retry_message" "$RETRY_LOG" | wc -l | tr -d ' ')"
[[ "$retry_prompt_count_after" == "$retry_prompt_count_before" ]] || \
  fail 'control-service retry duplicated the prompt'
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$retry_message")"
grep -q 'injected=1 .*wakes=1 wake_pending=0$' <<< "$output" || \
  fail 'automatic retry lacked a generation-bound start receipt'
coord "$SECOND" release --agent codex --lane retry-auto --reason 'automatic retry complete' >/dev/null
tmux -S "$SOCKET" kill-window -t recipient:retry-auto

SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$SECOND" send --agent codex \
  --lane legacy-recipient --to-agent sender --to-lane origin --kind info \
  --message 'establish a historical endpoint without registration' >/dev/null
legacy_secret='legacy-raw-message-must-not-be-injected'
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane legacy-recipient --kind request \
  --message "$legacy_secret")"
legacy_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$legacy_message" ]] || fail 'history-discovered send did not return a message id'
grep -q "^WAKE_STARTED message_id=$legacy_message .*discovery=history$" <<< "$output" || \
  fail 'stale client was not woken through its unique message history'
wait_for_text "$RECEIVER_LOG" "$legacy_message" || \
  fail 'history-discovered wake did not reach the recipient'
if grep -q "$legacy_secret" "$RECEIVER_LOG"; then
  fail 'history-discovered wake injected raw message content'
fi

session_id='6d7a2c7b-b721-447f-8f18-d8265889a6b7'
session_lane="session-${session_id:0:24}"
mkdir -p "$HISTORY_HOME/.claude/projects/physical-session"
printf '{"cwd":"%s","sessionId":"%s"}\n' "$SECOND" "$session_id" > \
  "$HISTORY_HOME/.claude/projects/physical-session/$session_id.jsonl"
SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$BACKGROUND" send --agent claude \
  --lane "$session_lane" --to-agent sender --to-lane origin --kind info \
  --message 'ownership is in a background worktree' >/dev/null
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent claude --to-lane "$session_lane" --kind request \
  --message 'wake the physical session, not its background claim')"
session_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$session_message" ]] || fail 'session-history send did not return a message id'
grep -q "^WAKE_STARTED message_id=$session_message .*address=$claude_pane .*discovery=session-history$" \
  <<< "$output" || fail 'session history did not bridge the physical and ownership worktrees'
wait_for_text "$CLAUDE_LOG" "$session_message" || \
  fail 'session-history wake did not reach the physical harness pane'

coord "$SECOND" claim --agent claude --lane "$session_lane" \
  --intent 'expired session endpoint rediscovery sabotage' \
  --files expired-session-endpoint.test >/dev/null
coord "$SECOND" endpoint-register --agent claude --lane "$session_lane" \
  --harness claude --transport tmux --address "$claude_pane" --socket "$SOCKET" \
  --ttl-seconds 1 >/dev/null
sleep 2
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent claude --to-lane "$session_lane" --kind request \
  --message 'expired endpoint must rediscover the verified physical session')"
expired_session_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$expired_session_message" ]] || fail 'expired session endpoint send returned no message id'
grep -q "^WAKE_STARTED message_id=$expired_session_message .*address=$claude_pane .*discovery=session-history$" \
  <<< "$output" || fail 'expired endpoint blocked verified session-history rediscovery'
wait_for_text "$CLAUDE_LOG" "$expired_session_message" || \
  fail 'expired endpoint rediscovery did not reach the physical harness pane'
coord "$SECOND" release --agent claude --lane "$session_lane" \
  --reason 'expired endpoint rediscovery sabotage complete' >/dev/null

SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$SECOND" send --agent grok-cli2 \
  --lane legacy-grok --to-agent sender --to-lane origin --kind info \
  --message 'establish a historical Grok endpoint' >/dev/null
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent grok-cli2 --to-lane legacy-grok --kind request \
  --message 'wake the Grok-compatible lane')"
grok_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$grok_message" ]] || fail 'cross-harness send did not return a message id'
grep -q "^WAKE_STARTED message_id=$grok_message .*address=$grok_pane .*discovery=identity-root$" \
  <<< "$output" || fail 'Grok lane was not woken through verified history'

tmux -S "$SOCKET" new-window -d -t recipient -n grok-duplicate -c "$GROK_HOME" \
  "$GROK_BIN '$RECEIVER' '$TEST_ROOT/grok-duplicate.log' '$RUNTIME' '$STATE' '$HISTORY_HOME' no-auto"
SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$SECOND" send --agent grok-cli2 \
  --lane ambiguous-grok --to-agent sender --to-lane origin --kind info \
  --message 'establish an ambiguous Grok identity' >/dev/null
output="$(SOUNIO_COORD_DISCOVERY_SOCKET="$SOCKET" coord "$REPO" send --agent sender \
  --lane origin --to-agent grok-cli2 --to-lane ambiguous-grok --kind info \
  --message 'must refuse two matching Grok panes')"
grep -q '^WAKE_UNAVAILABLE .*status=unavailable$' <<< "$output" || \
  fail 'identity-root discovery accepted two matching harness panes'
tmux -S "$SOCKET" kill-window -t recipient:grok-duplicate

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

# A hook from a successor process may see the durable message, but it cannot
# confirm a submission bound to the dead predecessor generation.
coord "$SECOND" claim --agent codex --lane stale-generation \
  --intent 'generation-bound start sabotage' --files stale-generation.test >/dev/null
tmux -S "$SOCKET" new-window -d -t recipient -n stale-generation -c "$SECOND" \
  "node '$RECEIVER' '$TEST_ROOT/stale-generation.log' '$RUNTIME' '$STATE' '$HISTORY_HOME' no-auto"
stale_pane="$(tmux -S "$SOCKET" display-message -p -t recipient:stale-generation '#{pane_id}')"
coord "$SECOND" endpoint-register --agent codex --lane stale-generation --harness codex \
  --transport tmux --address "$stale_pane" --socket "$SOCKET" --ttl-seconds 300 >/dev/null
output="$(SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS=100 coord "$REPO" send --agent sender \
  --lane origin --to-agent codex --to-lane stale-generation --kind request \
  --message 'successor hook cannot confirm predecessor submission')"
stale_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
grep -q "WAKE_PENDING message_id=$stale_message" <<< "$output" || \
  fail 'generation sabotage did not create a pending predecessor submission'
tmux -S "$SOCKET" respawn-pane -k -t "$stale_pane" -c "$SECOND" \
  "node '$RECEIVER' '$TEST_ROOT/stale-successor.log' '$RUNTIME' '$STATE' '$HISTORY_HOME' no-auto"
sleep 0.2
coord "$SECOND" endpoint-register --agent codex --lane stale-generation --harness codex \
  --transport tmux --address "$stale_pane" --socket "$SOCKET" --ttl-seconds 300 >/dev/null
output="$(coord "$SECOND" injected --agent codex --lane stale-generation \
  --messages "$stale_message")"
if grep -q '^WAKE_STARTED ' <<< "$output"; then
  fail 'successor generation confirmed a predecessor submission'
fi
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$stale_message")"
grep -q 'injected=1 .*wakes=0 wake_pending=1$' <<< "$output" || \
  fail 'generation sabotage fabricated or discarded the predecessor state'
sleep 1
output="$(SOUNIO_COORD_WAKE_RETRY_WAIT_MILLIS=100 coord "$REPO" wake-reconcile)"
grep -q '^WAKE_RECONCILE attempted=1 started=0 pending=1 ' <<< "$output" || \
  fail 'successor generation did not receive a fresh pending submission'
wait_for_text "$TEST_ROOT/stale-successor.log" "$stale_message" || \
  fail "successor generation did not receive the durable wake: reconcile=$output pane=$(tmux -S "$SOCKET" capture-pane -p -J -t "$stale_pane")"
output="$(coord "$SECOND" injected --agent codex --lane stale-generation \
  --messages "$stale_message")"
grep -q "^WAKE_STARTED message_id=$stale_message .*generation=" <<< "$output" || \
  fail 'successor hook did not confirm its own generation submission'
output="$(coord "$REPO" message-status --agent sender --lane origin --message "$stale_message")"
grep -q 'injected=1 .*wakes=1 wake_pending=0$' <<< "$output" || \
  fail 'successor start left a predecessor pending marker'
coord "$SECOND" release --agent codex --lane stale-generation \
  --reason 'generation sabotage complete' >/dev/null
tmux -S "$SOCKET" kill-window -t recipient:stale-generation

echo 'sounio-coord-wake-selftest: PASS'
