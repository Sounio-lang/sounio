#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-agentd-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
CONTEXT="$TEST_ROOT/context-worktree"
COORD_STATE="$TEST_ROOT/coord-state"
AGENTD_STATE="$TEST_ROOT/agentd-state"
RECEIVER="$TEST_ROOT/receiver.js"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
TMUX_SOCKET="$TEST_ROOT/tmux.sock"
SESSION_ID='c89fe8c8-7421-42c6-9321-agentd-selftest'
LANE='session-c89fe8c8-7421-42c6-9321-'

cleanup() {
  if [[ -d "$REPO" ]]; then
    (
      cd "$REPO"
      SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_AGENTD_DIR="$AGENTD_STATE" \
        bin/sounio-agentd stop --agent codex --lane "$LANE" --cwd "$REPO" \
        >/dev/null 2>&1 || true
    )
  fi
  tmux -S "$TMUX_SOCKET" kill-server >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$CONTEXT" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-coord-agentd-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

wait_for_text() {
  local file="$1" pattern="$2" attempt
  for attempt in $(seq 1 80); do
    [[ -f "$file" ]] && grep -q "$pattern" "$file" && return 0
    sleep 0.1
  done
  return 1
}

mkdir -p "$REPO/bin" "$REPO/scripts/dev"
cp "$ROOT_DIR/bin/sounio-coord" "$ROOT_DIR/bin/sounio-agentd" "$REPO/bin/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$REPO/scripts/dev/"
chmod +x "$REPO/bin/"* "$REPO/scripts/dev/"*
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Agentd Selftest'
git -C "$REPO" config user.email 'coord-agentd-selftest@sounio.local'
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b context-lane "$CONTEXT"

cat > "$RECEIVER" <<'JS'
const fs = require("fs");
const { spawnSync } = require("child_process");
const [repo, context, state, log, sessionId] = process.argv.slice(2);
const event = JSON.stringify({
  session_id: sessionId,
  cwd: context,
  hook_event_name: "SessionStart",
});
const hook = spawnSync(
  "python3",
  [`${repo}/scripts/dev/sounio_coord_agent_hook.py`, "--agent", "codex"],
  {
    cwd: repo,
    env: { ...process.env, SOUNIO_COORD_DIR: state, SOUNIO_COORD_RUNTIME_MODE: "local" },
    input: `${event}\n`,
    encoding: "utf8",
  },
);
fs.appendFileSync(log, `HOOK_RC=${hook.status}\n${hook.stdout}${hook.stderr}`);
process.stdin.on("data", (chunk) => fs.appendFileSync(log, chunk));
process.stdin.resume();
setInterval(() => {}, 1000);
JS

coord() {
  (
    cd "$REPO"
    SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_DIR="$COORD_STATE" \
      bin/sounio-coord "$@"
  )
}

agentd() {
  (
    cd "$REPO"
    SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_AGENTD_DIR="$AGENTD_STATE" \
      bin/sounio-agentd "$@"
  )
}

start_output="$(
  cd "$REPO"
  SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_DIR="$COORD_STATE" \
    SOUNIO_AGENTD_DIR="$AGENTD_STATE" \
    bin/sounio-agentd start --agent codex --lane "$LANE" --session-id "$SESSION_ID" \
      --cwd "$REPO" -- node "$RECEIVER" "$REPO" "$CONTEXT" "$COORD_STATE" \
      "$RECEIVER_LOG" "$SESSION_ID"
)"
grep -q '^AGENTD_STARTED ' <<< "$start_output" || fail 'supervisor did not start'
socket_path="$(sed -n 's/.* socket=\([^ ]*\).*/\1/p' <<< "$start_output")"
token_file="$(sed -n 's/.* token_file=\([^ ]*\).*/\1/p' <<< "$start_output")"
[[ -S "$socket_path" && -f "$token_file" ]] || fail 'supervisor omitted its protected endpoint'
[[ "$(stat -c %a "$token_file")" == 600 ]] || fail 'capability file mode is not 600'
output="$(agentd status --agent codex --lane "$LANE" --cwd "$REPO")"
argv_digest="$(python3 - "node" "$RECEIVER" "$REPO" "$CONTEXT" \
  "$COORD_STATE" "$RECEIVER_LOG" "$SESSION_ID" <<'PY'
import hashlib
import json
import sys
print(hashlib.sha256(json.dumps(sys.argv[1:], separators=(",", ":")).encode()).hexdigest())
PY
)"
grep -q "^argv_digest=$argv_digest$" <<< "$output" || \
  fail 'supervisor did not attest the complete argv vector'
if agentd start --agent codex --lane "$LANE" --session-id 'different-live-generation' \
  --cwd "$REPO" -- node "$RECEIVER" "$REPO" "$CONTEXT" "$COORD_STATE" \
  "$RECEIVER_LOG" 'different-live-generation' >/dev/null 2>&1; then
  fail 'start aliased a different UUID onto the live supervisor generation'
fi
if agentd start --agent codex --lane "$LANE" --session-id "$SESSION_ID" \
  --cwd "$REPO" -- node "$RECEIVER" "$REPO" "$CONTEXT" "$COORD_STATE" \
  "$RECEIVER_LOG" 'sabotaged-argv' >/dev/null 2>&1; then
  fail 'start reused a live generation with a different argv vector'
fi
wait_for_text "$RECEIVER_LOG" 'HOOK_RC=0' || fail 'harness hook did not start'
wait_for_text "$RECEIVER_LOG" "agent=codex lane=$LANE" || fail 'hook did not claim the supervised lane'
output="$(agentd list --cwd "$REPO")"
grep -q "^AGENTD_SESSION state=active agent=codex lane=$LANE " <<< "$output" || \
  fail 'supervisor fleet listing omitted the live session'

output="$(coord endpoint-status --agent codex --lane "$LANE")"
grep -q '^ENDPOINT_STATUS .* state=active .* transport=agentd ' <<< "$output" || \
  fail 'hook did not prefer the agentd endpoint'
grep -q " session_id=$SESSION_ID " <<< "$output" || fail 'endpoint lost the full session identity'
output="$(coord recover --agent codex --lane "$LANE")"
grep -q "^session_worktree=$REPO$" <<< "$output" || \
  fail 'recovery lost the physical supervisor worktree'
grep -q "^worktree=$CONTEXT$" <<< "$output" || \
  fail 'recovery collapsed context ownership into process presence'

coord claim --agent sender --lane origin --intent 'agentd sender' --files sender.test >/dev/null
secret='agentd-raw-message-must-not-enter-the-pty'
output="$(coord send --agent sender --lane origin --to-agent codex --to-lane "$LANE" \
  --kind request --message "$secret")"
message_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$message_id" ]] || fail 'agentd send did not persist a message'
grep -q "^WAKE_DELIVERED message_id=$message_id .*transport=agentd " <<< "$output" || \
  fail 'registered agentd endpoint did not receive the wake'
wait_for_text "$RECEIVER_LOG" "$message_id" || fail 'agentd wake did not reach the harness PTY'
if grep -q "$secret" "$RECEIVER_LOG"; then
  fail 'agentd wake injected the raw message body'
fi

original_token="$(<"$token_file")"
printf '%064d\n' 0 > "$token_file"
chmod 600 "$token_file"
output="$(coord send --agent sender --lane origin --to-agent codex --to-lane "$LANE" \
  --kind info --message 'capability drift sabotage' 2>&1)"
drift_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$drift_message" ]] || fail 'capability sabotage did not preserve the bus message'
grep -q "WAKE_REFUSED message_id=$drift_message .*reason=endpoint-drift" <<< "$output" || \
  fail 'capability drift did not fail closed'
grep -q "WAKE_UNAVAILABLE message_id=$drift_message status=failed-closed" <<< "$output" || \
  fail 'sender did not observe failed-closed capability drift'
printf '%s\n' "$original_token" > "$token_file"
chmod 600 "$token_file"
output="$(coord inbox --agent codex --lane "$LANE" --all)"
grep -q "MESSAGE id=$drift_message " <<< "$output" || \
  fail 'capability drift removed the durable bus message'

coord scope --agent codex --lane impostor --intent 'identity sabotage' \
  --resources api:agentd-impostor >/dev/null
if coord endpoint-register --agent codex --lane impostor --harness codex \
  --transport agentd --address "$socket_path" --socket "$socket_path" \
  --token-file "$token_file" >/dev/null 2>&1; then
  fail 'a second lane stole the live supervisor endpoint'
fi
coord release --agent codex --lane impostor --reason 'identity sabotage complete' >/dev/null

wrong_token="$TEST_ROOT/wrong-token"
printf '%064d\n' 0 > "$wrong_token"
chmod 600 "$wrong_token"
if agentd status --agent codex --lane "$LANE" --socket "$socket_path" \
  --token-file "$wrong_token" >/dev/null 2>&1; then
  fail 'supervisor accepted the wrong capability'
fi

if command -v tmux >/dev/null 2>&1; then
  tmux -S "$TMUX_SOCKET" new-session -d -s agentd-viewer -c "$REPO" \
    "SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_AGENTD_DIR='$AGENTD_STATE' \
      '$REPO/bin/sounio-agentd' attach --agent codex --lane '$LANE' --cwd '$REPO'"
  sleep 0.4
  viewer_status="$(agentd status --agent codex --lane "$LANE" --cwd "$REPO")"
  grep -q '^attached_clients=1$' <<< "$viewer_status" || \
    fail 'tmux viewer never attached to the detached supervisor'
  tmux -S "$TMUX_SOCKET" kill-server
  for _ in $(seq 1 40); do
    viewer_status="$(agentd status --agent codex --lane "$LANE" --cwd "$REPO")"
    grep -q '^attached_clients=0$' <<< "$viewer_status" && break
    sleep 0.1
  done
fi

status_output="$(agentd status --agent codex --lane "$LANE" --cwd "$REPO")"
grep -q '^state=active$' <<< "$status_output" || fail 'tmux crash killed the supervisor'
grep -q '^attached_clients=0$' <<< "$status_output" || fail 'dead tmux viewer remained attached'
harness_pid="$(sed -n 's/^harness_pid=//p' <<< "$status_output")"
kill -0 "$harness_pid" 2>/dev/null || fail 'tmux crash killed the harness process'

output="$(coord send --agent sender --lane origin --to-agent codex --to-lane "$LANE" \
  --kind info --message 'post-tmux-crash wake')"
post_crash_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
grep -q "^WAKE_DELIVERED message_id=$post_crash_message .*transport=agentd " <<< "$output" || \
  fail 'wake transport depended on the dead tmux server'
wait_for_text "$RECEIVER_LOG" "$post_crash_message" || \
  fail 'post-crash wake did not reach the surviving harness'

agentd stop --agent codex --lane "$LANE" --cwd "$REPO" >/dev/null
for _ in $(seq 1 50); do
  kill -0 "$harness_pid" 2>/dev/null || break
  sleep 0.1
done
kill -0 "$harness_pid" 2>/dev/null && fail 'supervisor stop left the harness alive'
(
  cd "$CONTEXT"
  SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_DIR="$COORD_STATE" \
    "$REPO/bin/sounio-coord" release --agent codex --lane "$LANE" \
    --reason 'agentd selftest complete' >/dev/null
)
coord release --agent sender --lane origin --reason 'agentd selftest complete' >/dev/null

"$ROOT_DIR/scripts/ci/sounio_coord_fleet_selftest.sh"
"$ROOT_DIR/scripts/ci/sounio_coord_fleetd_selftest.sh"
"$ROOT_DIR/scripts/ci/sounio_coord_fleet_model_selftest.sh"

echo 'sounio-coord-agentd-selftest: PASS tmux_crash=survived transport=agentd cross_worktree=1 generation_sabotage=refused capability_drift=failed-closed raw_body=absent'
