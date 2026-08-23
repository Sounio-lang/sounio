#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-fleet-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
RUNTIME="$TEST_ROOT/runtime"
STATE="$TEST_ROOT/agentd-state"
HOME_ROOT="$TEST_ROOT/home"
FAKE_BIN="$TEST_ROOT/bin"
RECEIVER="$TEST_ROOT/receiver.py"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
TMUX_SOCKET="$TEST_ROOT/tmux.sock"
SESSION_ID='01a02a17-f139-7613-98a1-76a7d516f4d7'
LANE='session-01a02a17-f139-7613-98a1-'
SLOT='codex-9'

cleanup() {
  tmux -S "$TMUX_SOCKET" kill-server >/dev/null 2>&1 || true
  if [[ -x "$RUNTIME/sounio-fleet-agent-runtime" && -d "$REPO" ]]; then
    SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot "$SLOT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-coord-fleet-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

wait_for() {
  local description="$1" command="$2" attempt
  for attempt in $(seq 1 100); do
    if eval "$command"; then return 0; fi
    sleep 0.1
  done
  fail "$description"
}

mkdir -p "$REPO" "$RUNTIME" "$HOME_ROOT" "$FAKE_BIN"
git -C "$REPO" init -q
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" \
  "$RUNTIME/sounio-agentd-runtime"
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" \
  "$RUNTIME/sounio-fleet-agent-runtime"

cat > "$RECEIVER" <<'PY'
#!/usr/bin/env python3
import os
import sys

log = sys.argv[1]
with open(log, "a", encoding="utf-8") as handle:
    handle.write(f"START pid={os.getpid()}\n")
    handle.flush()
print("RECEIVER_READY", flush=True)
for line in sys.stdin:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"INPUT {line.rstrip()}\n")
        handle.flush()
PY
chmod +x "$RECEIVER"

cat > "$FAKE_BIN/codex" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat > "$FAKE_BIN/claude" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$FAKE_BIN/codex" "$FAKE_BIN/claude"

fleet() {
  SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" "$@"
}

launch_command="SOUNIO_AGENTD_DIR=$STATE $RUNTIME/sounio-fleet-agent-runtime launch --slot $SLOT --agent codex --session-id $SESSION_ID --identity exact --cwd $REPO -- $RECEIVER $RECEIVER_LOG"
tmux -S "$TMUX_SOCKET" new-session -d -s fleet "$launch_command"
wait_for 'supervised receiver did not start' \
  "grep -q '^START pid=' '$RECEIVER_LOG' 2>/dev/null"
wait_for 'first tmux client did not attach' \
  "fleet status --cwd '$REPO' --slot '$SLOT' 2>/dev/null | grep -q 'state=active.*attached_clients=1'"

status="$(fleet status --cwd "$REPO" --slot "$SLOT")"
harness_pid="$(sed -n 's/.*harness_pid=\([0-9][0-9]*\).*/\1/p' <<< "$status")"
[[ -n "$harness_pid" ]] || fail 'status omitted the harness pid'

tmux -S "$TMUX_SOCKET" kill-server
wait_for 'agent did not survive tmux server loss' \
  "fleet status --cwd '$REPO' --slot '$SLOT' 2>/dev/null | grep -q 'state=active.*attached_clients=0'"
kill -0 "$harness_pid" 2>/dev/null || fail 'harness pid died with tmux'

mapping="$STATE/fleet-slots/$SLOT.json"
cp "$mapping" "$mapping.good"
python3 - "$mapping" <<'PY'
import json
import sys

path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["instance_id"] = "sabotaged-generation"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True)
    handle.write("\n")
PY
if fleet status --cwd "$REPO" --slot "$SLOT" >"$TEST_ROOT/drift-status" 2>&1; then
  fail 'status accepted a sabotaged slot generation'
fi
grep -q 'state=drifted' "$TEST_ROOT/drift-status" || \
  fail 'status did not classify the sabotaged generation as drifted'
if fleet launch --slot "$SLOT" --agent codex --session-id "$SESSION_ID" \
  --identity exact --cwd "$REPO" --no-attach -- "$RECEIVER" "$RECEIVER_LOG" \
  >"$TEST_ROOT/drift-launch" 2>&1; then
  fail 'launcher replaced a sabotaged live slot generation'
fi
grep -q 'identity drifted; refusing' "$TEST_ROOT/drift-launch" || \
  fail 'launcher did not fail closed on generation drift'
mv "$mapping.good" "$mapping"
kill -0 "$harness_pid" 2>/dev/null || fail 'sabotage control disturbed the live harness'

tmux -S "$TMUX_SOCKET" new-session -d -s fleet "$launch_command"
wait_for 'replacement tmux client did not reattach' \
  "fleet status --cwd '$REPO' --slot '$SLOT' 2>/dev/null | grep -q 'state=active.*attached_clients=1'"
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'reattach launched a duplicate harness'

fleet_status="$(fleet status --cwd "$REPO" --slot "$SLOT")"
grep -q "instance_id=$(python3 -c 'import json; print(json.load(open("'"$mapping"'"))["instance_id"])')" \
  <<< "$fleet_status" || fail 'reattach changed the supervisor generation'

SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-agentd-runtime" wake \
  --agent codex --lane "$LANE" --session-id "$SESSION_ID" \
  --cwd "$REPO" --message-id msg-fleet-selftest \
  --prompt 'FLEET_WAKE_SELFTEST' >/dev/null
wait_for 'reattached harness did not receive a wake' \
  "grep -q 'INPUT FLEET_WAKE_SELFTEST' '$RECEIVER_LOG'"

mkdir -p "$HOME_ROOT/.codex/sessions/2026/08/23"
PATH="$FAKE_BIN:$PATH" fleet plan-kind --slot codex-new --kind codex \
  --home "$TEST_ROOT/empty-home" > "$TEST_ROOT/codex-bootstrap"
grep -q '^identity=bootstrap$' "$TEST_ROOT/codex-bootstrap" || \
  fail 'fresh Codex plan did not expose bootstrap identity'

printf '{"type":"session_meta","payload":{"id":"%s","cwd":"%s","originator":"codex-tui","source":"cli"}}\n' \
  "$SESSION_ID" "$REPO" > \
  "$HOME_ROOT/.codex/sessions/2026/08/23/rollout-2026-08-23T00-00-00-$SESSION_ID.jsonl"
PATH="$FAKE_BIN:$PATH" fleet plan-kind --slot codex-existing --kind codex \
  --home "$HOME_ROOT" --cwd "$REPO" > "$TEST_ROOT/codex-exact"
grep -q '^identity=exact$' "$TEST_ROOT/codex-exact" || \
  fail 'persisted Codex plan was not exact'
grep -q "resume $SESSION_ID" "$TEST_ROOT/codex-exact" || \
  fail 'persisted Codex plan did not resume the exact UUID'

PATH="$FAKE_BIN:$PATH" fleet plan-kind --slot claude-new --kind claude \
  --home "$TEST_ROOT/empty-home" --cwd "$REPO" > "$TEST_ROOT/claude-exact"
grep -q '^identity=exact$' "$TEST_ROOT/claude-exact" || \
  fail 'fresh Claude plan was not exact'
grep -q -- '--session-id' "$TEST_ROOT/claude-exact" || \
  fail 'fresh Claude plan did not predeclare its UUID'

CLAUDE_SESSION_ID='c89fe8c8-7421-42c6-9321-3bc29cef07d3'
claude_project="$(printf '%s' "$REPO" | sed 's/[^A-Za-z0-9_-]/-/g')"
mkdir -p "$HOME_ROOT/.claude/projects/$claude_project" \
  "$HOME_ROOT/.claude/projects/-unrelated-worktree"
printf '{}\n' > "$HOME_ROOT/.claude/projects/$claude_project/$CLAUDE_SESSION_ID.jsonl"
printf '{}\n' > \
  "$HOME_ROOT/.claude/projects/-unrelated-worktree/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.jsonl"
touch -d 'next hour' \
  "$HOME_ROOT/.claude/projects/-unrelated-worktree/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.jsonl"
PATH="$FAKE_BIN:$PATH" fleet plan-kind --slot claude-existing --kind claude \
  --home "$HOME_ROOT" --cwd "$REPO" > "$TEST_ROOT/claude-project-exact"
grep -q "resume $CLAUDE_SESSION_ID" "$TEST_ROOT/claude-project-exact" || \
  fail 'Claude plan crossed project boundaries while resolving the latest session'

tmux -S "$TMUX_SOCKET" kill-server
fleet stop --cwd "$REPO" --slot "$SLOT" >/dev/null
[[ ! -e "$mapping" ]] || fail 'stop left the slot mapping behind'

echo 'sounio-coord-fleet-selftest: PASS tmux_crash=survived reattach=same-generation duplicate_harness=refused generation_sabotage=refused claude_identity=project-exact codex_resume=exact codex_fresh=bootstrap'
