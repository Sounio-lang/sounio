#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-crash-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/supervisor-worktree"
TMP_ROOT="$TEST_ROOT/tmp"
HISTORY_HOME="$TEST_ROOT/history-home"
LOOM_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
TEST_LOOM_RUNTIME="$REPO/tools/loom/_build/default/src/loom.exe"
LOOM_OBLIGATION_ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-obligation-runtime"
first_pid=''
second_pid=''
third_pid=''

cleanup() {
  for pid in "$first_pid" "$second_pid" "$third_pid"; do
    if [[ -n "$pid" ]]; then
      kill -9 "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-coord-crash-recovery-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
[[ -x "$LOOM_RUNTIME" ]] || fail "Loom runtime was not built: $LOOM_RUNTIME"
[[ -x "$LOOM_OBLIGATION_ADAPTER" ]] || \
  fail "Loom obligation adapter was not built: $LOOM_OBLIGATION_ADAPTER"
export SOUNIO_COORD_LOOM_RUNTIME="$LOOM_RUNTIME"
export SOUNIO_LOOM_OBLIGATION_ADAPTER="$LOOM_OBLIGATION_ADAPTER"
export SOUNIO_LOOM_HOOK_TEST_MODE=1
export SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$ROOT_DIR"
export SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT="$ROOT_DIR"
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-language-authority-runtime"
export SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-cutover"
export SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$ROOT_DIR/.codex/hooks.json"

process_start() {
  local pid="$1" tail
  tail="$(sed 's/^[^)]*) //' "/proc/$pid/stat")"
  awk '{print $20}' <<< "$tail"
}

spawn_process() {
  sleep 600 &
  SPAWNED_PID=$!
}

run_legacy() {
  (
    cd "$REPO"
    SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_DIR="$LEGACY_STATE" \
      bin/sounio-coord "$@"
  )
}

run_coord() {
  (
    cd "$REPO"
    env -u TMUX -u TMUX_PANE TMPDIR="$TMP_ROOT" \
      SOUNIO_COORD_HISTORY_HOME="$HISTORY_HOME" \
      SOUNIO_COORD_RUNTIME_MODE=local bin/sounio-coord "$@"
  )
}

run_hook() {
  local payload="$1"
  local hook_runtime="$TEST_LOOM_RUNTIME"
  [[ "$payload" != *"$SECOND"* ]] || \
    hook_runtime="$SECOND/tools/loom/_build/default/src/loom.exe"
  printf '%s\n' "$payload" | \
    env -u TMUX -u TMUX_PANE TMPDIR="$(dirname "$TEST_ROOT")" \
      SOUNIO_COORD_HISTORY_HOME="$HISTORY_HOME" \
      SOUNIO_COORD_DIR="$DURABLE_STATE" \
      SOUNIO_COORD_RUNTIME_MODE=local \
      SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
      "$hook_runtime" agent-hook --agent codex
}

mkdir -p "$REPO/bin" "$REPO/scripts/dev" "$TMP_ROOT" \
  "$REPO/tools/loom/_build/default/src"
cp "$ROOT_DIR/bin/sounio-coord" "$REPO/bin/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$REPO/scripts/dev/"
cp "$LOOM_RUNTIME" "$TEST_LOOM_RUNTIME"
chmod +x "$REPO/bin/sounio-coord" "$REPO/scripts/dev/"*
printf 'crash recovery marker\n' > "$REPO/marker.txt"
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Crash Recovery Selftest'
git -C "$REPO" config user.email 'coord-crash-selftest@sounio.local'
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b supervisor-lane "$SECOND"
rm "$SECOND/tools/loom/_build/default/src/loom.exe"
ln -s "$TEST_LOOM_RUNTIME" "$SECOND/tools/loom/_build/default/src/loom.exe"

common_dir="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)"
repo_key="$(printf '%s' "$common_dir" | cksum | awk '{print $1}')"
LEGACY_STATE="$TMP_ROOT/sounio-coord/$repo_key"
DURABLE_STATE="$common_dir/sounio-coord-state"

run_legacy claim --agent codex --lane crash-lane --intent 'survive fleet loss' \
  --ttl-seconds 600 --files marker.txt >/dev/null
run_legacy send --agent sender --lane control --to-agent codex --to-lane crash-lane \
  --kind request --message 'pending work survives the crash' >/dev/null
[[ -d "$LEGACY_STATE" ]] || fail 'legacy state was not seeded'

output="$(run_coord recover --agent codex --lane crash-lane)"
grep -Fq "state_dir=$DURABLE_STATE" <<< "$output" || \
  fail 'runtime did not switch to the Git-common durable state'
[[ -d "$DURABLE_STATE" ]] || fail 'durable state directory was not created'
[[ -L "$LEGACY_STATE" ]] || fail 'legacy state path was not fenced with an alias'
[[ "$(readlink -f "$LEGACY_STATE")" == "$DURABLE_STATE" ]] || \
  fail 'legacy state alias does not target the durable state'
grep -q '^pending_directed=1$' <<< "$output" || \
  fail 'pending message was lost during state migration'

legacy_session_id='deadbeef-1234-4567-89ab-0123456789ab'
legacy_lane="session-${legacy_session_id:0:24}"
mkdir -p "$HISTORY_HOME/.claude/projects/test-project"
printf '{"cwd":"%s","sessionId":"%s"}\n' "$REPO" "$legacy_session_id" > \
  "$HISTORY_HOME/.claude/projects/test-project/$legacy_session_id.jsonl"
run_coord scope --agent claude --lane "$legacy_lane" --intent 'legacy recovery witness' >/dev/null
output="$(run_coord recover --agent claude --lane "$legacy_lane")"
grep -q '^lane_state=legacy-recoverable$' <<< "$output" || \
  fail 'legacy session history did not become recoverable'
grep -q "^resume_session_id=$legacy_session_id$" <<< "$output" || \
  fail 'legacy recovery did not restore the complete session id'
grep -q "^session_worktree=$REPO$" <<< "$output" || \
  fail 'legacy recovery did not restore the physical session worktree'
run_coord release --agent claude --lane "$legacy_lane" --reason 'legacy witness complete' >/dev/null

cross_session_id='feedface-1234-4567-89ab-0123456789ab'
cross_lane="session-${cross_session_id:0:24}"
run_coord scope --agent codex --lane "$cross_lane" --intent 'cross-worktree session witness' >/dev/null
run_hook "{\"session_id\":\"$cross_session_id\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionStart\"}" \
  >"$TEST_ROOT/cross-session-start.log"
output="$(run_coord recover --agent codex --lane "$cross_lane")"
grep -q '^lane_state=live$' <<< "$output" || \
  fail 'physical session did not bind across an attached claim worktree'
grep -q "^worktree=$REPO$" <<< "$output" || \
  fail 'cross-worktree recovery lost the ownership worktree'
grep -q "^session_worktree=$SECOND$" <<< "$output" || \
  fail 'cross-worktree recovery lost the physical session worktree'
run_coord release --agent codex --lane "$cross_lane" --reason 'cross-worktree witness complete' >/dev/null
output="$(run_coord recover --agent codex --lane "$cross_lane")"
grep -q '^lane_state=missing$' <<< "$output" || \
  fail 'ownership release left cross-worktree process presence behind'

boot_id="$(cat /proc/sys/kernel/random/boot_id)"
pid_namespace="$(readlink /proc/self/ns/pid)"
host="$(uname -n)"
spawn_process
first_pid="$SPAWNED_PID"
first_start="$(process_start "$first_pid")"
output="$(run_coord presence-register --agent codex --lane crash-lane --harness codex \
  --session-id 01a02a17-f139-7613-98a1-full-session-id \
  --pid "$first_pid" --pid-start "$first_start" --boot-id "$boot_id" \
  --pid-namespace "$pid_namespace" --host "$host" --ttl-seconds 600)"
grep -q '^PRESENCE_REGISTERED .*generation=1 ' <<< "$output" || \
  fail 'first process generation was not registered'
output="$(run_coord recover --agent codex --lane crash-lane)"
grep -q '^lane_state=live$' <<< "$output" || fail 'verified process was not live'
grep -q '^delivery_state=unavailable$' <<< "$output" || \
  fail 'recovery unexpectedly depended on a tmux endpoint'
grep -q '^resume_session_id=01a02a17-f139-7613-98a1-full-session-id$' <<< "$output" || \
  fail 'full resume identity was not retained'

kill -9 "$first_pid"
wait "$first_pid" 2>/dev/null || true
first_pid=''
output="$(run_coord recover --agent codex --lane crash-lane)"
grep -q '^lane_state=orphaned$' <<< "$output" || \
  fail 'abrupt process death did not orphan the lane'
grep -q '^presence_reason=process-missing$' <<< "$output" || \
  fail 'crash reason was not classified precisely'
grep -q '^pending_directed=1$' <<< "$output" || \
  fail 'pending message disappeared after process death'

spawn_process
second_pid="$SPAWNED_PID"
second_start="$(process_start "$second_pid")"
output="$(run_coord presence-register --agent codex --lane crash-lane --harness codex \
  --session-id 01a02a17-f139-7613-98a1-full-session-id \
  --pid "$second_pid" --pid-start "$second_start" --boot-id "$boot_id" \
  --pid-namespace "$pid_namespace" --host "$host" --ttl-seconds 600)"
grep -q '^PRESENCE_RECOVERED .*generation=2 ' <<< "$output" || \
  fail 'replacement process did not advance the recovery generation'

# Sabotage control: a second live process must be unable to steal the lane.
spawn_process
third_pid="$SPAWNED_PID"
third_start="$(process_start "$third_pid")"
if output="$(run_coord presence-register --agent codex --lane crash-lane --harness codex \
  --session-id malicious-takeover --pid "$third_pid" --pid-start "$third_start" \
  --boot-id "$boot_id" --pid-namespace "$pid_namespace" --host "$host" \
  --ttl-seconds 600 2>&1)"; then
  fail 'live-generation sabotage stole the lane'
fi
grep -q 'lane is still bound to generation 2' <<< "$output" || \
  fail 'live-generation sabotage was refused by the wrong rule'
output="$(run_coord recover --agent codex --lane crash-lane)"
grep -q '^lane_state=live$' <<< "$output" || \
  fail 'sabotage changed the recovered lane state'
grep -q '^presence_generation=2$' <<< "$output" || \
  fail 'sabotage changed the process generation'

output="$(run_coord recover --all)"
grep -q 'LANE_RECOVERY agent=codex lane=crash-lane state=live ' <<< "$output" || \
  fail 'fleet recovery omitted the recovered lane'
grep -q '^fleet_lanes=1$' <<< "$output" || fail 'fleet recovery count is wrong'

# The hook must carry the generation fence into the structured write path.
hook_session_id='hook-crash-recovery-session-abcdef0123456789'
hook_lane="session-${hook_session_id:0:24}"
run_hook "{\"session_id\":\"$hook_session_id\",\"cwd\":\"$REPO\",\"hook_event_name\":\"SessionStart\"}" \
  >"$TEST_ROOT/hook-start.log"
output="$(run_coord recover --agent codex --lane "$hook_lane")"
grep -q '^lane_state=live$' <<< "$output" || fail 'hook did not bind its process generation'
grep -q "^resume_session_id=$hook_session_id$" <<< "$output" || \
  fail 'hook truncated the durable resume session id'
set +e
hook_sabotage="$(run_hook \
  "{\"session_id\":\"$hook_session_id\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"marker.txt\"}}" 2>&1)"
hook_rc=$?
set -e
[[ "$hook_rc" -eq 2 ]] || fail "hook generation sabotage returned $hook_rc instead of 2"
grep -q 'lane is still bound to generation 1' <<< "$hook_sabotage" || \
  fail "structured write sabotage was refused by the wrong rule: $hook_sabotage"
run_hook "{\"session_id\":\"$hook_session_id\",\"cwd\":\"$REPO\",\"hook_event_name\":\"SessionEnd\"}" \
  >"$TEST_ROOT/hook-end.log"

printf 'sounio-coord-crash-recovery-selftest: PASS migration=1 legacy_recovery=1 cross_worktree=1 crash=orphaned recovery_generation=2 sabotage_control=1 hook_fence=1 tmux=absent\n'
