#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/sounio-loom"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-selftest.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
AGENT=loom-test
LANE=cursor-replay
GUI_PID=''
OBSERVER_ONE=''
OBSERVER_TWO=''
HELD_ATTACH=''
COORD_LOOM_ACTIVE=0
COORD_AGENT=codex
COORD_LANE=loom-transport

fail() {
  echo "sounio-loom-selftest: FAIL: $*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

loom_status() {
  "$LOOM" status --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE"
}

wait_status() {
  local expected="$1" output='' attempt
  for attempt in $(seq 1 100); do
    output="$(loom_status 2>/dev/null || true)"
    [[ "$output" == *"$expected"* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "status did not reach $expected; last=$output"
}

cleanup() {
  [[ -z "$HELD_ATTACH" ]] || kill "$HELD_ATTACH" 2>/dev/null || true
  [[ -z "$OBSERVER_ONE" ]] || kill "$OBSERVER_ONE" 2>/dev/null || true
  [[ -z "$OBSERVER_TWO" ]] || kill "$OBSERVER_TWO" 2>/dev/null || true
  [[ -z "$GUI_PID" ]] || kill "$GUI_PID" 2>/dev/null || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  if [[ "$COORD_LOOM_ACTIVE" == 1 ]]; then
    SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" stop --state-dir "$TEST_ROOT/coord-loom" \
      --cwd "$ROOT_DIR" --agent "$COORD_AGENT" --lane "$COORD_LANE" >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

cat > "$TEST_ROOT/harness.sh" <<'HARNESS'
#!/bin/sh
stty -echo
printf 'BOOT_READY\n'
while IFS= read -r line; do
  printf 'ECHO:%s\n' "$line"
done
HARNESS
chmod +x "$TEST_ROOT/harness.sh"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
version="$($LOOM runtime-version)"
grep -q '^language=OCaml$' <<< "$version" || fail 'runtime does not identify as OCaml'
grep -q '^protocol_version=1$' <<< "$version" || fail 'unexpected Loom protocol version'

"$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
  --session-id loom-selftest --cwd "$TEST_ROOT" -- /bin/sh "$TEST_ROOT/harness.sh" >/dev/null

initial="$(wait_status 'output_cursor=')"
daemon_pid="$(field daemon_pid "$initial")"
harness_pid="$(field harness_pid "$initial")"
instance_id="$(field instance_id "$initial")"
journal="$(field journal "$initial")"
[[ -n "$daemon_pid" && -n "$harness_pid" && -n "$instance_id" && -f "$journal" ]] || \
  fail 'initial status omitted generation identity'
harness_fds="$(find "/proc/$harness_pid/fd" -maxdepth 1 -type l -printf '%l\n' 2>/dev/null || true)"
if grep -F -e '/daemon.lock' -e '/journal.tsv' -e '/output.bin' <<< "$harness_fds" >/dev/null; then
  fail 'harness inherited a kernel-owned durable descriptor'
fi
if grep -q '^socket:' <<< "$harness_fds"; then
  fail 'harness inherited a kernel-owned control socket'
fi

boot=''
for _ in $(seq 1 100); do
  boot="$($LOOM snapshot --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" --cursor 0 2>/dev/null || true)"
  [[ "$boot" == *BOOT_READY* ]] && break
  sleep 0.05
done
[[ "$boot" == *BOOT_READY* ]] || fail 'durable output omitted boot witness'

timeout 5 "$LOOM" observe --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor end > "$TEST_ROOT/observer-one.out" 2>&1 &
OBSERVER_ONE=$!
timeout 5 "$LOOM" observe --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor end > "$TEST_ROOT/observer-two.out" 2>&1 &
OBSERVER_TWO=$!
observed="$(wait_status 'observer_clients=2')"
[[ "$(field daemon_pid "$observed")" == "$daemon_pid" ]] || fail 'observer attach replaced daemon generation'
[[ "$(field harness_pid "$observed")" == "$harness_pid" ]] || fail 'observer attach replaced harness generation'
kill "$OBSERVER_ONE" "$OBSERVER_TWO" 2>/dev/null || true
wait "$OBSERVER_ONE" "$OBSERVER_TWO" 2>/dev/null || true
OBSERVER_ONE=''
OBSERVER_TWO=''
after_observers="$(wait_status 'observer_clients=0')"
[[ "$(field harness_pid "$after_observers")" == "$harness_pid" ]] || fail 'observer crash killed harness'

tail -f /dev/null | "$LOOM" attach --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor end --no-raw > "$TEST_ROOT/held.out" 2>&1 &
HELD_ATTACH=$!
held="$(wait_status 'interactive_clients=1')"
[[ "$(field harness_pid "$held")" == "$harness_pid" ]] || fail 'input lease changed harness generation'
if "$LOOM" attach --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor end --no-raw </dev/null \
  > "$TEST_ROOT/refused.out" 2> "$TEST_ROOT/refused.err"; then
  fail 'second interactive client acquired the same input lease'
fi
grep -q 'interactive-client-active' "$TEST_ROOT/refused.err" || \
  fail 'second interactive client failed for the wrong reason'
kill "$HELD_ATTACH" 2>/dev/null || true
wait "$HELD_ATTACH" 2>/dev/null || true
HELD_ATTACH=''
wait_status 'interactive_clients=0' >/dev/null

before_input="$(loom_status)"
cursor="$(field output_cursor "$before_input")"
printf 'round-trip\n' | "$LOOM" attach --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor end --no-raw >/dev/null
replay=''
for _ in $(seq 1 100); do
  replay="$($LOOM snapshot --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" --cursor "$cursor" 2>/dev/null || true)"
  [[ "$replay" == *ECHO:round-trip* ]] && break
  sleep 0.05
done
[[ "$replay" == *ECHO:round-trip* ]] || fail 'cursor replay lost post-detach output'
after_input="$(loom_status)"
[[ "$(field daemon_pid "$after_input")" == "$daemon_pid" ]] || fail 'reconnect replaced daemon generation'
[[ "$(field harness_pid "$after_input")" == "$harness_pid" ]] || fail 'reconnect replaced harness generation'
[[ "$(field instance_id "$after_input")" == "$instance_id" ]] || fail 'reconnect changed generation identity'

descriptor="$STATE_DIR/sessions/$AGENT--$LANE/session.state"
socket="$(sed -n 's/^socket=//p' "$descriptor")"
token_file="$(sed -n 's/^token_file=//p' "$descriptor")"
machine="$($LOOM status --machine --agent "$AGENT" --lane "$LANE" --cwd "$TEST_ROOT" \
  --socket "$socket" --token-file "$token_file")"
grep -q "^instance_id=$instance_id$" <<< "$machine" || fail 'machine status lost generation identity'
grep -q '^harness_pid_start=[1-9][0-9]*$' <<< "$machine" || fail 'machine status omitted process birth identity'
wake_cursor="$(field output_cursor "$after_input")"
"$LOOM" wake --agent "$AGENT" --lane "$LANE" --cwd "$TEST_ROOT" \
  --socket "$socket" --token-file "$token_file" --session-id loom-selftest \
  --message-id msg-loom-selftest --prompt wake-round-trip >/dev/null
wake_replay=''
for _ in $(seq 1 100); do
  wake_replay="$($LOOM snapshot --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" --cursor "$wake_cursor" 2>/dev/null || true)"
  [[ "$wake_replay" == *ECHO:wake-round-trip* ]] && break
  sleep 0.05
done
[[ "$wake_replay" == *ECHO:wake-round-trip* ]] || fail 'authenticated wake did not reach the PTY'

"$LOOM" serve --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" --bind 127.0.0.1 --port 0 \
  > "$TEST_ROOT/gui.log" 2>&1 &
GUI_PID=$!
gui_url=''
for _ in $(seq 1 100); do
  gui_url="$(sed -n 's/^LOOM_GUI url=\([^ ]*\).*/\1/p' "$TEST_ROOT/gui.log")"
  [[ -n "$gui_url" ]] && break
  sleep 0.05
done
[[ -n "$gui_url" ]] || fail 'read-only GUI did not start'
curl -fsS "$gui_url/" | grep -q 'SOUNIO LOOM' || fail 'GUI did not serve its operational view'
sessions_json="$(curl -fsS "$gui_url/api/sessions")"
[[ "$sessions_json" == *"\"instance_id\":\"$instance_id\""* ]] || fail 'GUI observed the wrong generation'
curl -fsS "$gui_url/api/snapshot?agent=$AGENT&lane=$LANE&cursor=0" | grep -q 'BOOT_READY' || \
  fail 'GUI read path could not observe durable output'
kill "$GUI_PID"
wait "$GUI_PID" 2>/dev/null || true
GUI_PID=''
after_gui="$(loom_status)"
[[ "$(field daemon_pid "$after_gui")" == "$daemon_pid" ]] || fail 'GUI crash killed daemon'
[[ "$(field harness_pid "$after_gui")" == "$harness_pid" ]] || fail 'GUI crash killed harness'

"$LOOM" verify-journal --journal "$journal" | grep -q 'phase=active' || \
  fail 'live journal did not verify'
forged="$TEST_ROOT/duplicate-lease.tsv"
"$LOOM" _forge-duplicate-lease --journal "$journal" --output "$forged" >/dev/null
if "$LOOM" verify-journal --journal "$forged" > "$TEST_ROOT/forge.out" 2> "$TEST_ROOT/forge.err"; then
  fail 'fully rehashed duplicate lease sabotage verified'
fi
grep -q 'semantic:duplicate-input-lease' "$TEST_ROOT/forge.err" || \
  fail 'sabotage was not refused by the exclusive-lease semantic rule'
if grep -q 'hash:' "$TEST_ROOT/forge.err"; then
  fail 'sabotage was refused by hashing before the semantic rule ran'
fi

"$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" >/dev/null
for _ in $(seq 1 100); do
  state="$(sed -n 's/^state=//p' "$STATE_DIR/sessions/$AGENT--$LANE/session.state" 2>/dev/null || true)"
  [[ "$state" == exited ]] && break
  sleep 0.05
done
[[ "$state" == exited ]] || fail 'session did not record a terminal state'
"$LOOM" verify-journal --journal "$journal" | grep -q 'phase=exited' || \
  fail 'terminal journal did not verify'

cp "$TEST_ROOT/harness.sh" "$TEST_ROOT/codex-test"
coord() {
  SOUNIO_COORD_WORKTREE="$ROOT_DIR" SOUNIO_COORD_STATE_DIR="$TEST_ROOT/coord-bus" \
    SOUNIO_COORD_RUNTIME_MODE=local "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$@"
}
SOUNIO_COORD_COMMAND="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" \
SOUNIO_COORD_STATE_DIR="$TEST_ROOT/coord-bus" SOUNIO_COORD_RUNTIME_MODE=local \
  "$LOOM" start --state-dir "$TEST_ROOT/coord-loom" --agent "$COORD_AGENT" \
  --lane "$COORD_LANE" --session-id loom-coord-selftest --cwd "$ROOT_DIR" -- \
  "$TEST_ROOT/codex-test" >/dev/null
COORD_LOOM_ACTIVE=1
endpoint=''
for _ in $(seq 1 100); do
  endpoint="$(coord endpoint-status --agent "$COORD_AGENT" --lane "$COORD_LANE" 2>/dev/null || true)"
  [[ "$endpoint" == *'state=active'* && "$endpoint" == *'transport=loom'* ]] && break
  sleep 0.05
done
[[ "$endpoint" == *'state=active'* && "$endpoint" == *'transport=loom'* ]] || \
  fail "Loom did not auto-register a live coordination endpoint: $endpoint"
coord scope --agent loom-sender --lane transport-test --intent 'exercise Loom wake transport' >/dev/null
send_output="$(coord send --agent loom-sender --lane transport-test \
  --to-agent "$COORD_AGENT" --to-lane "$COORD_LANE" --kind request \
  --message 'Loom transport selftest')"
message_id="$(sed -n 's/.*message_id=\([^ ]*\).*/\1/p' <<< "$send_output" | head -1)"
[[ -n "$message_id" ]] || fail 'coord send omitted its message identity'
message_status="$(coord message-status --agent loom-sender --lane transport-test --message "$message_id")"
grep -q 'transport=loom' <<< "$message_status" || fail 'coord bus omitted the Loom wake receipt'
coord_output=''
for _ in $(seq 1 100); do
  coord_output="$(SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" snapshot \
    --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
    --agent "$COORD_AGENT" --lane "$COORD_LANE" --cursor 0 2>/dev/null || true)"
  [[ "$coord_output" == *'Sounio coordination wake:'* ]] && break
  sleep 0.05
done
[[ "$coord_output" == *'Sounio coordination wake:'* ]] || fail 'bus wake did not reach the Loom-owned PTY'
SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" stop --state-dir "$TEST_ROOT/coord-loom" \
  --cwd "$ROOT_DIR" --agent "$COORD_AGENT" --lane "$COORD_LANE" >/dev/null
COORD_LOOM_ACTIVE=0

if rg -n '\btmux\b' "$ROOT_DIR/tools/loom" "$ROOT_DIR/bin/sounio-loom" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null; then
  fail 'Loom attach path contains a tmux dependency'
fi

echo "sounio-loom-selftest: PASS language=OCaml protocol=1 instance=$instance_id"
