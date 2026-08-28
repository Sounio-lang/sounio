#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/sounio-loom"
# This gate validates the source worktree before it can replace the shared runtime.
export SOUNIO_COORD_RUNTIME_MODE=local
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-selftest.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
AGENT=loom-test
LANE=cursor-replay
GUI_PID=''
OBSERVER_ONE=''
OBSERVER_TWO=''
HELD_ATTACH=''
GUARD_HELD_ATTACH=''
GUARD_STATE_DIR="$TEST_ROOT/guardian-state"
GUARD_AGENT=loom-guardian-test
GUARD_LANE=kernel-recovery
GUARD_ACTIVE=0
COORD_LOOM_ACTIVE=0
COORD_LOCK_HOLDER=''
COORD_AGENT=codex
COORD_LANE=loom-transport

fail() {
  echo "sounio-loom-selftest: FAIL: $* test_root=$TEST_ROOT" >&2
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

guardian_kernel_status() {
  "$LOOM" status --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$GUARD_AGENT" --lane "$GUARD_LANE"
}

guardian_owner_status() {
  "$LOOM" guardian-status --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$GUARD_AGENT" --lane "$GUARD_LANE"
}

wait_guardian_bridge_zero() {
  local output='' attempt
  for attempt in $(seq 1 100); do
    output="$(guardian_owner_status 2>/dev/null || true)"
    [[ "$output" == *'bridge_clients=0'* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "guardian did not release the dead kernel bridge; last=$output"
}

recover_guardian_kernel() {
  "$LOOM" recover --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$GUARD_AGENT" --lane "$GUARD_LANE" >/dev/null
  local output='' attempt
  for attempt in $(seq 1 100); do
    output="$(guardian_kernel_status 2>/dev/null || true)"
    [[ "$output" == *'state=active'* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "recovered kernel did not become active; last=$output"
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
  [[ -z "$GUARD_HELD_ATTACH" ]] || kill "$GUARD_HELD_ATTACH" 2>/dev/null || true
  [[ -z "$OBSERVER_ONE" ]] || kill "$OBSERVER_ONE" 2>/dev/null || true
  [[ -z "$OBSERVER_TWO" ]] || kill "$OBSERVER_TWO" 2>/dev/null || true
  [[ -z "$GUI_PID" ]] || kill "$GUI_PID" 2>/dev/null || true
  [[ -z "$COORD_LOCK_HOLDER" ]] || kill "$COORD_LOCK_HOLDER" 2>/dev/null || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  if [[ "$GUARD_ACTIVE" == 1 ]]; then
    "$LOOM" stop --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
      --agent "$GUARD_AGENT" --lane "$GUARD_LANE" >/dev/null 2>&1 || true
  fi
  if [[ "$COORD_LOOM_ACTIVE" == 1 ]]; then
    SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" stop --state-dir "$TEST_ROOT/coord-loom" \
      --cwd "$ROOT_DIR" --agent "$COORD_AGENT" --lane "$COORD_LANE" >/dev/null 2>&1 || true
  fi
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
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

cat > "$TEST_ROOT/guardian-harness.sh" <<'HARNESS'
#!/usr/bin/env bash
stty -echo
trap 'printf "OUTAGE_SIGNAL\n"' USR1
printf 'GUARDIAN_BOOT_READY\n'
while :; do
  if IFS= read -r -t 0.1 line; then
    printf 'GUARDIAN_ECHO:%s\n' "$line"
  fi
done
HARNESS
chmod +x "$TEST_ROOT/guardian-harness.sh"

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

archived_dir="$STATE_DIR/sessions/aaa--archived"
mkdir -p "$archived_dir"
sed -e 's/^state=.*/state=exited/' \
  -e 's/^agent=.*/agent=archived/' \
  -e 's/^lane=.*/lane=old-lane/' \
  -e 's/^instance_id=.*/instance_id=archived-generation/' \
  "$descriptor" > "$archived_dir/session.state"
lost_dir="$STATE_DIR/sessions/aab--lost"
mkdir -p "$lost_dir"
sed -e 's/^state=.*/state=active/' \
  -e 's/^agent=.*/agent=lost/' \
  -e 's/^lane=.*/lane=pod-loss/' \
  -e 's/^instance_id=.*/instance_id=lost-generation/' \
  -e 's/^daemon_pid=.*/daemon_pid=99999999/' \
  -e 's/^daemon_pid_start=.*/daemon_pid_start=1/' \
  -e 's/^guardian_pid=.*/guardian_pid=99999998/' \
  -e 's/^guardian_pid_start=.*/guardian_pid_start=1/' \
  "$descriptor" > "$lost_dir/session.state"
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
gui_html="$(curl -fsS "$gui_url/")"
grep -qi 'Sounio Loom' <<< "$gui_html" || fail 'GUI did not serve its operational view'
grep -Fq 'data-loom-ui="fusion-v1"' <<< "$gui_html" || fail 'GUI did not serve the Fusion cockpit'
grep -Fq 'navigator.gpu' <<< "$gui_html" || fail 'GUI omitted its WebGPU spectral path'
grep -Fq "list.find(s=>s.health_state==='WORKING')" <<< "$gui_html" || \
  fail 'GUI does not explicitly select Sounio-classified work'
sessions_json="$(curl -fsS "$gui_url/api/sessions")"
[[ "$sessions_json" == *"\"instance_id\":\"$instance_id\""* ]] || fail 'GUI observed the wrong generation'
[[ "$sessions_json" == "[{\"agent\":\"$AGENT\",\"lane\":\"$LANE\""* ]] || \
  fail 'operator session list did not rank active work before archived work'
[[ "$sessions_json" == *'"agent":"lost","lane":"pod-loss"'*'"state":"lost"'* ]] || \
  fail 'operator session list laundered a dead generation as active'
fleet_json="$(curl -fsS "$gui_url/api/fleet")"
[[ "$fleet_json" == *'"schema":"loom-authority-overlay-v2"'* ]] || \
  fail 'GUI omitted the authority overlay schema'
[[ "$fleet_json" == *'"health_authority":"Sounio"'* && \
  "$fleet_json" == *'"health_realization":"OCaml"'* && \
  "$fleet_json" == *'"health_semantics_sha256":"5eb48f9cb214f6018569fb24e1e419b3e800dccde2e6e8d775246f4c05e4c93f"'* ]] || \
  fail 'GUI omitted the frozen truthful-health authority chain'
[[ "$fleet_json" == *"\"instance_id\":\"$instance_id\""* || \
  "$fleet_json" == *"\"loom_instance\":\"$instance_id\""* ]] || \
  fail 'authority overlay observed the wrong Loom generation'
[[ "$fleet_json" == *'"loom_state":"active"'* ]] || \
  fail 'authority overlay omitted active Loom custody'
tui_machine="$("$LOOM" tui --machine --state-dir "$STATE_DIR" --cwd "$TEST_ROOT")"
grep -q '^LOOM_TUI schema=loom-truthful-fleet-tui-v1 authority=Sounio realization=OCaml ' \
  <<< "$tui_machine" || fail 'TUI omitted the truthful-health authority header'
grep -q 'semantics_sha256=5eb48f9cb214f6018569fb24e1e419b3e800dccde2e6e8d775246f4c05e4c93f' \
  <<< "$tui_machine" || fail 'TUI omitted the frozen Sounio semantics hash'
grep -Eq "^LOOM_TUI_LANE health=[A-Z]+ agent=$AGENT lane=$LANE .*custody=active " \
  <<< "$tui_machine" || fail 'TUI omitted the live Loom-owned lane'
events_json="$(curl -fsS "$gui_url/api/events")"
[[ "$events_json" == *"\"instance_id\":\"$instance_id\""* ]] || \
  fail 'event chronograph observed the wrong generation'
[[ "$events_json" == *'"verified":true'* ]] || \
  fail 'event chronograph accepted no verified journal'
[[ "$events_json" == *'"kind":"SESSION_STARTED"'* ]] || \
  fail 'event chronograph omitted the durable session start'
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
offline="$($LOOM snapshot --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor 0 --meta \
  2> "$TEST_ROOT/offline-snapshot.meta")"
[[ "$offline" == *BOOT_READY* ]] || fail 'offline replay omitted durable output'
grep -q "LOOM_SNAPSHOT instance=$instance_id .* source=offline" \
  "$TEST_ROOT/offline-snapshot.meta" || fail 'terminal replay did not use verified offline custody'
output_file="$(sed -n 's/^output_file=//p' \
  "$STATE_DIR/sessions/$AGENT--$LANE/session.state")"
ending="$(wc -c < "$output_file")"
[[ "$ending" -gt 0 ]] || fail 'terminal output is empty before custody sabotage'
clean_output="$TEST_ROOT/offline-output.clean"
cp "$output_file" "$clean_output"
printf 'X' | dd of="$output_file" bs=1 seek=0 conv=notrunc status=none
[[ "$(wc -c < "$output_file")" == "$ending" ]] || \
  fail 'same-size custody sabotage changed output length'
if cmp -s "$output_file" "$clean_output"; then
  fail 'same-size custody sabotage did not mutate output bytes'
fi
if "$LOOM" snapshot --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor 0 \
  > "$TEST_ROOT/offline-custody.out" 2> "$TEST_ROOT/offline-custody.err"; then
  fail 'offline replay accepted same-size durable-output mutation'
fi
grep -q 'guardian-output:digest-mismatch' "$TEST_ROOT/offline-custody.err" || \
  fail 'same-size output sabotage was refused by the wrong rule'
cp "$clean_output" "$output_file"
if "$LOOM" snapshot --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --cursor "$((ending + 1))" \
  > "$TEST_ROOT/offline-cursor.out" 2> "$TEST_ROOT/offline-cursor.err"; then
  fail 'offline replay accepted a cursor beyond durable output'
fi
grep -q 'cursor ahead of durable output' "$TEST_ROOT/offline-cursor.err" || \
  fail 'offline replay cursor sabotage was refused by the wrong rule'

"$LOOM" start --state-dir "$GUARD_STATE_DIR" --agent "$GUARD_AGENT" \
  --lane "$GUARD_LANE" --session-id loom-guardian-selftest --cwd "$TEST_ROOT" -- \
  /bin/bash "$TEST_ROOT/guardian-harness.sh" >/dev/null
GUARD_ACTIVE=1
guard_initial="$(guardian_kernel_status)"
guard_daemon="$(field daemon_pid "$guard_initial")"
guard_pid="$(field guardian_pid "$guard_initial")"
guard_harness="$(field harness_pid "$guard_initial")"
guard_instance="$(field instance_id "$guard_initial")"
guard_journal="$(field journal "$guard_initial")"
guard_descriptor="$GUARD_STATE_DIR/sessions/$GUARD_AGENT--$GUARD_LANE/session.state"
guardian_journal="$(sed -n 's/^guardian_journal_file=//p' "$guard_descriptor")"
[[ -n "$guard_daemon" && -n "$guard_pid" && -n "$guard_harness" && \
  -n "$guard_instance" && -f "$guard_journal" && -f "$guardian_journal" ]] || \
  fail 'guarded session omitted one of its three process identities or journals'
[[ "$guard_daemon" != "$guard_pid" && "$guard_pid" != "$guard_harness" ]] || \
  fail 'kernel, guardian, and harness did not receive distinct process identities'
if "$LOOM" recover --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$GUARD_AGENT" --lane "$GUARD_LANE" \
  > "$TEST_ROOT/recover-live.out" 2> "$TEST_ROOT/recover-live.err"; then
  fail 'recovery started a second kernel while the original kernel was active'
fi
grep -q 'active Loom kernel' "$TEST_ROOT/recover-live.err" || \
  fail 'live-kernel recovery was refused for the wrong reason'

guard_before="$(guardian_owner_status)"
guard_cursor="$(field output_cursor "$guard_before")"
"$LOOM" crash-kernel --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$GUARD_AGENT" --lane "$GUARD_LANE" --at now >/dev/null
wait_guardian_bridge_zero >/dev/null
kill -0 "$guard_pid" 2>/dev/null || fail 'guardian died with the first kernel'
kill -0 "$guard_harness" 2>/dev/null || fail 'harness died with the first kernel'
guard_token_path="$(sed -n 's/^token_file=//p' "$guard_descriptor")"
[[ -n "$guard_token_path" && -f "$guard_token_path" ]] || \
  fail 'guarded session omitted its capability path'
guard_token_before="$(cat "$guard_token_path")"
if "$LOOM" start --state-dir "$GUARD_STATE_DIR" --agent "$GUARD_AGENT" \
  --lane "$GUARD_LANE" --session-id laundering-attempt --cwd "$TEST_ROOT" -- \
  /bin/true > "$TEST_ROOT/start-over-guardian.out" \
  2> "$TEST_ROOT/start-over-guardian.err"; then
  fail 'start replaced a recoverable generation whose Guardian remained active'
fi
grep -q 'recoverable Guardian.*use recover' "$TEST_ROOT/start-over-guardian.err" || \
  fail 'start over a recoverable generation was refused for the wrong reason'
[[ "$(cat "$guard_token_path")" == "$guard_token_before" ]] || \
  fail 'refused start rotated the live Guardian capability'
guard_after_refusal="$(guardian_owner_status)"
[[ "$(field guardian_pid "$guard_after_refusal")" == "$guard_pid" && \
  "$(field harness_pid "$guard_after_refusal")" == "$guard_harness" && \
  "$(field instance_id "$guard_after_refusal")" == "$guard_instance" ]] || \
  fail 'refused start changed recoverable generation custody'
kill -USR1 "$guard_harness"
outage_owner=''
for _ in $(seq 1 100); do
  outage_owner="$(guardian_owner_status 2>/dev/null || true)"
  [[ -n "$outage_owner" && "$(field output_cursor "$outage_owner")" -gt "$guard_cursor" ]] && break
  sleep 0.05
done
[[ -n "$outage_owner" && "$(field output_cursor "$outage_owner")" -gt "$guard_cursor" ]] || \
  fail 'guardian did not durably capture output while no kernel existed'
guard_recovered="$(recover_guardian_kernel)"
[[ "$(field daemon_pid "$guard_recovered")" != "$guard_daemon" ]] || \
  fail 'recovery reused the dead kernel process identity'
[[ "$(field guardian_pid "$guard_recovered")" == "$guard_pid" ]] || \
  fail 'recovery replaced the guardian process'
[[ "$(field harness_pid "$guard_recovered")" == "$guard_harness" ]] || \
  fail 'recovery replaced the harness process'
[[ "$(field instance_id "$guard_recovered")" == "$guard_instance" ]] || \
  fail 'recovery changed the generation identity'
outage_replay="$($LOOM snapshot --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$GUARD_AGENT" --lane "$GUARD_LANE" --cursor "$guard_cursor")"
[[ "$outage_replay" == *OUTAGE_SIGNAL* ]] || \
  fail 'recovered kernel omitted output produced during its absence'

for boundary in after_guardian_read after_output_journal after_broadcast; do
  boundary_before="$(guardian_kernel_status)"
  boundary_cursor="$(field output_cursor "$boundary_before")"
  "$LOOM" crash-kernel --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$GUARD_AGENT" --lane "$GUARD_LANE" --at "$boundary" >/dev/null
  printf 'boundary-%s\n' "$boundary" | \
    "$LOOM" attach --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
      --agent "$GUARD_AGENT" --lane "$GUARD_LANE" --cursor end --no-raw \
      >/dev/null 2>&1 || true
  wait_guardian_bridge_zero >/dev/null
  kill -0 "$guard_pid" 2>/dev/null || fail "guardian died at $boundary"
  kill -0 "$guard_harness" 2>/dev/null || fail "harness died at $boundary"
  guard_recovered="$(recover_guardian_kernel)"
  [[ "$(field guardian_pid "$guard_recovered")" == "$guard_pid" && \
    "$(field harness_pid "$guard_recovered")" == "$guard_harness" && \
    "$(field instance_id "$guard_recovered")" == "$guard_instance" ]] || \
    fail "identity changed while recovering $boundary"
  boundary_replay="$($LOOM snapshot --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$GUARD_AGENT" --lane "$GUARD_LANE" --cursor "$boundary_cursor")"
  [[ "$boundary_replay" == *"GUARDIAN_ECHO:boundary-$boundary"* ]] || \
    fail "replay omitted the $boundary output"
  "$LOOM" verify-journal --journal "$guard_journal" | grep -q 'phase=active' || \
    fail "semantic journal became invalid after $boundary"
done

tail -f /dev/null | "$LOOM" attach --state-dir "$GUARD_STATE_DIR" \
  --cwd "$TEST_ROOT" --agent "$GUARD_AGENT" --lane "$GUARD_LANE" \
  --cursor end --no-raw > "$TEST_ROOT/guardian-held.out" 2>&1 &
GUARD_HELD_ATTACH=$!
for _ in $(seq 1 100); do
  held_guard="$(guardian_kernel_status 2>/dev/null || true)"
  [[ "$held_guard" == *'interactive_clients=1'* ]] && break
  sleep 0.05
done
[[ "$held_guard" == *'interactive_clients=1'* ]] || \
  fail 'guardian recovery control could not establish an input lease'
"$LOOM" crash-kernel --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$GUARD_AGENT" --lane "$GUARD_LANE" --at now >/dev/null
wait_guardian_bridge_zero >/dev/null
wait "$GUARD_HELD_ATTACH" 2>/dev/null || true
GUARD_HELD_ATTACH=''
guard_recovered="$(recover_guardian_kernel)"
[[ "$guard_recovered" == *'interactive_clients=0'* ]] || \
  fail 'recovery retained an input lease whose client died with the kernel'
"$LOOM" verify-journal --journal "$guard_journal" | grep -q 'phase=active' || \
  fail 'KERNEL_RECOVERED did not semantically revoke the orphaned input lease'
[[ "$(grep -c $'\tKERNEL_RECOVERED\t' "$guard_journal")" -ge 5 ]] || \
  fail 'recovery journal omitted one or more kernel generations'
"$LOOM" verify-guardian-journal --journal "$guardian_journal" | \
  grep -q 'phase=active' || fail 'live guardian journal did not verify'

"$LOOM" stop --state-dir "$GUARD_STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$GUARD_AGENT" --lane "$GUARD_LANE" >/dev/null
GUARD_ACTIVE=0
guard_terminal_state=''
for _ in $(seq 1 100); do
  guard_terminal_state="$(sed -n 's/^state=//p' "$guard_descriptor" 2>/dev/null || true)"
  [[ "$guard_terminal_state" == exited ]] && break
  sleep 0.05
done
[[ "$guard_terminal_state" == exited ]] || \
  fail 'recovered session did not record its terminal state'
"$LOOM" verify-journal --journal "$guard_journal" | grep -q 'phase=exited' || \
  fail 'recovered semantic journal did not reach a valid terminal state'
"$LOOM" verify-guardian-journal --journal "$guardian_journal" | \
  grep -q 'phase=exited' || fail 'guardian journal did not reach a valid terminal state'

cp "$TEST_ROOT/harness.sh" "$TEST_ROOT/codex-test"
cat > "$TEST_ROOT/flaky-coord.sh" <<'COORD'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == endpoint-register ]] && mkdir "$SOUNIO_LOOM_TEST_ENDPOINT_FAIL_ONCE" 2>/dev/null; then
  exit 75
fi
exec "$SOUNIO_LOOM_TEST_COORD_RUNTIME" "$@"
COORD
chmod +x "$TEST_ROOT/flaky-coord.sh"
coord() {
  SOUNIO_COORD_WORKTREE="$ROOT_DIR" SOUNIO_COORD_DIR="$TEST_ROOT/coord-bus" \
    SOUNIO_COORD_RUNTIME_MODE=local "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$@"
}
coord_retry() {
  local output='' attempt
  for attempt in $(seq 1 100); do
    if output="$(coord "$@" 2>&1)"; then
      printf '%s\n' "$output"
      return 0
    fi
    [[ "$output" == *'coordination state is being changed'* ]] || {
      printf '%s\n' "$output" >&2
      return 1
    }
    sleep 0.05
  done
  fail "coordination operation did not clear lock contention: $output"
}
start_coord_generation() {
  SOUNIO_COORD_COMMAND="$TEST_ROOT/flaky-coord.sh" \
  SOUNIO_COORD_DIR="$TEST_ROOT/coord-bus" SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_LOCK_WAIT_SECONDS=1 \
  SOUNIO_LOOM_TEST_COORD_RUNTIME="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" \
  SOUNIO_LOOM_TEST_ENDPOINT_FAIL_ONCE="$TEST_ROOT/coord-endpoint-failed-once" \
    "$LOOM" start --state-dir "$TEST_ROOT/coord-loom" --agent "$COORD_AGENT" \
    --lane "$COORD_LANE" --session-id loom-coord-selftest --cwd "$ROOT_DIR" -- \
    "$TEST_ROOT/codex-test" >/dev/null
}
coord_loom_status() {
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status \
    --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
    --agent "$COORD_AGENT" --lane "$COORD_LANE"
}
wait_coord_endpoint() {
  local output='' attempt
  for attempt in $(seq 1 300); do
    output="$(coord endpoint-status --agent "$COORD_AGENT" --lane "$COORD_LANE" 2>/dev/null || true)"
    [[ "$output" == *'state=active'* && "$output" == *'transport=loom'* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "Loom coordination endpoint did not become active: $output"
}
kill_coord_generation() {
  local status="$1" pid output='' attempt
  for pid in "$(field daemon_pid "$status")" "$(field guardian_pid "$status")" \
    "$(field harness_pid "$status")"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || fail "coordination generation omitted a process identity: $status"
    kill -9 "$pid" 2>/dev/null || true
  done
  for attempt in $(seq 1 100); do
    output="$(coord endpoint-status --agent "$COORD_AGENT" --lane "$COORD_LANE" 2>&1 || true)"
    [[ "$output" != *'state=active'* ]] && return 0
    sleep 0.05
  done
  fail "dead coordination generation retained an active endpoint: $output"
}

mkdir -p "$TEST_ROOT/coord-bus"
(
  exec 8> "$TEST_ROOT/coord-bus/.claims.lock"
  flock 8
  touch "$TEST_ROOT/coord-lock-ready"
  sleep 3
) &
COORD_LOCK_HOLDER=$!
for _ in $(seq 1 100); do
  [[ -f "$TEST_ROOT/coord-lock-ready" ]] && break
  sleep 0.01
done
[[ -f "$TEST_ROOT/coord-lock-ready" ]] || fail 'coordination contention fixture did not acquire its lock'
start_coord_generation
COORD_LOOM_ACTIVE=1
endpoint="$(wait_coord_endpoint)"
wait "$COORD_LOCK_HOLDER"
COORD_LOCK_HOLDER=''
coord_daemon_log="$(find "$TEST_ROOT/coord-loom" -name daemon.log -type f -print -quit)"
[[ -n "$coord_daemon_log" ]] || fail 'coordination generation omitted its daemon log'
grep -q 'LOOM_COORDINATION_RETRY failures=1 ' "$coord_daemon_log" || \
  fail 'Loom did not retry endpoint registration after transient lock contention'
grep -q 'LOOM_COORDINATION_WARNING operation=endpoint-register' "$coord_daemon_log" || \
  fail 'Loom endpoint sabotage did not reach the endpoint registration boundary'
grep -q 'LOOM_COORDINATION_RETRY failures=2 ' "$coord_daemon_log" || \
  fail 'Loom did not retry after the endpoint registration sabotage'
coord_before="$(coord_loom_status)"
coord_guardian="$(field guardian_pid "$coord_before")"
coord_harness="$(field harness_pid "$coord_before")"
coord_instance="$(field instance_id "$coord_before")"
SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" crash-kernel \
  --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
  --agent "$COORD_AGENT" --lane "$COORD_LANE" --at now >/dev/null
coord_guardian_status=''
for _ in $(seq 1 100); do
  coord_guardian_status="$(SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" guardian-status \
    --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
    --agent "$COORD_AGENT" --lane "$COORD_LANE" 2>/dev/null || true)"
  [[ "$coord_guardian_status" == *'bridge_clients=0'* ]] && break
  sleep 0.05
done
[[ "$coord_guardian_status" == *'bridge_clients=0'* ]] || \
  fail 'coordination guardian retained its dead kernel bridge'
dead_endpoint="$(coord endpoint-status --agent "$COORD_AGENT" \
  --lane "$COORD_LANE" 2>&1 || true)"
[[ "$dead_endpoint" != *'state=active'* ]] || \
  fail "coordination endpoint remained valid after its kernel died: $dead_endpoint"
SOUNIO_COORD_COMMAND="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" \
SOUNIO_COORD_DIR="$TEST_ROOT/coord-bus" SOUNIO_COORD_RUNTIME_MODE=local \
  "$LOOM" recover --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
    --agent "$COORD_AGENT" --lane "$COORD_LANE" >/dev/null
endpoint="$(wait_coord_endpoint)"
coord_after="$(coord_loom_status)"
[[ "$(field guardian_pid "$coord_after")" == "$coord_guardian" && \
  "$(field harness_pid "$coord_after")" == "$coord_harness" && \
  "$(field instance_id "$coord_after")" == "$coord_instance" ]] || \
  fail 'coordination endpoint recovery changed the live generation identity'
coord_retry scope --agent loom-sender --lane transport-test \
  --intent 'exercise Loom wake transport' >/dev/null
send_output="$(coord_retry send --agent loom-sender --lane transport-test \
  --to-agent "$COORD_AGENT" --to-lane "$COORD_LANE" --kind request \
  --message 'Loom transport selftest')"
message_id="$(sed -n 's/.*message_id=\([^ ]*\).*/\1/p' <<< "$send_output" | head -1)"
[[ -n "$message_id" ]] || fail 'coord send omitted its message identity'
message_status="$(coord_retry message-status --agent loom-sender --lane transport-test --message "$message_id")"
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

first_receipt="$(grep "^WAKE_RECEIPT message_id=$message_id " <<< "$message_status")"
[[ "$(field generation "$first_receipt")" == "$coord_instance" ]] || \
  fail 'first wake receipt was not bound to the live Loom generation'
same_generation_retry="$(coord_retry wake --agent loom-sender --lane transport-test \
  --message "$message_id")"
[[ "$same_generation_retry" == *'WAKE_SKIPPED'* && \
  "$same_generation_retry" == *"generation=$coord_instance"* && \
  "$same_generation_retry" == *'reason=already-delivered'* ]] || \
  fail "same-generation retry was not deduplicated: $same_generation_retry"

kill_coord_generation "$coord_after"
start_coord_generation
endpoint="$(wait_coord_endpoint)"
coord_generation_two="$(coord_loom_status)"
coord_instance_two="$(field instance_id "$coord_generation_two")"
[[ -n "$coord_instance_two" && "$coord_instance_two" != "$coord_instance" ]] || \
  fail 'full generation death did not produce a successor identity'
generation_two_wake="$(coord_retry wake --agent loom-sender --lane transport-test \
  --message "$message_id")"
[[ "$generation_two_wake" == *'WAKE_DELIVERED'* && \
  "$generation_two_wake" == *"generation=$coord_instance_two"* ]] || \
  fail "unacknowledged message did not replay into the successor generation: $generation_two_wake"
generation_two_output=''
for _ in $(seq 1 100); do
  generation_two_output="$(SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" snapshot \
    --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
    --agent "$COORD_AGENT" --lane "$COORD_LANE" --cursor 0 2>/dev/null || true)"
  [[ "$generation_two_output" == *"$message_id"* ]] && break
  sleep 0.05
done
[[ "$generation_two_output" == *"$message_id"* ]] || \
  fail 'successor generation PTY omitted the replayed wake'
generation_two_retry="$(coord_retry wake --agent loom-sender --lane transport-test \
  --message "$message_id")"
[[ "$generation_two_retry" == *'WAKE_SKIPPED'* && \
  "$generation_two_retry" == *"generation=$coord_instance_two"* && \
  "$generation_two_retry" == *'reason=already-delivered'* ]] || \
  fail "successor generation retry was not deduplicated: $generation_two_retry"

message_status="$(coord_retry message-status --agent loom-sender --lane transport-test \
  --message "$message_id")"
[[ "$message_status" == *'wakes=2'* ]] || \
  fail "cross-generation status did not retain two wake receipts: $message_status"
[[ "$(grep -c "^WAKE_RECEIPT message_id=$message_id " <<< "$message_status")" -eq 2 ]] || \
  fail 'cross-generation status did not expose exactly two receipt generations'
grep -q "generation=$coord_instance$" <<< "$message_status" || \
  fail 'cross-generation status lost the predecessor wake generation'
grep -q "generation=$coord_instance_two$" <<< "$message_status" || \
  fail 'cross-generation status lost the successor wake generation'

coord_retry ack --agent "$COORD_AGENT" --lane "$COORD_LANE" \
  --message "$message_id" >/dev/null
kill_coord_generation "$coord_generation_two"
start_coord_generation
endpoint="$(wait_coord_endpoint)"
coord_generation_three="$(coord_loom_status)"
coord_instance_three="$(field instance_id "$coord_generation_three")"
[[ -n "$coord_instance_three" && "$coord_instance_three" != "$coord_instance_two" ]] || \
  fail 'ACK control did not start a distinct third generation'
acked_retry="$(coord_retry wake --agent loom-sender --lane transport-test \
  --message "$message_id")"
[[ "$acked_retry" == *'WAKE_SKIPPED'* && "$acked_retry" == *'reason=acknowledged'* ]] || \
  fail "durable ACK did not suppress third-generation replay: $acked_retry"
sleep 0.1
generation_three_output="$(SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" snapshot \
  --state-dir "$TEST_ROOT/coord-loom" --cwd "$ROOT_DIR" \
  --agent "$COORD_AGENT" --lane "$COORD_LANE" --cursor 0 2>/dev/null || true)"
[[ "$generation_three_output" != *"$message_id"* ]] || \
  fail 'acknowledged message was injected into the third generation'
message_status="$(coord_retry message-status --agent loom-sender --lane transport-test \
  --message "$message_id")"
[[ "$message_status" == *'acknowledged=1'* && "$message_status" == *'wakes=2'* ]] || \
  fail "ACK control changed the durable wake count: $message_status"

SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" stop --state-dir "$TEST_ROOT/coord-loom" \
  --cwd "$ROOT_DIR" --agent "$COORD_AGENT" --lane "$COORD_LANE" >/dev/null
COORD_LOOM_ACTIVE=0
coord_snapshot=''
for _ in $(seq 1 100); do
  coord_snapshot="$(coord cockpit-snapshot)"
  if ! grep -Fq $'agent=codex\tlane=loom-transport' <<< "$coord_snapshot"; then
    break
  fi
  sleep 0.05
done
if grep -Fq $'agent=codex\tlane=loom-transport' <<< "$coord_snapshot"; then
  fail 'clean Loom exit retained coordination authority'
fi

FAKE_FLEET_STATE="$TEST_ROOT/fake-fleet-state"
FAKE_FLEET_AGENT="$TEST_ROOT/fake-fleet-agent"
mkdir -p "$FAKE_FLEET_STATE"
cat > "$FAKE_FLEET_AGENT" <<'FLEET_AGENT'
#!/usr/bin/env bash
set -euo pipefail
command_name="${1:-}"
shift || true
slot=''
while (($#)); do
  case "$1" in
    --slot) slot="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$slot" ]]
active="$SOUNIO_FAKE_FLEET_STATE/$slot.active"
case "$command_name" in
  status)
    if [[ -f "$active" ]]; then
      echo "FLEET_SLOT_STATUS state=active slot=$slot"
      echo 'fleet_slots=1 unhealthy=0'
      exit 0
    fi
    echo "FLEET_SLOT_STATUS state=absent slot=$slot"
    echo 'fleet_slots=1 unhealthy=1'
    exit 1
    ;;
  launch-kind)
    count_file="$SOUNIO_FAKE_FLEET_STATE/$slot.starts"
    count="$(cat "$count_file" 2>/dev/null || echo 0)"
    if [[ ! -f "$active" ]]; then
      printf '%s\n' "$((count + 1))" > "$count_file"
      : > "$active"
      action=started
    else
      action=reattached
    fi
    echo "FLEET_SLOT action=$action slot=$slot"
    ;;
  *) exit 2 ;;
esac
FLEET_AGENT
chmod +x "$FAKE_FLEET_AGENT"
fleet_loom() {
  SOUNIO_FAKE_FLEET_STATE="$FAKE_FLEET_STATE" \
    SOUNIO_LOOM_FLEET_AGENT_COMMAND="$FAKE_FLEET_AGENT" "$LOOM" "$@"
}
FLEET_CATALOG_STATE="$TEST_ROOT/fleet-catalog"
fleet_loom fleet-enroll --state-dir "$FLEET_CATALOG_STATE" \
  --slot post-pod --kind claude --home "$TEST_ROOT" --cwd "$TEST_ROOT" >/dev/null
fleet_loom fleet-enroll --state-dir "$FLEET_CATALOG_STATE" \
  --slot post-pod --kind claude --home "$TEST_ROOT" --cwd "$TEST_ROOT" >/dev/null
fleet_plan="$(fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR")"
[[ "$fleet_plan" == *'slot=post-pod state=DEAD action=start mode=plan'* ]] || \
  fail 'fleet dry-run did not plan the absent post-Pod lane'
[[ ! -e "$FAKE_FLEET_STATE/post-pod.starts" ]] || \
  fail 'fleet dry-run mutated launch state'
fleet_apply="$(fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR" --apply)"
[[ "$fleet_apply" == *'slot=post-pod state=DEAD action=started'* && \
  "$(cat "$FAKE_FLEET_STATE/post-pod.starts")" == 1 ]] || \
  fail 'fleet apply did not start exactly one post-Pod generation'
fleet_repeat="$(fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR" --apply)"
[[ "$fleet_repeat" == *'slot=post-pod state=DISCONNECTED action=operator-required'* && \
  "$(cat "$FAKE_FLEET_STATE/post-pod.starts")" == 1 ]] || \
  fail 'repeated fleet reconciliation did not fail closed on an active disconnected generation'
fleet_descriptor="$FLEET_CATALOG_STATE/fleet/post-pod.state"
cp "$fleet_descriptor" "$FLEET_CATALOG_STATE/fleet/duplicate.state"
if fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR" > "$TEST_ROOT/fleet-duplicate.out" 2>&1; then
  fail 'fleet catalog accepted duplicate desired authority for one slot'
fi
grep -q 'duplicate fleet slot' "$TEST_ROOT/fleet-duplicate.out" || \
  fail 'duplicate fleet authority was refused for the wrong reason'
rm "$FLEET_CATALOG_STATE/fleet/duplicate.state"
cp "$fleet_descriptor" "$TEST_ROOT/fleet-descriptor.backup"
sed 's/^kind=.*/kind=forged/' "$fleet_descriptor" > "$fleet_descriptor.tmp"
mv "$fleet_descriptor.tmp" "$fleet_descriptor"
if fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR" > "$TEST_ROOT/fleet-kind.out" 2>&1; then
  fail 'fleet catalog accepted a forged launcher kind'
fi
grep -q 'unsupported fleet kind' "$TEST_ROOT/fleet-kind.out" || \
  fail 'forged fleet kind was refused for the wrong reason'
mv "$TEST_ROOT/fleet-descriptor.backup" "$fleet_descriptor"
fleet_loom fleet-disable --state-dir "$FLEET_CATALOG_STATE" \
  --slot post-pod --cwd "$TEST_ROOT" >/dev/null
rm "$FAKE_FLEET_STATE/post-pod.active"
fleet_disabled="$(fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR" --apply)"
[[ "$fleet_disabled" == *'loom_fleet_slots=0'* && \
  "$(cat "$FAKE_FLEET_STATE/post-pod.starts")" == 1 ]] || \
  fail 'disabled fleet intent relaunched after simulated Pod loss'
fleet_loom fleet-enroll --state-dir "$FLEET_CATALOG_STATE" \
  --slot unauthorized-observer --kind claude --home "$TEST_ROOT" \
  --cwd "$TEST_ROOT" >/dev/null
unauthorized_reconcile="$(SOUNIO_COORD_COMMAND="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" \
  fleet_loom fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$ROOT_DIR" --apply)"
[[ "$unauthorized_reconcile" == *'slot=unauthorized-observer state=UNKNOWN action=operator-required'* && \
  "$unauthorized_reconcile" == *'observation_authorized=false'* && \
  ! -e "$FAKE_FLEET_STATE/unauthorized-observer.starts" ]] || \
  fail 'fleet reconciliation mutated an absent lane under an unauthorized observer'
if SOUNIO_LOOM_FLEET_AGENT_COMMAND="$TEST_ROOT/missing-fleet-agent" \
  "$LOOM" fleet-reconcile --state-dir "$FLEET_CATALOG_STATE" \
  --cwd "$TEST_ROOT" > "$TEST_ROOT/fleet-adapter.out" 2>&1; then
  fail 'fleet reconciliation silently replaced an unavailable configured adapter'
fi
grep -q 'configured fleet agent command is unavailable' \
  "$TEST_ROOT/fleet-adapter.out" || \
  fail 'unavailable configured fleet adapter was refused for the wrong reason'

if rg -n '\btmux\b' "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_ui.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" \
  "$ROOT_DIR/tools/loom/dune-project" "$ROOT_DIR/bin/sounio-loom" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null; then
  fail 'Loom native attach kernel contains a tmux dependency'
fi

"$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_ocaml_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_ocaml_freeze_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_material_admission_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_material_admission_freeze_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_host_promotion_capsule_selftest.sh" >/dev/null

echo "sounio-loom-selftest: PASS language=OCaml protocol=1 instance=$instance_id guardian_instance=$guard_instance kernel_crashes=6 coord_generations=3 unacked_replay=delivered acked_replay=suppressed post_pod_reconcile=idempotent"
