#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
TEST_ROOT="$(mktemp -d)"
CANARY_ROOT="$TEST_ROOT/state"
COORD_DIR="$CANARY_ROOT/coord"
LOOM_DIR="$CANARY_ROOT/loom"
LOOM="$ROOT_DIR/bin/sounio-loom"
RUNTIME="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh"
PANE_ID=loom-pod-replay:terminal
AGENT=beagle-workbench
LANE="pane-$(printf '%s' "$PANE_ID" | od -An -tx1 | tr -d ' \n')"

fail() {
  printf 'sounio-loom-pod-replay-canary-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

loom_status() {
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status --state-dir "$LOOM_DIR" \
    --cwd "$ROOT_DIR" --agent "$AGENT" --lane "$LANE"
}

coord() {
  SOUNIO_COORD_WORKTREE="$ROOT_DIR" SOUNIO_COORD_DIR="$COORD_DIR" \
    SOUNIO_COORD_RUNTIME_MODE=local "$RUNTIME" "$@"
}

kill_generation() {
  local status pid output='' attempt bridge_file bridge_pid
  status="$(loom_status 2>/dev/null || true)"
  [[ "$status" == *'state=active'* ]] || return 0
  for pid in "$(field daemon_pid "$status")" "$(field guardian_pid "$status")" \
    "$(field harness_pid "$status")"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || fail "generation omitted a process identity: $status"
    kill -9 "$pid" 2>/dev/null || true
  done
  for bridge_file in "$CANARY_ROOT"/bridge-*.pid; do
    [[ -f "$bridge_file" ]] || continue
    bridge_pid="$(cat "$bridge_file")"
    [[ "$bridge_pid" =~ ^[1-9][0-9]*$ ]] && kill -9 "$bridge_pid" 2>/dev/null || true
  done
  for attempt in $(seq 1 120); do
    output="$(coord endpoint-status --agent "$AGENT" --lane "$LANE" 2>&1 || true)"
    [[ "$output" != *'state=active'* ]] && return 0
    sleep 0.05
  done
  fail "dead simulated Pod retained an active endpoint: $output"
}

cleanup() {
  kill_generation || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

run_phase() {
  local uid="$1" name="$2" phase="$3"
  POD_UID="$uid" POD_NAME="$name" SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$CANARY_ROOT" bash "$CANARY" "$phase"
}

phase_one="$(run_phase pod-uid-one loom-replay-0 phase-one)"
[[ "$phase_one" == *'CANARY_PHASE_ONE'* && "$phase_one" == *'retry=deduplicated'* ]] || \
  fail "phase one did not establish first delivery: $phase_one"
if run_phase pod-uid-one loom-replay-0 phase-one >"$TEST_ROOT/repeat-one.log" 2>&1; then
  fail 'phase one accepted an existing state'
fi
grep -q 'phase one refuses an existing canary state' "$TEST_ROOT/repeat-one.log" || \
  fail 'phase-one refusal control failed for the wrong reason'

kill_generation
if run_phase pod-uid-one loom-replay-0 phase-two >"$TEST_ROOT/reused-uid.log" 2>&1; then
  fail 'phase two accepted the predecessor Pod UID'
fi
grep -q 'phase two is still running in the first Pod UID' "$TEST_ROOT/reused-uid.log" || \
  fail 'Pod-UID refusal control failed for the wrong reason'

phase_two="$(run_phase pod-uid-two loom-replay-0 phase-two)"
[[ "$phase_two" == *'CANARY_PHASE_TWO'* && "$phase_two" == *'wake=replayed'* && \
   "$phase_two" == *'receipts=2'* ]] || \
  fail "phase two did not replay the unacknowledged request: $phase_two"

kill_generation
phase_three="$(run_phase pod-uid-three loom-replay-0 phase-three)"
[[ "$phase_three" == *'CANARY_PHASE_THREE'* && "$phase_three" == *'depth_control=delivered'* && \
   "$phase_three" == *'ack=durable'* ]] || \
  fail "phase three did not establish the unacknowledged depth control: $phase_three"

kill_generation
phase_four="$(run_phase pod-uid-four loom-replay-0 phase-four)"
[[ "$phase_four" == *'CANARY_PHASE_FOUR'* && "$phase_four" == *'wake=ack-suppressed'* ]] || \
  fail "phase four did not suppress acknowledged replay: $phase_four"

report="$(SOUNIO_CANARY_SOURCE_ROOT=/missing \
  SOUNIO_LOOM_POD_CANARY_ROOT="$CANARY_ROOT" bash "$CANARY" report)"
[[ "$report" == *'SOUNIO_LOOM_SEPARATE_POD_REPLAY_PASS=true'* && \
   "$report" == *'unacked_successor_replay=delivered'* && \
   "$report" == *'unacked_third_generation_control=delivered'* && \
   "$report" == *'acked_fourth_generation_replay=suppressed'* ]] || \
  fail "final report omitted the replay proof: $report"
[[ "$(grep -c '^native_sounio_receipt_.*_sha256=' <<< "$report")" -eq 4 ]] || \
  fail 'final report omitted per-generation native Sounio receipts'
[[ "$(sed -n 's/^native_sounio_receipt_.*_sha256=//p' <<< "$report" | sort -u | wc -l | tr -d ' ')" -eq 4 ]] || \
  fail 'final report did not bind four distinct Sounio continuity receipts'

echo 'sounio-loom-pod-replay-canary-selftest: PASS simulated_pod_uids=4 native_sounio_receipts=4 unacked_depth_control=delivered ack_control=suppressed'
