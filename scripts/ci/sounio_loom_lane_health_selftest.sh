#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-lane-health-selftest.XXXXXX")"
RUNTIME="$TEST_ROOT/sounio-lane-health"
SABOTAGED="$TEST_ROOT/sounio-lane-health-sabotaged"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-lane-health-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_frame() {
  local runtime="$1" frame="$2"
  set +e
  FRAME_OUTPUT="$(printf '%s\n' "$frame" | "$runtime" 2>&1)"
  FRAME_RC=$?
  set -e
}

assert_state() {
  local runtime="$1" frame="$2" expected="$3"
  run_frame "$runtime" "$frame"
  [[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == *"state=$expected"* ]] ||
    fail "expected $expected: rc=$FRAME_RC output=$FRAME_OUTPUT frame=$frame"
}

SOUNIO_LOOM_LANE_HEALTH_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_LANE_HEALTH_SELFTEST PASS cases=28' ]] ||
  fail "Sounio-owned selftest did not pass: $selftest"

# schema policy expected claim residue pane process unresponsive process_absent
# endpoint endpoint_absent endpoint_stale custody custody_recoverable obligation blocker census
# progress progress_window liveness_window ready authority fresh
working='9030 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 1'
idle='9030 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 1 1 1'
blocked='9030 1 1 1 0 1 1 0 0 1 0 0 0 0 1 1 1 0 1 0 0 1 1'
disconnected='9030 1 1 1 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1'
fable_like='9030 1 1 1 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 1 0 1 1'
orphaned='9030 1 1 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 1'
dead='9030 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 1 1 0 1 1'
silent_incomplete='9030 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 1 0 1 1 1'
claim_only='9030 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1'
stale_sample='9030 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 1 1 0'
conflicted='9030 1 1 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1'

assert_state "$RUNTIME" "$working" WORKING
assert_state "$RUNTIME" "$idle" IDLE
assert_state "$RUNTIME" "$blocked" BLOCKED
assert_state "$RUNTIME" "$disconnected" DISCONNECTED
assert_state "$RUNTIME" "$fable_like" UNRESPONSIVE
assert_state "$RUNTIME" "$orphaned" ORPHANED
assert_state "$RUNTIME" "$dead" DEAD
assert_state "$RUNTIME" "$silent_incomplete" UNKNOWN
assert_state "$RUNTIME" "$claim_only" UNKNOWN
assert_state "$RUNTIME" "$stale_sample" UNKNOWN
assert_state "$RUNTIME" "$conflicted" CONFLICTED

run_frame "$RUNTIME" '9030 1 1'
[[ "$FRAME_RC" -eq 99 && "$FRAME_OUTPUT" == *'reason=malformed-frame'* ]] ||
  fail "malformed frame did not fail closed: rc=$FRAME_RC output=$FRAME_OUTPUT"

# Causal control: remove only the Sounio-owned obligation-census requirement.
# Rebuild the same Sounio program and replay the unchanged incomplete-silence
# frame. It must become IDLE, proving this exact rule prevents laundering.
module="$ROOT_DIR/stdlib/coordination/loom_lane_health.sio"
entrypoint="$ROOT_DIR/tools/loom/lane_health_main.sio"
needle='obligation_census_complete == 1 && progress_window_complete == 1 && ready_observed == 1'
replacement='progress_window_complete == 1 && ready_observed == 1'
[[ "$(grep -Fc "$needle" "$module")" -eq 1 ]] ||
  fail 'affirmative-absence sabotage point is not unique'
sed "s/$needle/$replacement/" "$module" > "$TEST_ROOT/module-sabotaged.sio"
sed -n '1,$p' "$TEST_ROOT/module-sabotaged.sio" "$entrypoint" > "$TEST_ROOT/runtime-sabotaged.sio"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile \
  "$TEST_ROOT/runtime-sabotaged.sio" -o "$SABOTAGED" >/dev/null
chmod 0755 "$SABOTAGED"
assert_state "$SABOTAGED" "$silent_incomplete" IDLE

printf '%s\n' \
  'sounio-loom-lane-health-selftest: PASS language=Sounio cases=28 fable=unresponsive silence=unknown claim_only=unknown stale=unknown malformed=refused sabotage_affirmative_absence=admits_idle'
