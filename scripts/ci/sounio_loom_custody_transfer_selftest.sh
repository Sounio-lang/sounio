#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-custody-transfer-selftest.XXXXXX")"
RUNTIME="$TEST_ROOT/sounio-custody-transfer"
SABOTAGED="$TEST_ROOT/sounio-custody-transfer-sabotaged"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-custody-transfer-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_frame() {
  local runtime="$1" frame="$2"
  set +e
  FRAME_OUTPUT="$(printf '%s\n' "$frame" | "$runtime" 2>&1)"
  FRAME_RC=$?
  set -e
}

assert_decision() {
  local runtime="$1" frame="$2" expected="$3"
  run_frame "$runtime" "$frame"
  [[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == *"decision=$expected"* ]] ||
    fail "expected $expected: rc=$FRAME_RC output=$FRAME_OUTPUT frame=$frame"
}

SOUNIO_LOOM_CUSTODY_TRANSFER_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_custody_transfer.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_CUSTODY_TRANSFER_SELFTEST PASS cases=30' ]] ||
  fail "Sounio-owned selftest did not pass: $selftest"

# schema phase policy source_catalog loom_catalog staged sealed resume_bound
# source_active source_identity source_quiesced target_active presence endpoint
# target_session rollback deadline observation_authority sample_fresh
prepare='9040 1 1 1 0 1 1 1 1 1 0 0 0 0 0 1 0 1 1'
source_quiesced='9040 2 1 1 0 1 1 1 0 1 1 0 0 0 0 1 0 1 1'
target_proven='9040 3 1 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1'
dual_authority='9040 3 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1'
target_timeout='9040 3 1 1 0 1 1 1 0 1 1 0 0 0 0 1 1 1 1'
committed='9040 5 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1'
rollback='9040 6 1 1 0 0 0 0 0 1 1 0 0 0 0 1 0 1 1'
rolled_back='9040 6 1 1 0 0 0 0 1 1 1 0 0 0 0 1 0 1 1'

assert_decision "$RUNTIME" "$prepare" QUIESCE_SOURCE
assert_decision "$RUNTIME" "$source_quiesced" START_TARGET
assert_decision "$RUNTIME" "$target_proven" COMMIT_TARGET
assert_decision "$RUNTIME" "$dual_authority" ABORT_TARGET
assert_decision "$RUNTIME" "$target_timeout" START_SOURCE_ROLLBACK
assert_decision "$RUNTIME" "$committed" COMPLETE
assert_decision "$RUNTIME" "$rollback" START_SOURCE_ROLLBACK
assert_decision "$RUNTIME" "$rolled_back" ROLLED_BACK

run_frame "$RUNTIME" '9040 1 1'
[[ "$FRAME_RC" -eq 119 && "$FRAME_OUTPUT" == *'decision=DENY_MALFORMED'* ]] ||
  fail "malformed frame did not fail closed: rc=$FRAME_RC output=$FRAME_OUTPUT"

# Causal control: remove only the Sounio-owned dual-authority guard. The same
# hostile frame must then reach COMMIT_TARGET, proving this exact rule refuses
# the laundering route.
module="$ROOT_DIR/stdlib/coordination/loom_custody_transfer.sio"
entrypoint="$ROOT_DIR/tools/loom/custody_transfer_main.sio"
needle='if source_active == 1 && target_active == 1 {'
replacement='if false {'
[[ "$(grep -Fc "$needle" "$module")" -eq 1 ]] ||
  fail 'dual-authority sabotage point is not unique'
sed "s/$needle/$replacement/" "$module" > "$TEST_ROOT/module-sabotaged.sio"
sed -n '1,$p' "$TEST_ROOT/module-sabotaged.sio" "$entrypoint" > \
  "$TEST_ROOT/runtime-sabotaged.sio"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile \
  "$TEST_ROOT/runtime-sabotaged.sio" -o "$SABOTAGED" >/dev/null
chmod 0755 "$SABOTAGED"
assert_decision "$SABOTAGED" "$dual_authority" COMMIT_TARGET

printf '%s\n' \
  'sounio-loom-custody-transfer-selftest: PASS language=Sounio cases=30 prepare=quiesce source_quiesced=start-target target_proven=commit dual_authority=abort-target timeout=rollback committed=complete malformed=refused sabotage_dual_authority=admits-commit'
