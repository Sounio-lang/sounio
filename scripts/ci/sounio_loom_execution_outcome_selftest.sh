#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-execution-outcome.XXXXXX")"
RUNTIME="$TEST_ROOT/execution-outcome"
SABOTAGED_RUNTIME="$TEST_ROOT/execution-outcome-sabotaged"
SABOTAGED_MODULE="$TEST_ROOT/loom_execution_outcome_authority.sio"
COMBINED="$TEST_ROOT/sabotaged.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-execution-outcome-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_EXECUTION_OUTCOME_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_execution_outcome.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_EXECUTION_OUTCOME_SELFTEST PASS cases=28' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
prefix='9022 3 1 1 0 0 1 1 1 1 1 1'
valid_frame="$prefix $one $one $one $one $one $one $one $one $one $one $one $one $one"
zero_result_frame="$prefix $one $one $one $one $one $one $one $one $one $one $one $one $zero"

valid="$(printf '%s\n' "$valid_frame" | "$RUNTIME")"
[[ "$valid" == 'SOUNIO_EXECUTION_OUTCOME_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "valid outcome refused: $valid"

zero_result="$(printf '%s\n' "$zero_result_frame" | "$RUNTIME" || true)"
[[ "$zero_result" == 'SOUNIO_EXECUTION_OUTCOME_DENY code=316 reason=result-digest-missing stage=SEMANTICS_FROZEN' ]] ||
  fail "zero result did not hit the dedicated refusal: $zero_result"

# Causal sabotage: remove only the nonzero result-digest refusal. The same
# zero-result frame must become ALLOW, proving that rule caused code 316.
sed '/if !execution_outcome_required_digest(result_hash) { return 316 }/d' \
  "$ROOT_DIR/stdlib/coordination/loom_execution_outcome_authority.sio" \
  > "$SABOTAGED_MODULE"
[[ "$(sha256sum "$SABOTAGED_MODULE" | cut -d' ' -f1)" != \
   "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_execution_outcome_authority.sio" | cut -d' ' -f1)" ]] ||
  fail 'sabotage did not alter the Sounio source'
sed -n '1,$p' "$SABOTAGED_MODULE" "$ROOT_DIR/tools/loom/execution_outcome_main.sio" > "$COMBINED"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$COMBINED" \
  -o "$SABOTAGED_RUNTIME" >/dev/null
chmod 0755 "$SABOTAGED_RUNTIME"
sabotaged="$(printf '%s\n' "$zero_result_frame" | "$SABOTAGED_RUNTIME")"
[[ "$sabotaged" == 'SOUNIO_EXECUTION_OUTCOME_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "causal sabotage did not admit the unchanged frame: $sabotaged"

printf '%s\n' \
  'sounio-loom-execution-outcome-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9022 cases=28 zero_result=DENY316 causal_sabotage=ALLOW'
