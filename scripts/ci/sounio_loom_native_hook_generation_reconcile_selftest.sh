#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
AUTHORITY="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-native-hook-generation-reconcile"

fail() {
  printf 'sounio-loom-native-hook-generation-reconcile-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_reconcile.sh" >/dev/null

expect_decision() {
  local frame="$1" expected="$2" expected_rc="${3:-0}" output rc
  set +e
  output="$(printf '%s\n' "$frame" | "$AUTHORITY" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -eq "$expected_rc" && "$output" == *" $expected "* ]] ||
    fail "expected $expected rc=$expected_rc, observed rc=$rc output=$output"
}

expect_decision '9047 1 3 16713727 0 0 0 1 2 3 4 8 8' KEEP
expect_decision '9047 1 3 16746495 0 0 0 1 2 3 4 8 8' KEEP
expect_decision '9047 1 3 16721919 2 2 3 1 2 3 4 8 8' QUARANTINE_ELIGIBLE
expect_decision '9047 2 3 16721919 2 2 3 1 2 3 4 8 8' QUARANTINE_READY
expect_decision '9047 2 3 16721917 2 2 3 1 2 3 4 8 8' DENY683 42
expect_decision '9047 2 3 16721903 2 2 3 1 2 3 4 8 8' DENY684 42
expect_decision '9047 2 3 16720895 2 2 3 1 2 3 4 8 8' DENY685 42
expect_decision '9047 2 3 16656383 2 1 3 1 2 3 4 8 8' DENY686 42
expect_decision '9047 2 3 12527615 2 2 3 1 2 3 4 8 8' DENY687 42
expect_decision '9047 2 3 14624767 2 2 3 1 2 3 4 8 8' DENY687 42
expect_decision '9047 2 3 16713727 2 2 3 1 2 3 4 8 8' DENY688 42
expect_decision '9047 2 3 14755839 2 2 3 1 2 3 4 8 8' DENY689 42

# Causal sabotage removes only the PID-absent fact from an otherwise ready
# frame. The claimed reason remains PID_ABSENT, so action 9047 must refuse.
control="$(printf '%s\n' '9047 2 3 16721919 2 2 3 1 2 3 4 8 8' | "$AUTHORITY")"
set +e
mutant="$(printf '%s\n' '9047 2 3 16713727 2 2 3 1 2 3 4 8 8' | "$AUTHORITY" 2>&1)"
mutant_rc=$?
set -e
[[ "$control" == *' QUARANTINE_READY '* && "$mutant_rc" -eq 42 &&
  "$mutant" == *' DENY688 '* ]] ||
  fail "causal absence sabotage was not load-bearing: control=$control mutant=$mutant"

set +e
malformed="$(printf '%s\n' '9047 2 3' | "$AUTHORITY" 2>&1)"
malformed_rc=$?
set -e
[[ "$malformed_rc" -eq 42 && "$malformed" == *' DENY424 '* ]] ||
  fail "malformed frame did not fail closed: $malformed"

printf '%s\n' \
  'sounio-loom-native-hook-generation-reconcile-selftest: PASS semantic_authority=Sounio action=9047 stage=SOUNIO_EXECUTABLE cases=13 live=KEEP heartbeat_only=KEEP pid_absent=QUARANTINE_READY unreadable=DENY683 identity_drift=DENY684 kernel_unbound=DENY685 related_artifact_drift=DENY686 oracle_or_sabotage_incomplete=DENY687 python_oracle_attempt=DENY687 causal_absence_missing=DENY688 transaction_incomplete=DENY689 malformed=DENY424 causal_sabotage=pid-absence-rule-removed python_executed=false rust_executed=false disposable_oracle_executed=false'
