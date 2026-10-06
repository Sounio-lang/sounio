#!/usr/bin/env bash
# Positive-control contract for compile-fail harness behaviour in
# scripts/dev/run_sio_test_suite_v2.sh:
#   matching diagnostic  → accept
#   wrong diagnostic     → reject
#   SEGV / signal        → reject (never a valid compile-fail)
#   timeout              → reject
#
# GATE_CONTRACT: v0
# GATE_ID: compile_fail_harness_contract
# GATE_CLAIMS: compile-fail means diagnostic, not crash or hang
# GATE_ENGINE: harness (fake compilers)
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT_DIR/scripts/dev/run_sio_test_suite_v2.sh"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-compile-fail-contract.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

[[ -f "$RUNNER" ]] || { echo "missing runner: $RUNNER" >&2; exit 1; }
bash -n "$RUNNER"

FIXTURE="$WORK_DIR/compile_fail_contract.sio"
TEST_LIST="$WORK_DIR/tests.txt"
cat >"$FIXTURE" <<'FIXT'
//@ compile-fail
//@ error-pattern: expected compiler diagnostic
//@ timeout: 1

fn main() {}
FIXT
printf '%s\n' "$FIXTURE" >"$TEST_LIST"

cat >"$WORK_DIR/diagnostic-souc" <<'EOFC'
#!/usr/bin/env bash
printf 'expected compiler diagnostic\n' >&2
exit 1
EOFC

cat >"$WORK_DIR/wrong-diagnostic-souc" <<'EOFC'
#!/usr/bin/env bash
printf 'different compiler failure\n' >&2
exit 1
EOFC

cat >"$WORK_DIR/signal-souc" <<'EOFC'
#!/usr/bin/env bash
printf 'expected compiler diagnostic\n' >&2
kill -s SEGV "$$"
EOFC

cat >"$WORK_DIR/timeout-souc" <<'EOFC'
#!/usr/bin/env bash
sleep 5
EOFC

chmod +x "$WORK_DIR"/*-souc

HARNESS_RC=0
run_harness() {
  local compiler="$1"
  local log="$2"
  if SOUNIO_TEST_SOUC_BIN="$compiler" SOUNIO_TEST_JOBS=1 \
      bash "$RUNNER" --test-list "$TEST_LIST" --jobs 1 >"$log" 2>&1; then
    HARNESS_RC=0
  else
    HARNESS_RC=$?
  fi
}

expect_failure() {
  local label="$1"
  local expected="$2"
  local log="$3"
  if [[ "$HARNESS_RC" -eq 0 ]]; then
    echo "compile-fail harness: ${label} was accepted" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -Fq "$expected" "$log"; then
    echo "compile-fail harness: ${label} did not report: ${expected}" >&2
    cat "$log" >&2
    exit 1
  fi
}

run_harness "$WORK_DIR/diagnostic-souc" "$WORK_DIR/diagnostic.log"
if [[ "$HARNESS_RC" -ne 0 ]]; then
  echo 'compile-fail harness: matching diagnostic was rejected' >&2
  cat "$WORK_DIR/diagnostic.log" >&2
  exit 1
fi
grep -Fq 'All tests passed!' "$WORK_DIR/diagnostic.log" \
  || grep -Fq 'Pass:' "$WORK_DIR/diagnostic.log" \
  || { echo 'compile-fail harness: no pass summary' >&2; cat "$WORK_DIR/diagnostic.log" >&2; exit 1; }

run_harness "$WORK_DIR/wrong-diagnostic-souc" "$WORK_DIR/wrong-diagnostic.log"
expect_failure 'wrong diagnostic' 'missing error: expected compiler diagnostic' "$WORK_DIR/wrong-diagnostic.log"

run_harness "$WORK_DIR/signal-souc" "$WORK_DIR/signal.log"
expect_failure 'signal termination' 'compile terminated by signal 11' "$WORK_DIR/signal.log"

run_harness "$WORK_DIR/timeout-souc" "$WORK_DIR/timeout.log"
expect_failure 'timeout' 'compile timed out after 1s' "$WORK_DIR/timeout.log"

echo 'COMPILE_FAIL_HARNESS_CONTRACT_PASS'
echo "[compile-fail-contract] GATE_RECEIPT id=compile_fail_harness_contract result=pass measured=1 inputs=4 assertions=4"
exit 0
