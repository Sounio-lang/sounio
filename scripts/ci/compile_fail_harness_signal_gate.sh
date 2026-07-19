#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT_DIR/scripts/dev/run_sio_test_suite_v2.sh"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-compile-fail-signal.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

FIXTURE="$WORK_DIR/compile_fail_signal.sio"
TEST_LIST="$WORK_DIR/tests.txt"

cat >"$FIXTURE" <<'EOF'
//@ compile-fail
//@ error-pattern: expected compiler diagnostic

fn main() {}
EOF
printf '%s\n' "$FIXTURE" >"$TEST_LIST"

cat >"$WORK_DIR/diagnostic-souc" <<'EOF'
#!/usr/bin/env bash
printf 'expected compiler diagnostic\n' >&2
exit 1
EOF

cat >"$WORK_DIR/wrong-diagnostic-souc" <<'EOF'
#!/usr/bin/env bash
printf 'different compiler diagnostic\n' >&2
exit 1
EOF

cat >"$WORK_DIR/signal-souc" <<'EOF'
#!/usr/bin/env bash
ulimit -c 0
printf 'expected compiler diagnostic\n' >&2
kill -s SEGV "$$"
EOF

chmod +x "$WORK_DIR"/*-souc

run_harness() {
    local compiler="$1"
    local log="$2"

    HARNESS_RC=0
    SOUNIO_TEST_SOUC_BIN="$compiler" SOUNIO_TEST_JOBS=1 \
        bash "$RUNNER" --test-list "$TEST_LIST" --jobs 1 >"$log" 2>&1 \
        || HARNESS_RC=$?
}

expect_rejection() {
    local label="$1"
    local expected="$2"
    local log="$3"

    if [[ $HARNESS_RC -eq 0 ]]; then
        echo "compile-fail harness accepted $label" >&2
        cat "$log" >&2
        exit 1
    fi
    if ! grep -Fq "$expected" "$log"; then
        echo "compile-fail harness did not classify $label as expected" >&2
        cat "$log" >&2
        exit 1
    fi
}

run_harness "$WORK_DIR/diagnostic-souc" "$WORK_DIR/diagnostic.log"
if [[ $HARNESS_RC -ne 0 ]]; then
    echo "compile-fail harness rejected a matching diagnostic" >&2
    cat "$WORK_DIR/diagnostic.log" >&2
    exit 1
fi
grep -Fqx 'All tests passed!' "$WORK_DIR/diagnostic.log"

run_harness "$WORK_DIR/wrong-diagnostic-souc" "$WORK_DIR/wrong-diagnostic.log"
expect_rejection \
    "a mismatched diagnostic" \
    "missing error: expected compiler diagnostic" \
    "$WORK_DIR/wrong-diagnostic.log"

run_harness "$WORK_DIR/signal-souc" "$WORK_DIR/signal.log"
expect_rejection \
    "signal termination" \
    "compile terminated by signal 11 (exit 139)" \
    "$WORK_DIR/signal.log"

echo "COMPILE_FAIL_HARNESS_SIGNAL_PASS"
