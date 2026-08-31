#!/usr/bin/env bash
# Harness self-test for //@ expect-stdout-contains and unknown expect-* keys.
#
# The control the defect requires: a garbage marker must go red. If this
# script stays green while a mutated marker still passes, the bug is back.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HARNESS="$ROOT_DIR/scripts/dev/run_sio_test_suite_v2.sh"
HELLO="$ROOT_DIR/tests/run-pass/hello.sio"
REAL_ONE="$ROOT_DIR/tests/run-pass/struct_array_elem_method_dispatch.sio"

fail() { echo "TEST_SUITE_ANNOTATION_SELFTEST_FAIL: $*" >&2; exit 1; }

[[ -f "$HARNESS" ]] || fail "harness missing: $HARNESS"
[[ -f "$HELLO" ]] || fail "hello fixture missing"
[[ -f "$REAL_ONE" ]] || fail "control fixture missing: $REAL_ONE"

bash -n "$HARNESS"

# Shape check: payload extraction must stay parameter-expansion, not quoted =~.
# The previous vacuous-regex bug made every expect-stdout assertion match
# the empty string. Reintroducing that shape for -contains would reimplement it.
if grep -nE '\[\[ "\$line" =~ "//@ expect-stdout-contains:\\ ' "$HARNESS" >/dev/null; then
    fail "expect-stdout-contains extraction uses quoted =~ (vacuous-regex shape)"
fi
grep -Fq 'expect-stdout-contains: "*' "$HARNESS" \
    || fail "harness does not prefix-match //@ expect-stdout-contains:"
grep -Fq 'unknown annotation:' "$HARNESS" \
    || fail "harness does not fail closed on unknown expect-* keys"

TMP="$(mktemp -d /tmp/sounio-ann-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

run_list() {
    local list="$1"
    # Unset CI so a one-file selection is allowed. Do not load the full-suite
    # known-failure manifest (that path is junit + no filter only, but be explicit).
    SOUNIO_TEST_KNOWN_FAILURES_FILE="" \
        bash "$HARNESS" --test-list "$list" --jobs 1 --verbose
}

expect_rc() {
    local want="$1"
    local log="$2"
    shift 2
    set +e
    "$@" >"$log" 2>&1
    local rc=$?
    set -e
    [[ "$rc" -eq "$want" ]] || {
        echo "----- log -----" >&2
        cat "$log" >&2
        fail "expected exit $want, got $rc"
    }
}

# --- 1. Live contains: hello prints "Hello, Sounio!" ---
cp "$HELLO" "$TMP/hello_contains.sio"
# Insert the assertion after the existing //@ run-pass line.
awk '
    NR==1 { print; print "//@ expect-stdout-contains: Hello, Sounio!"; next }
    { print }
' "$HELLO" > "$TMP/hello_contains.sio"
printf '%s\n' "$TMP/hello_contains.sio" > "$TMP/list_pass.txt"
expect_rc 0 "$TMP/pass.log" run_list "$TMP/list_pass.txt"
grep -Fq "PASS" "$TMP/pass.log" || fail "live contains marker did not PASS"

# --- 2. Garbage marker on one of the original 11 must go red ---
awk '
    { sub(/^\/\/@ expect-stdout-contains: .*/, "//@ expect-stdout-contains: THIS_MARKER_IS_GARBAGE") }
    { print }
' "$REAL_ONE" > "$TMP/garbage.sio"
grep -Fq 'THIS_MARKER_IS_GARBAGE' "$TMP/garbage.sio" \
    || fail "failed to rewrite control fixture marker"
printf '%s\n' "$TMP/garbage.sio" > "$TMP/list_garbage.txt"
expect_rc 1 "$TMP/garbage.log" run_list "$TMP/list_garbage.txt"
grep -Fq "missing stdout contains: THIS_MARKER_IS_GARBAGE" "$TMP/garbage.log" \
    || fail "garbage marker did not fail with missing stdout contains"

# --- 3. Unknown expect-* key must fail before it can pass vacuously ---
awk '
    NR==1 { print; print "//@ expect-stdout-not-a-thing: Hello, Sounio!"; next }
    { print }
' "$HELLO" > "$TMP/unknown.sio"
printf '%s\n' "$TMP/unknown.sio" > "$TMP/list_unknown.txt"
expect_rc 1 "$TMP/unknown.log" run_list "$TMP/list_unknown.txt"
grep -Fq "unknown annotation: expect-stdout-not-a-thing" "$TMP/unknown.log" \
    || fail "unknown expect-* key was not rejected"

echo "TEST_SUITE_ANNOTATION_SELFTEST_PASS"
