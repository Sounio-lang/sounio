#!/usr/bin/env bash
# Harness self-test for //@ expect-stdout-contains, unknown expect-* keys, and the
# known-failure manifest's optional `path|substring` reason pin.
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

# Like run_list, but loads the given scratch manifest instead of disabling it --
# for the known-failure reason-pin cases below.
run_list_with_manifest() {
    local list="$1" manifest="$2"
    SOUNIO_TEST_KNOWN_FAILURES_FILE="$manifest" \
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

# --- 4/5/6. Known-failure manifest reason pin (path|substring) ---
# A deterministic, non-timing-dependent failure: fixed exit code, fixed marker.
# Never listed under tests/, so the ordinary suite (which globs tests/run-pass/*.sio
# etc.) never sees it; only --test-list here does.
cat > "$TMP/reason_fixture.sio" <<'SIO'
//@ run-pass
fn main() -> i32 with IO {
    println("selftest_reason_marker_9f3c1")
    1
}
SIO
printf '%s\n' "$TMP/reason_fixture.sio" > "$TMP/list_reason.txt"

# 4. Matching reason: the pin's substring is in test_output ("run exited 1 | ...
#    selftest_reason_marker_9f3c1") -> laundered as a known failure, exit 0.
printf '%s|selftest_reason_marker_9f3c1\n' "$TMP/reason_fixture.sio" > "$TMP/manifest_match.txt"
expect_rc 0 "$TMP/reason_match.log" run_list_with_manifest "$TMP/list_reason.txt" "$TMP/manifest_match.txt"
grep -Fq "Known failures: 1" "$TMP/reason_match.log" \
    || fail "matching reason pin was not accepted as a known failure"

# 5. Mismatching reason: the pin's substring is not in test_output -> a fresh FAIL,
#    not a repeat of whatever the entry was audited for. This is the case the
#    plain-path form (no |) cannot catch: it would launder ANY failure.
printf '%s|THIS_SUBSTRING_IS_NOT_IN_THE_OUTPUT\n' "$TMP/reason_fixture.sio" > "$TMP/manifest_mismatch.txt"
expect_rc 1 "$TMP/reason_mismatch.log" run_list_with_manifest "$TMP/list_reason.txt" "$TMP/manifest_mismatch.txt"
grep -Fq "known-failure reason mismatch" "$TMP/reason_mismatch.log" \
    || fail "mismatching reason pin did not report a reason mismatch"
grep -Fq "FAIL  reason_fixture.sio" "$TMP/reason_mismatch.log" \
    || fail "mismatching reason pin did not fail the suite"

# 6. Legacy plain-path entry (no |): unaffected by the reason-pin feature, still
#    laundered whatever the failure is, exactly as before this feature existed.
printf '%s\n' "$TMP/reason_fixture.sio" > "$TMP/manifest_legacy.txt"
expect_rc 0 "$TMP/reason_legacy.log" run_list_with_manifest "$TMP/list_reason.txt" "$TMP/manifest_legacy.txt"
grep -Fq "Known failures: 1" "$TMP/reason_legacy.log" \
    || fail "legacy plain-path entry regressed"

# --- 7. Mixed environment: SOUNIO_MADAROS_AVAILABLE set but the resolved
#     compiler is actually lean_single -- `requires: lean_single` must still
#     RUN, not skip (Copilot review, sounio-lang/sounio#2694: OR'ing
#     SOUNIO_MADAROS_AVAILABLE, a statement of intent for the sibling
#     `requires: madaros` arm, into this arm's skip condition made a
#     Madaros-capable environment that explicitly selects lean_single via
#     SOUNIO_TEST_SOUC_BIN incorrectly skip a lean_single test on the exact
#     run meant to exercise it). Uses the real lean_single stage2 binary if
#     this checkout has one; skips itself, loudly, rather than passing
#     vacuously, if it doesn't.
LEAN_SINGLE_ELF="$ROOT_DIR/bin/souc-lean-single-x86_64"
if [[ -x "$LEAN_SINGLE_ELF" ]]; then
    cat > "$TMP/lean_single_gate_fixture.sio" <<'SIO'
//@ run-pass
//@ requires: lean_single
fn main() -> i32 with IO {
    println("selftest_lean_single_gate_marker_7be2")
    0
}
SIO
    printf '%s\n' "$TMP/lean_single_gate_fixture.sio" > "$TMP/list_lean_single_gate.txt"
    expect_rc 0 "$TMP/lean_single_gate.log" env SOUNIO_MADAROS_AVAILABLE=1 SOUNIO_TEST_SOUC_BIN="$LEAN_SINGLE_ELF" \
        bash "$HARNESS" --test-list "$TMP/list_lean_single_gate.txt" --jobs 1 --verbose
    grep -Fq "PASS  lean_single_gate_fixture.sio" "$TMP/lean_single_gate.log" \
        || fail "requires:lean_single skipped under SOUNIO_MADAROS_AVAILABLE=1 even though the resolved compiler is lean_single"
else
    echo "SKIP: case 7 (mixed-environment gate) -- no lean_single ELF at $LEAN_SINGLE_ELF" >&2
fi

# --- 8. Unrecognized wrapper identity: `requires: lean_single` must fail
#     CLOSED (skip), not default to "must be lean_single" (Copilot review,
#     sounio-lang/sounio#2694: a wrapper-style SOUNIO_TEST_SOUC_BIN with no
#     `--version` arm at all -- the kind
#     scripts/ci/build_ontology_validation_souc.sh generates -- can still
#     delegate `run` to a Madaros fallback underneath; the old code treated
#     any non-Madaros `--version` output as proof of lean_single and would
#     have run this fixture on that fallback instead of skipping it). This
#     fake wrapper never needs to actually compile anything: the fixture is
#     expected to be skipped before the wrapper's other subcommands are
#     ever invoked.
cat > "$TMP/unknown_wrapper.sh" <<'WRAPPER'
#!/usr/bin/env bash
echo "unknown_wrapper: unrecognized invocation: $*" >&2
exit 1
WRAPPER
chmod +x "$TMP/unknown_wrapper.sh"
cat > "$TMP/lean_single_unknown_fixture.sio" <<'SIO'
//@ run-pass
//@ requires: lean_single
fn main() -> i32 with IO {
    println("selftest_lean_single_unknown_marker_c4a1")
    0
}
SIO
printf '%s\n' "$TMP/lean_single_unknown_fixture.sio" > "$TMP/list_lean_single_unknown.txt"
expect_rc 0 "$TMP/lean_single_unknown.log" env SOUNIO_TEST_SOUC_BIN="$TMP/unknown_wrapper.sh" \
    bash "$HARNESS" --test-list "$TMP/list_lean_single_unknown.txt" --jobs 1 --verbose
grep -Fq "SKIP  lean_single_unknown_fixture.sio" "$TMP/lean_single_unknown.log" \
    || fail "requires:lean_single ran under an unrecognized wrapper identity instead of failing closed"

echo "TEST_SUITE_ANNOTATION_SELFTEST_PASS"
