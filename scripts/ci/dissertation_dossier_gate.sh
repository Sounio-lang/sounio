#!/usr/bin/env bash
# Lane 8c — Regulatory dossier generator gate.
#
# Verifies that scripts/dissertation/dossier_generator.sio renders a
# byte-stable Markdown dossier matching the committed golden snapshot
# at tests/golden/dissertation/dossier_rapamycin_snapshot.md.
#
# Acceptance:
#   1. souc check on the smoke test rc=0 (transitively checks the
#      generator module).
#   2. souc compile + run produces deterministic stdout.
#   3. stdout bytewise-equals the golden snapshot.
#   4. The compiled program ends with the line "PASS dossier_smoke".
#
# Determinism note: the smoke test embeds fixed values for
# generated_at_utc, commit_sha, and sounio_version, so no timestamp
# stripping is required.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
SRC_GEN="scripts/dissertation/dossier_generator.sio"
SRC_TEST="tests/run-pass/dossier_smoke.sio"
GOLDEN="tests/golden/dissertation/dossier_rapamycin_snapshot.md"

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

if [[ ! -x "$SOUC" ]]; then
    echo "FAIL: souc not executable at $SOUC" >&2
    exit 2
fi
for f in "$SRC_GEN" "$SRC_TEST" "$GOLDEN"; do
    if [[ ! -f "$f" ]]; then
        echo "FAIL: required file missing: $f" >&2
        exit 2
    fi
done

# (1) souc check on smoke test (transitive over generator)
if "$SOUC" check "$SRC_TEST" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc check failed: $SRC_TEST"
fi

# (2) compile + run + diff
TMP_BIN="$(mktemp -t dossier.XXXXXX)"
TMP_OUT="$(mktemp -t dossier.XXXXXX.out)"
trap 'rm -f "$TMP_BIN" "$TMP_OUT"' EXIT

if "$SOUC" compile "$SRC_TEST" -o "$TMP_BIN" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc compile failed: $SRC_TEST"
fi

if [[ -x "$TMP_BIN" ]]; then
    if "$TMP_BIN" >"$TMP_OUT" 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "compiled smoke test exited non-zero"
    fi
else
    note_fail "compiled binary not executable"
fi

if [[ -s "$TMP_OUT" ]]; then
    if diff -q "$GOLDEN" "$TMP_OUT" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "stdout differs from golden $GOLDEN"
        echo "  (diff -u $GOLDEN <runtime>):" >&2
        diff -u "$GOLDEN" "$TMP_OUT" | head -60 >&2 || true
    fi
fi

# (3) acceptance line: very last line of stdout
if tail -n1 "$TMP_OUT" 2>/dev/null | grep -qx "PASS dossier_smoke"; then
    PASS=$((PASS + 1))
else
    note_fail "last line of stdout != 'PASS dossier_smoke'"
fi

echo "=== Lane 8c regulatory dossier generator gate ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [[ "${#FIRST_FAILURES[@]}" -gt 0 ]]; then
    echo "  first failures:"
    for f in "${FIRST_FAILURES[@]}"; do
        echo "    $f"
    done
fi

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
