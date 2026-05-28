#!/usr/bin/env bash
# scripts/gates/g5a_formatter_idempotent.sh — G5a formatter idempotency gate
#
# Verifies that `souc format` is idempotent: format(format(x)) == format(x).
# Runs against a selection of stdlib files plus the formatter test fixture.
# Exits 0 on full PASS, 1 on any failure.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT_DIR/bin/souc"

pass=0
fail=0

check_file() {
    local f="$1"
    [[ -f "$f" ]] || return 0   # skip if absent (optional stdlib files)

    cp "$f" /tmp/g5a_f1.sio
    "$SOUC" format /tmp/g5a_f1.sio > /tmp/g5a_f2.sio
    "$SOUC" format /tmp/g5a_f2.sio > /tmp/g5a_f3.sio

    if diff -q /tmp/g5a_f2.sio /tmp/g5a_f3.sio > /dev/null 2>&1; then
        echo "PASS (idempotent): $f"
        pass=$((pass + 1))
    else
        echo "FAIL (NOT IDEMPOTENT): $f" >&2
        diff /tmp/g5a_f2.sio /tmp/g5a_f3.sio >&2 || true
        fail=$((fail + 1))
    fi
}

# Core stdlib targets
for f in \
    "$ROOT_DIR/stdlib/math/lib.sio" \
    "$ROOT_DIR/stdlib/io/lib.sio" \
    "$ROOT_DIR/stdlib/units/lib.sio"
do
    check_file "$f"
done

# Formatter test fixture
check_file "$ROOT_DIR/tests/run-pass/format_idempotent.sio"

echo ""
echo "G5a idempotency: $pass PASS, $fail FAIL"

if [[ $fail -gt 0 ]]; then
    echo "G5a idempotency FAIL" >&2
    exit 1
fi

echo "G5a idempotency PASS"
exit 0
