#!/usr/bin/env bash
# End-to-end acceptance of a native Madaros host and its native output.
# Usage: bash scripts/ci/madaros_macos_arm64_acceptance_gate.sh /path/to/madaros
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(uname -s)/$(uname -m)" != Darwin/arm64 ]]; then
    echo 'ERROR: this gate requires execution on macOS arm64' >&2
    exit 2
fi
MADAROS_BIN="${1:?pass the native Madaros executable path}"
case "$MADAROS_BIN" in
    /*) ;;
    *) MADAROS_BIN="$PWD/$MADAROS_BIN" ;;
esac
file "$MADAROS_BIN" | grep -Eq 'Mach-O 64-bit.*arm64'
codesign --verify "$MADAROS_BIN"
GATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-macos-acceptance.XXXXXX")"
echo "artifacts=$GATE_DIR"
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

"$MADAROS_BIN" --version >"$GATE_DIR/version.log" 2>&1
grep -Fq 'Madaros v' "$GATE_DIR/version.log"
SOURCE="$ROOT_DIR/tests/run-pass/a64_unit_literal_return.sio"
"$MADAROS_BIN" check "$SOURCE" >"$GATE_DIR/check.log" 2>&1
grep -Fq 'check: OK' "$GATE_DIR/check.log"
"$MADAROS_BIN" build "$SOURCE" --target aarch64-macos -o "$GATE_DIR/program" >"$GATE_DIR/build.log" 2>&1
if ! file "$GATE_DIR/program" | grep -Eq 'Mach-O 64-bit.*arm64'; then
    echo 'FAIL: Madaros build did not emit a Mach-O arm64 executable' >&2
    file "$GATE_DIR/program" >&2
    exit 1
fi
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$GATE_DIR/build.log"; then
    echo 'FAIL: backend reported a refused or missing function body' >&2
    exit 1
fi
chmod +x "$GATE_DIR/program"
codesign -s - -f "$GATE_DIR/program"
codesign --verify "$GATE_DIR/program"
"$GATE_DIR/program" >"$GATE_DIR/stdout.log" 2>"$GATE_DIR/stderr.log"
sed -n 's|^//@ expect-stdout: ||p' "$SOURCE" >"$GATE_DIR/expected.log"
test -s "$GATE_DIR/expected.log"
diff -u "$GATE_DIR/expected.log" "$GATE_DIR/stdout.log"
"$MADAROS_BIN" run "$SOURCE" --target aarch64-macos -o "$GATE_DIR/run-program" >"$GATE_DIR/run.log" 2>"$GATE_DIR/run.stderr.log"
tail -n 2 "$GATE_DIR/run.log" >"$GATE_DIR/run.stdout.log"
diff -u "$GATE_DIR/expected.log" "$GATE_DIR/run.stdout.log"
CONTROL_SOURCE="$ROOT_DIR/tests/run-pass/madaros_macos_integer_control.sio"
"$MADAROS_BIN" run "$CONTROL_SOURCE" --target aarch64-macos -o "$GATE_DIR/control-program" >"$GATE_DIR/control.log" 2>"$GATE_DIR/control.stderr.log"
sed -n 's|^//@ expect-stdout: ||p' "$CONTROL_SOURCE" >"$GATE_DIR/control.expected.log"
tail -n 2 "$GATE_DIR/control.log" >"$GATE_DIR/control.stdout.log"
diff -u "$GATE_DIR/control.expected.log" "$GATE_DIR/control.stdout.log"
set +e
"$MADAROS_BIN" run "$ROOT_DIR/tests/native/madaros_macos_exit_status.sio" --target aarch64-macos -o "$GATE_DIR/exit-program" >"$GATE_DIR/exit.log" 2>"$GATE_DIR/exit.stderr.log"
program_status=$?
set -e
if [[ "$program_status" != 7 ]]; then
    echo "FAIL: CLI run returned $program_status; expected program exit 7" >&2
    exit 1
fi
echo 'PASS: native Madaros check, Mach-O arm64 build, CLI run, struct/reference mutation, signed integer loop, stdout, and exit propagation'
