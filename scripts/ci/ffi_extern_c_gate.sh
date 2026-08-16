#!/usr/bin/env bash
# ffi_extern_c_gate.sh — prove extern "C" system() genuinely executes.
#
# Runs tests/run-pass/ffi_system_exec.sio under a lean_single ELF that is
# known to carry the Track B system() stub (append_extern_c_stubs, 2026-08-15;
# docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md). Asserts three
# things, in increasing strength:
#   1. the compile succeeds (a pre-Track-B seed fails here with E001)
#   2. the program exits 0 and prints its PASS line
#   3. the side-effect file exists  <- the actual regression; the silent
#      no-op satisfies 1 and 2 (a fabricated rc=0) and fails only this
#
# The default Madaros engine does not implement extern "C" system() yet
# (Track A); this gate is engine-forced by construction.
#
# SOUNIO_FFI_GATE_LEAN_BIN — override the lean_single ELF under test
#   (e.g. SOUNIO_FFI_GATE_LEAN_BIN=./gen3.elf after `make build`, to validate
#   the test against a fresh source build before refreshing the seed).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LEAN="${SOUNIO_FFI_GATE_LEAN_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
SRC="$ROOT_DIR/tests/run-pass/ffi_system_exec.sio"
PROBE=/tmp/sounio_ffi_system_probe
WORK="$(mktemp -d /tmp/sounio-ffi-gate.XXXXXX)"
trap 'rm -rf "$WORK" "$PROBE"' EXIT

fail() { echo "[ffi-extern-c] FAIL: $*" >&2; exit 1; }

[[ -x "$LEAN" ]] || fail "no executable lean_single ELF at $LEAN (set SOUNIO_FFI_GATE_LEAN_BIN?)"
[[ -f "$SRC" ]]  || fail "missing test source: $SRC"
rm -f "$PROBE"

"$LEAN" "$SRC" "$WORK/t.elf" >"$WORK/compile.log" 2>&1 \
  || fail "compile failed (pre-Track-B seed? see compile.log below; refresh per #725)
$(tail -5 "$WORK/compile.log")"
chmod +x "$WORK/t.elf"

set +e
"$WORK/t.elf" >"$WORK/run.log" 2>&1
RC=$?
set -e

grep -qF "ffi_system_exec: PASS" "$WORK/run.log" || fail "PASS line missing from stdout:
$(tail -5 "$WORK/run.log")"
[[ $RC -eq 0 ]] || fail "exit code $RC (expected 0)"
[[ -f "$PROBE" ]] || fail "side-effect file absent — system() claimed rc=0 without executing (the silent no-op this gate exists for)"

echo "[ffi-extern-c] PASS: system() genuinely forked+execed (rc=0, probe file created) under $LEAN"
