#!/usr/bin/env bash
# Gate: native-v2 write_file handle ABI (codegen_x86_linux::emit_builtin_write_file).
#
# Local fixed arrays are GC handles with 8-byte boxed element slots after a 32-byte
# header. The pre-fix emitter treated rsi as raw *u8, so write_file(path, buf, n)
# returned EFAULT (-14) and write_file(path, &buf, n) wrote garbage. The fixed emitter
# resolves the handle, unpacks low bytes into a packed buffer (same pattern as
# emit_builtin_str_from_bytes), then open/write/close.
#
# write_file for user programs is emitted by codegen_x86_linux (native-v2), which the
# checked-in bin/souc (prebuilt Madaros) LAGS — it must be a current-source Madaros that
# carries this change. Point the gate at one:
#   SOUNIO_TEST_SOUC_BIN=/path/to/current-source/madaros \
#     bash scripts/native_write_file_handle_abi_gate.sh
# Dev-tier (not wired into ci.yml; the shipping prebuilt is refreshed on a separate cadence).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
GOLDEN="tests/data_io_gated/write_file_csv_golden.csv"
PROBE="tests/data_io_gated/write_file_handle_abi.sio"
fail=0

if [[ ! -f "$GOLDEN" ]]; then
  echo "FAIL missing golden $GOLDEN"; exit 1
fi
if [[ "$(wc -c < "$GOLDEN" | tr -d ' ')" != "12" ]]; then
  echo "FAIL golden must be exactly 12 bytes"; exit 1
fi

if ! "$SOUC" compile "$PROBE" -o "$OUT/probe.elf" >/dev/null 2>"$OUT/cerr"; then
  echo "FAIL compile write_file_handle_abi.sio"
  tail -20 "$OUT/cerr" || true
  exit 1
fi
chmod +x "$OUT/probe.elf"

# Run in $OUT so the relative out path lands there.
got_stdout="$( cd "$OUT" && timeout 20 ./probe.elf 2>/dev/null || true )"
out_file="$OUT/write_file_handle_abi.out"

if [[ "$got_stdout" != "WRITE_OK" ]]; then
  echo "FAIL stdout: got '${got_stdout:-<crash/empty>}' want 'WRITE_OK'"
  fail=1
fi

if [[ ! -f "$out_file" ]]; then
  echo "FAIL missing output file $out_file"
  fail=1
elif ! cmp -s "$out_file" "$GOLDEN"; then
  echo "FAIL byte-exact: output != golden (12-byte CSV)"
  echo "--- golden (od -An -tx1c) ---"
  od -An -tx1c "$GOLDEN" || true
  echo "--- output (od -An -tx1c) ---"
  od -An -tx1c "$out_file" || true
  fail=1
else
  echo "PASS write_file(path, buf, 12) -> byte-exact golden CSV (12 bytes)"
fi

if [[ "$fail" = 0 ]]; then
  echo "NATIVE_WRITE_FILE_HANDLE_ABI_GATE_OK"
else
  echo "GATE FAILED"
  exit 1
fi
