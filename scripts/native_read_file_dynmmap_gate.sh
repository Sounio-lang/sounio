#!/usr/bin/env bash
# Gate: native-v2 read_file dynamic-mmap (codegen_x86_linux::emit_builtin_read_file).
#
# The old emitter mmap'd a fixed 1 MiB anon buffer per call and read up to 1 MiB, so any
# file >= 1 MiB either truncated silently or (all-NUL-free content, full buffer) made
# downstream str_len/str_char_at walk off the mapping and SIGSEGV. The dynamic emitter
# fstat's the file, mmap's round_up(size + 64, page), and reads mmap_len-1 so a zero tail
# always terminates the string. This gate proves the cap is gone.
#
# read_file for user programs is emitted by codegen_x86_linux (native-v2), which the
# checked-in bin/souc (prebuilt Madaros) LAGS — it must be a current-source Madaros that
# carries #1078 + this change. Point the gate at one:
#   SOUNIO_TEST_SOUC_BIN=/path/to/current-source/madaros bash scripts/native_read_file_dynmmap_gate.sh
# Dev-tier (not wired into ci.yml; the shipping prebuilt is refreshed on a separate cadence).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

if ! "$SOUC" compile tests/data_io_gated/read_file_dynmmap.sio -o "$OUT/probe.elf" >/dev/null 2>&1; then
  echo "FAIL compile read_file_dynmmap.sio"; exit 1
fi
chmod +x "$OUT/probe.elf"

# name:bytes:expected-sentinel
for case in "empty:0:CAP_SMALL" "100KiB:102400:CAP_SMALL" "2MiB:2097152:CAP_2M_OK" "20MiB:20971520:CAP_GE_20M"; do
  name="${case%%:*}"; rest="${case#*:}"; bytes="${rest%%:*}"; want="${rest##*:}"
  head -c "$bytes" /dev/zero | tr '\0' 'A' > "$OUT/big.dat"
  got="$( cd "$OUT" && timeout 20 ./probe.elf 2>/dev/null || true )"
  if [ "$got" = "$want" ]; then
    echo "PASS $name ($bytes bytes) -> $got"
  else
    echo "FAIL $name ($bytes bytes) -> got '${got:-<crash/empty>}' want '$want'"; fail=1
  fi
done

if [ "$fail" = 0 ]; then echo "NATIVE_READ_FILE_DYNMMAP_GATE_OK"; else echo "GATE FAILED"; exit 1; fi
