#!/usr/bin/env bash
# Gate: native-v2 packed-string IndexGet (s[i] on read_file / str_from_bytes).
#
# Pre-fix: s[i] on a packed *u8 (read_file / str_from_bytes result) was lowered
# as a GC-handle array load (resolve handle + [base+idx*8]) → SIGSEGV. The
# explicit alternative str_char_at / str_len worked. Fix: lower.sio routes string
# bases to IrIndexGet label_id=3 → native_v2_core_emit_raw_byte_array_load_into
# (movzx byte [base+idx]).
#
# Requires a current-source Madaros carrying the lower.sio change (shipping
# prebuilt bin/souc lags). Point the gate at one:
#   SOUNIO_TEST_SOUC_BIN=/path/to/current-source/madaros \
#     bash scripts/native_string_index_packed_gate.sh
# Dev-tier (not wired into ci.yml; shipping prebuilt refreshed on separate cadence).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
PROBE="tests/data_io_gated/string_index_packed.sio"
fail=0

# Seed the relative probe file the program opens.
printf 'hi\n' > "$OUT/probe.txt"

if ! "$SOUC" compile "$PROBE" -o "$OUT/probe.elf" >/dev/null 2>"$OUT/cerr"; then
  echo "FAIL compile string_index_packed.sio"
  tail -30 "$OUT/cerr" || true
  exit 1
fi
chmod +x "$OUT/probe.elf"

got="$( cd "$OUT" && timeout 20 ./probe.elf 2>/dev/null || true )"
want=$'SFB:72 105 33\nRF:3 104 105 10\nLIT:72 105\nAT:104 105\nARR:65 66\nSTRING_INDEX_OK'

if [[ "$got" == "$want" ]]; then
  echo "PASS packed-string s[i] (str_from_bytes + read_file + lit + arr regression)"
  echo "NATIVE_STRING_INDEX_PACKED_GATE_OK"
  exit 0
fi

echo "FAIL stdout mismatch"
echo "--- got ---"
printf '%s\n' "$got"
echo "--- want ---"
printf '%s\n' "$want"
# Distinguish SEGV (empty) from wrong values
if [[ -z "$got" ]]; then
  echo "(empty output — likely SIGSEGV on s[i] packed-string path)"
fi
exit 1
