#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
MADAROS="$ROOT/bin/madaros"
WITNESS="$ROOT/self-hosted/native/test_code_capacity.sio"

fail() {
  printf 'madaros-native-code-capacity gate: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN is not executable: $RAW_MADAROS"

grep -Fq 'pub var NC_BIG_CODE: [i8; 8388608] = [0; 8388608]' "$ROOT/self-hosted/native/encode.sio" \
  || fail '8 MiB BSS code tier is absent'
grep -Fq 'pub fn nc_code_capacity_bytes() -> i64 { 8388608 }' "$ROOT/self-hosted/native/encode.sio" \
  || fail 'checked code capacity does not match the BSS tier'
grep -Fq 'fn witness_code_capacity_bytes() -> i64 { 8388608 }' "$WITNESS" \
  || fail 'boundary witness is not tied to the checked code tier'
grep -Fq 'var NC_BIG_ELF: [i8; 16777216] = [0; 16777216]' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail '16 MiB BSS ELF tier is absent'
grep -Fq 'NC_CODE_DROPPED_BYTES = NC_CODE_DROPPED_BYTES + 1' "$ROOT/self-hosted/native/encode.sio" \
  || fail 'overflow accounting is absent from shared emitters'
grep -Fq 'if nc.code_overflow || NC_CODE_OVERFLOWED {' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'active finalizer does not preserve rc19 exhaustion handling'

if rg -n 'len < 2097152|jns_pos \+ 1 < 2097152|NC_BIG_ELF.*4194304|\(text_offset \+ i\) < 4194304|\(rodata_offset \+ i\) < 4194304' \
  "$ROOT/self-hosted/native/encode.sio" \
  "$ROOT/self-hosted/native/codegen.sio" \
  "$ROOT/self-hosted/native/codegen_x86_linux.sio"; then
  fail 'a stale code or ELF capacity guard remains'
fi

work="$(mktemp -d -t madaros-native-code-capacity.XXXXXX)"
trap 'rm -rf "$work"' EXIT

MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$WITNESS" >"$work/check.log" 2>&1 || {
  cat "$work/check.log" >&2
  fail 'boundary witness did not check'
}
MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$WITNESS" >"$work/run.log" 2>&1 || {
  cat "$work/run.log" >&2
  fail 'boundary witness did not run'
}
cat "$work/run.log"
grep -Fxq 'PASS native_code_capacity below_at_above rc19' "$work/run.log" \
  || fail 'exact boundary PASS receipt absent'

printf 'madaros-native-code-capacity gate: PASS\n'
