#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${MADAROS_EXPECTED_SHA256:-}"
MADAROS="$ROOT/bin/madaros"
WITNESS="$ROOT/self-hosted/native/test_code_capacity.sio"

fail() {
  printf 'madaros-native-code-capacity gate: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN is not executable: $RAW_MADAROS"
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail 'MADAROS_EXPECTED_SHA256 must pin the explicit Madaros ELF'
actual_raw_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$actual_raw_sha256" == "$EXPECTED_RAW_SHA256" ]] \
  || fail "Madaros SHA-256 mismatch: expected=$EXPECTED_RAW_SHA256 actual=$actual_raw_sha256"

grep -Fq 'pub var NC_BIG_CODE: [i8; 8388608] = [0; 8388608]' "$ROOT/self-hosted/native/encode.sio" \
  || fail '8 MiB BSS code tier is absent'
grep -Fq 'pub fn nc_code_capacity_bytes() -> i64 { 8388608 }' "$ROOT/self-hosted/native/encode.sio" \
  || fail 'checked code capacity does not match the BSS tier'
grep -Fq 'offset >= 0 && offset < nc_code_capacity_bytes()' "$ROOT/self-hosted/native/encode.sio" \
  || fail 'code boundary predicate is not strict at capacity'
grep -Fq 'if nc_code_offset_is_writable(offset) { return 0 }' "$ROOT/self-hosted/native/encode.sio" \
  || fail 'code exhaustion status is not tied to the boundary predicate'
grep -Fq 'fn witness_code_capacity_bytes() -> i64 { 8388608 }' "$WITNESS" \
  || fail 'boundary witness is not tied to the checked code tier'
grep -Fq 'fn witness_flat_reloc_capacity() -> i64 { 131072 }' "$WITNESS" \
  || fail 'boundary witness is not tied to the checked relocation tier'
grep -Fq 'var NC_BIG_ELF: [i8; 16777216] = [0; 16777216]' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail '16 MiB BSS ELF tier is absent'
grep -Fq 'fn nc_elf_capacity_bytes() -> i64 { 16777216 }' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'checked ELF capacity does not match the BSS tier'
grep -Fq 'if file_len <= 64 || file_len > nc_elf_capacity_bytes() { return 13 }' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'ELF preflight is not tied to checked capacity'
grep -Fq 'NC_CODE_DROPPED_BYTES = NC_CODE_DROPPED_BYTES + 1' "$ROOT/self-hosted/native/encode.sio" \
  || fail 'overflow accounting is absent from shared emitters'
grep -Fq 'if nc.code_overflow || NC_CODE_OVERFLOWED {' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'active finalizer does not preserve rc19 exhaustion handling'
grep -Fq '(*nc).flat_reloc_count = nc_flat_reloc_capacity() + 1' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'flat relocation overflow sentinel is absent'
grep -Fq 'fn nc_flat_reloc_capacity() -> i64 { 131072 }' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'checked relocation capacity does not match the measured tier'
for reloc_array in NC_FLAT_RELOC_OFFSETS NC_FLAT_RELOC_KIND_CODES NC_FLAT_RELOC_IS_FUNCTIONS NC_FLAT_RELOC_TARGET_INDICES; do
  grep -Fq "pub var ${reloc_array}: [i64; 131072] = [0; 131072]" "$ROOT/self-hosted/native/frame.sio" \
    || fail "relocation BSS tier mismatch for ${reloc_array}"
done
grep -Fq 'if NC_FLAT_RELOC_DROPPED > 0 || nc.flat_reloc_count > nc_flat_reloc_capacity() {' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'flat relocation overflow is not rejected before ELF write'
grep -Fq 'return 20' "$ROOT/self-hosted/native/codegen_x86_linux.sio" \
  || fail 'flat relocation exhaustion does not have a deterministic status'

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
grep -Fxq 'PASS native_code_capacity below_at_above rc19 reloc_rc20' "$work/run.log" \
  || fail 'exact boundary PASS receipt absent'

printf 'madaros-native-code-capacity gate: PASS\n'
printf 'madaros-native-code-capacity receipt madaros_sha256=%s\n' "$actual_raw_sha256"
