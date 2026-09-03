#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPILER_CLI="${SOUC_BIN:-$ROOT/bin/madaros}"
RAW_COMPILER="${SOIR_V6_BSS_LAYOUT_RAW_BIN:-${MADAROS_RAW_BIN:-$ROOT/bin/madaros-linux-x86_64}}"
EXPECTED_RAW_SHA256="${SOIR_V6_BSS_LAYOUT_EXPECTED_RAW_SHA256:-}"
COMPILER_SOURCE_SHA="${SOIR_V6_BSS_LAYOUT_COMPILER_SOURCE_SHA:-not_claimed}"
WRITER="self-hosted/ir/soir_v6_writer.sio"
READER="self-hosted/ir/soir_v6_reader.sio"
RUNTIME_TIMEOUT_SECONDS="${SOIR_V6_BSS_LAYOUT_RUNTIME_TIMEOUT_SECONDS:-15}"
SCHEMA_DESCRIPTOR='SOIRv6:bss-range-v1:le64:header64:dir48:kind5:critical+singleton:payload(module_total,range_count,range_offset,range_size):adler32'
SCHEMA_FINGERPRINT='3946524625'
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-soir-v6-bss-layout.XXXXXX")"

cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

fail() {
  printf 'SOIR_V6_BSS_LAYOUT_FAIL reason=%s\n' "$1" >&2
  exit 1
}

progress() {
  printf 'SOIR_V6_BSS_LAYOUT_PROGRESS stage=%s subject=%s status=%s\n' \
    "$1" "$2" "$3" >&2
}

run_compiler() {
  MADAROS_RAW_BIN="$RAW_COMPILER" "$COMPILER_CLI" "$@"
}

expected_paths() {
  printf '%s\n' \
    scripts/ci/soir_v6_bss_layout_gate.sh \
    self-hosted/ir/soir_v6_reader.sio \
    self-hosted/ir/soir_v6_writer.sio \
    tests/native-v2/soir_v6_bss_layout_reject_witness.sio \
    tests/native-v2/soir_v6_bss_layout_witness.sio
}

schema_adler32() {
  local descriptor="$1"
  local a=1
  local b=0
  local octet
  for octet in $(printf '%s' "$descriptor" | od -An -tu1 -v); do
    a=$(((a + octet) % 65521))
    b=$(((b + a) % 65521))
  done
  printf '%s\n' "$(((b << 16) | a))"
}

[[ -x "$COMPILER_CLI" ]] || fail compiler_cli_missing
[[ -x "$RAW_COMPILER" ]] || fail raw_compiler_missing
[[ "$(head -c 2 "$RAW_COMPILER" 2>/dev/null)" != '#!' ]] || fail raw_compiler_is_launcher
command -v timeout >/dev/null 2>&1 || fail timeout_command_missing
command -v od >/dev/null 2>&1 || fail od_command_missing
[[ "$RUNTIME_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail runtime_timeout_invalid
(( RUNTIME_TIMEOUT_SECONDS <= 60 )) || fail runtime_timeout_exceeds_cap_60

for path in "$WRITER" "$READER"; do
  [[ -f "$path" ]] || fail "missing_${path//\//_}"
done

[[ "$(schema_adler32 "$SCHEMA_DESCRIPTOR")" == "$SCHEMA_FINGERPRINT" ]] \
  || fail schema_descriptor_fingerprint_mismatch
for path in "$WRITER" "$READER"; do
  grep -Fq "// $SCHEMA_DESCRIPTOR" "$path" \
    || fail "schema_descriptor_missing_${path//\//_}"
  grep -Fq "SCHEMA_FINGERPRINT: i64 = $SCHEMA_FINGERPRINT" "$path" \
    || fail "schema_fingerprint_missing_${path//\//_}"
  grep -Fq 'Adler-32 is a non-cryptographic integrity checksum, not authentication.' "$path" \
    || fail "integrity_boundary_missing_${path//\//_}"
done

grep -Fq 'pub let SOIR_V6_WRITER_BSS_FRAME_SIZE: i64 = 144' "$WRITER" \
  || fail writer_frame_contract_missing
grep -Fq 'pub fn soir_v6_writer_preflight_bss_layout(' "$WRITER" \
  || fail writer_preflight_missing
grep -Fq 'pub fn soir_v6_writer_emit_bss_layout(' "$WRITER" \
  || fail writer_emit_missing
grep -Fq 'pub let SOIR_V6_READER_BSS_FRAME_SIZE: i64 = 144' "$READER" \
  || fail reader_frame_contract_missing
grep -Fq 'pub fn soir_v6_reader_decode_bss_layout(' "$READER" \
  || fail reader_decode_missing
grep -Fq 'var section_cursor = SoirV6ReaderCursor {' "$READER" \
  || fail section_local_cursor_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_WORDS: i64 = 7' "$READER" \
  || fail reader_receipt_contract_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_PHYSICAL_WORDS: i64 = 7' "$READER" \
  || fail reader_receipt_physical_contract_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_CODE: i64 = 0' "$READER" \
  || fail reader_receipt_code_slot_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_BYTE_OFFSET: i64 = 1' "$READER" \
  || fail reader_receipt_byte_offset_slot_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_DETAIL: i64 = 2' "$READER" \
  || fail reader_receipt_detail_slot_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_BSS_MODULE_TOTAL: i64 = 3' "$READER" \
  || fail reader_receipt_module_total_slot_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_BSS_RANGE_COUNT: i64 = 4' "$READER" \
  || fail reader_receipt_range_count_slot_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_BSS_RANGE_OFFSET: i64 = 5' "$READER" \
  || fail reader_receipt_range_offset_slot_missing
grep -Fq 'pub let SOIR_V6_READER_RECEIPT_BSS_RANGE_SIZE: i64 = 6' "$READER" \
  || fail reader_receipt_contract_missing

dependency_pattern='^use (parser|ir::ir|ir::serialize|ir::numeric_payload_wire)|\b(IrModule|IrInstr|TyF128|TyF256|TyF512)\b'
set +e
if command -v rg >/dev/null 2>&1; then
  rg -n "$dependency_pattern" "$WRITER" "$READER" >"$TMP/dependency-leak.log" 2>&1
else
  grep -En "$dependency_pattern" "$WRITER" "$READER" >"$TMP/dependency-leak.log" 2>&1
fi
dependency_scan_rc=$?
set -e
if [[ "$dependency_scan_rc" -eq 0 ]]; then
  cat "$TMP/dependency-leak.log" >&2
  fail shadow_dependency_expanded
fi
[[ "$dependency_scan_rc" -eq 1 ]] || fail dependency_scan_error

for default_surface in \
  self-hosted/ir/serialize.sio \
  self-hosted/ir/mod.sio \
  self-hosted/ir/lower.sio \
  self-hosted/check/check.sio \
  self-hosted/compiler/main.sio \
  self-hosted/compiler/module_frontend.sio; do
  [[ -f "$default_surface" ]] || continue
  if grep -Eq 'ir::soir_v6_(writer|reader)' "$default_surface"; then
    fail "shadow_imported_by_${default_surface//\//_}"
  fi
done

bash scripts/ci/madaros_f128_f256_format_identity_gate.sh --structural-only \
  >"$TMP/precision-identity.log" 2>&1 || {
    cat "$TMP/precision-identity.log" >&2
    fail precision_identity_regression
  }

head_sha="$(git rev-parse HEAD)"
tree_sha="$(git rev-parse 'HEAD^{tree}')"
mapfile -t evidence_paths < <(expected_paths)
for evidence_path in "${evidence_paths[@]}"; do
  git ls-files --error-unmatch -- "$evidence_path" >/dev/null 2>&1 \
    || fail evidence_path_untracked
done
dirty_tree="$(git status --porcelain --untracked-files=all)"
if [[ -n "$dirty_tree" ]]; then
  printf '%s\n' "$dirty_tree" >&2
  fail worktree_dirty
fi
if [[ "$COMPILER_SOURCE_SHA" != "not_claimed" && "$COMPILER_SOURCE_SHA" != "$head_sha" ]]; then
  fail compiler_source_sha_mismatch
fi

raw_compiler_sha256="$(sha256sum "$RAW_COMPILER" | awk '{print $1}')"
if [[ -n "$EXPECTED_RAW_SHA256" && "$raw_compiler_sha256" != "$EXPECTED_RAW_SHA256" ]]; then
  fail raw_compiler_sha256_mismatch
fi
compiler_cli_sha256="$(sha256sum "$COMPILER_CLI" | awk '{print $1}')"

for source in "$WRITER" "$READER"; do
  name="$(basename "$source" .sio)"
  progress source_check "$name" begin
  run_compiler check "$source" >"$TMP/$name.check.log" 2>&1 || {
    cat "$TMP/$name.check.log" >&2
    fail "source_check_$name"
  }
  progress source_check "$name" pass
done

compose_and_run() {
  local witness="$1"
  local marker="$2"
  local name composite elf build_log run_log runtime_rc
  name="$(basename "$witness" .sio)"
  composite="$TMP/$name.sio"
  elf="$TMP/$name.elf"
  build_log="$TMP/$name.build.log"
  run_log="$TMP/$name.run.log"

  {
    printf 'module ir::%s_gate\n\n' "$name"
    sed '/^module ir::soir_v6_writer$/d' "$WRITER"
    sed '/^module ir::soir_v6_reader$/d' "$READER"
    sed -e '/^use ir::soir_v6_writer::\*$/d' \
        -e '/^use ir::soir_v6_reader::\*$/d' \
        "$witness"
  } >"$composite"

  progress composite_check "$name" begin
  run_compiler check "$composite" >"$build_log" 2>&1 || {
    cat "$build_log" >&2
    fail "composite_check_$name"
  }
  progress composite_build "$name" begin
  run_compiler --native-v2-compile "$composite" -o "$elf" >>"$build_log" 2>&1 || {
    cat "$build_log" >&2
    fail "composite_build_$name"
  }
  chmod +x "$elf"
  progress composite_runtime "$name" begin
  set +e
  timeout --signal=TERM --kill-after=2s "${RUNTIME_TIMEOUT_SECONDS}s" "$elf" >"$run_log" 2>&1
  runtime_rc=$?
  set -e
  if [[ "$runtime_rc" -eq 124 ]]; then
    cat "$run_log" >&2
    fail "composite_runtime_timeout_$name"
  fi
  if [[ "$runtime_rc" -ne 0 ]]; then
    cat "$run_log" >&2
    fail "composite_runtime_${name}_rc_${runtime_rc}"
  fi
  grep -Fxq "$marker" "$run_log" || {
    cat "$run_log" >&2
    fail "marker_missing_$name"
  }
  progress composite_runtime "$name" pass
}

compose_and_run \
  tests/native-v2/soir_v6_bss_layout_witness.sio \
  SOIR_V6_BSS_LAYOUT_CANONICAL_PASS
compose_and_run \
  tests/native-v2/soir_v6_bss_layout_reject_witness.sio \
  SOIR_V6_BSS_LAYOUT_REJECT_PASS

while IFS= read -r path; do
  sha256sum "$path"
done < <(expected_paths) >"$TMP/lane-content.sha256"
lane_content_sha256="$(sha256sum "$TMP/lane-content.sha256" | awk '{print $1}')"

printf '%s\n' 'SOIR_V6_BSS_LAYOUT_CHECK source=2/2 composite=2/2 precision_identity=preserved'
printf '%s\n' 'SOIR_V6_BSS_LAYOUT_WIRE profile=v6-d0-single-bss-range bytes=144 header=64 directory_entries=1 directory_entry_bytes=48 section=BSS_LAYOUT section_bytes=32 schema_descriptor=canonical schema_fingerprint=adler32-derived little_endian=i64'
printf '%s\n' 'SOIR_V6_BSS_LAYOUT_INTEGRITY checksum=adler32 integrity_only=true authentication=not_claimed corruption_without_reseal=reject corruption_resealed_semantic_invalid=reject'
printf '%s\n' 'SOIR_V6_BSS_LAYOUT_WRITER preflight=exact emit_revalidation=pass validation_rejection_no_write=pass forged_plan=reject deterministic_bytes=pass offset_view=pass terminal_view=pass'
printf '%s\n' 'SOIR_V6_BSS_LAYOUT_READER frame_cursor=local section_cursor=local truncation_0_143=reject trailing=reject unknown_critical=reject unknown_optional=explicit_reject decoded_canary_on_failure=preserved receipt_storage=fixed_scalar_words semantic_words=7 physical_words=7 padding_words=0 aggregate_return=none status_slots=code,byte_offset,detail success_status=zeroed decoded_slots=module_total,range_count,range_offset,range_size slot_reuse=none'
printf '%s\n' 'SOIR_V6_BSS_LAYOUT_BOUNDARY module_graph=not_implemented opcode=not_implemented arena_install=not_implemented function_binding=not_claimed target_layout=not_claimed abi=not_claimed numeric_payload=not_imported multi_section_ordering=not_claimed issue_1162=partial_not_closed default_pipeline=unchanged legacy_v1_v4=kept soir_v5=unchanged'
printf 'SOIR_V6_BSS_LAYOUT_PROVENANCE head=%s tree=%s worktree_clean=pass compiler_source_sha=%s\n' \
  "$head_sha" "$tree_sha" "$COMPILER_SOURCE_SHA"
printf 'SOIR_V6_BSS_LAYOUT_PASS compiler_cli=%s compiler_cli_sha256=%s raw_compiler=%s raw_compiler_sha256=%s head=%s tree=%s lane_content_sha256=%s\n' \
  "$COMPILER_CLI" "$compiler_cli_sha256" "$RAW_COMPILER" "$raw_compiler_sha256" "$head_sha" "$tree_sha" "$lane_content_sha256"
