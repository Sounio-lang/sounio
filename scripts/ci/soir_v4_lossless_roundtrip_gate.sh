#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="self-hosted/ir/serialize.sio"
LOADER="self-hosted/compiler/module_loader.sio"
FIXTURE="tests/fixtures/soir_v4/lossless_roundtrip.sio"
WIRE_FIXTURE="tests/fixtures/soir_v4/lossless_wire_runtime.sio"
MODE="${SOUNIO_SOIR_V4_GATE_MODE:-source}"
CHECK_BIN="${SOUNIO_SOIR_V4_CHECK_BIN:-$ROOT_DIR/bin/souc}"

fail() {
  printf 'SOIR_V4_LOSSLESS_ROUNDTRIP_FAIL reason=%s\n' "$1" >&2
  exit 1
}

case "$MODE" in
  source|runtime) ;;
  *) fail "invalid_mode_${MODE}" ;;
esac

for path in "$SOURCE" "$LOADER" "$FIXTURE" "$WIRE_FIXTURE"; do
  [[ -f "$path" ]] || fail "missing_${path//\//_}"
done
[[ -x "$CHECK_BIN" ]] || fail check_compiler_missing

python3 - "$SOURCE" "$LOADER" "$FIXTURE" "$WIRE_FIXTURE" <<'PY'
from pathlib import Path
import sys

source_path, loader_path, fixture_path, wire_fixture_path = map(Path, sys.argv[1:])
source = source_path.read_text(encoding="utf-8")
loader = loader_path.read_text(encoding="utf-8")
fixture = fixture_path.read_text(encoding="utf-8")
wire_fixture = wire_fixture_path.read_text(encoding="utf-8")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise SystemExit(f"SOIR_V4_LOSSLESS_ROUNDTRIP_FAIL reason={reason}")


def section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    require(start >= 0, f"missing_{start_marker.split('(')[0]}")
    end = text.find(end_marker, start)
    require(end > start, f"missing_{end_marker.split('(')[0]}")
    return text[start:end]


require("let SOIR_VERSION: i8 = 4" in source, "writer_version_not_v4")
require("let SOIR_NAME_SIZE: i64 = 136" in source, "name_size_contract")
require("let SOIR_INSTR_SIZE: i64 = 232" in source, "instr_size_contract")
require("let SOIR_FUNCTION_HEADER_V4_SIZE: i64 = 704" in source, "function_header_contract")
require("let SOIR_EPISTEMIC_COUNT_FIELDS_V4: i64 = 36" in source, "epistemic_count_contract")
require("let SOIR_EMPTY_EPISTEMIC_SIZE_V4: i64 = 288" in source, "epistemic_size_contract")
require("let SOIR_ALGEBRA_INFO_SIZE_V4: i64 = 200" in source, "algebra_size_contract")
require("[IrFunction; 1024]" not in source, "stale_function_capacity_1024")
require("ieee754_bits_to_f64" not in source, "arithmetic_f64_reconstruction_present")
require("(bits_to_f64(pair.0), pair.1)" in source, "bitcast_f64_reader_missing")

primitive_read = section(source, "fn read_i8(", "fn read_f64(")
for marker in (
    "SOIR_V4_READ_LIMIT",
    "SOIR_V4_STATUS_TRUNCATED",
    "SOIR_V4_READ_LIMIT - pos",
):
    require(marker in primitive_read, f"bounded_primitive_missing_{marker}")

function_reader = section(source, "fn deserialize_ir_function_into(", "fn write_ir_algebra_info(")
for marker in (
    "SOIR_FUNCTION_HEADER_V4_SIZE > SOIR_V4_READ_LIMIT - pos",
    "instr_count > IR_MAX_INSTRS",
    "param_count > IR_MAX_PARAMS",
    "instr_count > (SOIR_V4_READ_LIMIT - p) / SOIR_INSTR_SIZE",
    "(*out).instr_count = instr_count",
):
    require(marker in function_reader, f"function_reader_missing_{marker}")
require("var instrs: [IrInstr;" not in function_reader, "function_reader_stack_copy_present")

preflight = section(source, "fn soir_v4_function_is_lossless(", "pub fn serialize_ir_module_into(")
for marker in (
    "defining_module_id != IR_DEFINING_MODULE_UNKNOWN",
    "returns_float != 0",
    "return_struct_name.len != 0",
    "is_sret != 0",
    "bss_size != 0",
    "first_param_is_ref != 0",
    "ontologies.count != 0",
    "prof_counter_count != 0",
    "export_count != 0",
    "bss_total_size != 0",
    "SOIR_V4_STATUS_SEMANTIC_LOSS",
    "SOIR_V4_STATUS_CAPACITY",
):
    require(marker in preflight, f"preflight_missing_{marker}")

writer_into = section(source, "pub fn serialize_ir_module_into(", "pub fn serialize_ir_module(")
for marker in (
    ") -> bool with Mut, Panic, Div, Alloc",
    "(*out_len) = 0",
    "if !serialize_ir_module_core_preflight(module) { return false }",
    "(*out_buf) = buf",
    "(*out_len) = pos",
    "\n    true\n}",
):
    require(marker in writer_into, f"status_bearing_writer_missing_{marker.strip()}")

instr_profile = section(source, "fn soir_v4_instr_is_lossless(", "fn soir_v4_function_is_lossless(")
for marker in (
    "imm_flags == 0",
    "arg_count == 0",
    "call_args_empty",
    "soir_v4_opcode_supported",
):
    require(marker in instr_profile, f"instruction_profile_missing_{marker}")

module_reader = section(source, "pub fn deserialize_ir_module_into(", "pub fn deserialize_ir_module(")
for marker in (
    "if len < 0 || len > SOIR_MAX_SIZE",
    "if len < 8",
    "if version != SOIR_VERSION",
    "SOIR_V4_STATUS_BAD_RESERVED",
    "fn_count > IR_MAX_FUNCS",
    "string_count > IR_MAX_STRINGS",
    "if pos != len",
    "SOIR_V4_STATUS_TRAILING_BYTES",
    "(*out).fn_count = fn_count",
    "(*out).string_count = string_count",
):
    require(marker in module_reader, f"module_reader_missing_{marker}")
for forbidden in ("version != 1", "version != 2", "version != 3"):
    require(forbidden not in module_reader, f"legacy_version_fail_open_{forbidden[-1]}")

wire_validator = section(source, "pub fn soir_v4_validate_lossless_wire(", "pub fn deserialize_ir_module_into(")
for marker in (
    "var scratch = ir_empty_function()",
    "deserialize_ir_function_into(buf, pos, &! scratch)",
    "SOIR_EPISTEMIC_COUNT_FIELDS_V4",
    "read_ir_algebra_info(buf, pos)",
    "if pos != len",
):
    require(marker in wire_validator, f"wire_validator_missing_{marker}")
require(
    "if !soir_v4_validate_lossless_wire(buf, len)" in module_reader,
    "module_decoder_does_not_prevalidate_wire",
)

empty_epistemic = section(
    source,
    "fn deserialize_ir_epistemic_empty_v4(",
    "fn soir_v4_name_valid(",
)
require("field_index < SOIR_EPISTEMIC_COUNT_FIELDS_V4" in empty_epistemic, "empty_epistemic_count_loop")
require("count_pair.0 != 0" in empty_epistemic, "nonempty_epistemic_not_rejected")
require("SOIR_V4_STATUS_SEMANTIC_LOSS" in empty_epistemic, "epistemic_loss_status")

require("fn thin_emit_ir_cache_quarantine_receipt(" in loader, "loader_quarantine_receipt_missing")
require("status=quarantined authority=0" in loader, "loader_authority_zero_missing")
write_cache = section(loader, "fn thin_try_write_ir_cache(", "fn thin_try_read_ir_cache(")
read_cache = section(loader, "fn thin_try_read_ir_cache(", "fn thin_try_read_ir_cache_no_stats(")
read_cache_no_stats = section(loader, "fn thin_try_read_ir_cache_no_stats(", "fn thin_try_write_binary_cache(")
require("serialize_ir_module(" not in write_cache, "persistent_v4_write_reactivated")
require("deserialize_ir_module(" not in read_cache, "persistent_v4_read_reactivated")
require("deserialize_ir_module(" not in read_cache_no_stats, "persistent_v4_read_no_stats_reactivated")
require("\n    false\n}" in write_cache, "persistent_v4_write_not_fail_closed")
require("(false, ir_empty_module())" in read_cache, "persistent_v4_read_not_fail_closed")
require("(false, ir_empty_module())" in read_cache_no_stats, "persistent_v4_read_no_stats_not_fail_closed")

runtime_self_test = section(
    source,
    "pub fn soir_v4_lossless_roundtrip_self_test(",
    "// ============================================================\n// Hyper type serialization",
)
for marker in (
    "len_a != 1624",
    "f64_to_bits(restored.instrs[0].imm_f64) != 9221120237041090626",
    "!soir_v4_probe_buffers_equal(&buf_a, &buf_b, len_a)",
    "SOIR_V4_STATUS_TRUNCATED",
    "SOIR_V4_STATUS_TRAILING_BYTES",
    "SOIR_V4_STATUS_BAD_VERSION",
    "SOIR_V4_STATUS_BAD_RESERVED",
    "SOIR_V4_STATUS_UNSUPPORTED_TAG",
    "SOIR_V4_STATUS_SEMANTIC_LOSS",
    "SOIR_V4_STATUS_CAPACITY",
    "SOIR_V4_STATUS_MALFORMED",
    "bad[8] = 0 as i8",
    "bad[9] = 8 as i8",
    "function.instr_count = 560",
    "len_b != 130944",
    "function.instr_count = 561",
):
    require(marker in runtime_self_test, f"self_test_missing_{marker}")
require("soir_v4_lossless_roundtrip_self_test()" in fixture, "fixture_self_test_call_missing")
require("SOIR_V4_LOSSLESS_ROUNDTRIP_RUNTIME_PASS" in fixture, "fixture_runtime_marker_missing")

for marker, reason in (
    ("let WIRE_SIZE: i64 = 1624", "wire_size_contract"),
    ("let ONE_FUNCTION_FIXED_SIZE: i64 = 1024", "wire_one_function_fixed_size"),
    ("let FUNCTION_HEADER_SIZE: i64 = 704", "wire_function_header_size"),
    ("let INSTRUCTION_SIZE: i64 = 232", "wire_instruction_size"),
    ("while param_index < 64", "wire_param_count_contract"),
    ("while epistemic_index < 36", "wire_epistemic_count_contract"),
    ("p = put_instr(buf, p, 9, -1, 0, 9221120237041090626", "wire_nan_payload_missing"),
    ("p = put_instr(buf, p, 10, 9, -37, 4614256656552045848", "wire_pi_payload_missing"),
    ("let first_status = instr_status(buf, 720)", "wire_first_instruction_offset"),
    ("let second_status = instr_status(buf, 952)", "wire_second_instruction_offset"),
    ("get_i64(buf, 1328 + epistemic_index * 8)", "wire_epistemic_offset"),
    ("if get_i64(buf, 1616) != 0", "wire_algebra_offset"),
    ("bad[4] = 5 as i8", "wire_bad_version_probe"),
    ("bad[5] = 1 as i8", "wire_bad_reserved_probe"),
    ("bad[720] = 35 as i8", "wire_bad_opcode_probe"),
    ("bad[944] = 1 as i8", "wire_call_args_loss_probe"),
    ("bad[1328] = 1 as i8", "wire_epistemic_loss_probe"),
    ("put_i64(&! bad, 8, 2048)", "wire_logical_limit_truncation_probe"),
    ("put_i64(&! bad, 8, 2049)", "wire_logical_overflow_probe"),
    ("one_function_capacity_status(560) != STATUS_OK", "wire_capacity_560_probe"),
    ("one_function_capacity_status(561) != STATUS_CAPACITY", "wire_capacity_561_probe"),
    ("SOIR_V4_LOSSLESS_WIRE_RUNTIME_PASS", "wire_runtime_marker"),
):
    require(marker in wire_fixture, reason)
require("bits_to_f64" not in wire_fixture, "wire_fixture_requires_new_frontend_bitcast")

print(
    "SOIR_V4_LOSSLESS_ROUNDTRIP_SOURCE_PASS "
    "version=4 function_capacity=2048 buffer=131072 "
    "name_bytes=136 function_header=704 instruction_bytes=232 "
    "epistemic_counts=36 one_function_instr_limit=560 persistent_authority=0"
)
PY

CHECK_TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-soir-v4-check.XXXXXX")"
trap 'rm -rf "$CHECK_TMP"' EXIT

check_source() {
  local label="$1"
  local path="$2"
  if ! SOUNIO_STDLIB_PATH="$ROOT_DIR/self-hosted" \
      timeout --signal=TERM --kill-after=10s 180 \
      "$CHECK_BIN" check "$path" >"$CHECK_TMP/${label}.log" 2>&1; then
    tail -n 160 "$CHECK_TMP/${label}.log" >&2 || true
    fail "${label}_check_failed"
  fi
  grep -Fq 'check: OK' "$CHECK_TMP/${label}.log" || fail "${label}_check_marker_missing"
}

check_source serializer "$SOURCE"
check_source fixture "$FIXTURE"
check_source wire_fixture "$WIRE_FIXTURE"

if [[ "$MODE" == "source" ]]; then
  printf '%s\n' \
    'SOIR_V4_LOSSLESS_ROUNDTRIP_BOUNDARY profile=function-core,string-table,empty-epistemic,algebra-table rejected=module-identity,call-args,imm-flags,return-abi,sret,bss,ontology,profiling,exports,nonempty-epistemic persistent-cache=quarantined capacity=560/561 runtime=not_run merge_ready=0'
  printf '%s\n' 'SOIR_V4_LOSSLESS_ROUNDTRIP_PASS mode=source'
  exit 0
fi

RAW_COMPILER="${SOUNIO_SOIR_V4_RAW_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_SOIR_V4_EXPECTED_COMPILER_SHA256:-}"
COMPILER_SOURCE_SHA="${SOUNIO_SOIR_V4_COMPILER_SOURCE_SHA:-not_claimed}"
BOOTSTRAP_COMPILER="${SOUNIO_SOIR_V4_BOOTSTRAP_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
[[ -n "$RAW_COMPILER" ]] || fail runtime_requires_explicit_source_fresh_compiler
[[ -x "$RAW_COMPILER" ]] || fail raw_compiler_missing
[[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail runtime_requires_expected_compiler_sha256
[[ "$(od -An -tx1 -N4 "$RAW_COMPILER" | tr -d ' \n')" == "7f454c46" ]] || fail raw_compiler_not_elf
compiler_sha256="$(sha256sum "$RAW_COMPILER" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] || fail compiler_sha256_mismatch

raw_check_source() {
  local label="$1"
  local path="$2"
  if ! SOUNIO_STDLIB_PATH="$ROOT_DIR/self-hosted" \
      timeout --signal=TERM --kill-after=10s 180 \
      "$RAW_COMPILER" check "$path" >"$CHECK_TMP/raw-${label}.log" 2>&1; then
    tail -n 160 "$CHECK_TMP/raw-${label}.log" >&2 || true
    fail "raw_${label}_check_failed"
  fi
  grep -Fq 'check: OK' "$CHECK_TMP/raw-${label}.log" || fail "raw_${label}_check_marker_missing"
}

raw_check_source serializer "$SOURCE"
raw_check_source fixture "$FIXTURE"
raw_check_source wire_fixture "$WIRE_FIXTURE"

[[ -x "$BOOTSTRAP_COMPILER" ]] || fail bootstrap_compiler_missing
[[ "$(od -An -tx1 -N4 "$BOOTSTRAP_COMPILER" | tr -d ' \n')" == "7f454c46" ]] || fail bootstrap_compiler_not_elf
bootstrap_sha256="$(sha256sum "$BOOTSTRAP_COMPILER" | awk '{print $1}')"

ELF="$CHECK_TMP/soir-v4-lossless-roundtrip.elf"
COMPILE_LOG="$CHECK_TMP/compile.log"
if ! timeout --signal=TERM --kill-after=10s 180 \
    "$BOOTSTRAP_COMPILER" "$WIRE_FIXTURE" "$ELF" >"$COMPILE_LOG" 2>&1; then
  tail -n 200 "$COMPILE_LOG" >&2 || true
  fail runtime_fixture_compile_failed
fi
[[ -s "$ELF" ]] || fail runtime_fixture_elf_missing
[[ "$(od -An -tx1 -N4 "$ELF" | tr -d ' \n')" == "7f454c46" ]] || fail runtime_fixture_not_elf
chmod +x "$ELF"

set +e
timeout --signal=TERM --kill-after=10s 120 "$ELF" >"$CHECK_TMP/runtime.stdout" 2>"$CHECK_TMP/runtime.stderr"
runtime_rc=$?
set -e
if [[ "$runtime_rc" -ne 0 ]]; then
  cat "$CHECK_TMP/runtime.stdout" >&2 || true
  cat "$CHECK_TMP/runtime.stderr" >&2 || true
  fail "runtime_rc_${runtime_rc}"
fi
grep -Fxq \
  'SOIR_V4_LOSSLESS_WIRE_RUNTIME_PASS bytes=1624 functions=1 instructions=2 strings=1 epistemic_counts=36 algebras=empty bit_exact=pass malformed=fail_closed capacity=560/561' \
  "$CHECK_TMP/runtime.stdout" || fail runtime_marker_missing

source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
fixture_sha256="$(sha256sum "$FIXTURE" | awk '{print $1}')"
wire_fixture_sha256="$(sha256sum "$WIRE_FIXTURE" | awk '{print $1}')"
elf_sha256="$(sha256sum "$ELF" | awk '{print $1}')"
printf '%s\n' \
  'SOIR_V4_LOSSLESS_ROUNDTRIP_BOUNDARY profile=function-core,string-table,empty-epistemic,algebra-table rejected=module-identity,call-args,imm-flags,return-abi,sret,bss,ontology,profiling,exports,nonempty-epistemic persistent-cache=quarantined capacity=560/561 runtime_scope=wire-schema-model bit_exact=pass malformed=fail_closed module_api_runtime=blocked_lower_array_seed raw_codegen=not_claimed merge_ready=0'
printf 'SOIR_V4_LOSSLESS_ROUNDTRIP_PASS mode=runtime compiler_sha256=%s compiler_source_sha=%s bootstrap_sha256=%s source_sha256=%s fixture_sha256=%s wire_fixture_sha256=%s elf_sha256=%s raw_check=pass runtime_rc=0 persistent_authority=0 merge_ready=0\n' \
  "$compiler_sha256" "$COMPILER_SOURCE_SHA" "$bootstrap_sha256" "$source_sha256" "$fixture_sha256" "$wire_fixture_sha256" "$elf_sha256"
