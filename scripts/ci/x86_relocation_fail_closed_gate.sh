#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="self-hosted/native/codegen_x86_linux.sio"
RELOC_SOURCE="self-hosted/native/reloc.sio"
FIXTURE="tests/fixtures/x86_relocation_fail_closed/policy_selftest.sio"
MODE="${SOUNIO_X86_RELOC_GATE_MODE:-source}"

fail() {
  printf 'X86_RELOCATION_FAIL_CLOSED_FAIL reason=%s\n' "$1" >&2
  exit 1
}

case "$MODE" in
  source|runtime) ;;
  *) fail "invalid_mode_${MODE}" ;;
esac

for path in "$SOURCE" "$RELOC_SOURCE" "$FIXTURE"; do
  [[ -f "$path" ]] || fail "missing_${path//\//_}"
done

python3 - "$SOURCE" "$RELOC_SOURCE" "$FIXTURE" <<'PY'
from pathlib import Path
import re
import sys

source_path, reloc_source_path, fixture_path = map(Path, sys.argv[1:])
source = source_path.read_text(encoding="utf-8")
reloc_source = reloc_source_path.read_text(encoding="utf-8")
fixture = fixture_path.read_text(encoding="utf-8")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise SystemExit(f"X86_RELOCATION_FAIL_CLOSED_FAIL reason={reason}")


def section(start_marker: str, end_marker: str) -> str:
    start = source.find(start_marker)
    require(start >= 0, f"missing_source_marker_{start_marker.split('(')[0]}")
    end = source.find(end_marker, start)
    require(end > start, f"missing_source_marker_{end_marker.split('(')[0]}")
    return source[start:end]


policy = section(
    "fn native_v2_flat_reloc_capacity()",
    "fn nc_emit_mov_rax_r15(",
)
flat_append = section("fn nc_add_flat_reloc(", "fn nc_add_call_reloc(")
persist = section(
    "fn native_v2_persist_builtin_emit_into(",
    "fn emit_builtin_str_len_into(",
)
flat_apply = section("fn apply_relocations_into(", "pub fn compile_module_streaming_finish_main_into(")
stream_finish = section(
    "pub fn compile_module_streaming_finish_into(",
    "fn native_v2_reloc_is_call_patch(",
)
module_v2 = section("pub fn compile_module_v2_mut(", "fn spill_ir_params(")
compile_v2 = section("fn compile_to_elf_v2(", "fn native_v2_file_put_u8(")
direct_writer = section(
    "fn native_v2_write_min_elf64_to_file(",
    "fn native_v2_ir_trace_enabled(",
)
preview_writer = section(
    "pub fn compile_native_v2_preview_to_file(",
    "fn compile_to_macho_arm64_preview(",
)
ref_writer = section(
    "fn compile_native_finalize_and_write_ref(",
    "pub fn compile_native_x86_linux_to_file(",
)
legacy_finalizers = section("fn compile_to_elf(", "// ET_REL object file emission")
rel_finalizer = section("fn finalize_elf64_relocatable(", "// Compile module to ET_REL")
obj_compile = section("fn compile_to_obj(", "// Sprint 110: ET_DYN")
shared_compile = section("fn compile_to_shared(", "}") + "}"
alloc = section("fn nc_core_emit_alloc_into(", "fn nc_rodata_add_ir_instr_name(")
write_file = section("fn emit_builtin_write_file(", "fn emit_builtin_sqrt(")

for marker in (
    "fn native_v2_flat_reloc_capacity() -> i64 { 65536 }",
    "fn native_legacy_reloc_capacity() -> i64 { 4096 }",
    "fn native_elf_rela_capacity() -> i64 { 341 }",
    "fn ERR_BACKEND_CAPACITY_EXCEEDED() -> i64 { 6 }",
    "out.len = -1",
    "native_v2_relocation_fail_closed_selftest",
):
    require(marker in source, f"policy_marker_missing_{marker.split('(')[0].strip()}")

require(
    "idx < native_v2_flat_reloc_capacity()" in flat_append,
    "flat_append_capacity_not_checked",
)
require(
    "(*nc).flat_reloc_count = native_v2_flat_reloc_capacity() + 1" in flat_append,
    "flat_append_overflow_not_marked",
)
require(
    "idx < native_legacy_reloc_capacity()" in policy,
    "legacy_append_capacity_not_checked",
)
require(
    "t.count = native_legacy_reloc_capacity() + 1" in policy,
    "legacy_append_overflow_not_marked",
)
require("t.count < 256" not in policy, "legacy_256_limit_still_authoritative")

require(
    "pub fn relocation_table_capacity() -> i64 { 4096 }" in reloc_source,
    "public_relocation_capacity_marker_missing",
)
require(
    reloc_source.count("idx >= 0 && idx < relocation_table_capacity()") == 4,
    "public_relocation_append_paths_not_all_checked",
)
require(
    reloc_source.count("t.count = relocation_table_capacity() + 1") == 4,
    "public_relocation_overflow_not_all_marked",
)
require("t.count < 256" not in reloc_source, "public_legacy_256_limit_still_authoritative")
reloc_apply = reloc_source[reloc_source.find("pub fn apply_relocations("):]
require(
    "table.count < 0 || table.count > relocation_table_capacity()" in reloc_apply,
    "public_apply_overflow_guard_missing",
)
require("b.len = -1" in reloc_apply, "public_apply_refusal_sentinel_missing")

bare_legacy_append = re.compile(r"(?<![A-Za-z0-9_])add_(?:call|rip|data_section)_reloc\(")
require(
    bare_legacy_append.search(source) is None,
    "unchecked_legacy_append_call_present",
)
for checked in (
    "native_checked_add_call_reloc(",
    "native_checked_add_rip_reloc(",
    "native_checked_add_data_section_reloc(",
):
    require(source.count(checked) > 1, f"checked_append_not_used_{checked.split('(')[0]}")

for marker in (
    "if !native_relocation_state_complete(nc) { return }",
    "while i < (*nc).flat_reloc_count {",
    "var applied = false",
    "if !applied {",
    "(*nc).flat_reloc_count = -1",
):
    require(marker in flat_apply, f"flat_apply_marker_missing_{marker.split('{')[0].strip()}")
require("&& i < 65536" not in flat_apply, "flat_apply_truncation_guard_present")
require("&& ri < 4096" not in persist, "persist_relocation_truncation_guard_present")
require(
    "(*nc).flat_reloc_count = -1" in persist,
    "persist_unsupported_relocation_not_refused",
)

for marker in (
    "native_legacy_relocations_resolvable(nc)",
    "(*nc).relocs.count = -1",
):
    require(marker in stream_finish, f"stream_finish_not_fail_closed_{marker.split('(')[0]}")
    require(marker in module_v2, f"module_v2_not_fail_closed_{marker.split('(')[0]}")

for marker in (
    "ERR_BACKEND_CAPACITY_EXCEEDED()",
    '"native_v2_relocation_capacity_exceeded"',
    '"native_v2_code_capacity_exceeded"',
):
    require(marker in compile_v2, f"compile_v2_marker_missing_{marker}")

for writer_name, writer in (
    ("direct", direct_writer),
    ("preview", preview_writer),
    ("ref", ref_writer),
):
    require(
        "native_relocation_tables_complete" in writer,
        f"{writer_name}_writer_missing_relocation_guard",
    )
    require("return 20" in writer, f"{writer_name}_writer_missing_relocation_rc20")

require("if (*nc).code_overflow" in direct_writer, "direct_writer_lost_code_overflow_guard")
require("return 19" in direct_writer, "direct_writer_lost_code_overflow_rc19")
require("if (*nc).code_overflow" in ref_writer, "ref_writer_lost_code_overflow_guard")
require("return 19" in ref_writer, "ref_writer_lost_code_overflow_rc19")

require(
    legacy_finalizers.count("native_relocation_state_finalizable(&nc)") == 2,
    "legacy_finalizers_not_both_guarded",
)
require(
    "rela.len < 0 || rela.len > 8192 || rela.len % 24 != 0" in rel_finalizer,
    "rela_finalizer_shape_not_checked",
)
for marker in (
    "native_relocation_state_complete_for_rela(&nc)",
    "rela.len != nc.relocs.count * 24",
):
    require(marker in obj_compile, f"object_finalizer_marker_missing_{marker.split('(')[0]}")
require(
    "native_relocation_state_complete_for_shared(&nc)" in shared_compile,
    "shared_pending_relocations_not_refused",
)
require(
    "if len <= 0" in section("fn native_compile_result_ok_elf(", "// Name comparison helpers"),
    "refusal_binary_could_be_reported_ok",
)

# Preserve the allocation and handle ABI landed immediately before this patch.
for marker in (
    "nc_core_emit_alloc_fail_into(nc, heap_slow_jnz, 181)",
    "nc_core_emit_alloc_fail_into(nc, handle_slow_jnz, 182)",
    "fn nc_core_emit_alloc_into(nc: &! NativeCompiler, dst: i64, type_tag: i64, elem_count: i64)",
    "nc_emit_mov_rax_imm(nc, elem_count)",
):
    require(marker in alloc, f"issue_919_abi_marker_missing_{marker.split('(')[0]}")
for marker in (
    "buf_handle",
    "native_v2_resolve_handle_to_object_base_rax(c)",
    "emit_store_rax_runtime_context_field(c, runtime_context_field_heap_cursor())",
    "emit_write_syscall_for_target(c, c.target_os_id)",
):
    require(marker in write_file, f"write_file_abi_marker_missing_{marker}")

require(
    "native_v2_relocation_fail_closed_selftest" in fixture,
    "fixture_does_not_call_policy_selftest",
)
require(
    "public_relocation_table_fail_closed" in fixture,
    "fixture_does_not_call_public_relocation_policy",
)
require(
    "X86_RELOCATION_FAIL_CLOSED_SELFTEST_PASS legacy_public=4096 legacy_x86=4096 flat=65536 rela=341 shared=pending_relocs_refused" in fixture,
    "fixture_pass_marker_missing",
)

print(
    "X86_RELOCATION_FAIL_CLOSED_SOURCE_RECEIPT "
    "status=pass flat_capacity=65536 legacy_capacity=4096 public_legacy_capacity=4096 "
    "legacy_256_limit=removed rela_capacity=341 apply=checked "
    "overflow=fail_closed shared_pending=refused "
    "code_overflow_rc=19 relocation_refusal_rc=20 issue_919_abi=preserved"
)
PY

if [[ "$MODE" == "source" ]]; then
  printf '%s\n' \
    'X86_RELOCATION_FAIL_CLOSED_BOUNDARY runtime=not-run source_fresh_raw_elf=required artifact_overflow=not-claimed' \
    'X86_RELOCATION_FAIL_CLOSED_PASS mode=source runtime=not_run merge_ready=0'
  exit 0
fi

if [[ "$(uname -s 2>/dev/null || true)" != "Linux" ]] ||
   [[ ! "$(uname -m 2>/dev/null || true)" =~ ^(x86_64|amd64)$ ]]; then
  fail "runtime_requires_x86_64_linux"
fi

SOUC="${SOUNIO_X86_RELOC_SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_X86_RELOC_EXPECTED_COMPILER_SHA256:-}"
[[ -n "$SOUC" ]] || fail "runtime_requires_explicit_source_fresh_compiler"
[[ -x "$SOUC" ]] || fail "compiler_not_executable"
[[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "runtime_requires_expected_compiler_sha256"
[[ "$(od -An -tx1 -N4 "$SOUC" | tr -d ' \n')" == "7f454c46" ]] ||
  fail "compiler_must_be_raw_elf"

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] ||
  fail "compiler_sha256_mismatch"

WORK_DIR="${SOUNIO_X86_RELOC_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-x86-reloc.XXXXXX")}"
KEEP_WORK="${SOUNIO_X86_RELOC_KEEP_WORK:-0}"
if [[ "$KEEP_WORK" != "1" && -z "${SOUNIO_X86_RELOC_WORK_DIR:-}" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi
mkdir -p "$WORK_DIR"

ELF="$WORK_DIR/policy-selftest.elf"
COMPILE_LOG="$WORK_DIR/policy-selftest.compile.log"
if ! timeout --signal=TERM --kill-after=10s 600 \
    "$SOUC" --native-compile "$FIXTURE" -o "$ELF" >"$COMPILE_LOG" 2>&1; then
  tail -n 160 "$COMPILE_LOG" >&2 || true
  fail "policy_selftest_compile_failed"
fi
[[ -s "$ELF" ]] || fail "policy_selftest_elf_missing"
[[ "$(od -An -tx1 -N4 "$ELF" | tr -d ' \n')" == "7f454c46" ]] ||
  fail "policy_selftest_output_not_elf"
chmod +x "$ELF"

set +e
timeout --signal=TERM --kill-after=5s 60 \
  "$ELF" >"$WORK_DIR/policy-selftest.stdout" 2>"$WORK_DIR/policy-selftest.stderr"
selftest_rc=$?
set -e
[[ "$selftest_rc" -eq 0 ]] || fail "policy_selftest_rc_${selftest_rc}"
grep -Fxq \
  'X86_RELOCATION_FAIL_CLOSED_SELFTEST_PASS legacy_public=4096 legacy_x86=4096 flat=65536 rela=341 shared=pending_relocs_refused' \
  "$WORK_DIR/policy-selftest.stdout" || fail "policy_selftest_marker_missing"

printf '%s\n' \
  "X86_RELOCATION_FAIL_CLOSED_PASS mode=runtime compiler_sha256=$compiler_sha256 policy_selftest_rc=0 flat_capacity=65536 legacy_capacity=4096 public_legacy_capacity=4096 rela_capacity=341 relocation_refusal_rc=20 artifact_overflow=not_run merge_ready=1"
