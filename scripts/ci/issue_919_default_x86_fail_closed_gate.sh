#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="self-hosted/native/codegen_x86_linux.sio"
SUCCESS_FIXTURE="tests/fixtures/issue_919/handle_boundary_success.sio"
EXHAUSTION_FIXTURE="tests/fixtures/issue_919/handle_boundary_exhaustion.sio"
MODE="${SOUNIO_ISSUE_919_GATE_MODE:-source}"

fail() {
  printf 'ISSUE_919_DEFAULT_X86_FAIL_CLOSED_FAIL reason=%s\n' "$1" >&2
  exit 1
}

case "$MODE" in
  source|runtime) ;;
  *) fail "invalid_mode_${MODE}" ;;
esac

for path in "$SOURCE" "$SUCCESS_FIXTURE" "$EXHAUSTION_FIXTURE"; do
  [[ -f "$path" ]] || fail "missing_${path//\//_}"
done

python3 - "$SOURCE" "$SUCCESS_FIXTURE" "$EXHAUSTION_FIXTURE" <<'PY'
from pathlib import Path
import sys

source_path, success_path, exhaustion_path = map(Path, sys.argv[1:])
source = source_path.read_text(encoding="utf-8")
success = success_path.read_text(encoding="utf-8")
exhaustion = exhaustion_path.read_text(encoding="utf-8")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise SystemExit(f"ISSUE_919_DEFAULT_X86_FAIL_CLOSED_FAIL reason={reason}")


def section(start_marker: str, end_marker: str) -> str:
    start = source.find(start_marker)
    require(start >= 0, f"missing_source_marker_{start_marker.split('(')[0]}")
    end = source.find(end_marker, start)
    require(end > start, f"missing_source_marker_{end_marker.split('(')[0]}")
    return source[start:end]


helper = section(
    "fn nc_core_emit_alloc_fail_into(",
    "fn nc_core_emit_alloc_into(",
)
alloc = section(
    "fn nc_core_emit_alloc_into(",
    "fn nc_rodata_add_ir_instr_name(",
)
write_file = section(
    "fn emit_builtin_write_file(",
    "fn emit_builtin_sqrt(",
)

require(
    "nc_patch_u32_le(nc, slow_jnz + 2, slow_path - (slow_jnz + 6))" in helper,
    "slow_branch_not_patched_to_fail_path",
)
require(
    helper.count("nc_emit_exit_code(nc, fail_code)") == 1,
    "fail_helper_must_emit_exactly_one_exit",
)
for forbidden in (
    "runtime_context_field_heap_base()",
    "runtime_context_field_heap_cursor()",
    "runtime_context_field_handle_count()",
    "runtime_context_field_pin_count()",
    "nc_emit_jmp_rel32",
    "retry",
    "reset",
):
    require(forbidden not in helper, f"fail_helper_contains_{forbidden}")

require(
    source.count("fn nc_core_emit_alloc_fail_into(") == 1,
    "fail_helper_definition_count",
)
require(
    "nc_core_emit_empty_frame_gc_reset_into" not in source,
    "legacy_core_reset_helper_present",
)
require(
    "nc_core_emit_alloc_retry_or_exit_into" not in source,
    "legacy_core_retry_helper_present",
)
require("alloc_retry_target" not in alloc, "core_alloc_retry_target_present")
require(
    alloc.count("nc_core_emit_alloc_fail_into(nc, heap_slow_jnz, 181)") == 1,
    "heap_exhaustion_exit_181_missing",
)
require(
    alloc.count("nc_core_emit_alloc_fail_into(nc, handle_slow_jnz, 182)") == 1,
    "handle_exhaustion_exit_182_missing",
)
require(
    alloc.find("nc_core_emit_alloc_fail_into(nc, heap_slow_jnz, 181)")
    < alloc.find("nc_core_emit_alloc_fail_into(nc, handle_slow_jnz, 182)"),
    "slow_path_order_changed",
)
require(
    "fn nc_core_emit_alloc_into(nc: &! NativeCompiler, dst: i64, type_tag: i64, elem_count: i64)" in alloc,
    "current_elem_count_abi_lost",
)
require(
    "nc_emit_mov_rax_imm(nc, elem_count)" in alloc,
    "array_logical_count_header_write_lost",
)

# Guard against transplanting the old file wholesale over the current write_file ABI.
for marker in (
    "buf_handle",
    "native_v2_resolve_handle_to_object_base_rax(c)",
    "emit_store_rax_runtime_context_field(c, runtime_context_field_heap_cursor())",
    "emit_write_syscall_for_target(c, c.target_os_id)",
):
    require(marker in write_file, f"current_write_file_abi_marker_missing_{marker}")

require("while i < 1048575" in success, "success_boundary_not_1048575")
require(
    "ISSUE_919_HANDLE_BOUNDARY_SUCCESS allocations=1048575" in success,
    "success_marker_missing",
)
require("while i < 1048576" in exhaustion, "exhaustion_attempt_not_1048576")
require(
    "ISSUE_919_HANDLE_EXHAUSTION_UNREACHABLE" in exhaustion,
    "exhaustion_unreachable_marker_missing",
)

print(
    "ISSUE_919_DEFAULT_X86_SOURCE_PASS "
    "heap_exit=181 handle_exit=182 reset_retry=absent "
    "usable_handles=1048575 attempted_handle=1048576 write_file_abi=preserved"
)
PY

if [[ "$MODE" == "source" ]]; then
  printf '%s\n' \
    'ISSUE_919_DEFAULT_X86_FAIL_CLOSED_PASS mode=source runtime=not_run merge_ready=0'
  exit 0
fi

if [[ "$(uname -s 2>/dev/null || true)" != "Linux" ]] ||
   [[ ! "$(uname -m 2>/dev/null || true)" =~ ^(x86_64|amd64)$ ]]; then
  fail "runtime_requires_x86_64_linux"
fi

SOUC="${SOUNIO_ISSUE_919_SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_ISSUE_919_EXPECTED_COMPILER_SHA256:-}"
[[ -n "$SOUC" ]] || fail "runtime_requires_explicit_source_fresh_compiler"
[[ -x "$SOUC" ]] || fail "compiler_not_executable"
[[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "runtime_requires_expected_compiler_sha256"
[[ "$(od -An -tx1 -N4 "$SOUC" | tr -d ' \n')" == "7f454c46" ]] ||
  fail "compiler_must_be_raw_elf"

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] ||
  fail "compiler_sha256_mismatch"

WORK_DIR="${SOUNIO_ISSUE_919_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-issue-919.XXXXXX")}"
KEEP_WORK="${SOUNIO_ISSUE_919_KEEP_WORK:-0}"
if [[ "$KEEP_WORK" != "1" && -z "${SOUNIO_ISSUE_919_WORK_DIR:-}" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi
mkdir -p "$WORK_DIR"

compile_fixture() {
  local label="$1"
  local fixture="$2"
  local elf="$WORK_DIR/${label}.elf"
  local log="$WORK_DIR/${label}.compile.log"

  if ! timeout --signal=TERM --kill-after=10s 300 \
      "$SOUC" --native-compile "$fixture" -o "$elf" >"$log" 2>&1; then
    tail -n 120 "$log" >&2 || true
    fail "${label}_compile_failed"
  fi
  [[ -s "$elf" ]] || fail "${label}_elf_missing"
  [[ "$(od -An -tx1 -N4 "$elf" | tr -d ' \n')" == "7f454c46" ]] ||
    fail "${label}_not_elf"
  chmod +x "$elf"
}

compile_fixture success "$SUCCESS_FIXTURE"
compile_fixture exhaustion "$EXHAUSTION_FIXTURE"

set +e
timeout --signal=TERM --kill-after=10s 180 \
  "$WORK_DIR/success.elf" >"$WORK_DIR/success.stdout" 2>"$WORK_DIR/success.stderr"
success_rc=$?
set -e
[[ "$success_rc" -eq 0 ]] || fail "success_boundary_rc_${success_rc}"
grep -Fxq 'ISSUE_919_HANDLE_BOUNDARY_SUCCESS allocations=1048575' \
  "$WORK_DIR/success.stdout" || fail "success_boundary_marker_missing"

set +e
timeout --signal=TERM --kill-after=10s 180 \
  "$WORK_DIR/exhaustion.elf" >"$WORK_DIR/exhaustion.stdout" 2>"$WORK_DIR/exhaustion.stderr"
exhaustion_rc=$?
set -e
[[ "$exhaustion_rc" -eq 182 ]] || fail "exhaustion_rc_${exhaustion_rc}_expected_182"
if grep -Fq 'ISSUE_919_HANDLE_EXHAUSTION_UNREACHABLE' "$WORK_DIR/exhaustion.stdout"; then
  fail "exhaustion_retried_and_reached_unreachable_marker"
fi

printf '%s\n' \
  "ISSUE_919_DEFAULT_X86_FAIL_CLOSED_PASS mode=runtime compiler_sha256=$compiler_sha256 usable_handles=1048575 success_rc=0 attempted_handle=1048576 exhaustion_rc=182 reset_retry=absent merge_ready=1"
