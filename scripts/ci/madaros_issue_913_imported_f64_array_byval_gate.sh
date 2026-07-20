#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/tests/compiler/madaros_issue_913_imported_f64_array_byval"
MAIN_SOURCE="$FIXTURE_DIR/main.sio"
LEAF_SOURCE="$FIXTURE_DIR/leaf.sio"
EXPECTED_STDOUT="$FIXTURE_DIR/expected.stdout"
LOWER_SOURCE="$ROOT_DIR/self-hosted/ir/lower.sio"
CODEGEN_SOURCE="$ROOT_DIR/self-hosted/native/codegen_x86_linux.sio"
WRAPPER="$ROOT_DIR/bin/madaros"

MODE=runtime
if [[ $# -gt 1 ]]; then
  printf 'MADAROS_ISSUE_913_FAIL stage=setup reason=unexpected_arguments\n' >&2
  exit 1
fi
if [[ $# -eq 1 ]]; then
  [[ "$1" == --source-only ]] || {
    printf 'MADAROS_ISSUE_913_FAIL stage=setup reason=unexpected_argument\n' >&2
    exit 1
  }
  MODE=source-only
fi

fail() {
  local stage="$1"
  local reason="$2"
  printf 'MADAROS_ISSUE_913_FAIL stage=%s reason=%s\n' "$stage" "$reason" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail setup "missing_command_$1"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual

  [[ -f "$path" ]] || fail fixtures "${label}_missing"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail fixtures "${label}_sha256_mismatch"
}

elf_magic() {
  od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n'
}

elf_u16() {
  od -An -tu2 -j "$2" -N2 "$1" 2>/dev/null | tr -d ' \n'
}

assert_executable_elf() {
  local path="$1"

  [[ -f "$path" && -s "$path" ]] || fail build final_elf_missing
  [[ -x "$path" ]] || fail build final_elf_not_executable
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail build final_artifact_not_elf
  [[ "$(elf_u16 "$path" 16)" == 2 ]] || fail build final_elf_not_et_exec
  [[ "$(elf_u16 "$path" 18)" == 62 ]] || fail build final_elf_not_x86_64
}

has_forbidden_path() {
  grep -Eq \
    'native_prebundle:|falling back to full IR path|compact modular IR table path|legacy compact IR differential enabled|SELFHOST=fallback|driver_orchestration.*status=fallback' \
    "$1"
}

require_command awk
require_command cmp
require_command grep
require_command od
require_command sha256sum
require_command tr

require_sha256 "$MAIN_SOURCE" 6beffe6b2559edc70f13770c15cb986bdf378f956e89081a877c6e5757d4a92d main
require_sha256 "$LEAF_SOURCE" 51c510e85dd7b5c1a30581026b32b6b3aa1e0f7c4316019a1babff2d9fb096fb leaf
require_sha256 "$EXPECTED_STDOUT" 478053105e65715920b4abea515ec230d9a645ad8d27f0221a69e44f46563e2a expected_stdout

grep -Fq 'LOWER_RETURN_TUPLE_FLOAT_BASE' "$LOWER_SOURCE" || fail source tuple_return_code_missing
grep -Fq 'tuple_float_masks' "$LOWER_SOURCE" || fail source tuple_float_reg_table_missing
grep -Fq 'lo5.bind_reg_tuple_float_mask(dst, tuple_float_mask)' "$LOWER_SOURCE" || fail source tuple_call_result_binding_missing
grep -Fq 'tuple_field_is_float' "$LOWER_SOURCE" || fail source tuple_field_lookup_missing
grep -Fq 'instr_op == IrOpcode::IrFieldGet' "$CODEGEN_SOURCE" || fail source field_get_codegen_missing
grep -Fq 'imm_flags == IR_FLOAT_REG_MARKER_FLAG || nc_core_field_is_float' "$CODEGEN_SOURCE" || fail source float_marker_codegen_missing

if [[ "$MODE" == source-only ]]; then
  printf 'MADAROS_ISSUE_913_SOURCE_PASS issue=913 fixture=two_module imported_by_value=f64_array local_by_value=control imported_by_reference=control lowering=tuple_return_float_mask codegen=field_get_float_marker\n'
  exit 0
fi

require_command env
require_command find
require_command mktemp
require_command timeout
require_command uname

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail setup linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail setup x86_64_required ;;
esac

RAW_MADAROS="${SOUNIO_ISSUE_913_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_ISSUE_913_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_ISSUE_913_KEEP:-0}"
BUILD_TIMEOUT_SECONDS="${SOUNIO_ISSUE_913_BUILD_TIMEOUT_SECONDS:-360}"
RUN_TIMEOUT_SECONDS="${SOUNIO_ISSUE_913_RUN_TIMEOUT_SECONDS:-30}"

[[ "$KEEP_WORK" == 0 || "$KEEP_WORK" == 1 ]] || fail setup invalid_keep_value
[[ "$BUILD_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail setup invalid_build_timeout
[[ "$RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail setup invalid_run_timeout
[[ -x "$WRAPPER" ]] || fail setup madaros_wrapper_missing
[[ -n "$RAW_MADAROS" ]] || fail setup explicit_raw_madaros_required
[[ -x "$RAW_MADAROS" ]] || fail setup raw_madaros_missing_or_not_executable
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail setup expected_raw_sha256_required

RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd -P)/$(basename "$RAW_MADAROS")"
[[ "$(elf_magic "$RAW_MADAROS")" == 7f454c46 ]] || fail setup raw_compiler_not_elf
[[ "$(elf_u16 "$RAW_MADAROS" 16)" == 2 ]] || fail setup raw_compiler_not_et_exec
[[ "$(elf_u16 "$RAW_MADAROS" 18)" == 62 ]] || fail setup raw_compiler_not_x86_64
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$RAW_SHA256" == "$EXPECTED_RAW_SHA256" ]] || fail setup raw_compiler_sha256_mismatch

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-issue913-array-byval.XXXXXX")"
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK/build-cwd"

INFO_LOG="$WORK/compiler.info"
env -u SOUNIO_MADAROS_BIN -u SOUNIO_SOUC_BIN \
  MADAROS_RAW_BIN="$RAW_MADAROS" \
  "$WRAPPER" info >"$INFO_LOG" 2>&1
grep -Fxq "raw_elf:      $RAW_MADAROS" "$INFO_LOG" || fail setup wrapper_raw_identity_mismatch

ELF="$WORK/issue913.elf"
BUILD_LOG="$WORK/build.log"
set +e
(
  cd "$WORK/build-cwd"
  exec env \
    -u SOUNIO_MADAROS_BIN \
    -u SOUNIO_SOUC_BIN \
    MADAROS_RAW_BIN="$RAW_MADAROS" \
    SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    SOUNIO_SOUC_ENGINE=madaros \
    SOUNIO_ENABLE_COMPACT_IMPORTED_IR=0 \
    SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
    timeout --signal=TERM --kill-after=10s "$BUILD_TIMEOUT_SECONDS" \
    "$WRAPPER" --science-boundary off build "$MAIN_SOURCE" -o "$ELF"
) >"$BUILD_LOG" 2>&1
BUILD_RC=$?
set -e

if [[ "$BUILD_RC" -ne 0 ]]; then
  tail -n 120 "$BUILD_LOG" >&2 || true
  fail build "compiler_rc_$BUILD_RC"
fi
has_forbidden_path "$BUILD_LOG" && fail build compact_or_fallback_path_observed
grep -Fq 'imported_compile: loaded 2' "$BUILD_LOG" || fail build two_module_closure_missing
grep -Fq 'module_frontend_full_ir: lower_node module_id=0 logical=main' "$BUILD_LOG" || fail build main_full_ir_missing
grep -Fq 'module_frontend_full_ir: lower_node module_id=1 logical=leaf' "$BUILD_LOG" || fail build leaf_full_ir_missing
grep -Eq '^Merged IR: 11$' "$BUILD_LOG" || fail build merged_ir_count_mismatch
if grep -Fq 'error[E' "$BUILD_LOG" || grep -Eq '^error:' "$BUILD_LOG"; then
  fail build diagnostic_on_success
fi
if find "$WORK/build-cwd" -type f -print -quit | grep -q .; then
  fail build unexpected_build_cwd_artifact
fi
assert_executable_elf "$ELF"

RUNTIME_STDOUT="$WORK/runtime.stdout"
RUNTIME_STDERR="$WORK/runtime.stderr"
set +e
timeout --signal=TERM --kill-after=5s "$RUN_TIMEOUT_SECONDS" "$ELF" >"$RUNTIME_STDOUT" 2>"$RUNTIME_STDERR"
RUNTIME_RC=$?
set -e
[[ "$RUNTIME_RC" -eq 0 ]] || fail runtime "elf_rc_$RUNTIME_RC"
[[ ! -s "$RUNTIME_STDERR" ]] || fail runtime stderr_not_empty
if ! cmp -s "$EXPECTED_STDOUT" "$RUNTIME_STDOUT"; then
  printf 'MADAROS_ISSUE_913_FAIL stage=runtime reason=stdout_mismatch expected_hex=' >&2
  od -An -tx1 "$EXPECTED_STDOUT" | tr -d ' \n' >&2
  printf ' actual_hex=' >&2
  od -An -tx1 "$RUNTIME_STDOUT" | tr -d ' \n' >&2
  printf '\n' >&2
  cat "$RUNTIME_STDOUT" >&2
  exit 1
fi

FINAL_RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$FINAL_RAW_SHA256" == "$RAW_SHA256" ]] || fail final raw_compiler_changed_during_gate
ELF_SHA256="$(sha256sum "$ELF" | awk '{print $1}')"
printf 'MADAROS_ISSUE_913_PASS issue=913 raw_authority=explicit_sha256 raw_sha256=%s driver=full_ir_noncompact modules=2 elf=ET_EXEC-x86_64 elf_sha256=%s runtime_exit=0 stdout=imported_by_value_26010+local_by_value_26010+imported_by_reference_26010 fallback=none work=%s\n' \
  "$RAW_SHA256" "$ELF_SHA256" "$WORK"
