#!/usr/bin/env bash
# Source-fresh runtime acceptance matrix for Sounio issues #921, #901, and #862.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${SOUNIO_MADAROS_IMPORTED_RUNTIME_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_MADAROS_IMPORTED_RUNTIME_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_KEEP:-0}"
TIMEOUT_SECONDS="${SOUNIO_MADAROS_IMPORTED_RUNTIME_TIMEOUT_SECONDS:-360}"
RUNTIME_TIMEOUT_SECONDS="${SOUNIO_MADAROS_IMPORTED_RUNTIME_EXEC_TIMEOUT_SECONDS:-30}"

ISSUE_921_SOURCE="$ROOT_DIR/docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio"
ISSUE_901_SOURCE="$ROOT_DIR/tests/stdlib/prob/test_prob_stdlib.sio"
FIXTURE_DIR="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance"
ISSUE_901_ITEM_CHAIN_MAIN="$FIXTURE_DIR/issue_901_item_chain_main.sio"
ISSUE_901_ITEM_CHAIN_LEAF="$FIXTURE_DIR/issue_901_item_chain_leaf.sio"
ISSUE_901_ITEM_CHAIN_NESTED="$FIXTURE_DIR/issue_901_item_chain_nested.sio"
ISSUE_901_LAYOUT_CAPACITY_MAIN="$FIXTURE_DIR/issue_901_layout_capacity_main.sio"
ISSUE_901_LAYOUT_CAPACITY_LEAF="$FIXTURE_DIR/issue_901_layout_capacity_leaf.sio"
ISSUE_862_POSITIVE="$FIXTURE_DIR/issue_862_positive.sio"
ISSUE_862_PUBLIC_LEAF="$FIXTURE_DIR/issue_862_public_leaf.sio"
ISSUE_862_PRIVATE_MAIN="$FIXTURE_DIR/issue_862_private_main.sio"
ISSUE_862_PRIVATE_LEAF="$FIXTURE_DIR/issue_862_private_leaf.sio"

fail() {
  local stage="$1"
  local reason="$2"
  printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_FAIL stage=%s reason=%s\n' "$stage" "$reason" >&2
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
  local label="$1"
  local path="$2"

  [[ -f "$path" && -s "$path" ]] || fail "$label" final_elf_missing
  [[ -x "$path" ]] || fail "$label" final_elf_not_executable
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail "$label" final_artifact_not_elf
  [[ "$(elf_u16 "$path" 16)" == 2 ]] || fail "$label" final_elf_not_et_exec
  [[ "$(elf_u16 "$path" 18)" == 62 ]] || fail "$label" final_elf_not_x86_64
}

artifact_state() {
  local path="${1:-}"
  if [[ -z "$path" ]]; then
    printf 'not_applicable\n'
  elif [[ -e "$path" || -L "$path" ]]; then
    printf 'present\n'
  else
    printf 'absent\n'
  fi
}

signal_name_from_rc() {
  local rc="$1"
  local number
  local name

  if (( rc < 129 || rc > 192 )); then
    return 1
  fi
  number=$((rc - 128))
  name="$(kill -l "$number" 2>/dev/null || true)"
  [[ -n "$name" ]] || name="SIGNAL_$number"
  [[ "$name" == SIG* || "$name" == SIGNAL_* ]] || name="SIG$name"
  printf '%s\n' "$name"
}

fatal_log_kind() {
  local log="$1"

  if grep -Eiq 'segmentation fault' "$log"; then printf 'SIGSEGV_LOG\n'; return 0; fi
  if grep -Eiq 'bus error' "$log"; then printf 'SIGBUS_LOG\n'; return 0; fi
  if grep -Eiq 'illegal instruction' "$log"; then printf 'SIGILL_LOG\n'; return 0; fi
  if grep -Eiq 'floating point exception' "$log"; then printf 'SIGFPE_LOG\n'; return 0; fi
  if grep -Eiq 'terminated by signal' "$log"; then printf 'SIGNAL_LOG\n'; return 0; fi
  if grep -Eiq 'core dumped' "$log"; then printf 'CORE_DUMP_LOG\n'; return 0; fi
  if grep -Eiq 'fatal:' "$log"; then printf 'FATAL_LOG\n'; return 0; fi
  return 1
}

show_failure_tail() {
  local log="$1"
  [[ -f "$log" ]] && tail -n 120 "$log" >&2 || true
}

report_execution_failure() {
  local stage="$1"
  local rc="$2"
  local log="$3"
  local artifact="${4:-}"
  local signal_name=""
  local fatal_kind=""
  local final_elf

  final_elf="$(artifact_state "$artifact")"
  if [[ "$rc" -eq 124 ]]; then
    printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_TIMEOUT stage=%s rc=%s final_elf=%s\n' \
      "$stage" "$rc" "$final_elf" >&2
    show_failure_tail "$log"
    exit 1
  fi

  signal_name="$(signal_name_from_rc "$rc" || true)"
  if [[ -n "$signal_name" ]]; then
    printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_SIGNAL stage=%s rc=%s signal=%s final_elf=%s\n' \
      "$stage" "$rc" "$signal_name" "$final_elf" >&2
    show_failure_tail "$log"
    exit 1
  fi

  fatal_kind="$(fatal_log_kind "$log" || true)"
  if [[ -n "$fatal_kind" ]]; then
    printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_CRASH stage=%s rc=%s classifier=%s final_elf=%s\n' \
      "$stage" "$rc" "$fatal_kind" "$final_elf" >&2
    show_failure_tail "$log"
    exit 1
  fi

  printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_FAIL stage=%s reason=unexpected_rc rc=%s final_elf=%s\n' \
    "$stage" "$rc" "$final_elf" >&2
  show_failure_tail "$log"
  exit 1
}

has_forbidden_path() {
  grep -Eq \
    'native_prebundle:|falling back to full IR path|compact modular IR table path|legacy compact IR differential enabled|SELFHOST=fallback backend=rust|driver_orchestration.*status=fallback' \
    "$1"
}

assert_no_fatal_log() {
  local stage="$1"
  local rc="$2"
  local log="$3"
  local artifact="${4:-}"

  if fatal_log_kind "$log" >/dev/null; then
    report_execution_failure "$stage" "$rc" "$log" "$artifact"
  fi
}

run_build() {
  local label="$1"
  local source="$2"
  local elf="$3"
  local log="$WORK/$label.build.log"
  local build_cwd="$WORK/$label.build-cwd"

  mkdir -p "$build_cwd"
  rm -f "$elf"
  set +e
  (
    cd "$build_cwd"
    exec env \
      -u SOUNIO_MADAROS_BIN \
      -u SOUNIO_SOUC_BIN \
      MADAROS_RAW_BIN="$RAW_MADAROS" \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      SOUNIO_SOUC_ENGINE=madaros \
      SOUNIO_ENABLE_COMPACT_IMPORTED_IR=0 \
      SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 \
      OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
      timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
      "$WRAPPER" --science-boundary off build "$source" -o "$elf"
  ) >"$log" 2>&1
  CASE_RC=$?
  set -e

  CASE_LOG="$log"
  CASE_BUILD_CWD="$build_cwd"
  CASE_ELF="$elf"
}

assert_empty_build_cwd() {
  local label="$1"
  local build_cwd="$2"

  if find "$build_cwd" -type f -print -quit | grep -q .; then
    find "$build_cwd" -type f -print >&2
    fail "$label" unexpected_build_cwd_artifact
  fi
}

assert_positive_build() {
  local label="$1"
  local rc="$2"
  local log="$3"
  local elf="$4"
  local build_cwd="$5"

  [[ "$rc" -eq 0 ]] || report_execution_failure "${label}_build" "$rc" "$log" "$elf"
  assert_no_fatal_log "${label}_build" "$rc" "$log" "$elf"
  has_forbidden_path "$log" && fail "${label}_build" compact_or_fallback_path_observed
  grep -Eq '^Merged IR: [1-9][0-9]*$' "$log" || fail "${label}_build" merged_ir_count_missing
  if grep -Fq 'error[E' "$log" || grep -Eq '^error:' "$log"; then
    fail "${label}_build" diagnostic_on_success
  fi
  assert_executable_elf "${label}_build" "$elf"
  assert_empty_build_cwd "${label}_build" "$build_cwd"
}

run_exact_runtime() {
  local label="$1"
  local elf="$2"
  local expected="$3"
  local stdout_token="$4"
  local stdout="$WORK/$label.runtime.stdout"
  local stderr="$WORK/$label.runtime.stderr"

  set +e
  (
    cd "$WORK"
    exec timeout --signal=TERM --kill-after=5s "$RUNTIME_TIMEOUT_SECONDS" "$elf"
  ) >"$stdout" 2>"$stderr"
  local rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    show_failure_tail "$stdout"
    report_execution_failure "${label}_runtime" "$rc" "$stderr" "$elf"
  fi
  assert_no_fatal_log "${label}_runtime" "$rc" "$stderr" "$elf"
  if ! cmp -s "$expected" "$stdout"; then
    printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_FAIL stage=%s_runtime reason=stdout_mismatch expected_hex=' "$label" >&2
    od -An -tx1 "$expected" | tr -d ' \n' >&2
    printf ' actual_hex=' >&2
    od -An -tx1 "$stdout" | tr -d ' \n' >&2
    printf '\n' >&2
    exit 1
  fi
  [[ ! -s "$stderr" ]] || fail "${label}_runtime" stderr_not_empty
  printf 'MADAROS_IMPORTED_RUNTIME_CASE_PASS issue=%s build=full_ir_noncompact elf=executable runtime_exit=0 stdout=%s fallback=none\n' \
    "${label#issue_}" "$stdout_token"
}

capture_issue_901_metrics() {
  local log="$1"
  local elf="$2"
  local value=""
  local text_hex=""
  local readelf_sections="$WORK/issue_901.readelf.sections"
  local readelf_segments="$WORK/issue_901.readelf.segments"
  local readelf_relocations="$WORK/issue_901.readelf.relocations"
  local readelf_symbols="$WORK/issue_901.readelf.symbols"

  ISSUE_901_FUNCTIONS="$(sed -n 's/^Merged IR: \([0-9][0-9]*\)$/\1/p' "$log" | tail -n 1)"
  ISSUE_901_FUNCTION_SOURCE=merged_ir
  if [[ -z "$ISSUE_901_FUNCTIONS" ]]; then
    ISSUE_901_FUNCTIONS="$(sed -n 's/.*fn_count=\([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1)"
    ISSUE_901_FUNCTION_SOURCE=compiler_log
  fi

  ISSUE_901_CODE_BYTES="$(sed -n 's/.*code_size=\([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1)"
  ISSUE_901_CODE_SOURCE=compiler_log
  ISSUE_901_RELOCATIONS="$(sed -n 's/.*reloc_count=\([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1)"
  ISSUE_901_RELOCATION_SOURCE=compiler_log

  if command -v readelf >/dev/null 2>&1; then
    if readelf -SW "$elf" >"$readelf_sections" 2>/dev/null; then
      if [[ -z "$ISSUE_901_CODE_BYTES" ]]; then
        text_hex="$(awk '{ for (i = 1; i <= NF; i++) if ($i == ".text") { print $(i + 4); exit } }' "$readelf_sections")"
        if [[ "$text_hex" =~ ^[0-9a-fA-F]+$ ]]; then
          ISSUE_901_CODE_BYTES="$((16#$text_hex))"
          ISSUE_901_CODE_SOURCE=elf_text
        fi
      fi
    fi

    if [[ -z "$ISSUE_901_CODE_BYTES" ]] && readelf -lW "$elf" >"$readelf_segments" 2>/dev/null; then
      text_hex="$(awk '$1 == "LOAD" { for (i = 7; i <= NF; i++) if ($i == "E") { print $5; exit } }' "$readelf_segments")"
      text_hex="${text_hex#0x}"
      if [[ "$text_hex" =~ ^[0-9a-fA-F]+$ ]]; then
        ISSUE_901_CODE_BYTES="$((16#$text_hex))"
        ISSUE_901_CODE_SOURCE=elf_executable_segment
      fi
    fi

    if [[ -z "$ISSUE_901_RELOCATIONS" ]] && readelf -rW "$elf" >"$readelf_relocations" 2>/dev/null; then
      ISSUE_901_RELOCATIONS="$(awk '/^[[:space:]]*[0-9a-fA-F]+[[:space:]]+[0-9a-fA-F]+[[:space:]]+R_/ { n++ } END { print n + 0 }' "$readelf_relocations")"
      ISSUE_901_RELOCATION_SOURCE=elf_relocation_sections
    fi

    if [[ -z "$ISSUE_901_FUNCTIONS" ]] && readelf -sW "$elf" >"$readelf_symbols" 2>/dev/null; then
      value="$(awk '$4 == "FUNC" && $7 != "UND" { n++ } END { if (n > 0) print n }' "$readelf_symbols")"
      if [[ -n "$value" ]]; then
        ISSUE_901_FUNCTIONS="$value"
        ISSUE_901_FUNCTION_SOURCE=elf_symbols
      fi
    fi
  fi

  [[ "$ISSUE_901_FUNCTIONS" =~ ^[1-9][0-9]*$ ]] || fail issue_901_metrics function_count_unavailable
  if [[ -z "$ISSUE_901_CODE_BYTES" ]]; then
    ISSUE_901_CODE_BYTES=unavailable
    ISSUE_901_CODE_SOURCE=unavailable
  fi
  if [[ -z "$ISSUE_901_RELOCATIONS" ]]; then
    ISSUE_901_RELOCATIONS=unavailable
    ISSUE_901_RELOCATION_SOURCE=unavailable
  fi
  ISSUE_901_ELF_BYTES="$(wc -c <"$elf" | tr -d ' ')"
}

assert_private_rejection() {
  local rc="$1"
  local log="$2"
  local elf="$3"
  local build_cwd="$4"

  if [[ "$rc" -eq 0 ]]; then
    printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_FAIL stage=issue_862_private_build reason=unexpected_acceptance final_elf=%s\n' \
      "$(artifact_state "$elf")" >&2
    exit 1
  fi
  [[ "$rc" -eq 1 ]] || report_execution_failure issue_862_private_build "$rc" "$log" "$elf"
  assert_no_fatal_log issue_862_private_build "$rc" "$log" "$elf"
  has_forbidden_path "$log" && fail issue_862_private_build compact_or_fallback_path_observed
  [[ ! -e "$elf" && ! -L "$elf" ]] || fail issue_862_private_build rejected_program_left_final_elf
  [[ "$(grep -Fc 'error[E' "$log" || true)" -eq 1 ]] || fail issue_862_private_build diagnostic_count_mismatch
  [[ "$(grep -Fc 'error[E175' "$log" || true)" -eq 1 ]] || fail issue_862_private_build e175_count_mismatch
  [[ "$(grep -Fc 'function is private in its defining module' "$log" || true)" -eq 1 ]] || \
    fail issue_862_private_build privacy_message_count_mismatch
  grep -Fq 'imported_compile: visibility_done' "$log" || \
    fail issue_862_private_build visibility_preflight_completion_missing
  grep -Fq 'Visibility/type preflight failed during imported compile' "$log" || \
    fail issue_862_private_build visibility_preflight_rejection_missing
  grep -Fq 'native_v2_compile: front-half/backend failed rc=1' "$log" || \
    fail issue_862_private_build frontend_rejection_status_missing
  if grep -Eq 'imported_compile: lower_begin|module_frontend_full_ir: lower_node|lower_array:|canonical AST closure full IR path|Merged IR:|Compilation successful!' "$log"; then
    fail issue_862_private_build lowering_reached_after_rejection
  fi
  assert_empty_build_cwd issue_862_private_build "$build_cwd"
  printf 'MADAROS_IMPORTED_RUNTIME_CASE_PASS issue=862-private diagnostic=E175 rejection_rc=1 final_elf=absent fallback=none\n'
}

[[ $# -eq 0 ]] || fail setup unexpected_argument
[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail setup linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail setup x86_64_required ;;
esac

for command_name in awk basename cmp dirname env find grep mkdir mktemp od rm sed sha256sum tail timeout tr uname wc; do
  require_command "$command_name"
done
[[ "$KEEP_WORK" == 0 || "$KEEP_WORK" == 1 ]] || fail setup invalid_keep_value
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail setup invalid_build_timeout
[[ "$RUNTIME_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail setup invalid_runtime_timeout
[[ -x "$WRAPPER" ]] || fail setup madaros_wrapper_missing
[[ -n "$RAW_MADAROS" ]] || fail setup explicit_raw_madaros_required
[[ -x "$RAW_MADAROS" ]] || fail setup explicit_raw_madaros_missing_or_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd -P)/$(basename "$RAW_MADAROS")"
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail setup expected_raw_sha256_required
[[ "$(elf_magic "$RAW_MADAROS")" == 7f454c46 ]] || fail setup raw_compiler_must_be_elf
[[ "$(elf_u16 "$RAW_MADAROS" 16)" == 2 ]] || fail setup raw_compiler_must_be_et_exec
[[ "$(elf_u16 "$RAW_MADAROS" 18)" == 62 ]] || fail setup raw_compiler_must_be_x86_64
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$RAW_SHA256" == "$EXPECTED_RAW_SHA256" ]] || fail setup raw_compiler_sha256_mismatch

require_sha256 "$ISSUE_921_SOURCE" 222e34365d37fee43d762c40db336211afdfb2d88ee7b840597fe3e1af3c7059 issue_921_exact_repro
require_sha256 "$ISSUE_901_SOURCE" 986e8a570310367e7b035d69653ee89bc849adb2d56166f3f4270bad2e988f99 issue_901_prob_stdlib
require_sha256 "$ISSUE_901_ITEM_CHAIN_MAIN" 84eb48da4d52acc53efd717b90809b51d56b1f77850e26e11e6d68b762ca272e issue_901_item_chain_main
require_sha256 "$ISSUE_901_ITEM_CHAIN_LEAF" 9977a30ac80ea1fae87d9b9be44f4707ede8f2962646d7099bf2ec4dfdebc38e issue_901_item_chain_leaf
require_sha256 "$ISSUE_901_ITEM_CHAIN_NESTED" 8dccd536f720dd292c823207cad0ee03f4e170f3dfc6a92c9bef2a37f932e22c issue_901_item_chain_nested
require_sha256 "$ISSUE_901_LAYOUT_CAPACITY_MAIN" d6d54e468297434e7b3f4b26cbbfc4a2f8d0af2a6a1f249daee6580ef34bf234 issue_901_layout_capacity_main
require_sha256 "$ISSUE_901_LAYOUT_CAPACITY_LEAF" f8aa765b7adb6c6366945c86c9a87d606c43891aa28ee8ea7a9f7ae135a55105 issue_901_layout_capacity_leaf
require_sha256 "$ISSUE_862_POSITIVE" 2fe4b0bee43ef8b55f349dc17320733c967c53ac595c4a1e4beffbae8a8d2094 issue_862_positive
require_sha256 "$ISSUE_862_PUBLIC_LEAF" 505b2f5255663bdefe49e4cfb94d97ccfb25a8acf1259d0835744ad389be9d19 issue_862_public_leaf
require_sha256 "$ISSUE_862_PRIVATE_MAIN" f6e69122a94788f4db5a82ded8dd62138455b098678f59dab01901fd48baa8fa issue_862_private_main
require_sha256 "$ISSUE_862_PRIVATE_LEAF" cbe28015500cff576991d1ef7a772c916833f622e75aa087a0cd4d6b2d7786af issue_862_private_leaf

if [[ -n "${SOUNIO_MADAROS_IMPORTED_RUNTIME_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_IMPORTED_RUNTIME_DIR"
  [[ ! -e "$WORK" ]] || fail setup work_directory_already_exists
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-imported-runtime.XXXXXX")"
fi
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW_VERSION_LOG="$WORK/raw-version.log"
set +e
timeout --signal=TERM --kill-after=5s 30 "$RAW_MADAROS" --version >"$RAW_VERSION_LOG" 2>&1
RAW_VERSION_RC=$?
set -e
[[ "$RAW_VERSION_RC" -eq 0 ]] || report_execution_failure raw_identity "$RAW_VERSION_RC" "$RAW_VERSION_LOG" "$RAW_MADAROS"
assert_no_fatal_log raw_identity "$RAW_VERSION_RC" "$RAW_VERSION_LOG" "$RAW_MADAROS"
grep -Fq 'Madaros v' "$RAW_VERSION_LOG" || fail raw_identity madaros_version_missing

WRAPPER_INFO_LOG="$WORK/wrapper-info.log"
set +e
env -u SOUNIO_MADAROS_BIN -u SOUNIO_SOUC_BIN \
  MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  timeout --signal=TERM --kill-after=5s 30 "$WRAPPER" info >"$WRAPPER_INFO_LOG" 2>&1
WRAPPER_INFO_RC=$?
set -e
[[ "$WRAPPER_INFO_RC" -eq 0 ]] || report_execution_failure wrapper_identity "$WRAPPER_INFO_RC" "$WRAPPER_INFO_LOG" "$RAW_MADAROS"
assert_no_fatal_log wrapper_identity "$WRAPPER_INFO_RC" "$WRAPPER_INFO_LOG" "$RAW_MADAROS"
grep -Fxq "raw_elf:      $RAW_MADAROS" "$WRAPPER_INFO_LOG" || fail raw_identity wrapper_raw_identity_mismatch

printf 'MADAROS_IMPORTED_RUNTIME_COMPILER_PASS raw_authority=explicit_sha256 raw_sha256=%s compact=disabled fallback=forbidden work=%s\n' \
  "$RAW_SHA256" "$WORK"

printf '11\n' >"$WORK/issue_921.expected"
run_build issue_921 "$ISSUE_921_SOURCE" "$WORK/issue_921.elf"
assert_positive_build issue_921 "$CASE_RC" "$CASE_LOG" "$CASE_ELF" "$CASE_BUILD_CWD"
run_exact_runtime issue_921 "$CASE_ELF" "$WORK/issue_921.expected" 11_LF

printf '42\n' >"$WORK/issue_901_item_chain.expected"
run_build issue_901_item_chain "$ISSUE_901_ITEM_CHAIN_MAIN" "$WORK/issue_901_item_chain.elf"
assert_positive_build issue_901_item_chain "$CASE_RC" "$CASE_LOG" "$CASE_ELF" "$CASE_BUILD_CWD"
run_exact_runtime issue_901_item_chain "$CASE_ELF" "$WORK/issue_901_item_chain.expected" 42_LF

printf 'ISSUE_901_LAYOUT_CAPACITY_OK\n' >"$WORK/issue_901_layout_capacity.expected"
run_build issue_901_layout_capacity "$ISSUE_901_LAYOUT_CAPACITY_MAIN" "$WORK/issue_901_layout_capacity.elf"
assert_positive_build issue_901_layout_capacity "$CASE_RC" "$CASE_LOG" "$CASE_ELF" "$CASE_BUILD_CWD"
run_exact_runtime issue_901_layout_capacity "$CASE_ELF" "$WORK/issue_901_layout_capacity.expected" LAYOUT_CAPACITY_257_LF

printf 'PROB_STDLIB_OK\n' >"$WORK/issue_901.expected"
run_build issue_901 "$ISSUE_901_SOURCE" "$WORK/issue_901.elf"
assert_positive_build issue_901 "$CASE_RC" "$CASE_LOG" "$CASE_ELF" "$CASE_BUILD_CWD"
run_exact_runtime issue_901 "$CASE_ELF" "$WORK/issue_901.expected" PROB_STDLIB_OK_LF
capture_issue_901_metrics "$CASE_LOG" "$CASE_ELF"
printf 'MADAROS_IMPORTED_RUNTIME_ISSUE_901_METRICS functions=%s function_source=%s code_bytes=%s code_source=%s relocations=%s relocation_source=%s elf_bytes=%s\n' \
  "$ISSUE_901_FUNCTIONS" "$ISSUE_901_FUNCTION_SOURCE" \
  "$ISSUE_901_CODE_BYTES" "$ISSUE_901_CODE_SOURCE" \
  "$ISSUE_901_RELOCATIONS" "$ISSUE_901_RELOCATION_SOURCE" "$ISSUE_901_ELF_BYTES"

printf '0.500000\n' >"$WORK/issue_862.expected"
run_build issue_862 "$ISSUE_862_POSITIVE" "$WORK/issue_862.elf"
assert_positive_build issue_862 "$CASE_RC" "$CASE_LOG" "$CASE_ELF" "$CASE_BUILD_CWD"
run_exact_runtime issue_862 "$CASE_ELF" "$WORK/issue_862.expected" 0.500000_LF

run_build issue_862_private "$ISSUE_862_PRIVATE_MAIN" "$WORK/issue_862_private.elf"
assert_private_rejection "$CASE_RC" "$CASE_LOG" "$CASE_ELF" "$CASE_BUILD_CWD"

FINAL_RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$FINAL_RAW_SHA256" == "$RAW_SHA256" ]] || fail final_receipt raw_compiler_changed_during_gate
printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_RECEIPT status=pass issues=921+901+862 raw_authority=explicit_sha256 raw_sha256=%s driver=full_ir_noncompact compact=disabled fallback=none issue_921=elf+exit0+11_LF issue_901_item_chain=elf+exit0+42_LF issue_901_layout_capacity=elf+exit0+257_catalog_layouts issue_901=elf+exit0+PROB_STDLIB_OK_LF issue_901_functions=%s issue_901_code_bytes=%s issue_901_relocations=%s issue_862=elf+exit0+0.500000_LF issue_862_true_private=E175+elf_absent crashes=separately_classified\n' \
  "$RAW_SHA256" "$ISSUE_901_FUNCTIONS" "$ISSUE_901_CODE_BYTES" "$ISSUE_901_RELOCATIONS"
printf 'MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_PASS\n'
