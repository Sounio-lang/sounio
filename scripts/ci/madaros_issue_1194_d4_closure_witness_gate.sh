#!/usr/bin/env bash
# Regression gate for #1194: D4-sized AST closure storage plus fixed-array runtime.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="$ROOT_DIR/bin/madaros"
SOURCE="$ROOT_DIR/tests/issue_1194/d4_closure_fixed_array_witness.sio"
EXPECTED_STDOUT="$ROOT_DIR/tests/issue_1194/d4_closure_fixed_array_witness.stdout"
RAW_MADAROS="${SOUNIO_MADAROS_ISSUE_1194_D4_RAW_BIN:-${MADAROS_RAW_BIN:-}}"
EXPECTED_RAW_SHA256="${SOUNIO_MADAROS_ISSUE_1194_D4_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_ISSUE_1194_D4_KEEP_WORK:-0}"
TIMEOUT_SECONDS="${SOUNIO_MADAROS_ISSUE_1194_D4_TIMEOUT_SECONDS:-180}"
RUNTIME_TIMEOUT_SECONDS="${SOUNIO_MADAROS_ISSUE_1194_D4_RUNTIME_TIMEOUT_SECONDS:-30}"

SOURCE_SHA256="6e463b9db920c85618b2a12f0918a70dc18432952a6801aa737e178f6e5a0817"
STDOUT_SHA256="76af54e5e747102a8daaa3636afc6beeff5772b4bbff37250721c6024b1a1d84"
PINNED_D4_SOURCE_SHA256="5e4f0cdc7643b21f99d890d8318e9eba870a82a909db619bd4d45f90621d6336"
PINNED_MAIN_STALE_SHA256="11e7730f01f5382f1f8a5afc3599d7069b3d917f6972e6e47ffb57aa6bf4421e"
PINNED_D8_CONTROL_SHA256="fa3bcbcb5f72c6d3d851f97521f60dfd6a277ea7faff984d31a10ca377335c2e"
SOURCE_ONLY=0

fail() {
  printf 'MADAROS_ISSUE_1194_D4_CLOSURE_WITNESS_FAIL reason=%s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/ci/madaros_issue_1194_d4_closure_witness_gate.sh [options]

Options:
  --source-only              Validate the pinned fixture contract without a compiler.
  --raw PATH                 Use this explicit raw Madaros ELF.
  --expected-sha256 SHA256   Require this exact raw Madaros SHA-256.
  -h, --help                 Show this help.

Runtime mode can instead use:
  SOUNIO_MADAROS_ISSUE_1194_D4_RAW_BIN=/path/to/madaros
  SOUNIO_MADAROS_ISSUE_1194_D4_EXPECTED_SHA256=<64 lowercase hex>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-only)
      SOURCE_ONLY=1
      shift
      ;;
    --raw)
      [[ $# -ge 2 ]] || fail raw_argument_missing
      RAW_MADAROS="$2"
      shift 2
      ;;
    --expected-sha256)
      [[ $# -ge 2 ]] || fail expected_sha256_argument_missing
      EXPECTED_RAW_SHA256="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail unexpected_argument
      ;;
  esac
done

for path in "$LAUNCHER" "$SOURCE" "$EXPECTED_STDOUT"; do
  [[ -f "$path" ]] || fail "missing_${path#"$ROOT_DIR"/}"
done
[[ -x "$LAUNCHER" ]] || fail launcher_not_executable

actual_source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
actual_stdout_sha256="$(sha256sum "$EXPECTED_STDOUT" | awk '{print $1}')"
[[ "$actual_source_sha256" == "$SOURCE_SHA256" ]] || fail fixture_sha256_mismatch
[[ "$actual_stdout_sha256" == "$STDOUT_SHA256" ]] || fail expected_stdout_sha256_mismatch

struct_count="$(grep -Ec '^struct D4Receipt[0-9][0-9] \{ value: i64 \}$' "$SOURCE" || true)"
function_count="$(grep -Ec '^fn ' "$SOURCE" || true)"
[[ "$struct_count" -eq 30 ]] || fail fixture_struct_count
[[ "$function_count" -eq 63 ]] || fail fixture_function_count
[[ $((struct_count + function_count)) -eq 93 ]] || fail fixture_top_level_item_count
grep -Fxq '    var values: [i64; 8] = [0; 8]' "$SOURCE" || fail fixed_array_shape_missing
[[ "$(grep -Ec '^    values\[[0-7]\] = ' "$SOURCE" || true)" -eq 8 ]] || fail fixed_array_write_count
grep -Fxq '    if checksum != 1194 { return 94 }' "$SOURCE" || fail runtime_exit_oracle_missing
printf '%s\n' 'ISSUE_1194_D4_CLOSURE_WITNESS checksum=1194 items=93' |
  cmp -s - "$EXPECTED_STDOUT" || fail expected_stdout_contract

printf 'MADAROS_ISSUE_1194_D4_CLOSURE_WITNESS_SOURCE_PASS fixture_sha256=%s original_d4_sha256=%s top_level_items=93 structs=30 functions=63 fixed_array_len=8 runtime=not_run\n' \
  "$actual_source_sha256" "$PINNED_D4_SOURCE_SHA256"

if [[ "$SOURCE_ONLY" -eq 1 ]]; then
  exit 0
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail runtime_requires_linux
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail runtime_requires_x86_64 ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail timeout_seconds_invalid
[[ "$RUNTIME_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail runtime_timeout_seconds_invalid
[[ -n "$RAW_MADAROS" ]] || fail explicit_raw_binary_required
[[ -n "$EXPECTED_RAW_SHA256" ]] || fail expected_raw_sha256_required
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail expected_raw_sha256_invalid
[[ -f "$RAW_MADAROS" ]] || fail raw_binary_missing
[[ -x "$RAW_MADAROS" ]] || fail raw_binary_not_executable

raw_dir="$(cd "$(dirname "$RAW_MADAROS")" && pwd -P)"
RAW_MADAROS="$raw_dir/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] ||
  fail raw_binary_not_elf

raw_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$raw_sha256" == "$EXPECTED_RAW_SHA256" ]] || fail raw_sha256_mismatch
case "$raw_sha256" in
  "$PINNED_MAIN_STALE_SHA256") fail known_issue_1194_main_stale_binary ;;
  "$PINNED_D8_CONTROL_SHA256") fail known_issue_1194_d8_control_is_not_current_source ;;
esac

if [[ -n "${SOUNIO_MADAROS_ISSUE_1194_D4_WORK_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_ISSUE_1194_D4_WORK_DIR"
  [[ ! -e "$WORK" ]] || fail work_directory_exists
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-issue-1194-d4.XXXXXX")"
fi
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

INFO_LOG="$WORK/launcher.info"
CHECK_LOG="$WORK/check.log"
COMPILE_LOG="$WORK/compile.log"
ELF="$WORK/d4-closure-fixed-array.elf"
RUNTIME_STDOUT="$WORK/runtime.stdout"
RUNTIME_STDERR="$WORK/runtime.stderr"

show_tail() {
  tail -n 120 "$1" >&2 || true
}

has_fatal_log() {
  grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$1"
}

has_forbidden_evidence() {
  grep -Eiq \
    'closure parser incomplete|AST closure incomplete|raw AST parser reported failure|native_prebundle:|falling back to full IR path|compact modular IR table path|legacy compact IR differential enabled|SELFHOST=fallback|driver_orchestration.*status=fallback' \
    "$1"
}

assert_clean_compiler_log() {
  local label="$1"
  local log="$2"

  if has_fatal_log "$log"; then
    show_tail "$log"
    fail "${label}_fatal_log"
  fi
  if has_forbidden_evidence "$log"; then
    show_tail "$log"
    fail "${label}_incomplete_closure_or_fallback"
  fi
  if grep -Fq 'error[E' "$log" || grep -Eq '^error:' "$log"; then
    show_tail "$log"
    fail "${label}_compiler_diagnostic"
  fi
}

if ! env \
    -u MADAROS_BIN -u SOUC_BIN -u SOUNIO_MADAROS_BIN -u SOUNIO_SOUC_BIN \
    MADAROS_RAW_BIN="$RAW_MADAROS" \
    "$LAUNCHER" info >"$INFO_LOG" 2>&1; then
  show_tail "$INFO_LOG"
  fail launcher_info_failed
fi
grep -Fxq "raw_elf:      $RAW_MADAROS" "$INFO_LOG" || fail launcher_raw_identity_mismatch
launcher_sha256="$(sha256sum "$LAUNCHER" | awk '{print $1}')"

set +e
timeout --signal=TERM --kill-after=5s "$TIMEOUT_SECONDS" \
  env -u MADAROS_BIN -u SOUC_BIN -u SOUNIO_MADAROS_BIN -u SOUNIO_SOUC_BIN \
    MADAROS_RAW_BIN="$RAW_MADAROS" \
    SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    SOUNIO_SOUC_ENGINE=madaros \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
    "$LAUNCHER" check "$SOURCE" >"$CHECK_LOG" 2>&1
check_rc=$?
set -e
if [[ "$check_rc" -ne 0 ]]; then
  show_tail "$CHECK_LOG"
  fail "check_rc_${check_rc}"
fi
assert_clean_compiler_log check "$CHECK_LOG"
[[ "$(grep -Fxc 'check: OK' "$CHECK_LOG" || true)" -eq 1 ]] || fail check_ok_marker_count

rm -f "$ELF"
set +e
SOUNIO_MODULE_CLOSURE_TRACE=1 \
timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
  env -u MADAROS_BIN -u SOUC_BIN -u SOUNIO_MADAROS_BIN -u SOUNIO_SOUC_BIN \
    MADAROS_RAW_BIN="$RAW_MADAROS" \
    SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    SOUNIO_SOUC_ENGINE=madaros \
    SOUNIO_ENABLE_COMPACT_IMPORTED_IR=0 \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
    "$LAUNCHER" compile "$SOURCE" -o "$ELF" >"$COMPILE_LOG" 2>&1
compile_rc=$?
set -e
if [[ "$compile_rc" -ne 0 ]]; then
  show_tail "$COMPILE_LOG"
  fail "compile_rc_${compile_rc}"
fi
assert_clean_compiler_log compile "$COMPILE_LOG"
[[ "$(grep -c '^module_closure_collect: phase=begin ' "$COMPILE_LOG" || true)" -eq 1 ]] ||
  fail compile_collection_count
grep -Fq 'imported_compile: collected_begin collection_id=' "$COMPILE_LOG" ||
  fail compile_collected_receipt_missing
grep -Fq 'imported_compile: visibility_begin' "$COMPILE_LOG" || fail compile_visibility_receipt_missing
grep -Fq 'imported_compile: typecheck_done' "$COMPILE_LOG" || fail compile_typecheck_receipt_missing
grep -Fq 'imported_compile: lower_begin' "$COMPILE_LOG" || fail compile_lower_begin_receipt_missing
grep -Fq 'imported_compile: lower_done' "$COMPILE_LOG" || fail compile_lower_done_receipt_missing
grep -Eq '^Merged IR: [1-9][0-9]*$' "$COMPILE_LOG" || fail compile_merged_ir_receipt_missing
grep -Fxq "native_v2_compile: emitted path=$ELF" "$COMPILE_LOG" || fail compile_emission_receipt_missing
if grep -Fq 'imported_compile: legacy_adapter_' "$COMPILE_LOG" ||
   grep -Fq 'imported_compile: snapshot_invalid' "$COMPILE_LOG"; then
  fail compile_legacy_or_invalid_snapshot
fi

[[ -s "$ELF" ]] || fail compiled_elf_missing
[[ -x "$ELF" ]] || fail compiled_elf_not_executable
[[ "$(od -An -tx1 -N4 "$ELF" | tr -d ' \n')" == 7f454c46 ]] || fail compiled_output_not_elf
if command -v file >/dev/null 2>&1; then
  file "$ELF" | grep -Fq 'ELF 64-bit LSB executable, x86-64' || fail compiled_output_not_x86_64
fi

set +e
timeout --signal=TERM --kill-after=5s "$RUNTIME_TIMEOUT_SECONDS" \
  "$ELF" >"$RUNTIME_STDOUT" 2>"$RUNTIME_STDERR"
runtime_rc=$?
set -e
if [[ "$runtime_rc" -ne 0 ]]; then
  show_tail "$RUNTIME_STDOUT"
  show_tail "$RUNTIME_STDERR"
  fail "runtime_rc_${runtime_rc}"
fi
[[ ! -s "$RUNTIME_STDERR" ]] || {
  show_tail "$RUNTIME_STDERR"
  fail runtime_stderr_not_empty
}
if ! cmp -s "$EXPECTED_STDOUT" "$RUNTIME_STDOUT"; then
  diff -u "$EXPECTED_STDOUT" "$RUNTIME_STDOUT" >&2 || true
  fail runtime_stdout_mismatch
fi

elf_sha256="$(sha256sum "$ELF" | awk '{print $1}')"
printf 'MADAROS_ISSUE_1194_D4_CLOSURE_WITNESS_PASS issue=1194 witness=d4_closure_fixed_array compiler_provenance=explicit_raw_hash raw_sha256=%s launcher_sha256=%s fixture_sha256=%s top_level_items=93 fixed_array_len=8 check=clean compile=full_ir elf_sha256=%s runtime_exit=0 stdout_sha256=%s fallback=none known_stale_hashes=rejected\n' \
  "$raw_sha256" "$launcher_sha256" "$actual_source_sha256" "$elf_sha256" "$actual_stdout_sha256"
