#!/usr/bin/env bash
# Prove imported named-function references through the canonical full-IR path.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${SOUNIO_MADAROS_FULL_PATH_HOF_RAW_BIN:-}"
KEEP_WORK="${SOUNIO_MADAROS_FULL_PATH_HOF_KEEP:-0}"
TIMEOUT_SECONDS="${SOUNIO_MADAROS_FULL_PATH_HOF_TIMEOUT_SECONDS:-360}"
SOURCE="$ROOT_DIR/tests/compiler/module_graph_fn_ref_hof/main.sio"
PRIVATE_SOURCE="$ROOT_DIR/tests/compiler/module_graph_fn_ref_hof/private_fn_ref_main.sio"

fail() {
  printf '[madaros-full-path-hof] FAIL: %s\n' "$1" >&2
  exit 1
}

is_fatal_log() {
  grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$1"
}

has_forbidden_path() {
  grep -Eq 'native_prebundle:|falling back to full IR path|compact modular IR table path|legacy compact IR differential enabled' "$1"
}

expect_checker_rejection() {
  local label="$1"
  local source="$2"
  local code="$3"
  local message="$4"
  local expected_count="$5"
  local log="$WORK/$label.check.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
    "$WRAPPER" --science-boundary off check "$source" >"$log" 2>&1
  rc=$?
  set -e

  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2 || true
    fail "${label}_expected_rc_1_got_$rc"
  }
  is_fatal_log "$log" && fail "${label}_fatal"
  [[ "$(grep -Fc 'error[E' "$log" || true)" -eq "$expected_count" ]] || fail "${label}_diagnostic_count_mismatch"
  [[ "$(grep -Fc "error[$code" "$log" || true)" -eq "$expected_count" ]] || fail "${label}_${code}_count_mismatch"
  [[ "$(grep -Fc "$message" "$log" || true)" -eq "$expected_count" ]] || fail "${label}_message_count_mismatch"
  grep -Fq 'run_check_mode: verdict=1' "$log" || fail "${label}_checker_verdict_missing"
}

if [[ -n "${SOUNIO_MADAROS_FULL_PATH_HOF_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_FULL_PATH_HOF_DIR"
  [[ ! -e "$WORK" ]] || fail "work_directory_already_exists path=$WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-full-path-hof.XXXXXX")"
fi

if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail x86_64_required ;;
esac

[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail invalid_timeout_seconds
[[ -x "$WRAPPER" ]] || fail madaros_wrapper_missing
[[ -n "$RAW_MADAROS" ]] || fail explicit_raw_required
[[ -x "$RAW_MADAROS" ]] || fail explicit_raw_missing_or_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] || fail raw_compiler_must_be_elf
[[ -f "$SOURCE" ]] || fail fixture_main_missing
[[ -f "$ROOT_DIR/tests/compiler/module_graph_fn_ref_hof/hof_leaf.sio" ]] || fail fixture_leaf_missing
[[ -f "$PRIVATE_SOURCE" ]] || fail fixture_private_ref_main_missing
[[ -f "$ROOT_DIR/tests/compiler/module_graph_fn_ref_hof/private_fn_ref_leaf.sio" ]] || fail fixture_private_ref_leaf_missing

expect_checker_rejection imported_private_fn_ref "$PRIVATE_SOURCE" E175 'function is private in its defining module' 2

CHECK_LOG="$WORK/check.log"
set +e
MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  SOUNIO_CHECKER_CONTEXTUAL_LOOKUP_TRACE=1 \
  timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
  "$WRAPPER" --science-boundary off check "$SOURCE" >"$CHECK_LOG" 2>&1
CHECK_RC=$?
set -e

[[ "$CHECK_RC" -eq 0 ]] || {
  tail -n 100 "$CHECK_LOG" >&2 || true
  fail "check_rc_$CHECK_RC"
}
is_fatal_log "$CHECK_LOG" && fail check_fatal
grep -Fq 'run_check_mode: verdict=0' "$CHECK_LOG" || fail checker_verdict_missing
grep -Fq 'check: OK' "$CHECK_LOG" || fail checker_ok_missing
[[ "$(grep -Fc 'checker_contextual_lookup: kind=fn_ref source=TyUnknown result=TyFn policy=local-first-global-unique' "$CHECK_LOG" || true)" -eq 2 ]] \
  || fail checker_contextual_fn_ref_receipt_mismatch
if grep -Fq 'error[E' "$CHECK_LOG" || grep -Eq '^error:' "$CHECK_LOG"; then
  fail checker_diagnostic_on_success
fi
tr -d '\r\n' <"$CHECK_LOG" >"$WORK/check.normalized.log"
grep -Fq 'run_check_mode: about to check 2 modules' "$WORK/check.normalized.log" || fail checker_closure_count_mismatch

BUILD_CWD="$WORK/build-cwd"
BUILD_LOG="$WORK/build.log"
ELF="$WORK/imported-hof.elf"
mkdir -p "$BUILD_CWD"
set +e
(
  cd "$BUILD_CWD"
  MADAROS_RAW_BIN="$RAW_MADAROS" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 \
  SOUNIO_ENABLE_COMPACT_IMPORTED_IR=0 \
    timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
    "$WRAPPER" --science-boundary off build "$SOURCE" -o "$ELF"
) >"$BUILD_LOG" 2>&1
BUILD_RC=$?
set -e

[[ "$BUILD_RC" -eq 0 ]] || {
  tail -n 120 "$BUILD_LOG" >&2 || true
  fail "build_rc_$BUILD_RC"
}
is_fatal_log "$BUILD_LOG" && fail build_fatal
has_forbidden_path "$BUILD_LOG" && fail forbidden_lowering_path
grep -Fq 'canonical AST closure full IR path' "$BUILD_LOG" || fail canonical_full_ir_marker_missing
grep -Fq 'Merged IR:' "$BUILD_LOG" || fail merged_ir_marker_missing
grep -Fq 'Compilation successful!' "$BUILD_LOG" || fail compilation_success_marker_missing
if grep -Fq 'error[E' "$BUILD_LOG" || grep -Eq '^error:' "$BUILD_LOG"; then
  fail compiler_diagnostic_on_build_success
fi
[[ "$(grep -c '^module_frontend_full_ir: lower_node ' "$BUILD_LOG" || true)" -eq 2 ]] || fail full_ir_node_receipt_mismatch
[[ "$(grep -c '^module_frontend_full_ir: lower_edge ' "$BUILD_LOG" || true)" -eq 1 ]] || fail full_ir_edge_receipt_mismatch
[[ -f "$ELF" ]] || fail elf_missing
[[ "$(od -An -tx1 -N4 "$ELF" | tr -d ' \n')" == 7f454c46 ]] || fail artifact_not_elf
if find "$BUILD_CWD" -type f -print -quit | grep -q .; then
  fail unexpected_build_cwd_artifact
fi

STDOUT="$WORK/runtime.stdout"
STDERR="$WORK/runtime.stderr"
chmod +x "$ELF"
set +e
timeout --signal=TERM --kill-after=5s 60 "$ELF" >"$STDOUT" 2>"$STDERR"
RUNTIME_RC=$?
set -e

[[ "$RUNTIME_RC" -eq 42 ]] || {
  cat "$STDOUT" >&2 || true
  cat "$STDERR" >&2 || true
  fail "runtime_rc_$RUNTIME_RC"
}
[[ ! -s "$STDOUT" ]] || fail runtime_stdout_not_empty
[[ ! -s "$STDERR" ]] || fail runtime_stderr_not_empty

printf 'MADAROS_FULL_PATH_IMPORTED_HOF_PASS closure_modules=2 checker_refinement=TyUnknown-to-TyFn paths=inplace+by-value private_fn_ref=E175x2 runtime_exit=42 stdout=empty fallback=none\n'
