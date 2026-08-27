#!/usr/bin/env bash
# Madaros must never route an unresolved print/println operand to the char*
# builtin. Positively identified strings and numeric scalars remain executable.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURES="$ROOT_DIR/scripts/ci/fixtures/madaros_print_dispatch_refusal"
KEEP_WORK="${SOUNIO_MADAROS_PRINT_DISPATCH_GATE_KEEP:-0}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
TOTAL=6
PASSED=0
FAILED=0
NOT_RUN=0

if [[ -n "${SOUNIO_MADAROS_PRINT_DISPATCH_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_PRINT_DISPATCH_GATE_DIR"
  if [[ -e "$WORK" ]]; then
    echo "[madaros-print-dispatch] FAIL: gate directory already exists: $WORK" >&2
    exit 1
  fi
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-print-dispatch.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

MADAROS="${SOUNIO_MADAROS_PRINT_DISPATCH_GATE_BIN:-$WORK/madaros}"
if [[ -z "${SOUNIO_MADAROS_PRINT_DISPATCH_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    echo "[madaros-print-dispatch] FAIL: current-source Madaros build failed" >&2
    echo "status=fail total=$TOTAL passed=0 failed=1 not_run=$((TOTAL - 1))"
    exit 1
  fi
fi

if [[ ! -x "$MADAROS" ]]; then
  echo "[madaros-print-dispatch] FAIL: Madaros is not executable: $MADAROS" >&2
  echo "status=fail total=$TOTAL passed=0 failed=1 not_run=$((TOTAL - 1))"
  exit 1
fi

record_pass() {
  PASSED=$((PASSED + 1))
  echo "[madaros-print-dispatch] PASS: $1"
}

record_fail() {
  FAILED=$((FAILED + 1))
  echo "[madaros-print-dispatch] FAIL: $1" >&2
}

expect_madaros_refusal() {
  local label="$1"
  local source="$2"
  local lean_marker="$3"
  local elf="$WORK/$label.elf"
  local compile_log="$WORK/$label.compile.log"
  local run_log="$WORK/$label.run.log"
  local lean_log="$WORK/$label.lean.log"
  local compile_rc=0

  set +e
  "$MADAROS" "$source" -o "$elf" >"$compile_log" 2>&1
  compile_rc=$?
  set -e

  if [[ "$compile_rc" -eq 0 ]]; then
    local run_rc=0
    chmod +x "$elf"
    set +e
    "$elf" >"$run_log" 2>&1
    run_rc=$?
    set -e
    record_fail "$label was accepted by Madaros (runtime rc=$run_rc)"
    return
  fi
  if [[ -e "$elf" ]]; then
    record_fail "$label refusal left an ELF behind"
    return
  fi
  if ! grep -Fq 'cannot safely lower print/println argument with unresolved scalar kind' "$compile_log"; then
    cat "$compile_log" >&2
    record_fail "$label refused without the dispatch diagnostic"
    return
  fi
  if grep -Fq 'Compilation successful' "$compile_log"; then
    record_fail "$label printed a success marker after refusal"
    return
  fi

  if ! SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$source" >"$lean_log" 2>&1; then
    cat "$lean_log" >&2
    record_fail "$label lean_single control did not execute"
    return
  fi
  if ! grep -Fq "$lean_marker" "$lean_log"; then
    cat "$lean_log" >&2
    record_fail "$label lean_single control produced the wrong value"
    return
  fi
  record_pass "$label is fail-closed in Madaros and executes in lean_single"
}

expect_madaros_run() {
  local label="$1"
  local source="$2"
  local marker="$3"
  local marker2="${4:-}"
  local elf="$WORK/$label.elf"
  local compile_log="$WORK/$label.compile.log"
  local run_log="$WORK/$label.run.log"

  if ! "$MADAROS" "$source" -o "$elf" >"$compile_log" 2>&1; then
    cat "$compile_log" >&2
    record_fail "$label did not compile"
    return
  fi
  chmod +x "$elf"
  if ! "$elf" >"$run_log" 2>&1; then
    cat "$run_log" >&2
    record_fail "$label did not execute"
    return
  fi
  if ! grep -Fq "$marker" "$run_log"; then
    cat "$run_log" >&2
    record_fail "$label output marker is missing"
    return
  fi
  if [[ -n "$marker2" ]] && ! grep -Fq "$marker2" "$run_log"; then
    cat "$run_log" >&2
    record_fail "$label second output marker is missing"
    return
  fi
  record_pass "$label compiled and executed under Madaros"
}

expect_madaros_refusal \
  unresolved-print-if \
  "$FIXTURES/unresolved_print_if.sio" \
  'LEAN_PRINT=41'
expect_madaros_refusal \
  unresolved-println-if \
  "$FIXTURES/unresolved_println_if.sio" \
  'LEAN_PRINTLN=41'
expect_madaros_refusal \
  unresolved-string-if \
  "$FIXTURES/unresolved_string_if.sio" \
  'LEAN_STRING=left'
expect_madaros_run string-param "$FIXTURES/string_param.sio" 'PARAM=param-ok'
expect_madaros_run string-return "$FIXTURES/string_return.sio" 'RETURN=return-ok'
expect_madaros_run scalar-controls "$FIXTURES/scalar_controls.sio" 'INT<17>' 'F64<2.500000>'

if [[ "$FAILED" -ne 0 ]]; then
  NOT_RUN=$((TOTAL - PASSED - FAILED))
  echo "status=fail total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN"
  exit 1
fi

echo "MADAROS_PRINT_DISPATCH_REFUSAL_GATE_OK"
echo "status=pass total=$TOTAL passed=$PASSED failed=0 not_run=0"
