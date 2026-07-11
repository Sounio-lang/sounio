#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CASE_TIMEOUT="${EISA_MADAROS_NATIVE_TIMEOUT:-300}"
WORK_DIR="${EISA_MADAROS_NATIVE_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-eisa-native-conformance.XXXXXX")}"
KEEP_WORK="${EISA_MADAROS_NATIVE_KEEP:-0}"
MADAROS="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-$ROOT_DIR/bin/madaros}}"
CORPUS="$ROOT_DIR/tools/eisa/eisa_evm_run.sio"
RECEIPT_PATH="${EISA_MADAROS_NATIVE_RECEIPT_PATH:-$WORK_DIR/eisa-native-conformance.receipt}"
COMPACT_SIMPLE_IR_CAP=128
COMPACT_OVERCAPACITY_FN_COUNT=$((COMPACT_SIMPLE_IR_CAP + 2))

if [[ "$KEEP_WORK" != "1" && -z "${EISA_MADAROS_NATIVE_WORK_DIR:-}" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

fail() {
  echo "[eisa-madaros-native] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros compiler is not executable: $MADAROS"
[[ -f "$CORPUS" ]] || fail "missing EISA corpus: $CORPUS"
grep -Fqx "const MODULE_FRONTEND_IMPORTED_SIMPLE_CAP: i64 = $COMPACT_SIMPLE_IR_CAP" \
  self-hosted/compiler/module_frontend.sio \
  || fail "compact simple IR witness capacity drifted from compiler source"
mkdir -p "$WORK_DIR"

run_vm() {
  local source="$1"
  local output="$2"
  local log="$3"
  set +e
  env SOUNIO_SOUC_ENGINE=lean_single SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    timeout "$CASE_TIMEOUT" "$ROOT_DIR/bin/souc" run "$source" >"$output" 2>"$log"
  local rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || {
    tail -80 "$log" >&2 || true
    fail "METRON/VM execution failed for $(basename "$source") (rc=$rc)"
  }
}

compile_and_run_native() {
  local source="$1"
  local elf="$2"
  local output="$3"
  local compile_log="$4"
  local run_log="$5"
  set +e
  env SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    timeout "$CASE_TIMEOUT" "$MADAROS" --native-compile "$source" -o "$elf" \
    >"$compile_log" 2>&1
  local compile_rc=$?
  set -e
  [[ "$compile_rc" -eq 0 ]] || {
    tail -120 "$compile_log" >&2 || true
    fail "Madaros native compilation failed for $(basename "$source") (rc=$compile_rc)"
  }
  [[ -s "$elf" ]] || fail "native compilation produced no ELF for $(basename "$source")"
  chmod +x "$elf"

  grep -Fq 'module_native_driver: imported source selected full modular IR path' "$compile_log" \
    || fail "$(basename "$source") did not select the full modular IR path"
  if grep -Eq 'compact simple IR|compact modular IR|falling back to full modular IR path|imported_simple_ir_over_capacity|IR lowering failed' "$compile_log"; then
    tail -120 "$compile_log" >&2 || true
    fail "$(basename "$source") reached a forbidden compact/fallback lowering path"
  fi

  set +e
  timeout "$CASE_TIMEOUT" "$elf" >"$output" 2>"$run_log"
  local run_rc=$?
  set -e
  [[ "$run_rc" -eq 0 ]] || {
    tail -80 "$run_log" >&2 || true
    fail "Madaros ELF execution failed for $(basename "$source") (rc=$run_rc)"
  }
}

require_receipts() {
  local output="$1"
  local label="$2"
  local count
  count="$(grep -c '^eisa-receipt: ' "$output" || true)"
  [[ "$count" -eq 39 ]] || fail "$label emitted $count receipts; expected 39"
  [[ "$(wc -l <"$output")" -eq 39 ]] || fail "$label emitted non-receipt stdout"
}

require_not_baked() {
  local elf="$1"
  local output="$2"
  local label="$3"
  grep -aq 'eisa-receipt: v=' "$elf" || fail "$label lacks the static receipt prefix"
  mapfile -t digit_runs < <(grep -Eo '[0-9]{12,}' "$output" | sort -u)
  local run
  for run in "${digit_runs[@]}"; do
    if grep -aq "$run" "$elf"; then
      fail "$label bakes source-observable receipt digits into the ELF: $run"
    fi
  done
}

require_compact_opt_in_fails_closed() {
  local source="$1"
  local elf="$2"
  local log="$3"
  rm -f "$elf"
  set +e
  env SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" SOUNIO_MADAROS_COMPACT_SIMPLE_IR=1 \
    timeout "$CASE_TIMEOUT" "$MADAROS" --native-compile "$source" -o "$elf" \
    >"$log" 2>&1
  local rc=$?
  set -e
  [[ "$rc" -ne 0 && "$rc" -ne 124 && "$rc" -ne 139 ]] \
    || fail "compact simple IR opt-in did not fail closed cleanly (rc=$rc)"
  [[ ! -e "$elf" ]] || fail "compact simple IR opt-in wrote an ELF after rejection"
  grep -Fq 'module_native_driver: imported source explicitly selected compact simple IR path' "$log" \
    || fail "compact simple IR negative did not enter the explicit opt-in path"
  grep -Fq 'module_native_driver: compact simple IR failed closed: imported_simple_ir_over_capacity' "$log" \
    || fail "compact simple IR negative lacked the classified capacity verdict"
  if grep -Fq 'module_native_driver: imported source selected full modular IR path' "$log"; then
    fail "compact simple IR rejection silently fell back to full modular IR"
  fi
}

make_compact_overcapacity_case() {
  # This replaces eisa_madaros_native_fail_closed_gate.sh: the success corpus,
  # EISA compact rejection, and synthetic capacity rejection now share one receipt.
  local case_dir="$WORK_DIR/compact-overcapacity"
  mkdir -p "$case_dir/overcap"
  {
    echo 'use overcap::mod::*'
    echo
    echo 'fn main() -> i64 {'
    echo '    var total: i64 = 0'
    local j=0
    while [[ "$j" -lt "$COMPACT_OVERCAPACITY_FN_COUNT" ]]; do
      printf '    total = total + f%03d()\n' "$j"
      j=$((j + 1))
    done
    echo '    total'
    echo '}'
  } >"$case_dir/main.sio"
  : >"$case_dir/overcap/mod.sio"
  local i=0
  while [[ "$i" -lt "$COMPACT_OVERCAPACITY_FN_COUNT" ]]; do
    printf 'fn f%03d() -> i64 {\n    %d\n}\n\n' "$i" "$i" >>"$case_dir/overcap/mod.sio"
    i=$((i + 1))
  done
  printf '%s\n' "$case_dir/main.sio"
}

VM_OUT="$WORK_DIR/metron-vm.stdout"
VM_LOG="$WORK_DIR/metron-vm.stderr"
NATIVE_ELF="$WORK_DIR/madaros-native.elf"
NATIVE_OUT="$WORK_DIR/madaros-native.stdout"
NATIVE_COMPILE_LOG="$WORK_DIR/madaros-native.compile.log"
NATIVE_RUN_LOG="$WORK_DIR/madaros-native.stderr"

run_vm "$CORPUS" "$VM_OUT" "$VM_LOG"
compile_and_run_native "$CORPUS" "$NATIVE_ELF" "$NATIVE_OUT" "$NATIVE_COMPILE_LOG" "$NATIVE_RUN_LOG"
require_receipts "$VM_OUT" "METRON/VM corpus"
require_receipts "$NATIVE_OUT" "Madaros native corpus"
cmp -s "$VM_OUT" "$NATIVE_OUT" || {
  diff -u "$VM_OUT" "$NATIVE_OUT" >&2 || true
  fail "Madaros ELF stdout is not bit-identical to METRON/VM"
}
require_not_baked "$NATIVE_ELF" "$NATIVE_OUT" "original corpus"

TAMPER_SOURCE="$WORK_DIR/eisa_evm_run_tampered.sio"
sed '0,/    b\.consts\[0\] = 7\.25/{s/    b\.consts\[0\] = 7\.25/    b.consts[0] = 7.5/}' \
  "$CORPUS" >"$TAMPER_SOURCE"
cmp -s "$CORPUS" "$TAMPER_SOURCE" && fail "tamper transform did not change the corpus"

TAMPER_VM_OUT="$WORK_DIR/tampered-metron-vm.stdout"
TAMPER_VM_LOG="$WORK_DIR/tampered-metron-vm.stderr"
TAMPER_ELF="$WORK_DIR/tampered-madaros-native.elf"
TAMPER_NATIVE_OUT="$WORK_DIR/tampered-madaros-native.stdout"
TAMPER_COMPILE_LOG="$WORK_DIR/tampered-madaros-native.compile.log"
TAMPER_RUN_LOG="$WORK_DIR/tampered-madaros-native.stderr"

run_vm "$TAMPER_SOURCE" "$TAMPER_VM_OUT" "$TAMPER_VM_LOG"
compile_and_run_native "$TAMPER_SOURCE" "$TAMPER_ELF" "$TAMPER_NATIVE_OUT" "$TAMPER_COMPILE_LOG" "$TAMPER_RUN_LOG"
require_receipts "$TAMPER_VM_OUT" "tampered METRON/VM corpus"
require_receipts "$TAMPER_NATIVE_OUT" "tampered Madaros native corpus"
cmp -s "$TAMPER_VM_OUT" "$TAMPER_NATIVE_OUT" || {
  diff -u "$TAMPER_VM_OUT" "$TAMPER_NATIVE_OUT" >&2 || true
  fail "tampered Madaros ELF stdout is not bit-identical to tampered METRON/VM"
}
cmp -s "$VM_OUT" "$TAMPER_NATIVE_OUT" && fail "tamper did not change source-observable behavior"
CHANGED_RECEIPTS="$(awk 'NR == FNR { original[NR] = $0; next } original[FNR] != $0 { changed += 1 } END { print changed + 0 }' \
  "$VM_OUT" "$TAMPER_NATIVE_OUT")"
[[ "$CHANGED_RECEIPTS" -ge 1 ]] \
  || fail "tamper did not change any source-observable receipt line"
require_not_baked "$TAMPER_ELF" "$TAMPER_NATIVE_OUT" "tampered corpus"

COMPACT_REJECT_ELF="$WORK_DIR/compact-opt-in-rejected.elf"
COMPACT_REJECT_LOG="$WORK_DIR/compact-opt-in-rejected.compile.log"
require_compact_opt_in_fails_closed "$CORPUS" "$COMPACT_REJECT_ELF" "$COMPACT_REJECT_LOG"

COMPACT_OVERCAPACITY_SOURCE="$(make_compact_overcapacity_case)"
COMPACT_OVERCAPACITY_ELF="$WORK_DIR/compact-overcapacity-rejected.elf"
COMPACT_OVERCAPACITY_LOG="$WORK_DIR/compact-overcapacity-rejected.compile.log"
require_compact_opt_in_fails_closed \
  "$COMPACT_OVERCAPACITY_SOURCE" "$COMPACT_OVERCAPACITY_ELF" "$COMPACT_OVERCAPACITY_LOG"

mkdir -p "$(dirname "$RECEIPT_PATH")"
RECEIPT_TMP="$RECEIPT_PATH.tmp.$$"
{
  echo "eisa_madaros_native_conformance_receipt_v1"
  echo "compiler=$MADAROS"
  echo "compiler_sha256=$(sha256sum "$MADAROS" | cut -d' ' -f1)"
  echo "corpus_sha256=$(sha256sum "$CORPUS" | cut -d' ' -f1)"
  echo "vm_stdout_sha256=$(sha256sum "$VM_OUT" | cut -d' ' -f1)"
  echo "native_stdout_sha256=$(sha256sum "$NATIVE_OUT" | cut -d' ' -f1)"
  echo "tampered_stdout_sha256=$(sha256sum "$TAMPER_NATIVE_OUT" | cut -d' ' -f1)"
  echo "tampered_receipts_changed=$CHANGED_RECEIPTS"
  echo "native_lowering=full_modular_ir_no_fallback"
  echo "compact_opt_in=fail_closed_no_fallback"
  echo "compact_overcapacity_witness=130_functions_fail_closed_no_elf"
  echo "receipts=39/39"
  echo "tamper=pass"
  echo "anti_vacuity=pass"
} >"$RECEIPT_TMP"
mv "$RECEIPT_TMP" "$RECEIPT_PATH"

grep -Fxq 'receipts=39/39' "$RECEIPT_PATH" || fail "receipt validation failed"
grep -Fxq 'tamper=pass' "$RECEIPT_PATH" || fail "tamper receipt validation failed"
grep -Fxq 'anti_vacuity=pass' "$RECEIPT_PATH" || fail "anti-vacuity receipt validation failed"

echo "[eisa-madaros-native] PASS: 39/39 METRON/VM == Madaros ELF, tamper-sensitive, anti-vacuous"
echo "[eisa-madaros-native] receipt=$RECEIPT_PATH"
