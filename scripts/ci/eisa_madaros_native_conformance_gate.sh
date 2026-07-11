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

if [[ "$KEEP_WORK" != "1" && -z "${EISA_MADAROS_NATIVE_WORK_DIR:-}" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

fail() {
  echo "[eisa-madaros-native] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros compiler is not executable: $MADAROS"
[[ -f "$CORPUS" ]] || fail "missing EISA corpus: $CORPUS"
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
