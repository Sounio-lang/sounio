#!/usr/bin/env bash
# compiler_stage_contract_gate.sh
#
# Classifies the Stage0/Stage1 compiler architecture contract. This is not the
# final parity gate. It is the gate that prevents parallel agents from hiding
# blockers or adding new semantics to the bootstrap monolith by accident.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=../lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK_DIR="${SOUNIO_COMPILER_STAGE_CONTRACT_DIR:-/tmp/sounio-compiler-stage-contract}"
LOG_DIR="$WORK_DIR/logs"
SUMMARY="$WORK_DIR/summary.tsv"
TIMEOUT_SECS="${SOUNIO_STAGE_CONTRACT_TIMEOUT_SECS:-120}"

rm -rf "$WORK_DIR"
mkdir -p "$LOG_DIR"
printf 'case\tstatus\treason\tlog\tcommand\n' > "$SUMMARY"

PASS_COUNT=0
KNOWN_COUNT=0
FAIL_COUNT=0
XPASS_COUNT=0

format_cmd() {
  local out="" quoted
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    out="$out $quoted"
  done
  printf '%s' "${out# }"
}

record_case() {
  local name="$1"
  local status="$2"
  local reason="$3"
  local log_path="$4"
  shift 4
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$status" "$reason" "$log_path" "$(format_cmd "$@")" >> "$SUMMARY"
}

pass_case() {
  PASS_COUNT=$((PASS_COUNT + 1))
  record_case "$@"
}

known_case() {
  KNOWN_COUNT=$((KNOWN_COUNT + 1))
  record_case "$@"
}

fail_case() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  record_case "$@"
}

xpass_case() {
  XPASS_COUNT=$((XPASS_COUNT + 1))
  record_case "$@"
}

required_file() {
  local name="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    pass_case "$name" "pass" "file_present" "-" test -f "$path"
  else
    fail_case "$name" "fail" "missing_file" "-" test -f "$path"
  fi
}

required_grep() {
  local name="$1"
  local path="$2"
  local pattern="$3"
  local reason="$4"
  if grep -Eq "$pattern" "$path"; then
    pass_case "$name" "pass" "$reason" "-" grep -Eq "$pattern" "$path"
  else
    fail_case "$name" "fail" "pattern_missing:$reason" "-" grep -Eq "$pattern" "$path"
  fi
}

required_rejects_with_pattern() {
  local name="$1"
  local reason="$2"
  local pattern="$3"
  shift 3
  local log_path="$LOG_DIR/$name.log"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]] && grep -Eq "$pattern" "$log_path"; then
    pass_case "$name" "pass" "$reason" "$log_path" "$@"
  elif [[ $rc -eq 0 ]]; then
    fail_case "$name" "fail" "expected_rejection_but_command_succeeded" "$log_path" "$@"
  else
    fail_case "$name" "fail" "rejected_without_expected_pattern:rc_$rc" "$log_path" "$@"
  fi
}

known_blocker_cmd() {
  local name="$1"
  local reason="$2"
  local pattern="$3"
  shift 3
  local log_path="$LOG_DIR/$name.log"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    xpass_case "$name" "xpass" "known_blocker_resolved_update_manifest:$reason" "$log_path" "$@"
  elif grep -Eq "$pattern" "$log_path"; then
    known_case "$name" "known_blocker" "$reason" "$log_path" "$@"
  else
    fail_case "$name" "fail" "unclassified_blocker:rc_$rc:$reason" "$log_path" "$@"
  fi
}

required_cmd_ok() {
  local name="$1"
  local reason="$2"
  shift 2
  local log_path="$LOG_DIR/$name.log"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    pass_case "$name" "pass" "$reason" "$log_path" "$@"
  else
    fail_case "$name" "fail" "required_command_failed:rc_$rc:$reason" "$log_path" "$@"
  fi
}

required_cmd_pattern() {
  local name="$1"
  local reason="$2"
  local pattern="$3"
  shift 3
  local log_path="$LOG_DIR/$name.log"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]] && grep -Eq "$pattern" "$log_path"; then
    pass_case "$name" "pass" "$reason" "$log_path" "$@"
  elif [[ $rc -eq 0 ]]; then
    fail_case "$name" "fail" "required_pattern_missing:$reason" "$log_path" "$@"
  else
    fail_case "$name" "fail" "required_command_failed:rc_$rc:$reason" "$log_path" "$@"
  fi
}

known_blocker_file_size_over() {
  local name="$1"
  local path="$2"
  local max_bytes="$3"
  local reason="$4"
  local bytes
  bytes="$(wc -c < "$path" | tr -d ' ')"
  if [[ "$bytes" -gt "$max_bytes" ]]; then
    known_case "$name" "known_blocker" "$reason:bytes_$bytes" "-" wc -c "$path"
  else
    xpass_case "$name" "xpass" "known_blocker_resolved_update_manifest:bytes_$bytes" "-" wc -c "$path"
  fi
}

required_file "manifest_anchor" "tests/compiler/stage_parity/manifest.tsv"
required_file "contract_doc_anchor" "docs/compiler/STAGE0_STAGE1_COMPILER_CONTRACT.md"
required_file "stage0_source_anchor" "self-hosted/compiler/lean_single.sio"
required_file "stage0_fixed_point_gate_anchor" "scripts/ci/souc_v2_gate.sh"
required_file "stage1_lean_source_anchor" "self-hosted/compiler/lean.sio"
required_file "stage1_frontend_source_anchor" "self-hosted/compiler/lean_frontend.sio"

required_grep \
  "stage1_native_driver_streaming_probe" \
  "self-hosted/compiler/module_native_driver.sio" \
  "compile_multimodule_native_maybe_streaming" \
  "streaming_probe_present"

required_grep \
  "stage1_native_driver_legacy_fallback" \
  "self-hosted/compiler/module_native_driver.sio" \
  "compile_native_x86_linux_to_file" \
  "legacy_native_fallback_present"

required_rejects_with_pattern \
  "diagnostic_assign_to_immut_rejects" \
  "B1_compile_fail_sentinel_rejected" \
  "assignment to immutable binding|cannot modify an immutable binding|typecheck: failed|type_check_failed|Mut" \
  "$SOUC_BIN" compile tests/ui/type/assign_to_immut.sio -o "$WORK_DIR/assign_to_immut.elf"

required_cmd_ok \
  "stage1_lean_check" \
  "B2_stage1_full_compiler_check_drift" \
  "$SOUC_BIN" check self-hosted/compiler/lean.sio

required_cmd_ok \
  "stage1_frontend_check" \
  "B2_stage1_frontend_check_drift" \
  "$SOUC_BIN" check self-hosted/compiler/lean_frontend.sio

required_cmd_pattern \
  "stage1_lean_self_test" \
  "B3_stage1_full_compiler_runtime_self_test_drift" \
  "souc-lean self-test: ok" \
  "$SOUC_BIN" run self-hosted/compiler/lean.sio -- --self-test

required_cmd_pattern \
  "stage1_frontend_self_test" \
  "B3_stage1_frontend_runtime_self_test_drift" \
  "souc-lean-frontend self-test: ok" \
  "$SOUC_BIN" run self-hosted/compiler/lean_frontend.sio -- --self-test

required_cmd_pattern \
  "stage1_frontend_hello_check" \
  "B3_stage1_frontend_examples_hello_check_drift" \
  "souc-lean-frontend check: ok" \
  "$SOUC_BIN" run self-hosted/compiler/lean_frontend.sio -- --check examples/hello.sio

known_blocker_file_size_over \
  "main_sio_shim_size" \
  "self-hosted/compiler/main.sio" \
  1000000 \
  "B5_main_sio_remains_compatibility_shim_not_architecture_target"

echo "Compiler stage contract summary: $SUMMARY"
cat "$SUMMARY"

if [[ "$FAIL_COUNT" -ne 0 || "$XPASS_COUNT" -ne 0 ]]; then
  echo "COMPILER_STAGE_CONTRACT_GATE_FAIL pass=$PASS_COUNT known_blocker=$KNOWN_COUNT fail=$FAIL_COUNT xpass=$XPASS_COUNT"
  exit 1
fi

echo "COMPILER_STAGE_CONTRACT_GATE_PASS pass=$PASS_COUNT known_blocker=$KNOWN_COUNT"
