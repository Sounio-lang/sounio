#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOUC="${SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_CONTEST_IR_AUTHORITY_EXPECTED_COMPILER_SHA256:-}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-contest-ir-authority.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() {
  printf 'MADAROS_CONTEST_IR_AUTHORITY_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -n "$SOUC" ]] || fail explicit_source_fresh_compiler_required
[[ -n "$EXPECTED_COMPILER_SHA256" ]] || fail expected_compiler_sha256_required
[[ -x "$SOUC" ]] || fail compiler_missing
[[ "$(od -An -tx1 -N4 "$SOUC" | tr -d ' \n')" == "7f454c46" ]] || fail compiler_must_be_elf
compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] || fail compiler_sha256_mismatch

export SOUNIO_MODULE_GRAPH_REQUIRED=0
export SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1
export SOUNIO_LOWER_LIVE_TRACE=1

run_positive() {
  local label="$1"
  local source="$2"
  local function_name="$3"
  local strategy="$4"
  local expected_count="$5"
  local expected_id="$6"
  local log="$TMP/${label}.log"

  set +e
  timeout --signal=TERM --kill-after=10s 180 \
    "$SOUC" --probe-ir-opt-strategy-boxed "$source" >"$log" 2>&1
  local rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "${label}_probe_rc_${rc}"
  }

  for marker in \
    "module_frontend_lower: epistemic_contest_count ${expected_count}" \
    "CONTEST_IR_INDEX_BIND constructor=summary contest_count=${expected_count}" \
    "CONTEST_IR_LOOKUP_CALL qualified=1 contest_count=${expected_count}" \
    "CONTEST_IR_LOOKUP_ENTER contest_count=${expected_count}" \
    "CONTEST_IR_LOOKUP_CANDIDATE contest_id=${expected_id}" \
    "CONTEST_IR_LOOKUP_RESULT contest_id=${expected_id}" \
    "CONTEST_IR_LOOKUP_CALL_RESULT contest_id=${expected_id}" \
    "CONTEST_IR_EMIT opcode=IrContest contest_id=${expected_id}" \
    "fn=${function_name} strategy=${strategy}" \
    'ir_contest_index=0' \
    "ir_contest_label_id=${expected_id}" \
    'ir_contest_model_count=1'; do
    grep -Fq "$marker" "$log" || {
      cat "$log" >&2
      fail "${label}_missing_${marker//[^a-zA-Z0-9]/_}"
    }
  done

  if grep -Fq 'CONTEST_IR_INDEX_INVALID' "$log"; then
    cat "$log" >&2
    fail "${label}_invalid_index_bounds"
  fi
  printf 'MADAROS_CONTEST_IR_AUTHORITY_CASE_PASS label=%s strategy=%s contest_count=%s contest_id=%s model_count=1\n' \
    "$label" "$strategy" "$expected_count" "$expected_id"
}

run_negative() {
  local label="$1"
  local source="$2"
  local expected_diagnostic="$3"
  local log="$TMP/${label}.log"
  local normalized_log="$TMP/${label}.normalized.log"

  set +e
  timeout --signal=TERM --kill-after=10s 180 \
    "$SOUC" --probe-ir-opt-strategy-boxed "$source" >"$log" 2>&1
  local rc=$?
  set -e
  [[ "$rc" -ne 0 ]] || {
    cat "$log" >&2
    fail "${label}_unexpectedly_accepted"
  }
  tr -d '\r\n' <"$log" >"$normalized_log"
  grep -Fq "error[${expected_diagnostic}]" "$normalized_log" || {
    cat "$log" >&2
    fail "${label}_diagnostic_missing"
  }
  printf 'MADAROS_CONTEST_IR_AUTHORITY_REJECT_PASS label=%s rc=%s diagnostic=%s\n' \
    "$label" "$rc" "$expected_diagnostic"
}

run_positive \
  instrumented \
  tests/frontend/contest_ir_authority_instrumented.sio \
  contest_ir_authority_instrumented \
  instrumented \
  2 \
  1
run_positive \
  standard \
  tests/frontend/contest_ir_authority_standard.sio \
  contest_ir_authority_standard \
  standard \
  1 \
  0

run_negative model_arity tests/compile-fail/contest_model_arity_reject.sio E063
run_negative non_model_family_member tests/compile-fail/contest_non_model_family_member_reject.sio E063
run_negative family_policy_target_mismatch tests/compile-fail/contest_family_policy_target_mismatch_reject.sio E059

printf 'MADAROS_CONTEST_IR_AUTHORITY_PASS compiler_sha256=%s checker_export=heap_owned authority=compiler_owned_contest_index instrumented_contest_count=2 instrumented_contest_id=1 standard_contest_count=1 standard_contest_id=0 final_ir_labels=1,0 model_count=1 negative_controls=3 legacy_module_copy=kept flat_path=unqualified module_epistemic_copy=not_claimed\n' \
  "$compiler_sha256"
