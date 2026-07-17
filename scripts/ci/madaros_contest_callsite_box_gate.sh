#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOUC="${SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_CONTEST_CALLSITE_BOX_EXPECTED_COMPILER_SHA256:-}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-contest-callsite-box.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() {
  printf 'MADAROS_CONTEST_CALLSITE_BOX_FAIL reason=%s\n' "$1" >&2
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

SOURCE=tests/frontend/chain_validated_param_contest.sio

run_callsite() {
  local label="$1"
  local caller="$2"
  local callee="$3"
  local expected_call="$4"
  local expected_forwarding="$5"
  local log="$TMP/${label}.log"

  set +e
  timeout --signal=TERM --kill-after=10s 180 \
    "$SOUC" --probe-ir-callsite "$SOURCE" "$caller" "$callee" >"$log" 2>&1
  local rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "${label}_probe_rc_${rc}"
  }

  for marker in \
    'module_frontend_lower: epistemic_contest_count 3' \
    "$expected_call" \
    "$expected_forwarding" \
    'probe_ir_callsite: ok'; do
    grep -Fq "$marker" "$log" || {
      cat "$log" >&2
      fail "${label}_missing_${marker//[^a-zA-Z0-9]/_}"
    }
  done

  printf 'MADAROS_CONTEST_CALLSITE_BOX_CASE_PASS label=%s caller=%s callee=%s\n' \
    "$label" "$caller" "$callee"
}

run_callsite \
  forwarded \
  outer_contested \
  inner_contested \
  'caller=outer_contested callee=inner_contested caller_strategy=instrumented callee_strategy=instrumented arg_count=2' \
  'forwards_param0=true fallback_load_imm1=false'

run_callsite \
  fallback \
  main \
  outer_contested \
  'caller=main callee=outer_contested caller_strategy=standard callee_strategy=instrumented arg_count=2' \
  'forwards_param0=false fallback_load_imm1=true'

boxed_log="$TMP/boxed.log"
set +e
timeout --signal=TERM --kill-after=10s 180 \
  "$SOUC" --probe-ir-opt-strategy-boxed "$SOURCE" >"$boxed_log" 2>&1
boxed_rc=$?
set -e
[[ "$boxed_rc" -eq 0 ]] || {
  cat "$boxed_log" >&2
  fail "boxed_control_rc_${boxed_rc}"
}
grep -Fq 'probe_ir_opt_strategy_boxed: functions=3' "$boxed_log" || {
  cat "$boxed_log" >&2
  fail boxed_control_function_count_missing
}

printf 'MADAROS_CONTEST_CALLSITE_BOX_PASS compiler_sha256=%s module_owner=Box_IrModule inspectors=by_reference forwarded=1 fallback=1 boxed_control=1 legacy_flat_lowerer=kept_non_authoritative imported_modules=unqualified production_default=unclaimed\n' \
  "$compiler_sha256"
