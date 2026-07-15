#!/usr/bin/env bash
# Local evidence and strict compiler acceptance for exact bitwise rebracketing.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_REBRACKET_COMPILER_BIN:-$ROOT_DIR/bin/souc}"
REQUIRE_COMPILER="${SOUNIO_REBRACKET_REQUIRE_COMPILER:-0}"
KEEP_WORK="${SOUNIO_REBRACKET_KEEP:-0}"
OPT="$ROOT_DIR/self-hosted/ir/opt_cleanup.sio"
RUNNER="$ROOT_DIR/self-hosted/ir/rebracket_authority_self_test_runner.sio"
KERNEL="$ROOT_DIR/tests/compiler/rebracket_authority_atomic_kernel.sio"
PRIVACY="$ROOT_DIR/tests/compiler/rebracket_authority_privacy"

fail() {
  echo "[rebracket-authority] FAIL: $*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "compiler is missing or not executable: $SOUC"
case "$REQUIRE_COMPILER" in
  0|1) ;;
  *) fail "SOUNIO_REBRACKET_REQUIRE_COMPILER must be 0 or 1" ;;
esac

if [[ -n "${SOUNIO_REBRACKET_WORK_DIR:-}" ]]; then
  WORK="$SOUNIO_REBRACKET_WORK_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing work directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-rebracket-authority.XXXXXX)"
fi
if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

require_text() {
  local pattern="$1"
  local file="$2"
  rg -q "$pattern" "$file" || fail "missing production anchor '$pattern' in $file"
}

reject_text() {
  local pattern="$1"
  local file="$2"
  if rg -q "$pattern" "$file"; then
    fail "forbidden production anchor '$pattern' found in $file"
  fi
}

count_struct_i64_fields() {
  local name="$1"
  awk -v name="$name" '
    $0 == "struct " name " {" || $0 == "pub struct " name " {" { inside=1; next }
    inside && $0 == "}" { print count + 0; exit }
    inside && /: i64,/ { count++ }
  ' "$OPT"
}

require_text '^struct OcpSealedExactBitwiseRebracketAuthority \{' "$OPT"
require_text '^let OCP_REBRACKET_TRACKED_REG_LIMIT: i64 = 256$' "$OPT"
reject_text '^pub struct OcpSealedExactBitwiseRebracketAuthority \{' "$OPT"
require_text '^fn ocp_issue_exact_bitwise_rebracket_authority\(' "$OPT"
reject_text '^pub fn ocp_issue_exact_bitwise_rebracket_authority\(' "$OPT"
require_text 'func: &! IrFunction,' "$OPT"
require_text 'ocp_try_exact_bitwise_rebracket\(&! result, i, la_inner_def\)' "$OPT"
require_text 'outer_const_use_count != 1' "$OPT"
require_text 'ocp_rebracket_has_canonical_scalar_operands' "$OPT"
require_text 'exact_bitwise_operand_model_ok' "$OPT"
require_text 'instr\.arg_count != 0' "$OPT"
require_text '!ocp_rebracket_has_no_call_args\(instr\.call_args\)' "$OPT"
require_text 'ocp_rebracket_has_float_marker_before' "$OPT"
require_text 'ocp_rebracket_occurrences_equal\(expected, current\.1\)' "$OPT"
require_text 'ocp_rebracket_instr_equal' "$OPT"
require_text 'boundary_claim_mask: 1' "$OPT"
reject_text 'runtime_d7_receipt_consumed|float_or_gum_authority_established|global_reassociation_authority_established' "$OPT"

occurrence_fields="$(count_struct_i64_fields OcpExactBitwiseRebracketOccurrence)"
audit_fields="$(count_struct_i64_fields OcpExactBitwiseRebracketAudit)"
receipt_fields="$(count_struct_i64_fields OcpExactBitwiseRebracketProbeReceipt)"
[[ "$occurrence_fields" == 6 ]] || fail "authority occurrence must remain 6 i64 fields, got $occurrence_fields"
[[ "$audit_fields" == 8 ]] || fail "authority audit must remain 8 i64 fields, got $audit_fields"
[[ "$receipt_fields" == 5 ]] || fail "probe receipt must remain 5 i64 fields, got $receipt_fields"
# These compact record counts are compatibility tripwires, not inferred ABIs:
# changing one must force an explicit gate review instead of passing silently.

combine_slice="$(awk '
  /^fn ocp_rebracket_combine\(/ { inside=1 }
  inside { print }
  inside && /^}/ { exit }
' "$OPT")"
[[ "$combine_slice" == *'BinaryOp::OpBitAnd'* ]] || fail "exact combiner is missing AND"
[[ "$combine_slice" == *'BinaryOp::OpBitOr'* ]] || fail "exact combiner is missing OR"
[[ "$combine_slice" == *'BinaryOp::OpBitXor'* ]] || fail "exact combiner is missing XOR"
[[ "$combine_slice" != *'BinaryOp::OpAdd'* ]] || fail "exact combiner must not admit Add"
[[ "$combine_slice" != *'BinaryOp::OpMul'* ]] || fail "exact combiner must not admit Mul"

operand_slice="$(awk '
  /^fn ocp_rebracket_has_canonical_scalar_operands\(/ { inside=1 }
  inside { print }
  inside && /^}/ { exit }
' "$OPT")"
for forbidden_opcode in IrLoadFloat IrIntToFloat IrFloatToInt IrCall IrCallExtern IrCallIndirect IrCallSret IrPhi IrJump IrBranchTrue IrBranchFalse IrLabel; do
  [[ "$operand_slice" != *"IrOpcode::$forbidden_opcode => true"* ]] ||
    fail "canonical scalar slice must refuse $forbidden_opcode"
done

kernel_log="$WORK/kernel.log"
if ! "$SOUC" run "$KERNEL" >"$kernel_log" 2>&1; then
  cat "$kernel_log" >&2
  fail "scalar authority kernel failed"
fi
kernel_marker='[rebracket-kernel] PASS cases=11 apps=3 refusals=8 replay=0 float=0 shared=0 collision=0'
rg -Fxq "$kernel_marker" "$kernel_log" || {
  cat "$kernel_log" >&2
  fail "scalar authority kernel omitted its exact receipt"
}

expect_private_rejection() {
  local label="$1"
  local source="$2"
  local code="$3"
  local message="$4"
  local log="$WORK/$label.log"
  local rc=0

  set +e
  "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e

  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  [[ "$(rg -c 'error\[E' "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one diagnostic"
  }
  [[ "$(rg -c "error\[$code" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code"
  }
  [[ "$(rg -c "$message" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label omitted the canonical privacy message"
  }
}

expect_private_rejection authority-private-struct \
  "$PRIVACY/authority_private_struct_main.sio" E176 \
  'struct constructor is private in its defining module'
expect_private_rejection authority-private-issuer \
  "$PRIVACY/authority_private_issuer_main.sio" E175 \
  'function is private in its defining module'

bash "$ROOT_DIR/scripts/ci/no_false_float_axioms.sh" >"$WORK/no-false-float.log" 2>&1 || {
  cat "$WORK/no-false-float.log" >&2
  fail "false-float-axiom guard failed"
}

compiler_check_log="$WORK/compiler.check.log"
compiler_state="unknown"
set +e
"$SOUC" check "$RUNNER" >"$compiler_check_log" 2>&1
compiler_check_rc=$?
set -e

if [[ "$compiler_check_rc" -eq 0 ]]; then
  compiler_run_log="$WORK/compiler.run.log"
  if ! "$SOUC" run "$RUNNER" >"$compiler_run_log" 2>&1; then
    cat "$compiler_run_log" >&2
    fail "production runner checked but did not execute"
  fi
  rg -Fq '[rebracket-compiler] PASS: cases=14 applications=3 unchanged_refusals=11' \
    "$compiler_run_log" || {
      cat "$compiler_run_log" >&2
      fail "production runner omitted its exact receipt"
    }
  compiler_state="executable"
else
  rg -q 'run_check_mode: verdict=1' "$compiler_check_log" || {
    cat "$compiler_check_log" >&2
    fail "production runner failed outside checker preflight"
  }
  reject_text 'parse error|error\[E015|error\[E039|error\[E137' "$compiler_check_log"
  require_text 'error\[E175' "$compiler_check_log"
  require_text 'function is private in its defining module' "$compiler_check_log"
  require_text 'error\[E016' "$compiler_check_log"
  require_text 'expected \[IrInstr; 2048' "$compiler_check_log"
  require_text 'found \[IrInstr; 1024' "$compiler_check_log"

  diagnostic_codes="$(rg -o 'error\[E[0-9]+' "$compiler_check_log" | sort -u)"
  while IFS= read -r code; do
    case "$code" in
      'error[E016'|'error[E035'|'error[E175') ;;
      '') ;;
      *) cat "$compiler_check_log" >&2; fail "unexpected production diagnostic: $code" ;;
    esac
  done <<<"$diagnostic_codes"
  compiler_state="blocked-known-baseline"
fi

if [[ "$REQUIRE_COMPILER" == 1 && "$compiler_state" != executable ]]; then
  echo '[rebracket-authority] BLOCKED BLK-20260715-REBRACKET-MODULAR-SOURCE: strict compiler execution is required' >&2
  exit 1
fi

merge_ready=0
if [[ "$compiler_state" == executable && "$REQUIRE_COMPILER" == 1 ]]; then
  merge_ready=1
fi

echo "[rebracket-authority] LOCAL_EVIDENCE_PASS kernel=11/11 privacy=E175,E176 occurrence_words=$occurrence_fields audit_words=$audit_fields receipt_words=$receipt_fields compiler_state=$compiler_state merge_ready=$merge_ready"
