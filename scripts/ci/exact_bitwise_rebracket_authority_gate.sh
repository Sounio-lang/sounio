#!/usr/bin/env bash
# Local evidence and strict compiler acceptance for exact bitwise rebracketing.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPLICIT_SOUC="${SOUNIO_REBRACKET_COMPILER_BIN:-}"
SOUC="${EXPLICIT_SOUC:-$ROOT_DIR/bin/souc}"
EXPECTED_COMPILER_SHA256="${SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256:-}"
REQUIRE_COMPILER="${SOUNIO_REBRACKET_REQUIRE_COMPILER:-0}"
KEEP_WORK="${SOUNIO_REBRACKET_KEEP:-0}"
OPT="$ROOT_DIR/self-hosted/ir/opt_cleanup.sio"
MAIN="$ROOT_DIR/self-hosted/compiler/main.sio"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
NATIVE_DRIVER="$ROOT_DIR/self-hosted/compiler/module_native_driver.sio"
RUNNER="$ROOT_DIR/self-hosted/ir/rebracket_authority_self_test_runner.sio"
KERNEL="$ROOT_DIR/tests/compiler/rebracket_authority_atomic_kernel.sio"
PRIVACY="$ROOT_DIR/tests/compiler/rebracket_authority_privacy"
DEFAULT_O_WITNESS="$ROOT_DIR/tests/compiler/rebracket_authority_default_o_reachability.sio"
IMPORTED_O_WITNESS="$ROOT_DIR/tests/compiler/rebracket_authority_imported_main.sio"

fail() {
  echo "[rebracket-authority] FAIL: $*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "compiler is missing or not executable: $SOUC"
case "$REQUIRE_COMPILER" in
  0|1) ;;
  *) fail "SOUNIO_REBRACKET_REQUIRE_COMPILER must be 0 or 1" ;;
esac

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
source_sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ "$REQUIRE_COMPILER" == 1 ]]; then
  [[ -n "$EXPLICIT_SOUC" ]] || fail "strict mode requires SOUNIO_REBRACKET_COMPILER_BIN"
  compiler_magic="$(od -An -tx1 -N4 "$SOUC" | tr -d ' \n')"
  [[ "$compiler_magic" == 7f454c46 ]] || fail "strict compiler must be an explicit ELF, got magic=$compiler_magic"
  [[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    fail "strict mode requires a lowercase 64-hex SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256"
  [[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] ||
    fail "compiler SHA-256 mismatch: expected=$EXPECTED_COMPILER_SHA256 actual=$compiler_sha256"
  [[ -z "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=no)" ]] ||
    fail "strict mode requires a clean tracked source worktree"
fi

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
require_text '^struct OcpExactBitwiseFlowUseCertificate \{' "$OPT"
require_text '^let OCP_REBRACKET_TRACKED_REG_LIMIT: i64 = 256$' "$OPT"
require_text '^let OCP_REBRACKET_CFG_CAPACITY: i64 = 2048$' "$OPT"
require_text 'var reached: \[bool; 2048\] = \[false; 2048\]' "$OPT"
reject_text '^pub struct OcpSealedExactBitwiseRebracketAuthority \{' "$OPT"
require_text '^fn ocp_issue_exact_bitwise_rebracket_authority\(' "$OPT"
reject_text '^pub fn ocp_issue_exact_bitwise_rebracket_authority\(' "$OPT"
require_text 'func: &! IrFunction,' "$OPT"
require_text 'ocp_try_exact_bitwise_rebracket\(&! result, i, la_inner_def\)' "$OPT"
require_text 'outer_const_use_count != 1' "$OPT"
require_text 'ocp_rebracket_has_canonical_scalar_operands' "$OPT"
require_text 'ocp_rebracket_function_has_complete_use_model' "$OPT"
require_text 'exact_bitwise_use_model_ok' "$OPT"
require_text 'ocp_rebracket_block_start_index' "$OPT"
require_text 'ocp_certify_exact_bitwise_flow_use' "$OPT"
require_text 'ocp_rebracket_forward_cfg_status' "$OPT"
require_text 'ocp_rebracket_cfg_has_duplicate_labels' "$OPT"
require_text 'ocp_rebracket_forward_cfg_reaches_excluding' "$OPT"
require_text 'ocp_rebracket_forward_cfg_dominates' "$OPT"
require_text 'ocp_rebracket_flow_path_binding\(inner_block, outer_block\)' "$OPT"
require_text 'ocp_rebracket_occurrence_failure' "$OPT"
require_text 'ocp_rebracket_occurrence_is_valid' "$OPT"
require_text 'ocp_rebracket_occurrence_reason' "$OPT"
require_text 'ocp_rebracket_flow_certificate_failure' "$OPT"
require_text 'ocp_rebracket_flow_certificate_reason' "$OPT"
reject_text 'captured\.[012]|current\.[012]|stale_capture\.[012]|certified\.[012]|current_flow\.[012]|stale_certificate\.[012]' "$OPT"
require_text 'instr\.arg_count == 0' "$OPT"
require_text 'ocp_rebracket_has_no_call_args\(instr\.call_args\)' "$OPT"
require_text 'ocp_rebracket_has_float_marker_before' "$OPT"
require_text 'ocp_rebracket_occurrences_equal\(expected, current\)' "$OPT"
require_text 'ocp_rebracket_instr_equal' "$OPT"
require_text 'boundary_claim_mask = boundary_claim_mask \| 16' "$OPT"
require_text 'boundary_claim_mask = boundary_claim_mask \| 32' "$OPT"
require_text 'nondom\.reason_code == 21' "$OPT"
require_text 'backward\.reason_code == 19' "$OPT"
require_text 'duplicate\.reason_code == 17' "$OPT"
require_text 'duplicate_label\.label_id = 2' "$OPT"
require_text '\(\(duplicate\.reason_code \+ 1\) << 8\)' "$OPT"
require_text 'missing\.reason_code == 18' "$OPT"
require_text 'base_kill\.reason_code == 10' "$OPT"
require_text 'phi\.reason_code == 14' "$OPT"
reject_text 'cfg_build\(|cfg_add_edge\(|cfg_compute_dominators\(' "$OPT"
reject_text 'runtime_d7_receipt_consumed|float_or_gum_authority_established|global_reassociation_authority_established' "$OPT"
require_text '^pub struct OcpCleanupModuleReceipt \{' "$OPT"
require_text '^pub fn opt_cleanup_module_inplace\(module: &! IrModule\)' "$OPT"
require_text '^pub fn ocp_exact_bitwise_rebracket_pipeline_probe\(\)' "$OPT"
require_text 'ocp_record_exact_bitwise_rebracket_audit\(exact_bitwise_audit, la_txn\)' "$OPT"
require_text 'exact_bitwise_application_count: audit\.application_count' "$OPT"
require_text 'exact_bitwise_same_region_application_count: audit\.same_region_application_count' "$OPT"
require_text 'exact_bitwise_cross_block_application_count: audit\.cross_block_application_count' "$OPT"
require_text 'exact_bitwise_transaction_ontology_parameter_link_count: transaction_ontology_parameter_link_count' "$OPT"
require_text 'exact_bitwise_application_ontology_parameter_link_count: application_ontology_parameter_link_count' "$OPT"
require_text 'last_transaction_ontology_class_hash: last_transaction_ontology_class_hash' "$OPT"
require_text 'OCP_CLEANUP_RECEIPT_SCHEMA_VERSION: i64 = 1' "$OPT"
require_text 'exact_bitwise_refusal_reason_mask: audit\.refusal_reason_mask' "$OPT"
require_text '^fn run_exact_bitwise_rebracket_authority_smoke\(\)' "$MAIN"
require_text 'let receipt = ocp_exact_bitwise_rebracket_authority_probe\(\)' "$MAIN"
require_text 'receipt\.boundary_claim_mask != 49' "$MAIN"
require_text 'pipeline\.exact_bitwise_same_region_application_count != 0' "$MAIN"
require_text 'pipeline\.exact_bitwise_cross_block_application_count != 1' "$MAIN"
require_text 'pipeline\.exact_bitwise_transaction_ontology_parameter_link_count != 0' "$MAIN"
require_text 'pipeline\.exact_bitwise_application_ontology_parameter_link_count != 0' "$MAIN"
require_text 'pipeline\.last_transaction_ontology_class_hash != 0' "$MAIN"
require_text 'pipeline\.exact_bitwise_refusal_reason_mask != 0' "$MAIN"
require_text 'mode == "--rebracket-authority-smoke"' "$MAIN"
require_text '\[rebracket-compiler\] PROBE: cases=' "$MAIN"
require_text '\[rebracket-compiler\] PASS: cases=21 applications=5 unchanged_refusals=16 pipeline=1' "$MAIN"
require_text 'module_frontend_compile_imported_to_file\(opts\.input_file, opts\.output_file, opts\.optimize\)' "$MAIN"
require_text '^pub fn module_frontend_compile_imported_to_file\(' "$FRONTEND"
require_text 'opt_cleanup_module_inplace\(&! \(\*module_box\)\)' "$FRONTEND"
require_text 'opt_cleanup_module_inplace\(&! merged_module\)' "$FRONTEND"
require_text '\[rebracket-native-v2\] phase=' "$FRONTEND"
require_text 'if !optimize \{' "$NATIVE_DRIVER"
require_text 'module_frontend_compile_imported_to_file\(main_path, output_path, optimize\)' "$NATIVE_DRIVER"

occurrence_fields="$(count_struct_i64_fields OcpExactBitwiseRebracketOccurrence)"
flow_certificate_fields="$(count_struct_i64_fields OcpExactBitwiseFlowUseCertificate)"
audit_fields="$(count_struct_i64_fields OcpExactBitwiseRebracketAudit)"
receipt_fields="$(count_struct_i64_fields OcpExactBitwiseRebracketProbeReceipt)"
module_receipt_fields="$(count_struct_i64_fields OcpCleanupModuleReceipt)"
[[ "$occurrence_fields" == 6 ]] || fail "authority occurrence must remain 6 i64 fields, got $occurrence_fields"
[[ "$flow_certificate_fields" == 2 ]] || fail "flow certificate must remain 2 i64 fields, got $flow_certificate_fields"
[[ "$audit_fields" == 9 ]] || fail "authority audit must remain 9 i64 fields, got $audit_fields"
[[ "$receipt_fields" == 5 ]] || fail "probe receipt must remain 5 i64 fields, got $receipt_fields"
[[ "$module_receipt_fields" == 14 ]] || fail "module cleanup receipt must remain 14 i64 fields, got $module_receipt_fields"
# These compact record counts are compatibility tripwires, not inferred ABIs:
# changing one, or reintroducing a heterogeneous capture-result tuple, must
# force an explicit gate review instead of passing silently.

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
for forbidden_opcode in IrLoadFloat IrIntToFloat IrFloatToInt IrCall IrCallExtern IrCallIndirect IrCallSret IrPhi IrJump IrBranchTrue IrBranchFalse IrLabel IrReturn IrReturnSret; do
  [[ "$operand_slice" != *"IrOpcode::$forbidden_opcode => true"* ]] ||
    fail "canonical scalar slice must refuse $forbidden_opcode"
done

control_slice="$(awk '
  /^fn ocp_rebracket_has_supported_control_operands\(/ { inside=1 }
  inside { print }
  inside && /^}/ { exit }
' "$OPT")"
[[ "$control_slice" == *'IrOpcode::IrLabel'* ]] ||
  fail "explicit control-use model is missing IrLabel"
[[ "$control_slice" == *'ocp_rebracket_is_control_terminator(op)'* ]] ||
  fail "explicit control-use model is not bound to its terminator set"
terminator_slice="$(awk '
  /^fn ocp_rebracket_is_control_terminator\(/ { inside=1 }
  inside { print }
  inside && /^}/ { exit }
' "$OPT")"
for admitted_control in IrJump IrBranchTrue IrBranchFalse IrReturn IrReturnSret; do
  [[ "$terminator_slice" == *"IrOpcode::$admitted_control"* ]] ||
    fail "explicit control-use model is missing $admitted_control"
done
for forbidden_control in IrCall IrCallExtern IrCallIndirect IrCallSret IrPhi; do
  [[ "$control_slice$terminator_slice" != *"IrOpcode::$forbidden_control"* ]] ||
    fail "explicit control-use model must refuse $forbidden_control"
done

pass_a1_slice="$(awk '
  /^fn ocp_const_fold_pass_a1\(/ { inside=1 }
  /^fn ocp_const_fold_pass_a2\(/ { exit }
  inside { print }
' "$OPT")"
[[ "$pass_a1_slice" == *'former direct'* || "$pass_a1_slice" == *'former Block BG direct'* ]] ||
  fail "pass A1 must declare its authority-owned rebracketing handoff"
[[ "$pass_a1_slice" != *'Block AE: And-mask chain fold'* ]] ||
  fail "pass A1 must not retain the pre-authority AND-chain mutation"
[[ "$pass_a1_slice" != *'Block BG: OR constant-chain fold'* ]] ||
  fail "pass A1 must not retain the pre-authority OR-chain mutation"
[[ "$pass_a1_slice" != *'Block BM: XOR constant-chain fold'* ]] ||
  fail "pass A1 must not retain the pre-authority XOR-chain mutation"

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

compiler_state="unknown"
compiler_path="unknown"
native_v2_reachability="not-required"
compiler_help_log="$WORK/compiler.help.log"
"$SOUC" --help >"$compiler_help_log" 2>&1 || true

if rg -Fq -- '--rebracket-authority-smoke' "$compiler_help_log"; then
  compiler_smoke_log="$WORK/compiler.smoke.log"
  if ! "$SOUC" --rebracket-authority-smoke >"$compiler_smoke_log" 2>&1; then
    cat "$compiler_smoke_log" >&2
    fail "current-source compiler advertised the authority smoke but failed it"
  fi
  rg -Fxq '[rebracket-compiler] PASS: cases=21 applications=5 unchanged_refusals=16 pipeline=1 replay=0 calls=0 packed=0 cross_block=1 nondom=0 backedge=0 duplicate=0 missing=0 base_kill=0 phi=0 control_elsewhere=1 float_ir=0 runtime_d7=0 float_gum=0 global=0' \
    "$compiler_smoke_log" || {
      cat "$compiler_smoke_log" >&2
      fail "current-source compiler authority smoke omitted its exact receipt"
    }
  compiler_state="executable"
  compiler_path="internal-smoke"
else
  # The checked-in compiler predates the focused internal mode. Keep the
  # modular import as a diagnostic classifier only: its historical real
  # cross-module privacy and IrFunction-capacity failures are not acceptance.
  compiler_check_log="$WORK/compiler.check.log"
  set +e
  "$SOUC" check "$RUNNER" >"$compiler_check_log" 2>&1
  compiler_check_rc=$?
  set -e

  [[ "$compiler_check_rc" -ne 0 ]] || {
    cat "$compiler_check_log" >&2
    fail "compiler lacks the internal smoke but unexpectedly accepts the diagnostic runner"
  }
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
  compiler_state="blocked-prebuilt-no-smoke"
  compiler_path="modular-baseline"
fi

if [[ "$REQUIRE_COMPILER" == 1 && "$compiler_state" != executable ]]; then
  echo '[rebracket-authority] BLOCKED BLK-20260715-REBRACKET-CURRENT-SOURCE-SMOKE: a fresh compiler with the internal smoke is required' >&2
  exit 1
fi

run_elf_expect_zero() {
  local label="$1"
  local elf="$2"
  local log="$WORK/$label.runtime.log"
  local rc=0
  local magic=""

  [[ -s "$elf" ]] || fail "$label did not produce a non-empty ELF: $elf"
  magic="$(od -An -tx1 -N4 "$elf" | tr -d ' \n')"
  [[ "$magic" == 7f454c46 ]] || fail "$label produced a non-ELF artifact: magic=$magic path=$elf"
  # The raw Madaros path writes the ELF payload; the public `build` launcher
  # normally supplies this file-mode step. The gate uses the raw `-O` ordering
  # to avoid source-first lean_single routing, so mirror that launcher action.
  chmod +x "$elf"
  set +e
  "$elf" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label runtime witness returned rc=$rc"
  }
}

if [[ "$REQUIRE_COMPILER" == 1 ]]; then
  wrapper_info_log="$WORK/wrapper.info.log"
  MADAROS_RAW_BIN="$SOUC" SOUNIO_SOUC_ENGINE=madaros \
    "$ROOT_DIR/bin/souc" info >"$wrapper_info_log" 2>&1 || {
      cat "$wrapper_info_log" >&2
      fail "public wrapper could not bind the explicit strict compiler"
    }
  rg -Fq "raw_elf:      $SOUC" "$wrapper_info_log" || {
    cat "$wrapper_info_log" >&2
    fail "public wrapper resolved a compiler other than the explicit strict ELF"
  }

  noopt_elf="$WORK/rebracket-noopt.elf"
  noopt_log="$WORK/rebracket-noopt.compile.log"
  SOUNIO_REBRACKET_TRACE=1 MADAROS_RAW_BIN="$SOUC" SOUNIO_SOUC_ENGINE=madaros \
    "$ROOT_DIR/bin/souc" -t native "$DEFAULT_O_WITNESS" -o "$noopt_elf" >"$noopt_log" 2>&1 || {
      cat "$noopt_log" >&2
      fail "default native-v2 no-opt control failed to compile"
    }
  rg -q '^\[rebracket-native-v2\] phase=single-disabled schema=1 optimize=0 functions=[0-9]+ transactions=0 authorizations=0 applications=0 refusals=0 same_region_applications=0 cross_block_applications=0 refusal_reason_mask=0 transaction_ontology_parameter_links=0 application_ontology_parameter_links=0 last_ontology_class_hash=0 operator_mask=0 last_function=-1 combined=0$' \
    "$noopt_log" || {
      cat "$noopt_log" >&2
      fail "no-opt control omitted its exact disabled receipt"
    }
  reject_text 'optimize=1' "$noopt_log"
  run_elf_expect_zero default-noopt "$noopt_elf"

  optimized_elf="$WORK/rebracket-optimized.elf"
  optimized_log="$WORK/rebracket-optimized.compile.log"
  SOUNIO_REBRACKET_TRACE=1 MADAROS_RAW_BIN="$SOUC" SOUNIO_SOUC_ENGINE=madaros \
    "$ROOT_DIR/bin/souc" -O "$DEFAULT_O_WITNESS" -o "$optimized_elf" >"$optimized_log" 2>&1 || {
      cat "$optimized_log" >&2
      fail "default native-v2 -O witness failed to compile"
    }
  rg -q '^\[rebracket-native-v2\] phase=single-post-resolve schema=1 optimize=1 functions=[0-9]+ transactions=1 authorizations=1 applications=1 refusals=0 same_region_applications=1 cross_block_applications=0 refusal_reason_mask=0 transaction_ontology_parameter_links=0 application_ontology_parameter_links=0 last_ontology_class_hash=0 operator_mask=1 last_function=[0-9]+ combined=15$' \
    "$optimized_log" || {
      cat "$optimized_log" >&2
      fail "default native-v2 -O witness omitted its exact application receipt"
    }
  [[ "$(rg -c '^\[rebracket-native-v2\]' "$optimized_log")" -eq 1 ]] || {
    cat "$optimized_log" >&2
    fail "default native-v2 -O witness emitted an ambiguous number of receipts"
  }
  reject_text 'falling back|lean_single' "$optimized_log"
  run_elf_expect_zero default-optimized "$optimized_elf"

  imported_elf="$WORK/rebracket-imported-optimized.elf"
  imported_log="$WORK/rebracket-imported-optimized.compile.log"
  SOUNIO_REBRACKET_TRACE=1 MADAROS_RAW_BIN="$SOUC" SOUNIO_SOUC_ENGINE=madaros \
    "$ROOT_DIR/bin/souc" -O "$IMPORTED_O_WITNESS" -o "$imported_elf" >"$imported_log" 2>&1 || {
      cat "$imported_log" >&2
      fail "merged native-v2 -O witness failed to compile"
    }
  require_text 'module_native_driver: -O selects finalized full IR cleanup path' "$imported_log"
  rg -q '^\[rebracket-native-v2\] phase=merged-post-finalize schema=1 optimize=1 functions=[0-9]+ transactions=1 authorizations=1 applications=1 refusals=0 same_region_applications=1 cross_block_applications=0 refusal_reason_mask=0 transaction_ontology_parameter_links=0 application_ontology_parameter_links=0 last_ontology_class_hash=0 operator_mask=1 last_function=[0-9]+ combined=15$' \
    "$imported_log" || {
      cat "$imported_log" >&2
      fail "merged native-v2 -O witness omitted its exact application receipt"
    }
  [[ "$(rg -c '^\[rebracket-native-v2\]' "$imported_log")" -eq 1 ]] || {
    cat "$imported_log" >&2
    fail "merged native-v2 -O witness emitted an ambiguous number of receipts"
  }
  reject_text 'falling back|lean_single' "$imported_log"
  run_elf_expect_zero merged-optimized "$imported_elf"

  native_v2_reachability="single-and-merged"
  compiler_path="internal-smoke+default-o"
fi

merge_ready=0
if [[ "$compiler_state" == executable && "$REQUIRE_COMPILER" == 1 && "$native_v2_reachability" == single-and-merged ]]; then
  merge_ready=1
fi

echo "[rebracket-authority] LOCAL_EVIDENCE_PASS kernel=11/11 privacy=E175,E176 occurrence_words=$occurrence_fields flow_certificate_words=$flow_certificate_fields audit_words=$audit_fields receipt_words=$receipt_fields module_receipt_words=$module_receipt_fields compiler_state=$compiler_state compiler_path=$compiler_path native_v2_reachability=$native_v2_reachability compiler_sha256=$compiler_sha256 source_sha=$source_sha merge_ready=$merge_ready"
