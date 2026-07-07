#!/usr/bin/env bash
# Madaros v2 S5 scalar program-MIR/ABI gate: build deterministic
# program-level receipts for compiler-exported MachineModule JSON plus the
# current scalar i64/bool ABI shadow contract. This is stronger than
# MIR-effect input receipts, but it still does not claim S5 FULL.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S5_PROGRAM_MIR_ABI_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s5-program-mir-abi.XXXXXX)}"
EFFECT_DIR="$OUT_DIR/mir_effect"
S5_RECEIPT_DIR="$OUT_DIR/canonical_s5_source_receipts"
SRET_RECEIPT_DIR="$OUT_DIR/sret_abi_receipt"
SOURCE_SRET_RECEIPT_DIR="$OUT_DIR/source_sret_receipt"
STACK_CALL_RECEIPT_DIR="$OUT_DIR/stack_call_receipt"
IMPORTED_SRET_RECEIPT_DIR="$OUT_DIR/imported_sret_receipt"
METHOD_SRET_RECEIPT_DIR="$OUT_DIR/method_sret_receipt"
F64_XMM0_RECEIPT_DIR="$OUT_DIR/f64_xmm0_receipt"
WIDE_INT_RECEIPT_DIR="$OUT_DIR/wide_int_receipt"
WIDE_MACHINE_SLOT_RECEIPT_DIR="$OUT_DIR/wide_machine_slot_metadata_receipt"
WIDE_ABI_CALL_RETURN_RECEIPT_DIR="$OUT_DIR/wide_abi_call_return_receipt"
GENERIC_AGG_RECEIPT_DIR="$OUT_DIR/generic_aggregate_sret_receipt"
F128_LITERAL_PROVENANCE_RECEIPT_DIR="$OUT_DIR/f128_literal_provenance_receipt"
F128_BINARY128_VALUE_RECEIPT_DIR="$OUT_DIR/f128_binary128_value_receipt"
F128_LITERAL_VALUE_BRIDGE_RECEIPT_DIR="$OUT_DIR/f128_literal_value_bridge_receipt"
MACHINE_SLOT_METADATA_RECEIPT_DIR="$OUT_DIR/machine_slot_metadata_receipt"
F128_ABI_METADATA_RECEIPT_DIR="$OUT_DIR/f128_abi_metadata_receipt"
F128_NATIVE_OPAQUE_STORAGE_RECEIPT_DIR="$OUT_DIR/f128_native_opaque_storage_receipt"
F128_OPAQUE_CALL_RETURN_ABI_RECEIPT_DIR="$OUT_DIR/f128_opaque_call_return_abi_receipt"
F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT_DIR="$OUT_DIR/f128_sret_internal_arg_boundary_receipt"
F128_BINARY128_NATIVE_ANCHOR_RECEIPT_DIR="$OUT_DIR/f128_binary128_native_anchor_receipt"
F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT_DIR="$OUT_DIR/f128_binary128_value_contract_native_receipt"
F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT_DIR="$OUT_DIR/f128_arithmetic_value_contract_receipt"
F128_IEEE_CLASS_HELPER_RECEIPT_DIR="$OUT_DIR/f128_ieee_class_helper_receipt"
F128_ORDERED_COMPARE_RECEIPT_DIR="$OUT_DIR/f128_ordered_compare_receipt"
F128_PARAM_SLOT_LAYOUT_RECEIPT_DIR="$OUT_DIR/f128_param_slot_layout_receipt"
DIAGNOSTICS_RECEIPT_DIR="$OUT_DIR/diagnostics_receipt"
DIFFERENTIAL_RECEIPT_DIR="$OUT_DIR/differential_receipt"
EFFECT_GATE="${ROOT_DIR}/scripts/dev/madaros_v2_s5_mir_effect_gate.sh"
SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_sret_abi_receipt.py"
SOURCE_SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_source_sret_receipt.py"
STACK_CALL_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_stack_call_receipt.py"
IMPORTED_SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_imported_sret_receipt.py"
METHOD_SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_method_sret_receipt.py"
F64_XMM0_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f64_xmm0_receipt.py"
WIDE_INT_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_wide_int_receipt.py"
WIDE_MACHINE_SLOT_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_wide_machine_slot_metadata_receipt.py"
WIDE_ABI_CALL_RETURN_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_wide_abi_call_return_receipt.py"
GENERIC_AGG_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_generic_aggregate_sret_receipt.py"
F128_LITERAL_PROVENANCE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_f128_literal_provenance_receipt.py"
F128_BINARY128_VALUE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_binary128_value_receipt.py"
F128_LITERAL_VALUE_BRIDGE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_literal_value_bridge_receipt.py"
MACHINE_SLOT_METADATA_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_machine_slot_metadata_receipt.py"
F128_ABI_METADATA_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_abi_metadata_receipt.py"
F128_NATIVE_OPAQUE_STORAGE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_native_opaque_storage_receipt.py"
F128_OPAQUE_CALL_RETURN_ABI_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_opaque_call_return_abi_receipt.py"
F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_sret_internal_arg_boundary_receipt.py"
F128_BINARY128_NATIVE_ANCHOR_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_binary128_native_anchor_receipt.py"
F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_binary128_value_contract_native_receipt.py"
F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_arithmetic_value_contract_receipt.py"
F128_IEEE_CLASS_HELPER_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_ieee_class_helper_receipt.py"
F128_ORDERED_COMPARE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_ordered_compare_receipt.py"
F128_PARAM_SLOT_LAYOUT_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_f128_param_slot_layout_receipt.py"
DIAGNOSTICS_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_diagnostics_receipt.py"
DIFFERENTIAL_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_differential_receipt.py"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
REFERENCE_SOUC="${SOUNIO_MADAROS_V2_S5_REFERENCE_SOUC:-${ROOT_DIR}/bin/souc}"
MANIFEST="${SOUNIO_MADAROS_V2_S5_SCALAR_MANIFEST:-tests/madaros/v2_s5/scalar_mir_abi_manifest.tsv}"
MODULE="$OUT_DIR/madaros_v2_s5_program_mir_abi.module.json"
RECEIPT="$OUT_DIR/madaros_v2_s5_program_mir_abi.receipt.json"
S5_RECEIPT_RESULTS="$OUT_DIR/madaros_v2_s5_source_receipts.tsv"
SRET_RECEIPT="$SRET_RECEIPT_DIR/madaros_v2_s5_sret_abi.receipt.json"
SOURCE_SRET_RECEIPT="$SOURCE_SRET_RECEIPT_DIR/madaros_v2_s5_source_sret.receipt.json"
STACK_CALL_RECEIPT="$STACK_CALL_RECEIPT_DIR/madaros_v2_s5_stack_call.receipt.json"
IMPORTED_SRET_RECEIPT="$IMPORTED_SRET_RECEIPT_DIR/madaros_v2_s5_imported_sret.receipt.json"
METHOD_SRET_RECEIPT="$METHOD_SRET_RECEIPT_DIR/madaros_v2_s5_method_sret.receipt.json"
F64_XMM0_RECEIPT="$F64_XMM0_RECEIPT_DIR/madaros_v2_s5_f64_xmm0.receipt.json"
WIDE_INT_RECEIPT="$WIDE_INT_RECEIPT_DIR/madaros_v2_s5_wide_int.receipt.json"
WIDE_MACHINE_SLOT_RECEIPT="$WIDE_MACHINE_SLOT_RECEIPT_DIR/madaros_v2_s5_wide_machine_slot_metadata.receipt.json"
WIDE_ABI_CALL_RETURN_RECEIPT="$WIDE_ABI_CALL_RETURN_RECEIPT_DIR/madaros_v2_s5_wide_abi_call_return.receipt.json"
GENERIC_AGG_RECEIPT="$GENERIC_AGG_RECEIPT_DIR/madaros_v2_s5_generic_aggregate_sret.receipt.json"
F128_LITERAL_PROVENANCE_RECEIPT="$F128_LITERAL_PROVENANCE_RECEIPT_DIR/madaros_v2_f128_literal_provenance.receipt.json"
F128_BINARY128_VALUE_RECEIPT="$F128_BINARY128_VALUE_RECEIPT_DIR/madaros_v2_s5_f128_binary128_value.receipt.json"
F128_LITERAL_VALUE_BRIDGE_RECEIPT="$F128_LITERAL_VALUE_BRIDGE_RECEIPT_DIR/madaros_v2_s5_f128_literal_value_bridge.receipt.json"
MACHINE_SLOT_METADATA_RECEIPT="$MACHINE_SLOT_METADATA_RECEIPT_DIR/madaros_v2_s5_machine_slot_metadata.receipt.json"
F128_ABI_METADATA_RECEIPT="$F128_ABI_METADATA_RECEIPT_DIR/madaros_v2_s5_f128_abi_metadata.receipt.json"
F128_NATIVE_OPAQUE_STORAGE_RECEIPT="$F128_NATIVE_OPAQUE_STORAGE_RECEIPT_DIR/madaros_v2_s5_f128_native_opaque_storage.receipt.json"
F128_OPAQUE_CALL_RETURN_ABI_RECEIPT="$F128_OPAQUE_CALL_RETURN_ABI_RECEIPT_DIR/madaros_v2_s5_f128_opaque_call_return_abi.receipt.json"
F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT="$F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT_DIR/madaros_v2_s5_f128_sret_internal_arg_boundary.receipt.json"
F128_BINARY128_NATIVE_ANCHOR_RECEIPT="$F128_BINARY128_NATIVE_ANCHOR_RECEIPT_DIR/madaros_v2_s5_f128_binary128_native_anchor.receipt.json"
F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT="$F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT_DIR/madaros_v2_s5_f128_binary128_value_contract_native.receipt.json"
F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT="$F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT_DIR/madaros_v2_s5_f128_arithmetic_value_contract.receipt.json"
F128_IEEE_CLASS_HELPER_RECEIPT="$F128_IEEE_CLASS_HELPER_RECEIPT_DIR/madaros_v2_s5_f128_ieee_class_helper.receipt.json"
F128_ORDERED_COMPARE_RECEIPT="$F128_ORDERED_COMPARE_RECEIPT_DIR/madaros_v2_s5_f128_ordered_compare.receipt.json"
F128_PARAM_SLOT_LAYOUT_RECEIPT="$F128_PARAM_SLOT_LAYOUT_RECEIPT_DIR/madaros_v2_s5_f128_param_slot_layout.receipt.json"
DIAGNOSTICS_RECEIPT="$DIAGNOSTICS_RECEIPT_DIR/madaros_v2_s5_diagnostics.receipt.json"
DIFFERENTIAL_RECEIPT="$DIFFERENTIAL_RECEIPT_DIR/madaros_v2_s5_differential.receipt.json"

mkdir -p "$EFFECT_DIR" "$S5_RECEIPT_DIR" "$SRET_RECEIPT_DIR" "$SOURCE_SRET_RECEIPT_DIR" "$STACK_CALL_RECEIPT_DIR" "$IMPORTED_SRET_RECEIPT_DIR" "$METHOD_SRET_RECEIPT_DIR" "$F64_XMM0_RECEIPT_DIR" "$WIDE_INT_RECEIPT_DIR" "$WIDE_MACHINE_SLOT_RECEIPT_DIR" "$WIDE_ABI_CALL_RETURN_RECEIPT_DIR" "$GENERIC_AGG_RECEIPT_DIR" "$F128_LITERAL_PROVENANCE_RECEIPT_DIR" "$F128_BINARY128_VALUE_RECEIPT_DIR" "$F128_LITERAL_VALUE_BRIDGE_RECEIPT_DIR" "$MACHINE_SLOT_METADATA_RECEIPT_DIR" "$F128_ABI_METADATA_RECEIPT_DIR" "$F128_NATIVE_OPAQUE_STORAGE_RECEIPT_DIR" "$F128_OPAQUE_CALL_RETURN_ABI_RECEIPT_DIR" "$F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT_DIR" "$F128_BINARY128_NATIVE_ANCHOR_RECEIPT_DIR" "$F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT_DIR" "$F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT_DIR" "$F128_IEEE_CLASS_HELPER_RECEIPT_DIR" "$F128_ORDERED_COMPARE_RECEIPT_DIR" "$F128_PARAM_SLOT_LAYOUT_RECEIPT_DIR" "$DIAGNOSTICS_RECEIPT_DIR" "$DIFFERENTIAL_RECEIPT_DIR"

echo "[madaros-v2-s5-program-mir-abi] START"
echo "[madaros-v2-s5-program-mir-abi] out=$OUT_DIR"

SOUNIO_MADAROS_V2_S5_MIR_EFFECT_DIR="$EFFECT_DIR" "$EFFECT_GATE"

if [[ ! -f "$MANIFEST" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing scalar ABI manifest: $MANIFEST" >&2
  exit 1
fi

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

printf 'case_id\tprogram\texpected_exit\ts5_receipt_sha256\ts5_receipt_path\tstatus\n' >"$S5_RECEIPT_RESULTS"
while IFS=$'\t' read -r case_id program expected_exit _abi_kind _required_machine_ops; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi
  if [[ ! -f "$program" ]]; then
    echo "[madaros-v2-s5-program-mir-abi] FAIL: missing scalar witness program for $case_id: $program" >&2
    exit 1
  fi
  log_path="$S5_RECEIPT_DIR/$case_id.s5-receipt.log"
  if ! "$COMPILER" s5-receipt "$program" \
    --out-dir "$S5_RECEIPT_DIR" \
    --expected-exit "$expected_exit" \
    --case-id "$case_id" >"$log_path" 2>&1; then
    echo "[madaros-v2-s5-program-mir-abi] FAIL: canonical s5-receipt failed for $case_id" >&2
    tail -n 80 "$log_path" >&2 || true
    exit 1
  fi
  receipt_path="$S5_RECEIPT_DIR/$case_id.s5.receipt.json"
  if [[ ! -f "$receipt_path" ]]; then
    echo "[madaros-v2-s5-program-mir-abi] FAIL: missing canonical s5 receipt for $case_id" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\tok\n' \
    "$case_id" "$program" "$expected_exit" "$(portable_sha256 "$receipt_path")" "$receipt_path" \
    >>"$S5_RECEIPT_RESULTS"
done <"$MANIFEST"

python3 "$SRET_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$SRET_RECEIPT_DIR"

if [[ ! -f "$SRET_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing SRET ABI receipt: $SRET_RECEIPT" >&2
  exit 1
fi

python3 "$SOURCE_SRET_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$SOURCE_SRET_RECEIPT_DIR"

if [[ ! -f "$SOURCE_SRET_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing source SRET receipt: $SOURCE_SRET_RECEIPT" >&2
  exit 1
fi

python3 "$STACK_CALL_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$STACK_CALL_RECEIPT_DIR"

if [[ ! -f "$STACK_CALL_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing stack-call receipt: $STACK_CALL_RECEIPT" >&2
  exit 1
fi

python3 "$IMPORTED_SRET_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$IMPORTED_SRET_RECEIPT_DIR"

if [[ ! -f "$IMPORTED_SRET_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing imported SRET receipt: $IMPORTED_SRET_RECEIPT" >&2
  exit 1
fi

python3 "$METHOD_SRET_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$METHOD_SRET_RECEIPT_DIR"

if [[ ! -f "$METHOD_SRET_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing method SRET receipt: $METHOD_SRET_RECEIPT" >&2
  exit 1
fi

python3 "$F64_XMM0_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F64_XMM0_RECEIPT_DIR"

if [[ ! -f "$F64_XMM0_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f64 XMM0 receipt: $F64_XMM0_RECEIPT" >&2
  exit 1
fi

python3 "$WIDE_INT_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$WIDE_INT_RECEIPT_DIR"

if [[ ! -f "$WIDE_INT_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing wide-int receipt: $WIDE_INT_RECEIPT" >&2
  exit 1
fi

python3 "$WIDE_MACHINE_SLOT_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$WIDE_MACHINE_SLOT_RECEIPT_DIR"

if [[ ! -f "$WIDE_MACHINE_SLOT_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing wide MachineIR slot receipt: $WIDE_MACHINE_SLOT_RECEIPT" >&2
  exit 1
fi

python3 "$WIDE_ABI_CALL_RETURN_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$WIDE_ABI_CALL_RETURN_RECEIPT_DIR"

if [[ ! -f "$WIDE_ABI_CALL_RETURN_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing wide ABI call-return receipt: $WIDE_ABI_CALL_RETURN_RECEIPT" >&2
  exit 1
fi

python3 "$GENERIC_AGG_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$GENERIC_AGG_RECEIPT_DIR"

if [[ ! -f "$GENERIC_AGG_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing generic aggregate SRET receipt: $GENERIC_AGG_RECEIPT" >&2
  exit 1
fi

python3 "$F128_LITERAL_PROVENANCE_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_LITERAL_PROVENANCE_RECEIPT_DIR"

if [[ ! -f "$F128_LITERAL_PROVENANCE_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 literal provenance receipt: $F128_LITERAL_PROVENANCE_RECEIPT" >&2
  exit 1
fi

python3 "$F128_BINARY128_VALUE_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_BINARY128_VALUE_RECEIPT_DIR"

if [[ ! -f "$F128_BINARY128_VALUE_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 binary128 value receipt: $F128_BINARY128_VALUE_RECEIPT" >&2
  exit 1
fi

python3 "$F128_LITERAL_VALUE_BRIDGE_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F128_LITERAL_VALUE_BRIDGE_RECEIPT_DIR"

if [[ ! -f "$F128_LITERAL_VALUE_BRIDGE_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 literal value bridge receipt: $F128_LITERAL_VALUE_BRIDGE_RECEIPT" >&2
  exit 1
fi

python3 "$MACHINE_SLOT_METADATA_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$MACHINE_SLOT_METADATA_RECEIPT_DIR"

if [[ ! -f "$MACHINE_SLOT_METADATA_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing MachineIR slot metadata receipt: $MACHINE_SLOT_METADATA_RECEIPT" >&2
  exit 1
fi

python3 "$F128_ABI_METADATA_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_ABI_METADATA_RECEIPT_DIR"

if [[ ! -f "$F128_ABI_METADATA_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 ABI metadata receipt: $F128_ABI_METADATA_RECEIPT" >&2
  exit 1
fi

python3 "$DIAGNOSTICS_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$DIAGNOSTICS_RECEIPT_DIR"

if [[ ! -f "$DIAGNOSTICS_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing diagnostics receipt: $DIAGNOSTICS_RECEIPT" >&2
  exit 1
fi

python3 "$DIFFERENTIAL_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --reference-souc "$REFERENCE_SOUC" \
  --out-dir "$DIFFERENTIAL_RECEIPT_DIR"

if [[ ! -f "$DIFFERENTIAL_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing differential receipt: $DIFFERENTIAL_RECEIPT" >&2
  exit 1
fi

python3 "$F128_NATIVE_OPAQUE_STORAGE_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_NATIVE_OPAQUE_STORAGE_RECEIPT_DIR"

if [[ ! -f "$F128_NATIVE_OPAQUE_STORAGE_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 native opaque storage receipt: $F128_NATIVE_OPAQUE_STORAGE_RECEIPT" >&2
  exit 1
fi

python3 "$F128_OPAQUE_CALL_RETURN_ABI_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F128_OPAQUE_CALL_RETURN_ABI_RECEIPT_DIR"

if [[ ! -f "$F128_OPAQUE_CALL_RETURN_ABI_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 opaque call-return ABI receipt: $F128_OPAQUE_CALL_RETURN_ABI_RECEIPT" >&2
  exit 1
fi

python3 "$F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT_DIR"

if [[ ! -f "$F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 SRET internal arg-boundary receipt: $F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT" >&2
  exit 1
fi

python3 "$F128_BINARY128_NATIVE_ANCHOR_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_BINARY128_NATIVE_ANCHOR_RECEIPT_DIR"

if [[ ! -f "$F128_BINARY128_NATIVE_ANCHOR_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 binary128 native anchor receipt: $F128_BINARY128_NATIVE_ANCHOR_RECEIPT" >&2
  exit 1
fi

python3 "$F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT_DIR"

if [[ ! -f "$F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 binary128 value-contract native receipt: $F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT" >&2
  exit 1
fi

python3 "$F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT_DIR"

if [[ ! -f "$F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 arithmetic value-contract receipt: $F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT" >&2
  exit 1
fi

python3 "$F128_IEEE_CLASS_HELPER_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F128_IEEE_CLASS_HELPER_RECEIPT_DIR"

if [[ ! -f "$F128_IEEE_CLASS_HELPER_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 IEEE class-code helper receipt: $F128_IEEE_CLASS_HELPER_RECEIPT" >&2
  exit 1
fi

python3 "$F128_ORDERED_COMPARE_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --out-dir "$F128_ORDERED_COMPARE_RECEIPT_DIR"

if [[ ! -f "$F128_ORDERED_COMPARE_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 ordered compare receipt: $F128_ORDERED_COMPARE_RECEIPT" >&2
  exit 1
fi

python3 "$F128_PARAM_SLOT_LAYOUT_RECEIPT_TOOL" emit \
  --compiler "$COMPILER" \
  --root "$ROOT_DIR" \
  --out-dir "$F128_PARAM_SLOT_LAYOUT_RECEIPT_DIR"

if [[ ! -f "$F128_PARAM_SLOT_LAYOUT_RECEIPT" ]]; then
  echo "[madaros-v2-s5-program-mir-abi] FAIL: missing f128 parameter slot-layout receipt: $F128_PARAM_SLOT_LAYOUT_RECEIPT" >&2
  exit 1
fi

python3 - "$EFFECT_DIR" "$S5_RECEIPT_RESULTS" "$SRET_RECEIPT" "$SOURCE_SRET_RECEIPT" "$STACK_CALL_RECEIPT" "$IMPORTED_SRET_RECEIPT" "$METHOD_SRET_RECEIPT" "$F64_XMM0_RECEIPT" "$WIDE_INT_RECEIPT" "$WIDE_MACHINE_SLOT_RECEIPT" "$WIDE_ABI_CALL_RETURN_RECEIPT" "$GENERIC_AGG_RECEIPT" "$F128_LITERAL_PROVENANCE_RECEIPT" "$F128_BINARY128_VALUE_RECEIPT" "$F128_LITERAL_VALUE_BRIDGE_RECEIPT" "$MACHINE_SLOT_METADATA_RECEIPT" "$F128_ABI_METADATA_RECEIPT" "$F128_NATIVE_OPAQUE_STORAGE_RECEIPT" "$F128_OPAQUE_CALL_RETURN_ABI_RECEIPT" "$F128_SRET_INTERNAL_ARG_BOUNDARY_RECEIPT" "$F128_BINARY128_NATIVE_ANCHOR_RECEIPT" "$F128_BINARY128_VALUE_CONTRACT_NATIVE_RECEIPT" "$F128_ARITHMETIC_VALUE_CONTRACT_RECEIPT" "$F128_IEEE_CLASS_HELPER_RECEIPT" "$F128_ORDERED_COMPARE_RECEIPT" "$F128_PARAM_SLOT_LAYOUT_RECEIPT" "$DIAGNOSTICS_RECEIPT" "$DIFFERENTIAL_RECEIPT" "$MODULE" "$RECEIPT" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_function_count(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Merged IR:\s*(\d+)\s+functions", text)
    if not match:
        raise SystemExit(f"cannot parse Merged IR function count from {log_path}")
    return int(match.group(1))


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    parsed = json.loads(first)
    second = stable_json(parsed)
    if first != second:
        raise SystemExit("program MIR/ABI canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


effect_dir = Path(sys.argv[1])
source_receipts_path = Path(sys.argv[2])
sret_receipt_path = Path(sys.argv[3])
source_sret_receipt_path = Path(sys.argv[4])
stack_call_receipt_path = Path(sys.argv[5])
imported_sret_receipt_path = Path(sys.argv[6])
method_sret_receipt_path = Path(sys.argv[7])
f64_xmm0_receipt_path = Path(sys.argv[8])
wide_int_receipt_path = Path(sys.argv[9])
wide_machine_slot_receipt_path = Path(sys.argv[10])
wide_abi_call_return_receipt_path = Path(sys.argv[11])
generic_agg_receipt_path = Path(sys.argv[12])
f128_literal_provenance_receipt_path = Path(sys.argv[13])
f128_binary128_value_receipt_path = Path(sys.argv[14])
f128_literal_value_bridge_receipt_path = Path(sys.argv[15])
machine_slot_metadata_receipt_path = Path(sys.argv[16])
f128_abi_metadata_receipt_path = Path(sys.argv[17])
f128_native_opaque_storage_receipt_path = Path(sys.argv[18])
f128_opaque_call_return_abi_receipt_path = Path(sys.argv[19])
f128_sret_internal_arg_boundary_receipt_path = Path(sys.argv[20])
f128_binary128_native_anchor_receipt_path = Path(sys.argv[21])
f128_binary128_value_contract_native_receipt_path = Path(sys.argv[22])
f128_arithmetic_value_contract_receipt_path = Path(sys.argv[23])
f128_ieee_class_helper_receipt_path = Path(sys.argv[24])
f128_ordered_compare_receipt_path = Path(sys.argv[25])
f128_param_slot_layout_receipt_path = Path(sys.argv[26])
diagnostics_receipt_path = Path(sys.argv[27])
differential_receipt_path = Path(sys.argv[28])
module_path = Path(sys.argv[29])
receipt_path = Path(sys.argv[30])

effect_receipt_path = effect_dir / "madaros_v2_s5_mir_effect.receipt.json"
effect_module_path = effect_dir / "madaros_v2_s5_mir_effect.module.json"
effect_receipt = load_json(effect_receipt_path)
effect_module = load_json(effect_module_path)
sret_receipt = load_json(sret_receipt_path)
source_sret_receipt = load_json(source_sret_receipt_path)
stack_call_receipt = load_json(stack_call_receipt_path)
imported_sret_receipt = load_json(imported_sret_receipt_path)
method_sret_receipt = load_json(method_sret_receipt_path)
f64_xmm0_receipt = load_json(f64_xmm0_receipt_path)
wide_int_receipt = load_json(wide_int_receipt_path)
wide_machine_slot_receipt = load_json(wide_machine_slot_receipt_path)
wide_abi_call_return_receipt = load_json(wide_abi_call_return_receipt_path)
generic_agg_receipt = load_json(generic_agg_receipt_path)
f128_literal_provenance_receipt = load_json(f128_literal_provenance_receipt_path)
f128_binary128_value_receipt = load_json(f128_binary128_value_receipt_path)
f128_literal_value_bridge_receipt = load_json(f128_literal_value_bridge_receipt_path)
machine_slot_metadata_receipt = load_json(machine_slot_metadata_receipt_path)
f128_abi_metadata_receipt = load_json(f128_abi_metadata_receipt_path)
f128_native_opaque_storage_receipt = load_json(f128_native_opaque_storage_receipt_path)
f128_opaque_call_return_abi_receipt = load_json(f128_opaque_call_return_abi_receipt_path)
f128_sret_internal_arg_boundary_receipt = load_json(f128_sret_internal_arg_boundary_receipt_path)
f128_binary128_native_anchor_receipt = load_json(f128_binary128_native_anchor_receipt_path)
f128_binary128_value_contract_native_receipt = load_json(f128_binary128_value_contract_native_receipt_path)
f128_arithmetic_value_contract_receipt = load_json(f128_arithmetic_value_contract_receipt_path)
f128_ieee_class_helper_receipt = load_json(f128_ieee_class_helper_receipt_path)
f128_ordered_compare_receipt = load_json(f128_ordered_compare_receipt_path)
f128_param_slot_layout_receipt = load_json(f128_param_slot_layout_receipt_path)
diagnostics_receipt = load_json(diagnostics_receipt_path)
differential_receipt = load_json(differential_receipt_path)

if effect_receipt.get("s4_applied_extraction_consumed") is not True:
    raise SystemExit("program MIR/ABI gate requires MIR-effect receipt to consume S4 applied extraction")
if effect_receipt.get("input_applied_extraction_contract") != "madaros.v2.s4.applied_extraction/0.1":
    raise SystemExit("program MIR/ABI gate requires S4 applied-extraction input contract")
if not effect_receipt.get("input_applied_extraction_sha256"):
    raise SystemExit("program MIR/ABI gate requires S4 applied-extraction hash")
if effect_module.get("s4_applied_extraction_consumed") is not True:
    raise SystemExit("program MIR/ABI gate requires MIR-effect module to carry S4 applied extraction")
if effect_module.get("input_applied_extraction_sha256") != effect_receipt.get("input_applied_extraction_sha256"):
    raise SystemExit("program MIR/ABI gate MIR-effect applied-extraction hash mismatch")
for effect in effect_module.get("mir_effects", []):
    if effect.get("input_applied_extraction_sha256") != effect_receipt["input_applied_extraction_sha256"]:
        raise SystemExit(f"program MIR/ABI gate effect applied-extraction hash mismatch: {effect.get('rewrite_id')}")
    if not effect.get("source_applied_effect_sha256"):
        raise SystemExit(f"program MIR/ABI gate effect missing source applied-effect hash: {effect.get('rewrite_id')}")
    if not effect.get("post_apply_s5_input_hlir_sha256") or not effect.get("post_apply_s5_input_egraph_sha256"):
        raise SystemExit(f"program MIR/ABI gate effect missing post-apply S5 input hashes: {effect.get('rewrite_id')}")

if sret_receipt.get("schema") != "madaros.v2.s5.sret_abi_receipt/0.1":
    raise SystemExit("bad S5 SRET ABI receipt schema")
if sret_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing SRET ABI receipt")
if sret_receipt.get("s5_sret_machine_module_abi_discriminator_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed SRET ABI discriminator")
if sret_receipt.get("positive", {}).get("actual_exit") != 14:
    raise SystemExit("SRET ABI receipt positive witness must return 14")
if sret_receipt.get("negative_plaincall", {}).get("actual_exit") == 14:
    raise SystemExit("SRET ABI receipt negative discriminator must not return 14")

if source_sret_receipt.get("schema") != "madaros.v2.s5.source_sret_receipt/0.1":
    raise SystemExit("bad S5 source SRET receipt schema")
if source_sret_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing source SRET receipt")
if source_sret_receipt.get("s5_source_sret_local_one_arg_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed source local one-arg SRET receipt")
if source_sret_receipt.get("s5_source_sret_local_register_multi_arg_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed source local register multi-arg SRET receipt")
if source_sret_receipt.get("s5_source_sret_local_stack_arg_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed source local stack-arg SRET receipt")
if source_sret_receipt.get("source_frontend_lowers_local_aggregate_return_to_IrCallSret") is not True:
    raise SystemExit("program MIR/ABI gate requires source front-end SRET lowering evidence")
if source_sret_receipt.get("source_frontend_lowers_local_register_multi_arg_aggregate_return_to_IrCallSret") is not True:
    raise SystemExit("program MIR/ABI gate requires source register multi-arg SRET lowering evidence")
if source_sret_receipt.get("source_frontend_lowers_local_stack_arg_aggregate_return_to_IrCallSret") is not True:
    raise SystemExit("program MIR/ABI gate requires source stack-arg SRET lowering evidence")
source_sret_cases = {row.get("case_id"): row for row in source_sret_receipt.get("cases", [])}
one_arg_case = source_sret_cases.get("source_sret_local_i64_triple_return_14")
multi_arg_case = source_sret_cases.get("source_sret_local_register_multi_arg_return_43")
stack_one_case = source_sret_cases.get("source_sret_local_stack_one_arg_return_49")
stack_two_case = source_sret_cases.get("source_sret_local_stack_two_arg_return_57")
if not one_arg_case or not multi_arg_case or not stack_one_case or not stack_two_case:
    raise SystemExit("source SRET receipt must contain one-arg, register multi-arg, and stack-arg cases")
if one_arg_case.get("actual_exit") != 14:
    raise SystemExit("source SRET one-arg witness must return 14")
if one_arg_case.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1]:
    raise SystemExit("source SRET one-arg receipt must pass hidden dest then explicit arg")
if one_arg_case.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [1, 0]:
    raise SystemExit("source SRET one-arg receipt must prove hidden dest from slot1 and explicit arg from slot0")
if multi_arg_case.get("actual_exit") != 43:
    raise SystemExit("source SRET register multi-arg witness must return 43")
if multi_arg_case.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("source SRET register multi-arg receipt must pass hidden dest plus five register args")
if multi_arg_case.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [5, 0, 1, 2, 3, 4]:
    raise SystemExit("source SRET register multi-arg receipt must prove hidden dest slot5 then explicit arg slots 0..4")
if stack_one_case.get("actual_exit") != 49:
    raise SystemExit("source SRET one-stack-arg witness must return 49")
if stack_one_case.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("source SRET one-stack-arg receipt must pass hidden dest plus five register args")
if stack_one_case.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [6, 0, 1, 2, 3, 4]:
    raise SystemExit("source SRET one-stack-arg receipt must prove hidden dest slot6 then explicit arg slots 0..4")
if stack_one_case.get("machine_shape", {}).get("main_stack_arg_push_indices") != [6]:
    raise SystemExit("source SRET one-stack-arg receipt must push explicit arg5 as machine arg6")
if stack_one_case.get("machine_shape", {}).get("main_stack_arg_push_source_stack_slots") != [5]:
    raise SystemExit("source SRET one-stack-arg receipt must load stack arg from explicit slot5")
if stack_one_case.get("machine_shape", {}).get("main_stack_adjust_immediates") != [-8, 16]:
    raise SystemExit("source SRET one-stack-arg receipt must record padding -8 and cleanup 16")
if stack_two_case.get("actual_exit") != 57:
    raise SystemExit("source SRET two-stack-arg witness must return 57")
if stack_two_case.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("source SRET two-stack-arg receipt must pass hidden dest plus five register args")
if stack_two_case.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [7, 0, 1, 2, 3, 4]:
    raise SystemExit("source SRET two-stack-arg receipt must prove hidden dest slot7 then explicit arg slots 0..4")
if stack_two_case.get("machine_shape", {}).get("main_stack_arg_push_indices") != [7, 6]:
    raise SystemExit("source SRET two-stack-arg receipt must push explicit args in reverse stack order")
if stack_two_case.get("machine_shape", {}).get("main_stack_arg_push_source_stack_slots") != [6, 5]:
    raise SystemExit("source SRET two-stack-arg receipt must load stack args from explicit slots 6 then 5")
if stack_two_case.get("machine_shape", {}).get("main_stack_adjust_immediates") != [16]:
    raise SystemExit("source SRET two-stack-arg receipt must record cleanup 16 without padding")

if stack_call_receipt.get("schema") != "madaros.v2.s5.stack_call_receipt/0.1":
    raise SystemExit("bad S5 normal stack-call receipt schema")
if stack_call_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing normal stack-call receipt")
if stack_call_receipt.get("s5_normal_call_stack_args_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed normal stack-call receipt")
stack_call_cases = {row.get("case_id"): row for row in stack_call_receipt.get("cases", [])}
normal_stack_one = stack_call_cases.get("normal_call_stack_one_arg_return_28")
normal_stack_two = stack_call_cases.get("normal_call_stack_two_arg_return_36")
if not normal_stack_one or not normal_stack_two:
    raise SystemExit("normal stack-call receipt must contain one-stack and two-stack cases")
if normal_stack_one.get("actual_exit") != 28:
    raise SystemExit("normal one-stack-call witness must return 28")
if normal_stack_one.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("normal one-stack-call receipt must pass first six args in registers")
if normal_stack_one.get("machine_shape", {}).get("main_stack_arg_push_indices") != [6]:
    raise SystemExit("normal one-stack-call receipt must push explicit arg6")
if normal_stack_one.get("machine_shape", {}).get("main_stack_arg_push_source_stack_slots") != [6]:
    raise SystemExit("normal one-stack-call receipt must load stack arg from slot6")
if normal_stack_one.get("machine_shape", {}).get("main_stack_adjust_immediates") != [-8, 16]:
    raise SystemExit("normal one-stack-call receipt must record padding -8 and cleanup 16")
if normal_stack_two.get("actual_exit") != 36:
    raise SystemExit("normal two-stack-call witness must return 36")
if normal_stack_two.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("normal two-stack-call receipt must pass first six args in registers")
if normal_stack_two.get("machine_shape", {}).get("main_stack_arg_push_indices") != [7, 6]:
    raise SystemExit("normal two-stack-call receipt must push stack args in reverse order")
if normal_stack_two.get("machine_shape", {}).get("main_stack_arg_push_source_stack_slots") != [7, 6]:
    raise SystemExit("normal two-stack-call receipt must load stack args from slots 7 then 6")
if normal_stack_two.get("machine_shape", {}).get("main_stack_adjust_immediates") != [16]:
    raise SystemExit("normal two-stack-call receipt must record cleanup 16 without padding")

if imported_sret_receipt.get("schema") != "madaros.v2.s5.imported_sret_receipt/0.1":
    raise SystemExit("bad S5 imported SRET receipt schema")
if imported_sret_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing imported SRET receipt")
if imported_sret_receipt.get("s5_imported_sret_module_boundary_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed imported SRET module-boundary receipt")
if imported_sret_receipt.get("compiler_machine_module_exported_for_imported_path") is not True:
    raise SystemExit("program MIR/ABI gate requires MachineModule JSON on imported native-v2 path")
imported_sret_cases = {row.get("case_id"): row for row in imported_sret_receipt.get("cases", [])}
imported_one = imported_sret_cases.get("imported_sret_one_arg_return_29")
imported_reg = imported_sret_cases.get("imported_sret_register_multi_arg_return_43")
imported_stack = imported_sret_cases.get("imported_sret_stack_two_arg_return_57")
if not imported_one or not imported_reg or not imported_stack:
    raise SystemExit("imported SRET receipt must contain one-arg, register multi-arg, and stack-arg cases")
if imported_one.get("actual_exit") != 29:
    raise SystemExit("imported SRET one-arg witness must return 29")
if imported_one.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1]:
    raise SystemExit("imported SRET one-arg receipt must pass hidden dest then explicit arg")
if imported_one.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [1, 0]:
    raise SystemExit("imported SRET one-arg receipt must prove hidden dest from slot1 and explicit arg from slot0")
if imported_one.get("machine_shape", {}).get("main_field_load_indices") != [0, 1, 2]:
    raise SystemExit("imported SRET one-arg receipt must prove imported aggregate field indices 0,1,2")
if imported_reg.get("actual_exit") != 43:
    raise SystemExit("imported SRET register multi-arg witness must return 43")
if imported_reg.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("imported SRET register multi-arg receipt must pass hidden dest plus five register args")
if imported_reg.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [5, 0, 1, 2, 3, 4]:
    raise SystemExit("imported SRET register multi-arg receipt must prove hidden dest slot5 then explicit arg slots 0..4")
if imported_reg.get("machine_shape", {}).get("main_field_load_indices") != [0, 1, 2]:
    raise SystemExit("imported SRET register multi-arg receipt must prove imported aggregate field indices 0,1,2")
if imported_stack.get("actual_exit") != 57:
    raise SystemExit("imported SRET stack-arg witness must return 57")
if imported_stack.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("imported SRET stack receipt must pass hidden dest plus five register args")
if imported_stack.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [7, 0, 1, 2, 3, 4]:
    raise SystemExit("imported SRET stack receipt must prove hidden dest slot7 then explicit arg slots 0..4")
if imported_stack.get("machine_shape", {}).get("main_stack_arg_push_indices") != [7, 6]:
    raise SystemExit("imported SRET stack receipt must push explicit stack args in reverse order")
if imported_stack.get("machine_shape", {}).get("main_stack_arg_push_source_stack_slots") != [6, 5]:
    raise SystemExit("imported SRET stack receipt must load stack args from explicit slots 6 then 5")
if imported_stack.get("machine_shape", {}).get("main_stack_adjust_immediates") != [16]:
    raise SystemExit("imported SRET stack receipt must record cleanup 16 without padding")
if imported_stack.get("machine_shape", {}).get("main_field_load_indices") != [0, 1, 2]:
    raise SystemExit("imported SRET stack receipt must prove imported aggregate field indices 0,1,2")

if method_sret_receipt.get("schema") != "madaros.v2.s5.method_sret_receipt/0.1":
    raise SystemExit("bad S5 method SRET receipt schema")
if method_sret_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing method SRET receipt")
if method_sret_receipt.get("s5_method_sret_receiver_only_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed receiver-only method SRET receipt")
if method_sret_receipt.get("s5_method_sret_receiver_register_args_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed receiver+register-args method SRET receipt")
if method_sret_receipt.get("s5_method_sret_receiver_stack_args_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed receiver+stack-args method SRET receipt")
if method_sret_receipt.get("source_frontend_lowers_method_aggregate_return_to_IrCallSret") is not True:
    raise SystemExit("program MIR/ABI gate requires method aggregate return lowering evidence")
method_sret_cases = {row.get("case_id"): row for row in method_sret_receipt.get("cases", [])}
method_recv = method_sret_cases.get("method_sret_receiver_only_return_24")
method_reg = method_sret_cases.get("method_sret_receiver_register_args_return_43")
method_stack = method_sret_cases.get("method_sret_receiver_stack_args_return_57")
if not method_recv or not method_reg or not method_stack:
    raise SystemExit("method SRET receipt must contain receiver-only, register-arg, and stack-arg cases")
if method_recv.get("actual_exit") != 24:
    raise SystemExit("method SRET receiver-only witness must return 24")
if method_recv.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1]:
    raise SystemExit("method SRET receiver-only receipt must pass hidden dest then receiver")
if method_recv.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [2, 0]:
    raise SystemExit("method SRET receiver-only receipt must prove hidden dest slot2 then receiver slot0")
if method_recv.get("machine_shape", {}).get("main_field_load_indices") != [0, 1, 2]:
    raise SystemExit("method SRET receiver-only receipt must prove aggregate field indices 0,1,2")
if method_recv.get("machine_shape", {}).get("method_source_is_sret") != 1:
    raise SystemExit("method SRET receiver-only callee must be source_is_sret")
if method_reg.get("actual_exit") != 43:
    raise SystemExit("method SRET register-arg witness must return 43")
if method_reg.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("method SRET register receipt must pass hidden dest, receiver, and register args")
if method_reg.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [6, 0, 2, 3, 4, 5]:
    raise SystemExit("method SRET register receipt must prove hidden dest slot6, receiver slot0, explicit slots 2..5")
if method_reg.get("machine_shape", {}).get("main_field_load_indices") != [0, 1, 2]:
    raise SystemExit("method SRET register receipt must prove aggregate field indices 0,1,2")
if method_reg.get("machine_shape", {}).get("method_source_is_sret") != 1:
    raise SystemExit("method SRET register callee must be source_is_sret")
if method_stack.get("actual_exit") != 57:
    raise SystemExit("method SRET stack-arg witness must return 57")
if method_stack.get("machine_shape", {}).get("main_arg_move_indices") != [0, 1, 2, 3, 4, 5]:
    raise SystemExit("method SRET stack receipt must pass hidden dest, receiver, and first register args")
if method_stack.get("machine_shape", {}).get("main_arg_move_source_stack_slots") != [8, 0, 2, 3, 4, 5]:
    raise SystemExit("method SRET stack receipt must prove hidden dest slot8, receiver slot0, explicit slots 2..5")
if method_stack.get("machine_shape", {}).get("main_stack_arg_push_indices") != [7, 6]:
    raise SystemExit("method SRET stack receipt must push explicit stack args in reverse order")
if method_stack.get("machine_shape", {}).get("main_stack_arg_push_source_stack_slots") != [7, 6]:
    raise SystemExit("method SRET stack receipt must load stack args from explicit slots 7 then 6")
if method_stack.get("machine_shape", {}).get("main_stack_adjust_immediates") != [16]:
    raise SystemExit("method SRET stack receipt must record cleanup 16 without padding")
if method_stack.get("machine_shape", {}).get("main_field_load_indices") != [0, 1, 2]:
    raise SystemExit("method SRET stack receipt must prove aggregate field indices 0,1,2")
if method_stack.get("machine_shape", {}).get("method_source_is_sret") != 1:
    raise SystemExit("method SRET stack callee must be source_is_sret")

if f64_xmm0_receipt.get("schema") != "madaros.v2.s5.f64_xmm0_receipt/0.1":
    raise SystemExit("bad S5 f64 XMM0 receipt schema")
if f64_xmm0_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f64 XMM0 receipt")
if f64_xmm0_receipt.get("stage_contract_level") != "S5_F64_XMM0_CALL_RETURN_PROMOTED":
    raise SystemExit("f64 XMM0 receipt must declare promoted stage contract")
if f64_xmm0_receipt.get("s5_f64_xmm0_call_return_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed f64 XMM0 call-return receipt")
if f64_xmm0_receipt.get("f64_xmm0_promoted") is not True:
    raise SystemExit("program MIR/ABI gate requires f64 XMM0 promotion flag")
if f64_xmm0_receipt.get("case_count") != 7:
    raise SystemExit("program MIR/ABI gate requires exact seven f64 XMM0 cases")
required_f64_flags = [
    "source_frontend_dispatches_print_to_print_f64",
    "source_frontend_dispatches_println_f64_to_print_f64",
    "source_frontend_tracks_let_bound_f64_identifiers",
    "ir_lowers_f64_literals_to_IrLoadFloat",
    "native_v2_lowers_f64_to_i64_cast",
    "native_v2_lowers_fractional_f64_binops",
    "native_v2_materializes_print_f64_fraction_scale_without_rodata_relocation",
    "native_v2_bridges_print_f64_arg0_to_xmm0",
    "compiler_machine_module_exported",
    "real_program_mir_emitted",
    "real_abi_layout_emitted",
]
for field in required_f64_flags:
    if f64_xmm0_receipt.get(field) is not True:
        raise SystemExit(f"f64 XMM0 receipt missing required true flag: {field}")
f64_cases = {row.get("case_id"): row for row in f64_xmm0_receipt.get("cases", [])}
required_f64_cases = {
    "f64_cast_literal_to_i64_return_4": {"exit": 4, "stdout": ""},
    "f64_fractional_binop_cast_return_50": {"exit": 50, "stdout": ""},
    "f64_return_compare_exit_45": {"exit": 45, "stdout": ""},
    "f64_mixed_args_return_compare_exit_55": {"exit": 55, "stdout": ""},
    "f64_println_call_stdout_4_5": {"exit": 0, "stdout": "4.500000\n"},
    "f64_print_call_stdout_4_5": {"exit": 0, "stdout": "4.500000"},
    "f64_let_bound_println_stdout_4_5": {"exit": 0, "stdout": "4.500000\n"},
}
if set(f64_cases) != set(required_f64_cases):
    raise SystemExit(f"f64 XMM0 receipt cases mismatch: {sorted(f64_cases)}")
for case_id, expected in required_f64_cases.items():
    row = f64_cases[case_id]
    if row.get("actual_exit") != expected["exit"]:
        raise SystemExit(f"{case_id} expected exit {expected['exit']}, got {row.get('actual_exit')}")
    if row.get("actual_stdout") != expected["stdout"]:
        raise SystemExit(f"{case_id} expected stdout {expected['stdout']!r}, got {row.get('actual_stdout')!r}")
    shape = row.get("machine_shape", {})
    if "main" not in shape.get("function_names", []):
        raise SystemExit(f"{case_id} must include main in MachineModule functions")
    if not row.get("machine_module_json_sha256"):
        raise SystemExit(f"{case_id} missing MachineModule sha256")
    if not row.get("elf_sha256"):
        raise SystemExit(f"{case_id} missing ELF sha256")

if wide_int_receipt.get("schema") != "madaros.v2.s5.wide_int_receipt/0.1":
    raise SystemExit("bad S5 wide-int receipt schema")
if wide_int_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing wide-int receipt")
if wide_int_receipt.get("stage_contract_level") != "S5_WIDE_INT_I128_I256_PROMOTED_NOT_F128":
    raise SystemExit("wide-int receipt must declare promoted i128/i256 but not f128 stage contract")
if wide_int_receipt.get("case_count") != 25:
    raise SystemExit("program MIR/ABI gate requires exact 25 wide-int cases")
if wide_int_receipt.get("check_ok_case_count") != 7:
    raise SystemExit("wide-int receipt must contain seven positive checker cases")
if wide_int_receipt.get("check_reject_case_count") != 2:
    raise SystemExit("wide-int receipt must contain two negative checker cases")
if wide_int_receipt.get("source_native_case_count") != 6:
    raise SystemExit("wide-int receipt must contain six source native-v2 cases")
if wide_int_receipt.get("native_emit_case_count") != 10:
    raise SystemExit("wide-int receipt must contain ten hand-built IR native-v2 cases")
required_wide_flags = [
    "s5_wide_int_i128_i256_complete",
    "wide_i128_i256_promoted",
    "wide_u128_u256_promoted",
    "source_level_wide_arithmetic_promoted",
    "native_v2_wide_limb_backend_promoted",
    "wide_type_identity_and_safety_promoted",
    "numeric_tower_width_receipts_partial",
    "compiler_machine_module_exported",
    "real_program_mir_emitted",
    "real_abi_layout_emitted",
]
for field in required_wide_flags:
    if wide_int_receipt.get(field) is not True:
        raise SystemExit(f"wide-int receipt missing required true flag: {field}")
if wide_int_receipt.get("f128_promoted") is not False:
    raise SystemExit("wide-int receipt must not promote f128")
wide_cases = {row.get("case_id"): row for row in wide_int_receipt.get("cases", [])}
required_wide_exits = {
    "source_i128_mul_gt": 42,
    "source_i256_mul_gt": 42,
    "source_u128_mul_add_gt": 42,
    "source_u256_mul_add_ne": 42,
    "source_i128_sub_eq_zero": 42,
    "source_i256_add_eq": 7,
    "irwide_add4_i256_carry_chain": 1,
    "irwide_sub_i128_borrow_chain": 1,
    "irwide_mul_i128_cross_limb": 1,
    "irwide_shr_limb_i128": 1,
    "irwide_div_single_limb_i128": 5,
    "irwide_mod_single_limb_i128": 3,
    "irwide_cmp_high_limb_i128": 1,
    "irwide_shr_unaligned_i128": 16,
    "irwide_divfull_multilimb_i128": 5,
    "irwide_modfull_multilimb_i128": 193,
}
required_wide_checks = {
    "i128_type_identity",
    "i256_type_identity",
    "u128_type_identity",
    "u256_type_identity",
    "wide_explicit_casts",
    "i128_param_return_check",
    "i256_param_return_check",
    "reject_i128_from_i256",
    "reject_u128_from_i128",
}
if not (set(required_wide_exits) | required_wide_checks).issubset(set(wide_cases)):
    raise SystemExit(f"wide-int receipt cases missing required ids: {sorted((set(required_wide_exits) | required_wide_checks) - set(wide_cases))}")
for case_id, expected_exit in required_wide_exits.items():
    row = wide_cases[case_id]
    if row.get("actual_exit") != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {row.get('actual_exit')}")
    if not row.get("elf_sha256"):
        raise SystemExit(f"{case_id} missing ELF sha256")
for case_id in required_wide_checks:
    row = wide_cases[case_id]
    if not row.get("check_log_sha256"):
        raise SystemExit(f"{case_id} missing checker log sha256")

if wide_machine_slot_receipt.get("schema") != "madaros.v2.s5.wide_machine_slot_metadata_receipt/0.1":
    raise SystemExit("bad S5 wide MachineIR slot metadata receipt schema")
if wide_machine_slot_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing wide MachineIR slot metadata receipt")
if wide_machine_slot_receipt.get("stage_contract_level") != "S5_WIDE_INT_MACHINE_SLOT_METADATA_PROMOTED_NOT_F128":
    raise SystemExit("wide MachineIR slot receipt must declare promoted wide slots but not f128")
if wide_machine_slot_receipt.get("case_count") != 2:
    raise SystemExit("wide MachineIR slot receipt must contain exact two cases")
for field in [
    "wide_machine_slot_metadata_complete",
    "wide_i256_u256_machine_slots_promoted",
    "wide_slot_width_words_exported",
    "machine_module_supported_for_wide_ints",
]:
    if wide_machine_slot_receipt.get(field) is not True:
        raise SystemExit(f"wide MachineIR slot receipt missing required true flag: {field}")
for field in [
    "f128_execution_slot_emitted",
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if wide_machine_slot_receipt.get(field) is not False:
        raise SystemExit(f"wide MachineIR slot receipt must not overclaim {field}")
wide_slot_cases = {row.get("case_id"): row for row in wide_machine_slot_receipt.get("cases", [])}
required_wide_slot_cases = {
    "i256_add_eq_machine_slots": 7,
    "u256_mul_add_ne_machine_slots": 42,
}
if set(wide_slot_cases) != set(required_wide_slot_cases):
    raise SystemExit(f"wide MachineIR slot receipt cases mismatch: {sorted(wide_slot_cases)}")
for case_id, expected_exit in required_wide_slot_cases.items():
    row = wide_slot_cases[case_id]
    if row.get("actual_exit") != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {row.get('actual_exit')}")
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} MachineModule must be supported")
    if row.get("wide_slot_kind") != 4:
        raise SystemExit(f"{case_id} must use wide slot kind 4")
    if row.get("wide_slot_width_words") != 4:
        raise SystemExit(f"{case_id} must use width_words=4")
    if row.get("wide_slot_row_count", 0) < 4:
        raise SystemExit(f"{case_id} must record at least four wide limb rows")
    if 3 in row.get("slot_kinds_seen", []):
        raise SystemExit(f"{case_id} must not emit f128 slot kind 3")

if wide_abi_call_return_receipt.get("schema") != "madaros.v2.s5.wide_abi_call_return_receipt/0.1":
    raise SystemExit("bad S5 wide ABI call-return receipt schema")
if wide_abi_call_return_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing wide ABI call-return receipt")
if wide_abi_call_return_receipt.get("stage_contract_level") != "S5_WIDE_I256_U256_LOCAL_AND_IMPORTED_ABI_CALL_RETURN_PROMOTED_NOT_F128":
    raise SystemExit("wide ABI call-return receipt must declare local+imported i256/u256 promotion but not f128")
if wide_abi_call_return_receipt.get("case_count") != 9:
    raise SystemExit("wide ABI call-return receipt must contain exact nine cases")
if wide_abi_call_return_receipt.get("i256_case_count") != 5:
    raise SystemExit("wide ABI call-return receipt must contain five i256 cases")
if wide_abi_call_return_receipt.get("u256_case_count") != 4:
    raise SystemExit("wide ABI call-return receipt must contain four u256 cases")
if wide_abi_call_return_receipt.get("two_wide_arg_case_count") != 4:
    raise SystemExit("wide ABI call-return receipt must contain four two-wide-arg cases")
if wide_abi_call_return_receipt.get("imported_module_case_count") != 4:
    raise SystemExit("wide ABI call-return receipt must contain four imported module cases")
if wide_abi_call_return_receipt.get("public_native_imported_case_count") != 4:
    raise SystemExit("wide ABI call-return receipt must check four imported public native cases")
for field in [
    "s5_wide_i256_u256_local_abi_call_return_complete",
    "s5_wide_i256_u256_imported_abi_call_return_complete",
    "wide_i256_u256_local_abi_call_return_promoted",
    "wide_i256_u256_imported_abi_call_return_promoted",
    "imported_module_wide_abi_promoted",
    "public_native_imported_route_checked",
    "public_native_imported_route_uses_full_modular_native_v2",
    "stale_compact_modular_ir_table_path_blocked",
    "wide_return_uses_sret",
    "wide_arg_limb_expansion_promoted",
    "wide_two_arg_order_preserved",
    "wide_second_arg_preserved",
    "wide_callee_arithmetic_return_promoted",
    "compiler_machine_module_exported",
    "real_program_mir_emitted",
    "real_abi_layout_emitted",
]:
    if wide_abi_call_return_receipt.get(field) is not True:
        raise SystemExit(f"wide ABI call-return receipt missing required true flag: {field}")
for field in [
    "legacy_fallback_for_wide_abi",
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if wide_abi_call_return_receipt.get(field) is not False:
        raise SystemExit(f"wide ABI call-return receipt must not overclaim {field}")
if wide_abi_call_return_receipt.get("wide_machine_slot_kind") != 4:
    raise SystemExit("wide ABI call-return receipt must use wide slot kind 4")
if wide_abi_call_return_receipt.get("wide_machine_slot_width_words") != 4:
    raise SystemExit("wide ABI call-return receipt must use width_words=4")
wide_abi_cases = {row.get("case_id"): row for row in wide_abi_call_return_receipt.get("cases", [])}
required_wide_abi_cases = {
    "i256_return_only_sret_return_41": 41,
    "i256_arg_return_sret_return_31": 31,
    "u256_first_of_two_wide_args_return_43": 43,
    "u256_second_of_two_wide_args_return_47": 47,
    "u256_two_arg_add_return_37": 37,
    "imported_i256_return_only_sret_return_52": 52,
    "imported_i256_arg_return_sret_return_54": 54,
    "imported_u256_second_of_two_wide_args_return_53": 53,
    "imported_i256_mixed_param_order_return_55": 55,
}
if set(wide_abi_cases) != set(required_wide_abi_cases):
    raise SystemExit(f"wide ABI call-return receipt cases mismatch: {sorted(wide_abi_cases)}")
for case_id, expected_exit in required_wide_abi_cases.items():
    row = wide_abi_cases[case_id]
    if row.get("actual_exit") != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {row.get('actual_exit')}")
    if row.get("actual_exit") == row.get("fake_scalar_exit"):
        raise SystemExit(f"{case_id} matched fake scalar/truncated discriminator")
    if not row.get("elf_sha256") or not row.get("machine_module_json_sha256"):
        raise SystemExit(f"{case_id} missing ELF or MachineModule sha256")
    if row.get("imported_module") is True:
        if row.get("public_native_compile_checked") is not True:
            raise SystemExit(f"{case_id} must check the public native compile route")
        if row.get("public_native_compile_rc") != 0:
            raise SystemExit(f"{case_id} public native compile rc must be 0")
        if row.get("public_native_actual_exit") != expected_exit:
            raise SystemExit(f"{case_id} public native expected exit {expected_exit}, got {row.get('public_native_actual_exit')}")
        if not row.get("public_native_elf_sha256"):
            raise SystemExit(f"{case_id} missing public native ELF sha256")
    shape = row.get("machine_shape", {})
    if shape.get("callee_source_is_sret") != 1:
        raise SystemExit(f"{case_id} callee must be SRET lowered")
    if shape.get("wide_slot_kind_seen") != [4]:
        raise SystemExit(f"{case_id} must record only wide slot kind 4 for wide slots")
    if shape.get("wide_slot_width_words_seen") != [4]:
        raise SystemExit(f"{case_id} must record width_words=4 for wide slots")
    if shape.get("wide_slot_row_count", 0) < 4:
        raise SystemExit(f"{case_id} must record at least one four-limb wide value")
second_case = wide_abi_cases["u256_second_of_two_wide_args_return_47"]
if second_case.get("machine_shape", {}).get("callee_source_param_count") != 8:
    raise SystemExit("u256 second-arg receipt must prove two wide args expand to eight callee params")
if second_case.get("trace_matched") is not True and second_case.get("trace_satisfied_by_machine_module") is not True:
    raise SystemExit("u256 second-arg receipt must include matched lowerer trace or MachineModule param_count=8 evidence")
imported_second_case = wide_abi_cases["imported_u256_second_of_two_wide_args_return_53"]
if imported_second_case.get("machine_shape", {}).get("callee_source_param_count") != 8:
    raise SystemExit("imported u256 second-arg receipt must prove two wide args expand to eight callee params")
imported_mixed_case = wide_abi_cases["imported_i256_mixed_param_order_return_55"]
if imported_mixed_case.get("machine_shape", {}).get("callee_source_param_count") != 5:
    raise SystemExit("imported mixed-left receipt must prove i256+i64 expands to five callee params")
if imported_mixed_case.get("extra_callee_param_counts", {}).get("mixed_right") != 5:
    raise SystemExit("imported mixed-right receipt must prove i64+i256 expands to five callee params")

if generic_agg_receipt.get("schema") != "madaros.v2.s5.generic_aggregate_sret_receipt/0.1":
    raise SystemExit("bad S5 generic aggregate SRET receipt schema")
if generic_agg_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing generic aggregate SRET receipt")
if generic_agg_receipt.get("stage_contract_level") != "S5_GENERIC_AGGREGATE_SRET_LAYOUT_PROMOTED":
    raise SystemExit("generic aggregate SRET receipt must declare promoted layout stage contract")
if generic_agg_receipt.get("case_count") != 5:
    raise SystemExit("generic aggregate SRET receipt must contain exact five cases")
required_generic_flags = [
    "s5_generic_aggregate_sret_layout_complete",
    "generic_aggregate_return_promoted",
    "generic_aggregate_local_layout_promoted",
    "generic_aggregate_imported_layout_promoted",
    "generic_aggregate_method_layout_promoted",
    "layout_derived_sret_alloc_promoted",
    "wide9_sret_alloc_72_bytes_promoted",
    "compiler_machine_module_exported",
    "real_program_mir_emitted",
    "real_abi_layout_emitted",
]
for field in required_generic_flags:
    if generic_agg_receipt.get(field) is not True:
        raise SystemExit(f"generic aggregate SRET receipt missing required true flag: {field}")
generic_cases = {row.get("case_id"): row for row in generic_agg_receipt.get("cases", [])}
required_generic_cases = {
    "source_sret_generic_pair2_return_23": {"exit": 23, "declared_layout_bytes": 16, "field_count": 2},
    "source_sret_generic_quad4_return_26": {"exit": 26, "declared_layout_bytes": 32, "field_count": 4},
    "source_sret_generic_wide9_return_45": {"exit": 45, "declared_layout_bytes": 72, "field_count": 9},
    "imported_sret_generic_wide9_return_45": {"exit": 45, "declared_layout_bytes": 72, "field_count": 9},
    "method_sret_generic_wide9_return_45": {"exit": 45, "declared_layout_bytes": 72, "field_count": 9},
}
if set(generic_cases) != set(required_generic_cases):
    raise SystemExit(f"generic aggregate SRET receipt cases mismatch: {sorted(generic_cases)}")
for case_id, expected in required_generic_cases.items():
    row = generic_cases[case_id]
    if row.get("actual_exit") != expected["exit"]:
        raise SystemExit(f"{case_id} expected exit {expected['exit']}, got {row.get('actual_exit')}")
    if row.get("declared_layout_bytes") != expected["declared_layout_bytes"]:
        raise SystemExit(f"{case_id} expected declared layout bytes {expected['declared_layout_bytes']}, got {row.get('declared_layout_bytes')}")
    if row.get("field_count") != expected["field_count"]:
        raise SystemExit(f"{case_id} expected field count {expected['field_count']}, got {row.get('field_count')}")
    shape = row.get("machine_shape", {})
    if not row.get("machine_module_json_sha256") or not row.get("elf_sha256"):
        raise SystemExit(f"{case_id} missing MachineModule or ELF sha256")
    if expected["declared_layout_bytes"] == 72:
        main_allocs = shape.get("main_alloc_bytes", [])
        if 72 not in main_allocs:
            raise SystemExit(f"{case_id} must allocate 72-byte Wide9 SRET destination, got {main_allocs}")
        if 64 in main_allocs:
            raise SystemExit(f"{case_id} must not retain fixed 64-byte Wide9 SRET destination, got {main_allocs}")
        if shape.get("main_field_load_indices") != list(range(9)):
            raise SystemExit(f"{case_id} must load all Wide9 fields in order")

if f128_literal_provenance_receipt.get("schema") != "madaros.v2.f128_literal_provenance_receipt/0.3":
    raise SystemExit("bad f128 literal provenance receipt schema")
if f128_literal_provenance_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 literal provenance receipt")
if f128_literal_provenance_receipt.get("stage_contract_level") != "S4_S5_F128_LITERAL_DECIMAL_METADATA_AND_TYPE_PROMOTED_NOT_F128_EXECUTION":
    raise SystemExit("f128 literal provenance receipt must declare parser decimal-metadata stage contract")
for field in [
    "raw_literal_capture_before_advance",
    "float_literal_ast_name_preserved",
    "float_literal_f64_value_still_preserved",
    "float_literal_decimal_metadata_fields_present",
    "float_literal_truncated_tail_metadata_present",
    "float_literal_decimal_metadata_helper_present",
    "float_literal_decimal_metadata_attached_in_parser",
    "f128_literal_decimal_metadata_independent_from_f64",
    "f128_type_kind_present",
    "f128_type_constructor_present",
    "f128_type_name_recognized_by_checker",
    "f128_type_mangle_and_print_present",
    "f128_type_positive_checker_test_present",
    "f128_type_manifest_printers_present",
    "f128_type_soir_serialization_present",
    "f128_type_byte_width_recorded",
    "f128_type_system_awareness_promoted",
    "f128_literal_provenance_preserved_for_future_binary128",
    "f128_decimal_not_forced_through_f64_only_ast",
]:
    if f128_literal_provenance_receipt.get(field) is not True:
        raise SystemExit(f"f128 literal provenance receipt missing required true flag: {field}")
for field in [
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if f128_literal_provenance_receipt.get(field) is not False:
        raise SystemExit(f"f128 literal provenance receipt must not overclaim {field}")

if f128_binary128_value_receipt.get("schema") != "madaros.v2.s5.f128_binary128_value_receipt/0.1":
    raise SystemExit("bad f128 binary128 value receipt schema")
if f128_binary128_value_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 binary128 value receipt")
if f128_binary128_value_receipt.get("stage_contract_level") != "S5_F128_BINARY128_VALUE_CONTRACT_PROMOTED_NOT_EXECUTION":
    raise SystemExit("f128 binary128 value receipt must declare value-contract stage contract")
if f128_binary128_value_receipt.get("case_count") != 40:
    raise SystemExit("f128 binary128 value receipt must contain exact forty cases")
for field in [
    "f128_binary128_value_contract_complete",
    "f128_binary128_round_ties_to_even_recorded",
    "f128_binary128_subnormal_underflow_overflow_recorded",
    "f128_binary128_sign_exponent_fraction_recorded",
    "f128_binary128_anchor_cases_verified",
    "f128_binary128_decimal_metadata_bridge_recorded",
]:
    if f128_binary128_value_receipt.get(field) is not True:
        raise SystemExit(f"f128 binary128 value receipt missing required true flag: {field}")
for field in [
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if f128_binary128_value_receipt.get(field) is not False:
        raise SystemExit(f"f128 binary128 value receipt must not overclaim {field}")
f128_value_cases = {row.get("case_id"): row for row in f128_binary128_value_receipt.get("cases", [])}
required_f128_value_hex = {
    "positive_zero": "00000000000000000000000000000000",
    "negative_zero": "80000000000000000000000000000000",
    "one": "3fff0000000000000000000000000000",
    "half": "3ffe0000000000000000000000000000",
    "two": "40000000000000000000000000000000",
    "smallest_normal": "00010000000000000000000000000000",
    "one_tenth_rounded": "3ffb999999999999999999999999999a",
    "high_precision_probe": "3fff3c0ca428c59fb71a7be16b6b6d5b",
    "quarter_exact": "3ffd0000000000000000000000000000",
    "eighth_exact": "3ffc0000000000000000000000000000",
    "one_and_half_exact": "3fff8000000000000000000000000000",
    "twelve_and_three_quarters_exact": "40029800000000000000000000000000",
    "negative_two_and_half_exact": "c0004000000000000000000000000000",
    "thirty_two_exact": "40040000000000000000000000000000",
    "ten_twenty_four_exact": "40090000000000000000000000000000",
    "one_e3_exact": "4008f400000000000000000000000000",
    "two_tenths_rounded": "3ffc999999999999999999999999999a",
    "three_tenths_rounded": "3ffd3333333333333333333333333333",
    "six_tenths_rounded": "3ffe3333333333333333333333333333",
    "seven_tenths_rounded": "3ffe6666666666666666666666666666",
    "nine_tenths_rounded": "3ffecccccccccccccccccccccccccccd",
    "one_point_one_rounded": "3fff199999999999999999999999999a",
    "negative_one_point_one_rounded": "bfff199999999999999999999999999a",
    "one_hundredth_rounded": "3ff847ae147ae147ae147ae147ae147b",
    "one_thousandth_rounded": "3ff50624dd2f1a9fbe76c8b439581062",
    "one_point_2345_rounded": "3fff3c083126e978d4fdf3b645a1cac1",
    "twelve_point_345_rounded": "40028b0a3d70a3d70a3d70a3d70a3d71",
    "one_twenty_three_point_456_rounded": "4005edd2f1a9fbe76c8b4395810624dd",
    "pi_scale10_rounded": "4000921fb54411743e0ccd6545767925",
    "one_seventeenth_prefix_scale16_rounded": "3ffae1e1e1e1e1e1d4518dd6a9289864",
    "scale17_rounded": "3ffbf9add3746f65e780cb23f138e780",
    "scale18_rounded": "3fc32725dd1d243aba0e75fe645cc487",
    "negative_scale18_rounded": "bfc32725dd1d243aba0e75fe645cc487",
    "large_scale6_rounded": "4023cbe991a14587e5a78f25a250f840",
    "large_all_nines_scale6_rounded": "4026d1a94a1fffffffde7210be9424e6",
    "minimum_subnormal_rounded": "00000000000000000000000000000001",
    "negative_minimum_subnormal_rounded": "80000000000000000000000000000001",
    "underflow_to_positive_zero": "00000000000000000000000000000000",
    "overflow_to_positive_infinity": "7fff0000000000000000000000000000",
    "overflow_to_negative_infinity": "ffff0000000000000000000000000000",
}
required_f128_value_classes = {
    "minimum_subnormal_rounded": "subnormal",
    "negative_minimum_subnormal_rounded": "subnormal",
    "underflow_to_positive_zero": "zero",
    "overflow_to_positive_infinity": "infinity",
    "overflow_to_negative_infinity": "infinity",
}
if set(f128_value_cases) != set(required_f128_value_hex):
    raise SystemExit(f"f128 binary128 value receipt cases mismatch: {sorted(f128_value_cases)}")
for case_id, expected_hex in required_f128_value_hex.items():
    row = f128_value_cases[case_id]
    if row.get("hex") != expected_hex:
        raise SystemExit(f"{case_id} expected binary128 hex {expected_hex}, got {row.get('hex')}")
    if "fraction_hi" not in row or "fraction_lo" not in row or "exponent_field" not in row:
        raise SystemExit(f"{case_id} must record sign/exponent/fraction limbs")
for case_id, expected_class in required_f128_value_classes.items():
    if f128_value_cases.get(case_id, {}).get("class") != expected_class:
        raise SystemExit(f"{case_id} expected binary128 class {expected_class}")
if f128_value_cases.get("high_precision_probe", {}).get("decimal_digit_count") != 35:
    raise SystemExit("high_precision_probe must preserve 35 decimal digits in the binary128 value receipt")
if f128_value_cases.get("high_precision_probe", {}).get("decimal_scale10") != 34:
    raise SystemExit("high_precision_probe must preserve decimal scale10=34 in the binary128 value receipt")
if f128_value_cases.get("one_e3_exact", {}).get("decimal_scale10") != -3:
    raise SystemExit("one_e3_exact must preserve decimal scale10=-3 in the binary128 value receipt")
if f128_value_cases.get("one_twenty_three_point_456_rounded", {}).get("decimal_scale10") != 3:
    raise SystemExit("one_twenty_three_point_456_rounded must preserve decimal scale10=3 in the binary128 value receipt")
if f128_value_cases.get("scale18_rounded", {}).get("decimal_scale10") != 18:
    raise SystemExit("scale18_rounded must preserve decimal scale10=18 in the binary128 value receipt")

if f128_literal_value_bridge_receipt.get("schema") != "madaros.v2.s5.f128_literal_value_bridge_receipt/0.3":
    raise SystemExit("bad S5 f128 literal value bridge receipt schema")
if f128_literal_value_bridge_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 literal value bridge receipt")
if f128_literal_value_bridge_receipt.get("stage_contract_level") != "S5_2_F128_LITERAL_VALUE_BRIDGED_WITH_NATIVE_OPAQUE_LOCAL_STORAGE":
    raise SystemExit("f128 literal value bridge receipt must declare IR/MIR/JSON bridge stage contract")
if f128_literal_value_bridge_receipt.get("case_count") != 3:
    raise SystemExit("f128 literal value bridge receipt must contain exact three cases")
for field in [
    "f128_literal_value_bridge_promoted",
    "f128_literal_decimal_metadata_bridged_to_ir",
    "f128_literal_decimal_metadata_bridged_to_machine_ir",
    "f128_literal_decimal_metadata_bridged_to_machine_module",
    "f128_literal_decimal_metadata_machine_module_supported",
    "f128_binary128_slot_metadata_emitted",
    "f128_machine_ir_opaque_literal_promoted",
    "f128_native_opaque_local_storage_promoted",
    "f128_native_v2_local_opaque_execution_promoted",
]:
    if f128_literal_value_bridge_receipt.get(field) is not True:
        raise SystemExit(f"f128 literal value bridge receipt missing required true flag: {field}")
for field in [
    "f128_native_ieee_binary128_materialization_promoted",
    "f128_native_arithmetic_promoted",
    "f128_native_call_abi_promoted",
    "f128_native_return_abi_promoted",
    "s5_ready",
]:
    if f128_literal_value_bridge_receipt.get(field) is not False:
        raise SystemExit(f"f128 literal value bridge receipt must not overclaim {field}")
bridge_cases = {row.get("case_id"): row for row in f128_literal_value_bridge_receipt.get("cases", [])}
required_bridge_cases = {
    "f128_literal_one_point_zero_bridge": [1, 0, 10, 2, 1, 0],
    "f128_literal_zero_point_five_bridge": [1, 0, 5, 2, 1, 0],
    "f128_literal_long_decimal_bridge": [1, 90123456789012345, 123456789012345678, 35, 34, 0],
}
if set(bridge_cases) != set(required_bridge_cases):
    raise SystemExit(f"f128 literal value bridge cases mismatch: {sorted(bridge_cases)}")
for case_id, expected in required_bridge_cases.items():
    row = bridge_cases[case_id]
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} must support f128 literal metadata at MachineIR level")
    if row.get("machine_module_unsupported_detail") not in ("", None):
        raise SystemExit(f"{case_id} must not carry a MachineModule unsupported detail")
    if row.get("expected_decimal_metadata") != expected:
        raise SystemExit(f"{case_id} expected decimal metadata mismatch")
    if row.get("f128_native_opaque_local_storage_promoted") is not True:
        raise SystemExit(f"{case_id} must promote local opaque f128 storage")
    if row.get("f128_ieee_binary128_execution_promoted") is not False:
        raise SystemExit(f"{case_id} must not promote IEEE binary128 execution")
    if row.get("run_rc") != 0:
        raise SystemExit(f"{case_id} emitted local opaque ELF must run rc=0")
    literal_rows = row.get("f128_literal_metadata_rows", [])
    slot_rows = row.get("f128_slot_rows", [])
    if len(literal_rows) != 1:
        raise SystemExit(f"{case_id} must emit exactly one f128 literal metadata row")
    if not slot_rows:
        raise SystemExit(f"{case_id} must emit at least one f128 slot row")
    literal_row = literal_rows[0]
    if [literal_row.get("decimal_sign"), literal_row.get("sig_hi"), literal_row.get("sig_lo"), literal_row.get("digit_count"), literal_row.get("scale10"), literal_row.get("truncated_digits")] != expected:
        raise SystemExit(f"{case_id} f128 literal metadata row mismatch")
    if literal_row.get("truncated_tail_info", 0) != 0:
        raise SystemExit(f"{case_id} should not carry truncated tail info for non-truncated bridge literal")

if machine_slot_metadata_receipt.get("schema") != "madaros.v2.s5.machine_slot_metadata_receipt/0.4":
    raise SystemExit("bad MachineIR slot metadata receipt schema")
if machine_slot_metadata_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing MachineIR slot metadata receipt")
if machine_slot_metadata_receipt.get("stage_contract_level") != "S5_2_MACHINEIR_F128_SLOT_METADATA_AND_NATIVE_OPAQUE_LOCAL_STORAGE":
    raise SystemExit("MachineIR slot metadata receipt must declare slot-kind/width stage contract")
if machine_slot_metadata_receipt.get("case_count") != 3:
    raise SystemExit("MachineIR slot metadata receipt must contain exact three cases")
required_slot_metadata_true_flags = [
    "machine_ir_slot_metadata_exported",
    "slot_kind_encoding_complete_for_current_scalars",
    "slot_width_words_exported",
    "i64_slot_kind_width_promoted",
    "f64_slot_kind_width_promoted",
    "f64_slot_kind_distinguished_from_i64",
    "f128_binary128_slot_kind_reserved",
    "f128_binary128_slot_kind_width_promoted",
    "f128_binary128_limb_contract_recorded",
    "f128_binary128_slot_metadata_emitted",
    "f128_machine_ir_opaque_slot_promoted",
    "f128_machine_ir_local_metadata_copy_promoted",
    "f128_native_opaque_local_storage_promoted",
    "wide_int_limb_slot_kind_reserved",
]
for field in required_slot_metadata_true_flags:
    if machine_slot_metadata_receipt.get(field) is not True:
        raise SystemExit(f"MachineIR slot metadata receipt missing required true flag: {field}")
if machine_slot_metadata_receipt.get("f128_binary128_limb_count") != 2:
    raise SystemExit("MachineIR slot metadata receipt must reserve two limbs for binary128")
if machine_slot_metadata_receipt.get("f128_binary128_limb_bits") != 64:
    raise SystemExit("MachineIR slot metadata receipt must reserve 64-bit binary128 limbs")
if machine_slot_metadata_receipt.get("f128_binary128_slot_metadata_emitted") is not True:
    raise SystemExit("MachineIR slot metadata receipt must emit f128 binary128 slot metadata")
if machine_slot_metadata_receipt.get("f128_execution_slot_emitted") is not True:
    raise SystemExit("MachineIR slot metadata receipt must record local opaque f128 execution-slot promotion")
for field in [
    "f128_native_ieee_binary128_materialization_promoted",
    "f128_native_arithmetic_promoted",
    "f128_native_call_abi_promoted",
    "f128_native_return_abi_promoted",
]:
    if machine_slot_metadata_receipt.get(field) is not False:
        raise SystemExit(f"MachineIR slot metadata receipt must not overclaim {field}")
for field in [
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if machine_slot_metadata_receipt.get(field) is not False:
        raise SystemExit(f"MachineIR slot metadata receipt must not overclaim {field}")
slot_metadata_cases = {row.get("case_id"): row for row in machine_slot_metadata_receipt.get("cases", [])}
if set(slot_metadata_cases) != {"i64_slot_kind_width_metadata", "f64_slot_kind_width_metadata", "f128_binary128_slot_kind_width_metadata"}:
    raise SystemExit(f"MachineIR slot metadata receipt cases mismatch: {sorted(slot_metadata_cases)}")
if 1 not in slot_metadata_cases["i64_slot_kind_width_metadata"].get("slot_kinds_seen", []):
    raise SystemExit("i64 slot metadata case must emit i64 kind")
if 2 not in slot_metadata_cases["f64_slot_kind_width_metadata"].get("slot_kinds_seen", []):
    raise SystemExit("f64 slot metadata case must emit f64 kind")
if 3 not in slot_metadata_cases["f128_binary128_slot_kind_width_metadata"].get("slot_kinds_seen", []):
    raise SystemExit("f128 slot metadata case must emit f128 kind")
if slot_metadata_cases["f128_binary128_slot_kind_width_metadata"].get("machine_module_supported") is not True:
    raise SystemExit("f128 slot metadata case must support MachineIR-level f128 metadata")
if slot_metadata_cases["f128_binary128_slot_kind_width_metadata"].get("machine_module_unsupported_detail") not in ("", None):
    raise SystemExit("f128 slot metadata case must not carry a MachineModule unsupported detail")

if f128_abi_metadata_receipt.get("schema") != "madaros.v2.s5.f128_abi_metadata_receipt/0.2":
    raise SystemExit("bad S5 f128 ABI metadata receipt schema")
if f128_abi_metadata_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 ABI metadata receipt")
if f128_abi_metadata_receipt.get("stage_contract_level") != "S5_1_F128_ABI_METADATA_PROMOTED":
    raise SystemExit("f128 ABI metadata receipt must declare ABI metadata stage contract")
if f128_abi_metadata_receipt.get("case_count") != 3:
    raise SystemExit("f128 ABI metadata receipt must contain exact three cases")
if f128_abi_metadata_receipt.get("imported_module_case_count") != 1:
    raise SystemExit("f128 ABI metadata receipt must contain one imported module case")
for field in [
    "f128_local_param_metadata_promoted",
    "f128_local_return_metadata_promoted",
    "f128_imported_param_metadata_promoted",
    "f128_imported_return_metadata_promoted",
    "f128_call_result_slot_metadata_promoted",
    "f128_abi_metadata_promoted",
]:
    if f128_abi_metadata_receipt.get(field) is not True:
        raise SystemExit(f"f128 ABI metadata receipt missing required true flag: {field}")
if f128_abi_metadata_receipt.get("f128_binary128_slot_kind") != 3:
    raise SystemExit("f128 ABI metadata receipt must use slot kind 3")
if f128_abi_metadata_receipt.get("f128_binary128_width_words") != 2:
    raise SystemExit("f128 ABI metadata receipt must use two 64-bit words")
if f128_abi_metadata_receipt.get("f128_sysv_classes") != "SSE,SSEUP":
    raise SystemExit("f128 ABI metadata receipt must record binary128 SysV SSE/SSEUP classes")
for field in [
    "f128_full_execution_promoted",
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
    "legacy_fallback_for_f128_abi",
]:
    if f128_abi_metadata_receipt.get(field) is not False:
        raise SystemExit(f"f128 ABI metadata receipt must not overclaim {field}")
f128_abi_cases = {row.get("case_id"): row for row in f128_abi_metadata_receipt.get("cases", [])}
required_f128_abi_cases = {
    "local_f128_arg_return_metadata",
    "local_f128_return_only_metadata",
    "imported_f128_arg_return_metadata",
}
if set(f128_abi_cases) != required_f128_abi_cases:
    raise SystemExit(f"f128 ABI metadata receipt cases mismatch: {sorted(f128_abi_cases)}")
for case_id, row in f128_abi_cases.items():
    if row.get("machine_supported") is not True:
        raise SystemExit(f"{case_id} must keep MachineModule supported for promoted direct shape")
    if row.get("machine_unsupported_detail") not in ("", None):
        raise SystemExit(f"{case_id} must not record an unsupported detail")
    if row.get("elf_emitted") is not True:
        raise SystemExit(f"{case_id} must emit an ELF for promoted opaque direct shape")
    shape = row.get("machine_shape", {})
    if shape.get("callee_source_returns_f128") is not True:
        raise SystemExit(f"{case_id} callee must export f128 return metadata")
    if shape.get("callee_source_return_slot_kind") != 3:
        raise SystemExit(f"{case_id} callee return slot kind must be 3")
    if shape.get("callee_source_return_width_words") != 2:
        raise SystemExit(f"{case_id} callee return width words must be 2")
    if shape.get("callee_source_f128_sysv_classes") != "SSE,SSEUP":
        raise SystemExit(f"{case_id} callee must export f128 SysV classes")
    if shape.get("callee_source_f128_execution_pending") is not True:
        raise SystemExit(f"{case_id} callee must keep full f128 execution pending")
    if shape.get("callee_source_f128_opaque_direct_call_return_promoted") is not True:
        raise SystemExit(f"{case_id} callee must export direct opaque call/return promotion")
    if int(shape.get("main_f128_slot_row_count", 0)) <= 0:
        raise SystemExit(f"{case_id} main must contain f128 slot metadata")
if f128_abi_cases["local_f128_arg_return_metadata"].get("machine_shape", {}).get("callee_source_f128_param_count") != 1:
    raise SystemExit("local f128 arg-return case must record one f128 callee param")
if f128_abi_cases["imported_f128_arg_return_metadata"].get("machine_shape", {}).get("callee_source_f128_param_count") != 1:
    raise SystemExit("imported f128 arg-return case must record one f128 callee param")
if f128_abi_cases["local_f128_return_only_metadata"].get("machine_shape", {}).get("callee_source_f128_param_count") != 0:
    raise SystemExit("local f128 return-only case must record zero f128 callee params")

if f128_native_opaque_storage_receipt.get("schema") != "madaros.v2.s5.f128_native_opaque_storage_receipt/0.1":
    raise SystemExit("bad S5 f128 native opaque storage receipt schema")
if f128_native_opaque_storage_receipt.get("stage_contract_level") != "S5_2_F128_NATIVE_OPAQUE_LOCAL_STORAGE_COPY":
    raise SystemExit("f128 native opaque storage receipt must declare S5.2 stage contract")
claims = f128_native_opaque_storage_receipt.get("claims", {})
for field in [
    "f128_native_opaque_local_storage_copy_promoted",
    "f128_native_executes_local_no_observe_program",
    "f128_truncated_arbitrary_decimal_materialization_promoted_elsewhere",
]:
    if claims.get(field) is not True:
        raise SystemExit(f"f128 native opaque storage receipt missing required true claim: {field}")
for field in [
    "f128_native_ieee_binary128_materialization_promoted",
    "f128_native_arithmetic_promoted",
    "f128_external_sysv_abi_promoted",
    "f128_sret_abi_promoted",
    "f128_overwide_call_shape_promoted",
    "legacy_fallback_used",
]:
    if claims.get(field) is not False:
        raise SystemExit(f"f128 native opaque storage receipt must not overclaim {field}")
if claims.get("f128_opaque_direct_call_return_abi_promoted_elsewhere") is not True:
    raise SystemExit("f128 native opaque storage receipt must acknowledge S5.5 direct call/return promotion")
if claims.get("f128_runtime_add_sub_helper_promoted_elsewhere") is not True:
    raise SystemExit("f128 native opaque storage receipt must acknowledge S5.8 runtime add/sub helper promotion")
if claims.get("f128_runtime_positive_rounded_tenths_add_helper_promoted_elsewhere") is not True:
    raise SystemExit("f128 native opaque storage receipt must acknowledge S5.18 rounded-tenths add helper promotion")
if claims.get("f128_runtime_positive_rounded_decimal_add_matrix_promoted_elsewhere") is not True:
    raise SystemExit("f128 native opaque storage receipt must acknowledge S5.19 rounded-decimal add matrix promotion")
if claims.get("f128_direct_expanded_gpr_call_shape_promoted_elsewhere") is not True:
    raise SystemExit("f128 native opaque storage receipt must acknowledge expanded-GPR direct call promotion")
if claims.get("f128_direct_stack_call_shape_promoted_elsewhere") is not True:
    raise SystemExit("f128 native opaque storage receipt must acknowledge stack direct call promotion")
if claims.get("f128_native_payload_words") != ["binary128_hi64", "binary128_lo64"]:
    raise SystemExit("f128 native opaque storage receipt must use binary128 payload words for supported literals")
f128_native_cases = {row.get("case_id"): row for row in f128_native_opaque_storage_receipt.get("cases", [])}
required_f128_native_cases = {
    "local_literal_copy_executes",
    "f128_rounded_decimal_arithmetic_runtime_helper_executes",
    "f128_overwide_arg_shape_stays_blocked",
    "truncated_arbitrary_decimal_materialization_executes",
}
if set(f128_native_cases) != required_f128_native_cases:
    raise SystemExit(f"f128 native opaque storage receipt cases mismatch: {sorted(f128_native_cases)}")
if f128_native_cases["local_literal_copy_executes"].get("native_v2_emitted") is not True:
    raise SystemExit("f128 local literal/copy witness must emit native-v2 ELF")
if f128_native_cases["local_literal_copy_executes"].get("run_rc") != 0:
    raise SystemExit("f128 local literal/copy witness must execute with rc=0")
if f128_native_cases["truncated_arbitrary_decimal_materialization_executes"].get("native_v2_emitted") is not True:
    raise SystemExit("f128 truncated arbitrary decimal witness must emit native-v2 ELF")
if f128_native_cases["truncated_arbitrary_decimal_materialization_executes"].get("run_rc") != 0:
    raise SystemExit("f128 truncated arbitrary decimal witness must execute with rc=0")
rounded_row = f128_native_cases["f128_rounded_decimal_arithmetic_runtime_helper_executes"]
if rounded_row.get("native_v2_emitted") is not True:
    raise SystemExit("rounded f128 arithmetic helper witness must emit native-v2 ELF")
if rounded_row.get("run_rc") != 0:
    raise SystemExit("rounded f128 arithmetic helper witness must execute with rc=0")
row = f128_native_cases["f128_overwide_arg_shape_stays_blocked"]
if row.get("native_v2_emitted") is not False:
    raise SystemExit("f128_overwide_arg_shape_stays_blocked must not emit native-v2 ELF")
if row.get("blocked_fail_closed") is not True:
    raise SystemExit("f128_overwide_arg_shape_stays_blocked must fail closed")
if row.get("expected_detail") != "call_arity_gt_8":
    raise SystemExit("f128_overwide_arg_shape_stays_blocked expected detail mismatch")
if row.get("machine_unsupported_detail") != "call_arity_gt_8":
    raise SystemExit("f128_overwide_arg_shape_stays_blocked machine unsupported detail mismatch")
arb_row = f128_native_cases["truncated_arbitrary_decimal_materialization_executes"]
arb_literals = arb_row.get("f128_literal_rows", [])
if not any(lit.get("truncated_tail_info") == 71 for lit in arb_literals):
    raise SystemExit("arbitrary decimal witness must preserve truncated_tail_info=71 for future IEEE rounding")

if f128_opaque_call_return_abi_receipt.get("schema") != "madaros.v2.s5.f128_opaque_call_return_abi_receipt/0.1":
    raise SystemExit("bad S5 f128 opaque call-return ABI receipt schema")
if f128_opaque_call_return_abi_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 opaque call-return ABI receipt")
if f128_opaque_call_return_abi_receipt.get("stage_contract_level") != "S5_5_F128_OPAQUE_DIRECT_CALL_RETURN_ABI_PROMOTED":
    raise SystemExit("f128 opaque call-return ABI receipt must declare S5.5 stage contract")
if f128_opaque_call_return_abi_receipt.get("case_count") != 16:
    raise SystemExit("f128 opaque call-return ABI receipt must contain exact sixteen cases")
if f128_opaque_call_return_abi_receipt.get("positive_case_count") != 15:
    raise SystemExit("f128 opaque call-return ABI receipt must contain exact fifteen positive cases")
if f128_opaque_call_return_abi_receipt.get("negative_case_count") != 1:
    raise SystemExit("f128 opaque call-return ABI receipt must contain exact one negative case")
for field in [
    "f128_opaque_direct_call_return_abi_promoted",
    "f128_opaque_direct_expanded_gpr_call_abi_promoted",
    "f128_opaque_direct_stack_call_abi_promoted",
    "f128_opaque_imported_direct_call_return_abi_promoted",
    "f128_native_internal_call_abi_promoted",
    "f128_native_internal_return_abi_promoted",
    "f128_machineir_return_high_word_capture_promoted",
]:
    if f128_opaque_call_return_abi_receipt.get(field) is not True:
        raise SystemExit(f"f128 opaque call-return ABI receipt missing required true flag: {field}")
if f128_opaque_call_return_abi_receipt.get("f128_runtime_add_sub_helper_promoted_elsewhere") is not True:
    raise SystemExit("f128 opaque call-return ABI receipt must acknowledge S5.8 runtime add/sub helper promotion")
if f128_opaque_call_return_abi_receipt.get("f128_runtime_positive_rounded_tenths_add_helper_promoted_elsewhere") is not True:
    raise SystemExit("f128 opaque call-return ABI receipt must acknowledge S5.18 rounded-tenths add helper promotion")
if f128_opaque_call_return_abi_receipt.get("f128_runtime_positive_rounded_decimal_add_matrix_promoted_elsewhere") is not True:
    raise SystemExit("f128 opaque call-return ABI receipt must acknowledge S5.19 rounded-decimal add matrix promotion")
for field in [
    "f128_external_sysv_abi_promoted",
    "f128_sret_abi_promoted",
    "f128_arithmetic_promoted",
    "f128_software_helpers_promoted",
    "f128_nan_inf_contract_promoted",
]:
    if f128_opaque_call_return_abi_receipt.get(field) is not False:
        raise SystemExit(f"f128 opaque call-return ABI receipt must not overclaim {field}")
f128_call_cases = {row.get("case_id"): row for row in f128_opaque_call_return_abi_receipt.get("cases", [])}
required_f128_call_cases = {
    "local_f128_identity_arg_return",
    "local_f128_return_only",
    "local_f128_arg_i64_return",
    "imported_f128_identity_arg_return",
    "imported_f128_return_only",
    "imported_f128_arg_i64_return",
    "imported_f128_plus_i64_arg_return",
    "imported_two_f128_args_return",
    "local_f128_plus_i64_arg_return",
    "local_i64_plus_f128_arg_return",
    "local_two_f128_args_return",
    "local_mixed_arg_f128_return",
    "local_four_f128_args_stack_return",
    "local_five_f128_args_deeper_stack_return",
    "f128_rounded_decimal_arithmetic_runtime_helper_return",
    "f128_nine_arg_arity_still_blocked",
}
if set(f128_call_cases) != required_f128_call_cases:
    raise SystemExit(f"f128 opaque call-return ABI receipt cases mismatch: {sorted(f128_call_cases)}")
for case_id in [
    "local_f128_identity_arg_return",
    "local_f128_return_only",
    "local_f128_arg_i64_return",
    "imported_f128_identity_arg_return",
    "imported_f128_return_only",
    "imported_f128_arg_i64_return",
    "imported_f128_plus_i64_arg_return",
    "imported_two_f128_args_return",
    "local_f128_plus_i64_arg_return",
    "local_i64_plus_f128_arg_return",
    "local_two_f128_args_return",
    "local_mixed_arg_f128_return",
    "local_four_f128_args_stack_return",
    "local_five_f128_args_deeper_stack_return",
    "f128_rounded_decimal_arithmetic_runtime_helper_return",
]:
    row = f128_call_cases[case_id]
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} must be MachineModule supported")
    if row.get("callee_direct_promoted") is not True:
        raise SystemExit(f"{case_id} must carry direct promotion flag")
    if int(row.get("run_rc", -1)) != int(row.get("expected_exit", -2)):
        raise SystemExit(f"{case_id} run exit mismatch")
    if not row.get("f128_slot_rows"):
        raise SystemExit(f"{case_id} must record f128 slot rows")
explicit_f128_capture_cases = 0
param_return_metadata_cases = 0
for case_id, row in f128_call_cases.items():
    flow = row.get("f128_return_word_flow")
    if isinstance(flow, dict):
        if flow.get("caller_capture_ret_word_selectors") == [0, 1]:
            explicit_f128_capture_cases += 1
        if flow.get("param_return_literal_metadata_propagated") is True:
            param_return_metadata_cases += 1
if explicit_f128_capture_cases < 1:
    raise SystemExit("f128 opaque call-return ABI receipt must retain at least one explicit low/high capture witness")
if param_return_metadata_cases < 1:
    raise SystemExit("f128 opaque call-return ABI receipt must record at least one param-return metadata propagation witness")
rounded_call_row = f128_call_cases["f128_rounded_decimal_arithmetic_runtime_helper_return"]
if rounded_call_row.get("machine_module_supported") is not True:
    raise SystemExit("f128 rounded decimal arithmetic call receipt helper must emit supported MachineModule")
if rounded_call_row.get("run_rc") != 0:
    raise SystemExit("f128 rounded decimal arithmetic call receipt helper must run rc=0")
row = f128_call_cases["f128_nine_arg_arity_still_blocked"]
if row.get("machine_module_supported") is not False:
    raise SystemExit("f128_nine_arg_arity_still_blocked must remain MachineModule unsupported")
if row.get("machine_module_unsupported_detail") != "call_arity_gt_8":
    raise SystemExit("f128_nine_arg_arity_still_blocked expected detail mismatch")

if f128_sret_internal_arg_boundary_receipt.get("schema") != "madaros.v2.s5.f128_sret_internal_arg_boundary_receipt/0.1":
    raise SystemExit("bad S5 f128 SRET internal arg-boundary receipt schema")
if f128_sret_internal_arg_boundary_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 SRET internal arg-boundary receipt")
if f128_sret_internal_arg_boundary_receipt.get("stage_contract_level") != "S5_5_F128_INTERNAL_SRET_ARG_BOUNDARY_PROMOTED":
    raise SystemExit("f128 SRET internal arg-boundary receipt must declare S5.5 stage contract")
if f128_sret_internal_arg_boundary_receipt.get("case_count") != 4:
    raise SystemExit("f128 SRET internal arg-boundary receipt must contain exact four cases")
if f128_sret_internal_arg_boundary_receipt.get("direct_control_case_count") != 1:
    raise SystemExit("f128 SRET internal arg-boundary receipt must contain one direct classifier-control case")
if f128_sret_internal_arg_boundary_receipt.get("sret_case_count") != 3:
    raise SystemExit("f128 SRET internal arg-boundary receipt must contain three SRET cases")
if f128_sret_internal_arg_boundary_receipt.get("sret_stack_case_count") != 1:
    raise SystemExit("f128 SRET internal arg-boundary receipt must contain one stack-boundary SRET case")
for field in [
    "f128_internal_sret_arg_boundary_promoted",
    "f128_internal_sret_arg_register_boundary_promoted",
    "f128_internal_sret_arg_stack_boundary_promoted",
    "f128_compact_vreg_classifier_base_only_promoted",
]:
    if f128_sret_internal_arg_boundary_receipt.get(field) is not True:
        raise SystemExit(f"f128 SRET internal arg-boundary receipt missing required true flag: {field}")
for field in [
    "f128_external_sysv_abi_promoted",
    "f128_sret_abi_promoted",
    "f128_arithmetic_promoted",
    "f128_software_helpers_promoted",
    "f128_nan_inf_contract_promoted",
]:
    if f128_sret_internal_arg_boundary_receipt.get(field) is not False:
        raise SystemExit(f"f128 SRET internal arg-boundary receipt must not overclaim {field}")
f128_sret_boundary_cases = {row.get("case_id"): row for row in f128_sret_internal_arg_boundary_receipt.get("cases", [])}
required_f128_sret_boundary_cases = {
    "direct_f128_then_i64_arithmetic_classifier_guard",
    "sret_f128_arg_then_i64_arithmetic",
    "sret_f128_arg_copied_to_f128_field_payload",
    "sret_three_f128_args_crosses_stack_boundary",
}
if set(f128_sret_boundary_cases) != required_f128_sret_boundary_cases:
    raise SystemExit(f"f128 SRET internal arg-boundary receipt cases mismatch: {sorted(f128_sret_boundary_cases)}")
for case_id in required_f128_sret_boundary_cases:
    row = f128_sret_boundary_cases[case_id]
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} must be MachineModule supported")
    if int(row.get("run_rc", -1)) != int(row.get("expected_exit", -2)):
        raise SystemExit(f"{case_id} run exit mismatch")
    shape = row.get("machine_shape", {})
    if shape.get("callee_internal_call_promoted") is not True:
        raise SystemExit(f"{case_id} must carry internal call promotion flag")
    if case_id != "direct_f128_then_i64_arithmetic_classifier_guard" and shape.get("callee_source_is_sret") is not True:
        raise SystemExit(f"{case_id} must be an SRET callee")
if f128_sret_boundary_cases["direct_f128_then_i64_arithmetic_classifier_guard"]["machine_shape"].get("main_arg_move_indices") != [0, 1, 2]:
    raise SystemExit("direct f128 classifier guard must pass f128 low/high plus trailing i64")
if f128_sret_boundary_cases["sret_f128_arg_then_i64_arithmetic"]["machine_shape"].get("main_arg_move_indices") != [0, 1, 2, 3]:
    raise SystemExit("f128 SRET register case must pass hidden dest, f128 low/high, trailing i64")
if f128_sret_boundary_cases["sret_three_f128_args_crosses_stack_boundary"]["machine_shape"].get("main_stack_arg_push_indices") != [7, 6]:
    raise SystemExit("f128 SRET stack case must push expanded ABI words 7 then 6")

if f128_binary128_native_anchor_receipt.get("schema") != "madaros.v2.s5.f128_binary128_native_anchor_receipt/0.1":
    raise SystemExit("bad S5 f128 binary128 native anchor receipt schema")
if f128_binary128_native_anchor_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 binary128 native anchor receipt")
if f128_binary128_native_anchor_receipt.get("stage_contract_level") != "S5_3_F128_NATIVE_BINARY128_ANCHOR_LITERAL_MATERIALIZATION":
    raise SystemExit("f128 binary128 native anchor receipt must declare S5.3 stage contract")
if f128_binary128_native_anchor_receipt.get("case_count") != 2:
    raise SystemExit("f128 binary128 native anchor receipt must contain exact two cases")
anchor_claims = f128_binary128_native_anchor_receipt.get("claims", {})
if anchor_claims.get("f128_binary128_native_anchor_materialization_promoted") is not True:
    raise SystemExit("f128 binary128 native anchor materialization must be promoted")
if anchor_claims.get("f128_binary128_native_anchor_classes") != ["positive finite exact 0.5", "positive finite exact 1.0"]:
    raise SystemExit("f128 binary128 native anchor receipt must declare exactly the promoted anchor classes")
if anchor_claims.get("f128_native_payload_words") != ["binary128_hi64", "binary128_lo64"]:
    raise SystemExit("f128 binary128 native anchor receipt must use binary128 payload words")
for field in [
    "f128_native_general_decimal_binary128_materialization_promoted",
    "f128_native_arithmetic_promoted",
    "f128_native_call_abi_promoted",
    "f128_native_return_abi_promoted",
    "legacy_fallback_used",
]:
    if anchor_claims.get(field) is not False:
        raise SystemExit(f"f128 binary128 native anchor receipt must not overclaim {field}")
anchor_cases = {row.get("case_id"): row for row in f128_binary128_native_anchor_receipt.get("cases", [])}
required_anchor_cases = {
    "binary128_anchor_half": {
        "literal": "0.5",
        "expected_hex": "3ffe0000000000000000000000000000",
        "expected_hi": 4611123068473966592,
    },
    "binary128_anchor_one": {
        "literal": "1.0",
        "expected_hex": "3fff0000000000000000000000000000",
        "expected_hi": 4611404543450677248,
    },
}
if set(anchor_cases) != set(required_anchor_cases):
    raise SystemExit(f"f128 binary128 native anchor receipt cases mismatch: {sorted(anchor_cases)}")
for case_id, expected in required_anchor_cases.items():
    row = anchor_cases[case_id]
    if row.get("literal") != expected["literal"]:
        raise SystemExit(f"{case_id} literal mismatch")
    if row.get("expected_binary128_hex") != expected["expected_hex"]:
        raise SystemExit(f"{case_id} expected binary128 hex mismatch")
    if row.get("expected_hi") != expected["expected_hi"]:
        raise SystemExit(f"{case_id} expected high word mismatch")
    if row.get("expected_lo") != 0:
        raise SystemExit(f"{case_id} expected low word must be zero")
    if row.get("hi_mov_imm64_pattern_found") is not True:
        raise SystemExit(f"{case_id} must prove binary128 high-word immediate in emitted ELF")
    if row.get("run_rc") != 0:
        raise SystemExit(f"{case_id} emitted ELF must run rc=0")
    if not row.get("elf_sha256") or not row.get("machine_module_sha256"):
        raise SystemExit(f"{case_id} missing ELF or MachineModule hash")

if f128_binary128_value_contract_native_receipt.get("schema") != "madaros.v2.s5.f128_binary128_value_contract_native_receipt/0.1":
    raise SystemExit("bad S5 f128 binary128 value-contract native receipt schema")
if f128_binary128_value_contract_native_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 binary128 value-contract native receipt")
if f128_binary128_value_contract_native_receipt.get("stage_contract_level") != "S5_17_F128_NATIVE_SIGNED_EXTREME_BINARY128_MATERIALIZATION":
    raise SystemExit("f128 binary128 value-contract native receipt must declare S5.17 signed extreme binary128 stage contract")
if f128_binary128_value_contract_native_receipt.get("case_count") != 73:
    raise SystemExit("f128 binary128 value-contract native receipt must contain exact seventy-three cases")
if f128_binary128_value_contract_native_receipt.get("negative_case_count") != 4:
    raise SystemExit("f128 binary128 value-contract native receipt must contain exact four negative fail-closed cases")
value_native_claims = f128_binary128_value_contract_native_receipt.get("claims", {})
for field in [
    "f128_binary128_value_contract_native_materialization_promoted",
    "f128_binary128_value_contract_case_set_complete",
    "f128_native_exact_dyadic_decimal_binary128_materialization_promoted",
    "f128_native_bounded_rounded_decimal_binary128_materialization_promoted",
    "f128_native_general_bounded_decimal_siglo_scale18_materialization_promoted",
    "f128_native_two_limb_integer_decimal_binary128_materialization_promoted",
    "f128_native_two_limb_fractional_decimal_binary128_materialization_promoted",
    "f128_native_truncated_decimal_binary128_value_contract_promoted",
    "f128_native_subnormal_underflow_overflow_value_contract_promoted",
    "f128_native_signed_minimum_subnormal_binary128_materialization_promoted",
    "uncontracted_f128_decimal_materialization_fails_closed",
]:
    if value_native_claims.get(field) is not True:
        raise SystemExit(f"f128 binary128 value-contract native receipt missing required true claim: {field}")
for field in [
    "f128_native_arbitrary_decimal_binary128_materialization_promoted",
    "f128_native_arithmetic_promoted",
    "f128_native_call_abi_promoted",
    "f128_native_return_abi_promoted",
    "legacy_fallback_used",
]:
    if value_native_claims.get(field) is not False:
        raise SystemExit(f"f128 binary128 value-contract native receipt must not overclaim {field}")
value_native_cases = {row.get("case_id"): row for row in f128_binary128_value_contract_native_receipt.get("cases", [])}
required_value_native_cases = {
    "positive_zero": {"literal": "0.0", "hex": "00000000000000000000000000000000", "metadata": [1, 0, 0, 2, 1, 0]},
    "negative_zero": {"literal": "-0.0", "hex": "80000000000000000000000000000000", "metadata": [-1, 0, 0, 2, 1, 0]},
    "one": {"literal": "1.0", "hex": "3fff0000000000000000000000000000", "metadata": [1, 0, 10, 2, 1, 0]},
    "half": {"literal": "0.5", "hex": "3ffe0000000000000000000000000000", "metadata": [1, 0, 5, 2, 1, 0]},
    "two": {"literal": "2.0", "hex": "40000000000000000000000000000000", "metadata": [1, 0, 20, 2, 1, 0]},
    "smallest_normal": {"literal": "3.36210314311209350626267781732175260259807934484647e-4932", "hex": "00010000000000000000000000000000", "metadata": [1, 626267781732175260, 336210314311209350, 51, 4982, 15]},
    "one_tenth_rounded": {"literal": "0.1", "hex": "3ffb999999999999999999999999999a", "metadata": [1, 0, 1, 2, 1, 0]},
    "high_precision_probe": {"literal": "1.2345678901234567890123456789012345", "hex": "3fff3c0ca428c59fb71a7be16b6b6d5b", "metadata": [1, 90123456789012345, 123456789012345678, 35, 34, 0]},
    "two_limb_int_36_digit_exact": {"literal": "123456789012345678901234567890123456e0", "hex": "40737c6e3bfd70fdeeaec417172dcbac", "metadata": [1, 901234567890123456, 123456789012345678, 36, 0, 0]},
    "two_limb_int_35_digit_exact": {"literal": "12345678901234567890123456789012345e0", "hex": "407030582ffdf3fe588bd01278f16fbc", "metadata": [1, 90123456789012345, 123456789012345678, 35, 0, 0]},
    "two_limb_int_all_nines_rounded": {"literal": "999999999999999999999999999999999999e0", "hex": "4076812f9cf7920e2b66973e20000000", "metadata": [1, 999999999999999999, 999999999999999999, 36, 0, 0]},
    "two_limb_int_rounding_pair_even_low": {"literal": "123456789012345678500000000000000000e0", "hex": "40737c6e3bfd70fdee55ac8bac02a000", "metadata": [1, 500000000000000000, 123456789012345678, 36, 0, 0]},
    "two_limb_int_rounding_pair_even_high": {"literal": "123456789012345678500000000000000001e0", "hex": "40737c6e3bfd70fdee55ac8bac02a000", "metadata": [1, 500000000000000001, 123456789012345678, 36, 0, 0]},
    "two_limb_int_36_digit_negative": {"literal": "-123456789012345678901234567890123456e0", "hex": "c0737c6e3bfd70fdeeaec417172dcbac", "metadata": [-1, 901234567890123456, 123456789012345678, 36, 0, 0]},
    "two_limb_dec_scale18_rounded": {"literal": "123456789012345678.901234567890123456", "hex": "4037b69b4ba630f34ee6b74f031cdea0", "metadata": [1, 901234567890123456, 123456789012345678, 36, 18, 0]},
    "two_limb_dec_scale18_sticky_low": {"literal": "123456789012345678.000000000000000001", "hex": "4037b69b4ba630f34e00000000000000", "metadata": [1, 1, 123456789012345678, 36, 18, 0]},
    "two_limb_dec_scale18_all_nines": {"literal": "999999999999999999.999999999999999999", "hex": "403abc16d674ec800000000000000000", "metadata": [1, 999999999999999999, 999999999999999999, 36, 18, 0]},
    "two_limb_dec_scale17_rounded": {"literal": "1234567890123456789.01234567890123456", "hex": "403b12210f47de981150329161f20b24", "metadata": [1, 901234567890123456, 123456789012345678, 36, 17, 0]},
    "two_limb_dec_scale16_rounded": {"literal": "12345678901234567890.1234567890123456", "hex": "403e56a95319d63e15a43f35ba6e8ded", "metadata": [1, 901234567890123456, 123456789012345678, 36, 16, 0]},
    "two_limb_dec_scale18_negative": {"literal": "-123456789012345678.901234567890123456", "hex": "c037b69b4ba630f34ee6b74f031cdea0", "metadata": [-1, 901234567890123456, 123456789012345678, 36, 18, 0]},
    "quarter_exact": {"literal": "0.25", "hex": "3ffd0000000000000000000000000000", "metadata": [1, 0, 25, 3, 2, 0]},
    "eighth_exact": {"literal": "0.125", "hex": "3ffc0000000000000000000000000000", "metadata": [1, 0, 125, 4, 3, 0]},
    "one_and_half_exact": {"literal": "1.5", "hex": "3fff8000000000000000000000000000", "metadata": [1, 0, 15, 2, 1, 0]},
    "twelve_and_three_quarters_exact": {"literal": "12.75", "hex": "40029800000000000000000000000000", "metadata": [1, 0, 1275, 4, 2, 0]},
    "negative_two_and_half_exact": {"literal": "-2.5", "hex": "c0004000000000000000000000000000", "metadata": [-1, 0, 25, 2, 1, 0]},
    "thirty_two_exact": {"literal": "32.0", "hex": "40040000000000000000000000000000", "metadata": [1, 0, 320, 3, 1, 0]},
    "ten_twenty_four_exact": {"literal": "1024.0", "hex": "40090000000000000000000000000000", "metadata": [1, 0, 10240, 5, 1, 0]},
    "one_e3_exact": {"literal": "1e3", "hex": "4008f400000000000000000000000000", "metadata": [1, 0, 1, 1, -3, 0]},
    "two_tenths_rounded": {"literal": "0.2", "hex": "3ffc999999999999999999999999999a", "metadata": [1, 0, 2, 2, 1, 0]},
    "three_tenths_rounded": {"literal": "0.3", "hex": "3ffd3333333333333333333333333333", "metadata": [1, 0, 3, 2, 1, 0]},
    "six_tenths_rounded": {"literal": "0.6", "hex": "3ffe3333333333333333333333333333", "metadata": [1, 0, 6, 2, 1, 0]},
    "seven_tenths_rounded": {"literal": "0.7", "hex": "3ffe6666666666666666666666666666", "metadata": [1, 0, 7, 2, 1, 0]},
    "nine_tenths_rounded": {"literal": "0.9", "hex": "3ffecccccccccccccccccccccccccccd", "metadata": [1, 0, 9, 2, 1, 0]},
    "one_point_one_rounded": {"literal": "1.1", "hex": "3fff199999999999999999999999999a", "metadata": [1, 0, 11, 2, 1, 0]},
    "negative_one_point_one_rounded": {"literal": "-1.1", "hex": "bfff199999999999999999999999999a", "metadata": [-1, 0, 11, 2, 1, 0]},
    "one_hundredth_rounded": {"literal": "0.01", "hex": "3ff847ae147ae147ae147ae147ae147b", "metadata": [1, 0, 1, 3, 2, 0]},
    "one_thousandth_rounded": {"literal": "0.001", "hex": "3ff50624dd2f1a9fbe76c8b439581062", "metadata": [1, 0, 1, 4, 3, 0]},
    "one_point_2345_rounded": {"literal": "1.2345", "hex": "3fff3c083126e978d4fdf3b645a1cac1", "metadata": [1, 0, 12345, 5, 4, 0]},
    "twelve_point_345_rounded": {"literal": "12.345", "hex": "40028b0a3d70a3d70a3d70a3d70a3d71", "metadata": [1, 0, 12345, 5, 3, 0]},
    "one_twenty_three_point_456_rounded": {"literal": "123.456", "hex": "4005edd2f1a9fbe76c8b4395810624dd", "metadata": [1, 0, 123456, 6, 3, 0]},
    "pi_scale10_rounded": {"literal": "3.1415926535", "hex": "4000921fb54411743e0ccd6545767925", "metadata": [1, 0, 31415926535, 11, 10, 0]},
    "one_seventeenth_prefix_scale16_rounded": {"literal": "0.0588235294117647", "hex": "3ffae1e1e1e1e1e1d4518dd6a9289864", "metadata": [1, 0, 588235294117647, 17, 16, 0]},
    "scale17_rounded": {"literal": "0.12345678901234567", "hex": "3ffbf9add3746f65e780cb23f138e780", "metadata": [1, 0, 12345678901234567, 18, 17, 0]},
    "scale18_rounded": {"literal": "1e-18", "hex": "3fc32725dd1d243aba0e75fe645cc487", "metadata": [1, 0, 1, 1, 18, 0]},
    "negative_scale18_rounded": {"literal": "-1e-18", "hex": "bfc32725dd1d243aba0e75fe645cc487", "metadata": [-1, 0, 1, 1, 18, 0]},
    "bounded_1e_minus_4": {"literal": "1e-4", "hex": "3ff1a36e2eb1c432ca57a786c226809d", "metadata": [1, 0, 1, 1, 4, 0]},
    "bounded_1e_minus_5": {"literal": "1e-5", "hex": "3fee4f8b588e368f08461f9f01b866e4", "metadata": [1, 0, 1, 1, 5, 0]},
    "bounded_1e_minus_6": {"literal": "1e-6", "hex": "3feb0c6f7a0b5ed8d36b4c7f34938583", "metadata": [1, 0, 1, 1, 6, 0]},
    "bounded_1e_minus_7": {"literal": "1e-7", "hex": "3fe7ad7f29abcaf485787a6520ec08d2", "metadata": [1, 0, 1, 1, 7, 0]},
    "bounded_1e_minus_8": {"literal": "1e-8", "hex": "3fe45798ee2308c39df9fb841a566d75", "metadata": [1, 0, 1, 1, 8, 0]},
    "bounded_1e_minus_9": {"literal": "1e-9", "hex": "3fe112e0be826d694b2e62d01511f12a", "metadata": [1, 0, 1, 1, 9, 0]},
    "bounded_1e_minus_10": {"literal": "1e-10", "hex": "3fddb7cdfd9d7bdbab7d6ae6881cb511", "metadata": [1, 0, 1, 1, 10, 0]},
    "bounded_1e_minus_11": {"literal": "1e-11", "hex": "3fda5fd7fe17964955fdef1ed34a2a74", "metadata": [1, 0, 1, 1, 11, 0]},
    "bounded_1e_minus_12": {"literal": "1e-12", "hex": "3fd719799812dea11197f27f0f6e885d", "metadata": [1, 0, 1, 1, 12, 0]},
    "bounded_1e_minus_13": {"literal": "1e-13", "hex": "3fd3c25c268497681c2650cb4be40d61", "metadata": [1, 0, 1, 1, 13, 0]},
    "bounded_1e_minus_14": {"literal": "1e-14", "hex": "3fd06849b86a12b9b01ea70909833de7", "metadata": [1, 0, 1, 1, 14, 0]},
    "bounded_1e_minus_15": {"literal": "1e-15", "hex": "3fcd203af9ee756159b21f3a6e0297ec", "metadata": [1, 0, 1, 1, 15, 0]},
    "bounded_1e_minus_16": {"literal": "1e-16", "hex": "3fc9cd2b297d889bc2b6985d7cd0f313", "metadata": [1, 0, 1, 1, 16, 0]},
    "bounded_1e_minus_17": {"literal": "1e-17", "hex": "3fc670ef54646d496892137dfd73f5a9", "metadata": [1, 0, 1, 1, 17, 0]},
    "bounded_7_8125": {"literal": "7.8125", "hex": "4001f400000000000000000000000000", "metadata": [1, 0, 78125, 5, 4, 0]},
    "bounded_42_0625": {"literal": "42.0625", "hex": "40045080000000000000000000000000", "metadata": [1, 0, 420625, 6, 4, 0]},
    "bounded_large_pi_prefix_scale1": {"literal": "314159265358979.3", "hex": "402f1db9e76a24834ccccccccccccccd", "metadata": [1, 0, 3141592653589793, 16, 1, 0]},
    "bounded_large_e_prefix_scale1": {"literal": "271828182845904.5", "hex": "402eee73dc8e93a10000000000000000", "metadata": [1, 0, 2718281828459045, 16, 1, 0]},
    "large_scale6_rounded": {"literal": "123456789012.345678", "hex": "4023cbe991a14587e5a78f25a250f840", "metadata": [1, 0, 123456789012345678, 18, 6, 0]},
    "large_all_nines_scale6_rounded": {"literal": "999999999999.999999", "hex": "4026d1a94a1fffffffde7210be9424e6", "metadata": [1, 0, 999999999999999999, 18, 6, 0]},
    "minimum_subnormal_rounded": {"literal": "6.475175119438025110924438958227646552499569338034681e-4966", "hex": "00000000000000000000000000000001", "metadata": [1, 92443895822764655, 647517511943802511, 52, 5017, 16]},
    "negative_minimum_subnormal_rounded": {"literal": "-6.475175119438025110924438958227646552499569338034681e-4966", "hex": "80000000000000000000000000000001", "metadata": [-1, 92443895822764655, 647517511943802511, 52, 5017, 16]},
    "underflow_to_positive_zero": {"literal": "1e-5000", "hex": "00000000000000000000000000000000", "metadata": [1, 0, 1, 1, 5000, 0]},
    "overflow_to_positive_infinity": {"literal": "1e5000", "hex": "7fff0000000000000000000000000000", "metadata": [1, 0, 1, 1, -5000, 0]},
    "overflow_to_negative_infinity": {"literal": "-1e5000", "hex": "ffff0000000000000000000000000000", "metadata": [-1, 0, 1, 1, -5000, 0]},
    "truncated_arbitrary_1p23456789012345678901234567890123456789": {"literal": "1.23456789012345678901234567890123456789", "hex": "3fff3c0ca428c59fb71a7be16b6b6d5b", "metadata": [1, 901234567890123456, 123456789012345678, 39, 38, 3]},
    "truncated_pi_40_digits": {"literal": "3.14159265358979323846264338327950288419", "hex": "4000921fb54442d18469898cc51701b8", "metadata": [1, 846264338327950288, 314159265358979323, 39, 38, 3]},
    "truncated_one_third_39_repeating": {"literal": "0.333333333333333333333333333333333333333", "hex": "3ffd5555555555555555555555555555", "metadata": [1, 333333333333333333, 33333333333333333, 40, 39, 4]},
}
if set(value_native_cases) != set(required_value_native_cases):
    raise SystemExit(f"f128 binary128 value-contract native receipt cases mismatch: {sorted(value_native_cases)}")
for case_id, expected in required_value_native_cases.items():
    row = value_native_cases[case_id]
    if row.get("literal") != expected["literal"]:
        raise SystemExit(f"{case_id} value-contract native literal mismatch")
    if row.get("expected_binary128_hex") != expected["hex"]:
        raise SystemExit(f"{case_id} value-contract native expected hex mismatch")
    if row.get("expected_decimal_metadata") != expected["metadata"]:
        raise SystemExit(f"{case_id} value-contract native metadata mismatch")
    if row.get("run_rc") != 0:
        raise SystemExit(f"{case_id} value-contract native ELF must run rc=0")
    if row.get("expected_hi_u64") != 0 and row.get("hi_mov_imm64_pattern_found") is not True:
        raise SystemExit(f"{case_id} value-contract native must prove high-word immediate")
    if row.get("expected_lo_u64") != 0 and row.get("lo_mov_imm64_pattern_found") is not True:
        raise SystemExit(f"{case_id} value-contract native must prove nonzero low-word immediate")
    if not row.get("elf_sha256") or not row.get("machine_module_sha256"):
        raise SystemExit(f"{case_id} value-contract native missing ELF or MachineModule hash")
value_native_negative_cases = {row.get("case_id"): row for row in f128_binary128_value_contract_native_receipt.get("negative_cases", [])}
required_value_native_negative = {
    "uncontracted_near_half_min_subnormal_fails_closed": "f128_decimal_materialization_pending",
    "uncontracted_truncated_pi_tail_fails_closed": "f128_decimal_materialization_pending",
    "uncontracted_positive_overflow_boundary_fails_closed": "f128_decimal_materialization_pending",
    "uncontracted_positive_underflow_boundary_fails_closed": "f128_decimal_materialization_pending",
}
if set(value_native_negative_cases) != set(required_value_native_negative):
    raise SystemExit(f"f128 binary128 value-contract native negative cases mismatch: {sorted(value_native_negative_cases)}")
for case_id, expected_detail in required_value_native_negative.items():
    row = value_native_negative_cases[case_id]
    if row.get("kind") != "negative":
        raise SystemExit(f"{case_id} must be marked as a negative case")
    if row.get("elf_emitted") is not False:
        raise SystemExit(f"{case_id} must not emit an ELF")
    if row.get("machine_module_supported") is not False:
        raise SystemExit(f"{case_id} must be MachineModule unsupported")
    if row.get("machine_module_unsupported_detail") != expected_detail:
        raise SystemExit(f"{case_id} unsupported detail mismatch")
    if not row.get("machine_module_sha256"):
        raise SystemExit(f"{case_id} missing MachineModule hash")

if f128_arithmetic_value_contract_receipt.get("schema") != "madaros.v2.s5.f128_arithmetic_value_contract_receipt/0.1":
    raise SystemExit("bad S5 f128 arithmetic value-contract receipt schema")
if f128_arithmetic_value_contract_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 arithmetic value-contract receipt")
if f128_arithmetic_value_contract_receipt.get("stage_contract_level") != "S5_19_F128_RUNTIME_ROUNDED_DECIMAL_ADD_MATRIX":
    raise SystemExit("f128 arithmetic value-contract receipt must declare S5.19 rounded-decimal add matrix stage contract")
if f128_arithmetic_value_contract_receipt.get("case_count") != 33:
    raise SystemExit("f128 arithmetic value-contract receipt must contain exact thirty-three cases")
if f128_arithmetic_value_contract_receipt.get("positive_case_count") != 31:
    raise SystemExit("f128 arithmetic value-contract receipt must contain thirty-one positive cases")
if f128_arithmetic_value_contract_receipt.get("negative_case_count") != 2:
    raise SystemExit("f128 arithmetic value-contract receipt must contain two negative cases")
for field in [
    "f128_arithmetic_value_contract_promoted",
    "f128_native_arithmetic_promoted",
    "f128_runtime_callee_add_sub_mul_div_value_contract_promoted",
    "f128_runtime_positive_rounded_tenths_add_helper_promoted",
    "f128_runtime_positive_rounded_decimal_add_matrix_promoted",
]:
    if f128_arithmetic_value_contract_receipt.get(field) is not True:
        raise SystemExit(f"f128 arithmetic value-contract receipt missing required true flag: {field}")
for field in [
    "f128_native_ieee_binary128_materialization_promoted",
    "f128_native_general_decimal_binary128_materialization_promoted",
    "f128_native_arbitrary_decimal_binary128_materialization_promoted",
    "f128_software_helpers_promoted",
    "f128_nan_inf_contract_promoted",
    "f128_external_sysv_abi_promoted",
    "f128_sret_abi_promoted",
    "f128_native_call_abi_promoted",
    "f128_native_return_abi_promoted",
    "f128_promoted",
]:
    if f128_arithmetic_value_contract_receipt.get(field) is not False:
        raise SystemExit(f"f128 arithmetic value-contract receipt must not overclaim {field}")
arith_cases = {row.get("case_id"): row for row in f128_arithmetic_value_contract_receipt.get("cases", [])}
required_arith_cases = {
    "f128_add_one_two_to_three",
    "f128_mul_one_two_to_two",
    "f128_add_half_half_to_one",
    "f128_div_one_two_to_half",
    "f128_chain_add_sub_to_one",
    "f128_add_half_one_to_one_and_half",
    "f128_mul_half_half_to_quarter",
    "f128_mul_one_and_half_half_to_three_quarters",
    "f128_add_quarter_one_to_one_and_quarter",
    "f128_sub_half_one_to_negative_half",
    "f128_add_negative_half_one_to_half",
    "f128_add_negative_half_negative_half_to_negative_one",
    "f128_mul_negative_half_half_to_negative_quarter",
    "f128_div_negative_one_two_to_negative_half",
    "f128_call_literal_return_then_add_to_three",
    "f128_call_identity_return_then_add_to_three",
    "f128_call_pick_first_return_then_add_to_three",
    "f128_call_pick_second_return_then_add_to_three",
    "f128_add_rounded_tenths_runtime_helper_to_sum",
    "f128_add_one_tenth_seven_tenths_runtime_helper_to_binary_sum",
    "f128_add_two_tenths_seven_tenths_runtime_helper_to_binary_sum",
    "f128_add_three_tenths_six_tenths_runtime_helper_to_binary_sum",
    "f128_add_six_tenths_seven_tenths_runtime_helper_to_binary_sum",
    "f128_add_nine_tenths_two_tenths_runtime_helper_to_literal_equivalent_sum",
    "f128_add_hundredth_thousandth_runtime_helper_to_binary_sum",
    "f128_add_one_point_2345_thousandth_runtime_helper_to_binary_sum",
    "f128_callee_add_args_runtime_helper_to_three",
    "f128_callee_sub_args_runtime_helper_to_one",
    "f128_callee_mul_args_runtime_helper_to_three_quarters",
    "f128_callee_div_args_runtime_helper_to_half",
    "f128_callee_div_negative_args_runtime_helper_to_negative_half",
    "f128_callee_div_one_three_runtime_fail_closed",
    "f128_callee_div_by_zero_runtime_fail_closed",
}
if set(arith_cases) != required_arith_cases:
    raise SystemExit(f"f128 arithmetic value-contract receipt cases mismatch: {sorted(arith_cases)}")
required_arith_positive = {
    "f128_add_one_two_to_three": {"hex": "40008000000000000000000000000000", "metadata": [1, 0, 30, 2, 1, 0]},
    "f128_mul_one_two_to_two": {"hex": "40000000000000000000000000000000", "metadata": [1, 0, 20, 2, 1, 0]},
    "f128_add_half_half_to_one": {"hex": "3fff0000000000000000000000000000", "metadata": [1, 0, 10, 2, 1, 0]},
    "f128_div_one_two_to_half": {"hex": "3ffe0000000000000000000000000000", "metadata": [1, 0, 5, 2, 1, 0]},
    "f128_chain_add_sub_to_one": {"hex": "3fff0000000000000000000000000000", "metadata": [1, 0, 10, 2, 1, 0]},
    "f128_add_half_one_to_one_and_half": {"hex": "3fff8000000000000000000000000000", "metadata": [1, 0, 15, 2, 1, 0]},
    "f128_mul_half_half_to_quarter": {"hex": "3ffd0000000000000000000000000000", "metadata": [1, 0, 25, 3, 2, 0]},
    "f128_mul_one_and_half_half_to_three_quarters": {"hex": "3ffe8000000000000000000000000000", "metadata": [1, 0, 75, 3, 2, 0]},
    "f128_add_quarter_one_to_one_and_quarter": {"hex": "3fff4000000000000000000000000000", "metadata": [1, 0, 125, 3, 2, 0]},
    "f128_sub_half_one_to_negative_half": {"hex": "bffe0000000000000000000000000000", "metadata": [-1, 0, 5, 2, 1, 0]},
    "f128_add_negative_half_one_to_half": {"hex": "3ffe0000000000000000000000000000", "metadata": [1, 0, 5, 2, 1, 0]},
    "f128_add_negative_half_negative_half_to_negative_one": {"hex": "bfff0000000000000000000000000000", "metadata": [-1, 0, 10, 2, 1, 0]},
    "f128_mul_negative_half_half_to_negative_quarter": {"hex": "bffd0000000000000000000000000000", "metadata": [-1, 0, 25, 3, 2, 0]},
    "f128_div_negative_one_two_to_negative_half": {"hex": "bffe0000000000000000000000000000", "metadata": [-1, 0, 5, 2, 1, 0]},
    "f128_call_literal_return_then_add_to_three": {"hex": "40008000000000000000000000000000", "metadata": [1, 0, 30, 2, 1, 0]},
    "f128_call_identity_return_then_add_to_three": {"hex": "40008000000000000000000000000000", "metadata": [1, 0, 30, 2, 1, 0]},
    "f128_call_pick_first_return_then_add_to_three": {"hex": "40008000000000000000000000000000", "metadata": [1, 0, 30, 2, 1, 0]},
    "f128_call_pick_second_return_then_add_to_three": {"hex": "40008000000000000000000000000000", "metadata": [1, 0, 30, 2, 1, 0]},
    "f128_add_rounded_tenths_runtime_helper_to_sum": {"hex": "3ffd3333333333333333333333333334", "metadata": None, "opcode": 131},
    "f128_add_one_tenth_seven_tenths_runtime_helper_to_binary_sum": {"hex": "3ffe9999999999999999999999999999", "metadata": None, "opcode": 131},
    "f128_add_two_tenths_seven_tenths_runtime_helper_to_binary_sum": {"hex": "3ffecccccccccccccccccccccccccccc", "metadata": None, "opcode": 131},
    "f128_add_three_tenths_six_tenths_runtime_helper_to_binary_sum": {"hex": "3ffecccccccccccccccccccccccccccc", "metadata": None, "opcode": 131},
    "f128_add_six_tenths_seven_tenths_runtime_helper_to_binary_sum": {"hex": "3fff4ccccccccccccccccccccccccccc", "metadata": None, "opcode": 131},
    "f128_add_nine_tenths_two_tenths_runtime_helper_to_literal_equivalent_sum": {"hex": "3fff199999999999999999999999999a", "metadata": None, "opcode": 131},
    "f128_add_hundredth_thousandth_runtime_helper_to_binary_sum": {"hex": "3ff86872b020c49ba5e353f7ced91687", "metadata": None, "opcode": 131},
    "f128_add_one_point_2345_thousandth_runtime_helper_to_binary_sum": {"hex": "3fff3c49ba5e353f7ced916872b020c5", "metadata": None, "opcode": 131},
    "f128_callee_add_args_runtime_helper_to_three": {"hex": "40008000000000000000000000000000", "metadata": None, "opcode": 131},
    "f128_callee_sub_args_runtime_helper_to_one": {"hex": "3fff0000000000000000000000000000", "metadata": None, "opcode": 131},
    "f128_callee_mul_args_runtime_helper_to_three_quarters": {"hex": "3ffe8000000000000000000000000000", "metadata": None, "opcode": 131},
    "f128_callee_div_args_runtime_helper_to_half": {"hex": "3ffe0000000000000000000000000000", "metadata": None, "opcode": 131},
    "f128_callee_div_negative_args_runtime_helper_to_negative_half": {"hex": "bffe0000000000000000000000000000", "metadata": None, "opcode": 131},
}
for case_id, expected in required_arith_positive.items():
    row = arith_cases[case_id]
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} must be MachineModule supported")
    if row.get("run_rc") != 0:
        raise SystemExit(f"{case_id} ELF must run rc=0")
    if row.get("expected_binary128_hex") != expected["hex"]:
        raise SystemExit(f"{case_id} binary128 hex mismatch")
    if row.get("expected_result_metadata") != expected["metadata"]:
        raise SystemExit(f"{case_id} result metadata mismatch")
    expected_opcode = int(expected.get("opcode", 0) or 0)
    if expected_opcode != 0:
        if int(row.get("expected_machine_opcode", 0) or 0) != expected_opcode:
            raise SystemExit(f"{case_id} expected MachineIR opcode declaration mismatch")
        if row.get("expected_machine_opcode_found") is not True:
            raise SystemExit(f"{case_id} must prove MachineIR runtime helper opcode")
    if row.get("hi_mov_imm_pattern_found") is not True:
        raise SystemExit(f"{case_id} must prove high-word immediate")
    if not row.get("elf_sha256") or not row.get("machine_module_sha256"):
        raise SystemExit(f"{case_id} missing ELF or MachineModule hash")
for case_id in [
    "f128_callee_div_one_three_runtime_fail_closed",
    "f128_callee_div_by_zero_runtime_fail_closed",
]:
    row = arith_cases[case_id]
    if row.get("negative_mode") != "runtime_fail_closed":
        raise SystemExit(f"{case_id} must be runtime fail-closed")
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} must emit supported helper MachineModule")
    if row.get("run_rc") != 12 or row.get("expected_runtime_rc") != 12:
        raise SystemExit(f"{case_id} must trap with rc=12")
    if row.get("expected_machine_opcode_found") is not True:
        raise SystemExit(f"{case_id} must prove runtime helper opcode")

if f128_ieee_class_helper_receipt.get("schema") != "madaros.v2.s5.f128_ieee_class_helper_receipt/0.2":
    raise SystemExit("bad S5 f128 IEEE class-code helper receipt schema")
if f128_ieee_class_helper_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 IEEE class-code helper receipt")
if f128_ieee_class_helper_receipt.get("stage_contract_level") != "S5_14_F128_NATIVE_IEEE_CLASS_CODE_HELPER_WITH_NAN_SOURCE":
    raise SystemExit("f128 IEEE class-code helper receipt must declare S5.14 NaN-source stage contract")
if f128_ieee_class_helper_receipt.get("case_count") != 12:
    raise SystemExit("f128 IEEE class-code helper receipt must contain exact twelve positive cases")
if f128_ieee_class_helper_receipt.get("negative_case_count") != 0:
    raise SystemExit("f128 IEEE class-code helper receipt must contain zero negative fail-closed cases")
class_helper_claims = f128_ieee_class_helper_receipt.get("claims", {})
for field in [
    "f128_native_ieee_class_code_helper_promoted",
    "f128_native_ieee_class_code_source_observable_zero_subnormal_normal_infinity_promoted",
    "f128_native_ieee_class_code_source_observable_signed_subnormal_promoted",
    "f128_native_ieee_class_code_nan_branch_emitted",
    "f128_native_ieee_class_code_nan_source_surface_promoted",
    "f128_native_canonical_quiet_nan_constructor_promoted",
]:
    if class_helper_claims.get(field) is not True:
        raise SystemExit(f"f128 IEEE class-code helper receipt missing required true claim: {field}")
for field in [
    "f128_native_generic_ieee_arithmetic_promoted",
    "f128_external_sysv_abi_promoted",
    "f128_native_arbitrary_decimal_binary128_materialization_promoted",
    "legacy_fallback_used",
]:
    if class_helper_claims.get(field) is not False:
        raise SystemExit(f"f128 IEEE class-code helper receipt must not overclaim {field}")
if f128_ieee_class_helper_receipt.get("class_code_contract") != {
    "zero": 0,
    "subnormal": 1,
    "normal": 2,
    "infinity": 3,
    "nan": 4,
}:
    raise SystemExit("f128 IEEE class-code helper contract changed")
class_helper_cases = {row.get("case_id"): row for row in f128_ieee_class_helper_receipt.get("cases", [])}
required_class_helper_cases = {
    "zero_positive": 0,
    "zero_negative": 0,
    "normal_one": 2,
    "normal_one_tenth": 2,
    "normal_negative_one_tenth": 2,
    "normal_smallest_binary128": 2,
    "subnormal_min_positive": 1,
    "subnormal_min_negative": 1,
    "underflow_positive_zero": 0,
    "infinity_positive_overflow": 3,
    "infinity_negative_overflow": 3,
    "nan_canonical_quiet_builtin": 4,
}
if set(class_helper_cases) != set(required_class_helper_cases):
    raise SystemExit(f"f128 IEEE class-code helper cases mismatch: {sorted(class_helper_cases)}")
for case_id, expected_rc in required_class_helper_cases.items():
    row = class_helper_cases[case_id]
    if row.get("run_rc") != expected_rc or row.get("expected_class_code") != expected_rc:
        raise SystemExit(f"{case_id} expected class code {expected_rc}, got {row.get('run_rc')}")
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} MachineModule must be supported")
    if row.get("machine_module_legacy_fallback") is not False:
        raise SystemExit(f"{case_id} must not use legacy fallback")
    if row.get("contains_exponent_mask_imm64") is not True:
        raise SystemExit(f"{case_id} must prove binary128 exponent-mask immediate in emitted ELF")
    if row.get("contains_fraction_high_mask_imm64") is not True:
        raise SystemExit(f"{case_id} must prove binary128 fraction-high-mask immediate in emitted ELF")
    if not row.get("elf_sha256") or not row.get("machine_module_json_sha256"):
        raise SystemExit(f"{case_id} missing ELF or MachineModule hash")
class_helper_negative_cases = {
    row.get("case_id"): row for row in f128_ieee_class_helper_receipt.get("negative_cases", [])
}
if class_helper_negative_cases:
    raise SystemExit(f"f128 IEEE class-code helper negative cases must be empty: {sorted(class_helper_negative_cases)}")

if f128_ordered_compare_receipt.get("schema") != "madaros.v2.s5.f128_ordered_compare_receipt/0.1":
    raise SystemExit("bad S5 f128 ordered binary128 comparison receipt schema")
if f128_ordered_compare_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 ordered comparison receipt")
if f128_ordered_compare_receipt.get("stage_contract_level") != "S5_20_F128_ORDERED_BINARY128_COMPARE":
    raise SystemExit("f128 ordered comparison receipt must declare S5.20 stage contract")
if f128_ordered_compare_receipt.get("case_count") != 17:
    raise SystemExit("f128 ordered comparison receipt must contain exact seventeen cases")
if f128_ordered_compare_receipt.get("operators_covered") != ["!=", "<", "<=", "==", ">", ">="]:
    raise SystemExit("f128 ordered comparison receipt must cover all six comparison operators")
if f128_ordered_compare_receipt.get("ordered_compare_contract") != {
    "finite_sign_magnitude_ordered": True,
    "infinities_ordered": True,
    "nan_not_equal": True,
    "nan_ordered_predicates": False,
    "signed_zero_equal": True,
}:
    raise SystemExit("f128 ordered comparison contract changed")
ordered_compare_claims = f128_ordered_compare_receipt.get("claims", {})
for field in [
    "f128_ordered_binary128_compare_promoted",
    "f128_ordered_binary128_compare_source_observable_promoted",
    "f128_ordered_binary128_compare_nan_unordered_promoted",
    "f128_ordered_binary128_compare_signed_zero_promoted",
    "f128_ordered_binary128_compare_infinity_promoted",
    "f128_ordered_binary128_compare_subnormal_promoted",
]:
    if ordered_compare_claims.get(field) is not True:
        raise SystemExit(f"f128 ordered comparison receipt missing required true claim: {field}")
for field in [
    "f128_native_generic_ieee_arithmetic_promoted",
    "f128_software_helpers_promoted",
    "f128_external_sysv_abi_promoted",
    "f128_native_arbitrary_decimal_binary128_materialization_promoted",
    "f128_promoted",
    "legacy_fallback_used",
]:
    if ordered_compare_claims.get(field) is not False:
        raise SystemExit(f"f128 ordered comparison receipt must not overclaim {field}")
ordered_compare_cases = {row.get("case_id"): row for row in f128_ordered_compare_receipt.get("cases", [])}
required_ordered_compare_cases = {
    "f128_cmp_eq_one_one_true": {"expected": True, "op": "=="},
    "f128_cmp_ne_one_two_true": {"expected": True, "op": "!="},
    "f128_cmp_lt_one_two_true": {"expected": True, "op": "<"},
    "f128_cmp_le_one_one_true": {"expected": True, "op": "<="},
    "f128_cmp_gt_two_one_true": {"expected": True, "op": ">"},
    "f128_cmp_ge_two_two_true": {"expected": True, "op": ">="},
    "f128_cmp_signed_zero_eq_true": {"expected": True, "op": "=="},
    "f128_cmp_signed_zero_ne_false": {"expected": False, "op": "!="},
    "f128_cmp_negative_order_true": {"expected": True, "op": "<"},
    "f128_cmp_negative_reverse_false": {"expected": False, "op": "<"},
    "f128_cmp_subnormal_less_normal_true": {"expected": True, "op": "<"},
    "f128_cmp_positive_infinity_gt_true": {"expected": True, "op": ">"},
    "f128_cmp_negative_infinity_lt_true": {"expected": True, "op": "<"},
    "f128_cmp_nan_eq_false": {"expected": False, "op": "=="},
    "f128_cmp_nan_ne_true": {"expected": True, "op": "!="},
    "f128_cmp_nan_lt_false": {"expected": False, "op": "<"},
    "f128_cmp_nan_le_false": {"expected": False, "op": "<="},
}
if set(ordered_compare_cases) != set(required_ordered_compare_cases):
    raise SystemExit(f"f128 ordered comparison cases mismatch: {sorted(ordered_compare_cases)}")
for case_id, expected in required_ordered_compare_cases.items():
    row = ordered_compare_cases[case_id]
    expected_rc = 0 if expected["expected"] else 1
    if row.get("op") != expected["op"]:
        raise SystemExit(f"{case_id} comparison operator mismatch")
    if row.get("expected_bool") is not expected["expected"]:
        raise SystemExit(f"{case_id} expected bool mismatch")
    if row.get("run_rc") != expected_rc or row.get("expected_exit") != expected_rc:
        raise SystemExit(f"{case_id} expected exit {expected_rc}, got {row.get('run_rc')}")
    if row.get("machine_module_supported") is not True:
        raise SystemExit(f"{case_id} MachineModule must be supported")
    if row.get("machine_module_legacy_fallback") is not False:
        raise SystemExit(f"{case_id} must not use legacy fallback")
    if row.get("contains_binary128_exponent_mask") is not True:
        raise SystemExit(f"{case_id} must prove binary128 exponent-mask immediate in emitted ELF")
    if not row.get("elf_sha256") or not row.get("machine_module_json_sha256"):
        raise SystemExit(f"{case_id} missing ELF or MachineModule hash")
if not any(row.get("contains_binary128_sign_mask") is True for row in ordered_compare_cases.values()):
    raise SystemExit("f128 ordered comparison receipt must prove binary128 sign-mask immediate")
if not any(row.get("contains_binary128_abs_mask") is True for row in ordered_compare_cases.values()):
    raise SystemExit("f128 ordered comparison receipt must prove binary128 abs-mask immediate")

if f128_param_slot_layout_receipt.get("schema") != "madaros.v2.s5.f128_param_slot_layout_receipt/0.1":
    raise SystemExit("bad S5 f128 parameter slot-layout receipt schema")
if f128_param_slot_layout_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 parameter slot-layout receipt")
if f128_param_slot_layout_receipt.get("stage_contract_level") != "S5_7_F128_NON_OVERLAPPING_PARAMETER_SLOTS":
    raise SystemExit("f128 parameter slot-layout receipt must declare S5.7 stage contract")
if f128_param_slot_layout_receipt.get("case_count") != 4:
    raise SystemExit("f128 parameter slot-layout receipt must contain exact four cases")
for field in [
    "f128_param_slot_layout_promoted",
    "f128_param_slots_non_overlapping",
    "f128_callee_add_sub_value_contract_helper_layout_promoted",
]:
    if f128_param_slot_layout_receipt.get(field) is not True:
        raise SystemExit(f"f128 parameter slot-layout receipt missing required true flag: {field}")
for field in ["f128_full_execution_promoted", "f128_promoted", "s5_ready", "s5_full_complete"]:
    if f128_param_slot_layout_receipt.get(field) is not False:
        raise SystemExit(f"f128 parameter slot-layout receipt must not overclaim {field}")
slot_layout_cases = {row.get("case_id"): row for row in f128_param_slot_layout_receipt.get("cases", [])}
required_slot_layout = {
    "local_two_f128_params_non_overlapping": {"rows": [[0, 3, 2], [2, 3, 2]], "supported": True, "detail": ""},
    "local_f128_i64_f128_params_non_overlapping": {"rows": [[0, 3, 2], [3, 3, 2]], "supported": True, "detail": ""},
    "imported_two_f128_params_non_overlapping": {"rows": [[0, 3, 2], [2, 3, 2]], "supported": True, "detail": ""},
    "f128_callee_add_args_slot_layout_feeds_runtime_helper": {"rows": [[0, 3, 2], [2, 3, 2], [4, 3, 2]], "supported": True, "detail": ""},
}
if set(slot_layout_cases) != set(required_slot_layout):
    raise SystemExit(f"f128 parameter slot-layout receipt cases mismatch: {sorted(slot_layout_cases)}")
for case_id, expected in required_slot_layout.items():
    row = slot_layout_cases[case_id]
    if row.get("observed_f128_rows") != expected["rows"]:
        raise SystemExit(f"{case_id} observed f128 rows mismatch")
    if row.get("non_overlapping_f128_param_slots") is not True:
        raise SystemExit(f"{case_id} must prove non-overlapping f128 parameter slots")
    if row.get("machine_supported") is not expected["supported"]:
        raise SystemExit(f"{case_id} supported mismatch in slot-layout receipt")
    if row.get("machine_unsupported_detail") != expected["detail"]:
        raise SystemExit(f"{case_id} unsupported detail mismatch in slot-layout receipt")

if diagnostics_receipt.get("schema") != "madaros.v2.s5.diagnostics_receipt/0.3":
    raise SystemExit("bad S5 diagnostics receipt schema")
if diagnostics_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing diagnostics receipt")
if diagnostics_receipt.get("stage_contract_level") != "S5_1_UNSUPPORTED_NUMERIC_AND_F128_BLOCKER_DIAGNOSTICS_PROMOTED":
    raise SystemExit("diagnostics receipt must declare unsupported numeric diagnostic stage contract")
if diagnostics_receipt.get("case_count") != 5:
    raise SystemExit("diagnostics receipt must contain exact five cases")
if diagnostics_receipt.get("negative_case_count") != 3:
    raise SystemExit("diagnostics receipt must contain exact three negative cases")
if diagnostics_receipt.get("positive_guard_case_count") != 2:
    raise SystemExit("diagnostics receipt must contain exact two positive guard cases")
required_diagnostics_true_flags = [
    "s5_diagnostics_unsupported_numeric_complete",
    "unsupported_numeric_widths_fail_closed",
    "unsupported_widths_do_not_emit_elf",
    "front_half_unsupported_widths_do_not_emit_machine_module_json",
    "f128_blockers_emit_machine_module_json",
    "unsupported_widths_do_not_segfault",
    "f128_full_execution_not_promoted",
    "f128_opaque_direct_call_return_abi_promoted_elsewhere",
    "f128_direct_expanded_gpr_call_shape_promoted_elsewhere",
    "f128_direct_stack_call_shape_promoted_elsewhere",
    "f128_runtime_positive_rounded_tenths_add_helper_promoted_elsewhere",
    "f128_runtime_positive_rounded_decimal_add_matrix_promoted_elsewhere",
    "i512_u512_rejected_not_promoted",
    "promoted_i256_width_preserved",
]
for field in required_diagnostics_true_flags:
    if diagnostics_receipt.get(field) is not True:
        raise SystemExit(f"diagnostics receipt missing required true flag: {field}")
if diagnostics_receipt.get("f128_machine_module_supported") != "mixed":
    raise SystemExit("diagnostics receipt must record mixed f128 MachineModule support after runtime helper promotion")
if diagnostics_receipt.get("f128_runtime_fail_closed_rc12") is not False:
    raise SystemExit("diagnostics receipt must record that rounded f128 runtime rc=12 is no longer the active S5.19 guard")
if diagnostics_receipt.get("f128_machine_module_unsupported_details") != [
    "call_arity_gt_8",
]:
    raise SystemExit("diagnostics receipt must record specific f128 blocker details")
for field in [
    "legacy_fallback_for_unsupported_widths",
    "f128_overwide_call_shape_promoted",
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if diagnostics_receipt.get(field) is not False:
        raise SystemExit(f"diagnostics receipt must not overclaim {field}")
diagnostic_cases = {row.get("case_id"): row for row in diagnostics_receipt.get("cases", [])}
required_diagnostic_negative = {
    "reject_f128_overwide_arg_shape_native_v2": {"width": "f128", "detail": "call_arity_gt_8", "fragment": "call_arity_gt_8", "machine_module": True},
    "reject_i512_let_annotation_native_v2": {"width": "i512", "detail": "let annotation"},
    "reject_u512_cast_native_v2": {"width": "u512", "detail": "cast"},
}
required_diagnostic_positive = {
    "preserve_f128_rounded_tenths_add_helper_native_v2": {"width": "f128", "exit": 0},
    "preserve_i256_promoted_width_native_v2": {"width": "i256", "exit": 7},
}
if set(diagnostic_cases) != set(required_diagnostic_negative) | set(required_diagnostic_positive):
    raise SystemExit(f"diagnostics receipt cases mismatch: {sorted(diagnostic_cases)}")
for case_id, expected in required_diagnostic_negative.items():
    row = diagnostic_cases[case_id]
    expected_status = expected.get("status", "fail_closed")
    if row.get("status") != expected_status:
        raise SystemExit(f"{case_id} must fail closed")
    if row.get("unsupported_width") != expected["width"]:
        raise SystemExit(f"{case_id} expected unsupported width {expected['width']}, got {row.get('unsupported_width')}")
    if row.get("expected_detail") != expected["detail"]:
        raise SystemExit(f"{case_id} expected detail {expected['detail']}, got {row.get('expected_detail')}")
    if row.get("native_v2_compile_rc") == 0 and not expected.get("machine_module", False):
        raise SystemExit(f"{case_id} unexpectedly has rc=0")
    if row.get("diagnostic_fragment") != expected.get("fragment", "native-v2 S5 unsupported numeric width"):
        raise SystemExit(f"{case_id} missing stable diagnostic fragment")
    if expected_status == "runtime_fail_closed":
        if row.get("elf_emitted") is not True:
            raise SystemExit(f"{case_id} must emit a runtime fail-closed ELF")
        if row.get("run_rc") != expected.get("run_rc"):
            raise SystemExit(f"{case_id} runtime fail-closed rc mismatch")
    elif row.get("elf_emitted") is not False:
        raise SystemExit(f"{case_id} must not emit an ELF")
    if expected.get("machine_module", False):
        if row.get("machine_module_json_emitted") is not True:
            raise SystemExit(f"{case_id} must emit unsupported f128 MachineModule JSON")
        if expected_status == "runtime_fail_closed":
            if row.get("machine_module_supported") is not True:
                raise SystemExit(f"{case_id} runtime helper MachineModule must be supported")
            if row.get("expected_machine_opcode_found") is not True:
                raise SystemExit(f"{case_id} must prove runtime helper opcode")
        else:
            if row.get("machine_module_supported") is not False:
                raise SystemExit(f"{case_id} MachineModule must be unsupported")
            if row.get("machine_module_unsupported_detail") != expected["detail"]:
                raise SystemExit(f"{case_id} must record expected f128 blocker detail")
    elif row.get("machine_module_json_emitted") is not False:
        raise SystemExit(f"{case_id} must not emit MachineModule JSON")
    if row.get("segfault") is not False:
        raise SystemExit(f"{case_id} must not segfault")
    if row.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id} must not use legacy fallback")
    if not row.get("check_log_sha256") or not row.get("compile_log_sha256"):
        raise SystemExit(f"{case_id} missing diagnostic log hashes")
for case_id, expected in required_diagnostic_positive.items():
    row = diagnostic_cases[case_id]
    if row.get("status") != "promoted_width_preserved":
        raise SystemExit(f"{case_id} must preserve promoted-width status")
    if row.get("promoted_width") != expected["width"]:
        raise SystemExit(f"{case_id} expected promoted width {expected['width']}, got {row.get('promoted_width')}")
    if row.get("actual_exit") != expected["exit"]:
        raise SystemExit(f"{case_id} expected exit {expected['exit']}, got {row.get('actual_exit')}")
    if row.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id} must not use legacy fallback")
    if not row.get("machine_module_json_sha256") or not row.get("elf_sha256"):
        raise SystemExit(f"{case_id} missing MachineModule or ELF hashes")

if differential_receipt.get("schema") != "madaros.v2.s5.differential_receipt/0.1":
    raise SystemExit("bad S5 differential receipt schema")
if differential_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing differential receipt")
if differential_receipt.get("stage_contract_level") != "S5_11_NATIVE_V2_LEAN_SINGLE_DIFFERENTIAL_WITH_F128_PROMOTED_SURFACES":
    raise SystemExit("differential receipt must declare S5.11 promoted native-v2/lean_single stage contract")
if differential_receipt.get("case_count") != 87:
    raise SystemExit("differential receipt must contain exact 87 cases")
if differential_receipt.get("matched_case_count") != 79:
    raise SystemExit("differential receipt must contain exact 79 matched comparable cases")
if differential_receipt.get("reference_unavailable_case_count") != 8:
    raise SystemExit("differential receipt must contain exact eight reference-unavailable cases")
required_differential_flags = [
    "native_v2_vs_lean_single_differential_complete",
    "s5_differential_native_v2_lean_single_complete",
    "differential_native_v2_vs_lean_single_promoted",
    "all_reference_available_cases_match_exit_and_stdout",
    "all_native_v2_cases_compile_without_legacy_fallback",
    "all_native_v2_cases_return_expected_exit",
    "all_reference_available_lean_single_cases_return_expected_exit",
    "known_reference_unavailable_cases_recorded",
    "f128_promoted_surface_differentials_complete",
    "f128_arithmetic_value_contract_differential_complete",
    "f128_opaque_call_return_abi_differential_complete",
    "f128_sret_internal_arg_boundary_differential_complete",
    "f128_param_slot_layout_differential_complete",
]
for field in required_differential_flags:
    if differential_receipt.get(field) is not True:
        raise SystemExit(f"differential receipt missing required true flag: {field}")
for field in ["f128_promoted", "s5_ready", "s5_implemented", "s5_full_complete"]:
    if differential_receipt.get(field) is not False:
        raise SystemExit(f"differential receipt must not overclaim {field}")
required_differential_categories = {
    "scalar_i64",
    "scalar_bool",
    "normal_call_stack_args",
    "source_sret_local",
    "imported_sret_module_boundary",
    "method_sret",
    "f64_xmm0",
    "wide_int_source",
    "generic_aggregate_sret",
    "f128_arithmetic_value_contract",
    "f128_opaque_call_return_abi",
    "f128_sret_internal_arg_boundary",
    "f128_param_slot_layout",
}
if set(differential_receipt.get("categories_compared", [])) != required_differential_categories:
    raise SystemExit("differential receipt categories mismatch")
differential_cases = {row.get("case_id"): row for row in differential_receipt.get("cases", [])}
required_unavailable = {
    "f64_println_call_stdout_4_5",
    "f64_let_bound_println_stdout_4_5",
    "imported_f128_identity_arg_return",
    "imported_f128_return_only",
    "imported_f128_arg_i64_return",
    "imported_f128_plus_i64_arg_return",
    "imported_two_f128_args_return",
    "imported_two_f128_params_non_overlapping",
}
if {case_id for case_id, row in differential_cases.items() if row.get("status") == "reference_unavailable"} != required_unavailable:
    raise SystemExit("differential receipt reference-unavailable cases mismatch")
required_scalar_differential_cases = {
    "scalar_i64_literal_return_42",
    "scalar_i64_direct_call_return_42",
    "scalar_bool_direct_call_return_1",
}
required_wide_source_differential_cases = {
    "source_i128_mul_gt",
    "source_i256_mul_gt",
    "source_u128_mul_add_gt",
    "source_u256_mul_add_ne",
    "source_i128_sub_eq_zero",
    "source_i256_add_eq",
}
required_f128_arithmetic_differential_cases = {
    "f128_add_one_two_to_three",
    "f128_mul_one_two_to_two",
    "f128_add_half_half_to_one",
    "f128_div_one_two_to_half",
    "f128_chain_add_sub_to_one",
    "f128_add_half_one_to_one_and_half",
    "f128_mul_half_half_to_quarter",
    "f128_mul_one_and_half_half_to_three_quarters",
    "f128_add_quarter_one_to_one_and_quarter",
    "f128_sub_half_one_to_negative_half",
    "f128_add_negative_half_one_to_half",
    "f128_add_negative_half_negative_half_to_negative_one",
    "f128_mul_negative_half_half_to_negative_quarter",
    "f128_div_negative_one_two_to_negative_half",
    "f128_call_literal_return_then_add_to_three",
    "f128_call_identity_return_then_add_to_three",
    "f128_call_pick_first_return_then_add_to_three",
    "f128_call_pick_second_return_then_add_to_three",
    "f128_add_rounded_tenths_runtime_helper_to_sum",
    "f128_add_one_tenth_seven_tenths_runtime_helper_to_binary_sum",
    "f128_add_two_tenths_seven_tenths_runtime_helper_to_binary_sum",
    "f128_add_three_tenths_six_tenths_runtime_helper_to_binary_sum",
    "f128_add_six_tenths_seven_tenths_runtime_helper_to_binary_sum",
    "f128_add_nine_tenths_two_tenths_runtime_helper_to_literal_equivalent_sum",
    "f128_add_hundredth_thousandth_runtime_helper_to_binary_sum",
    "f128_add_one_point_2345_thousandth_runtime_helper_to_binary_sum",
    "f128_callee_add_args_runtime_helper_to_three",
    "f128_callee_sub_args_runtime_helper_to_one",
    "f128_callee_mul_args_runtime_helper_to_three_quarters",
    "f128_callee_div_args_runtime_helper_to_half",
    "f128_callee_div_negative_args_runtime_helper_to_negative_half",
}
required_f128_abi_differential_cases = {
    "local_f128_identity_arg_return",
    "local_f128_return_only",
    "local_f128_arg_i64_return",
    "imported_f128_identity_arg_return",
    "imported_f128_return_only",
    "imported_f128_arg_i64_return",
    "imported_f128_plus_i64_arg_return",
    "imported_two_f128_args_return",
    "local_f128_plus_i64_arg_return",
    "local_i64_plus_f128_arg_return",
    "local_two_f128_args_return",
    "local_mixed_arg_f128_return",
    "local_four_f128_args_stack_return",
    "local_five_f128_args_deeper_stack_return",
    "f128_rounded_decimal_arithmetic_runtime_helper_return",
}
required_f128_sret_differential_cases = {
    "direct_f128_then_i64_arithmetic_classifier_guard",
    "sret_f128_arg_then_i64_arithmetic",
    "sret_f128_arg_copied_to_f128_field_payload",
    "sret_three_f128_args_crosses_stack_boundary",
}
required_f128_param_differential_cases = {
    "local_two_f128_params_non_overlapping",
    "local_f128_i64_f128_params_non_overlapping",
    "imported_two_f128_params_non_overlapping",
    "f128_callee_add_args_slot_layout_feeds_runtime_helper",
}
required_matched_cases = (
    required_scalar_differential_cases
    | {"normal_call_stack_one_arg_return_28", "normal_call_stack_two_arg_return_36"}
    | {
        "source_sret_local_i64_triple_return_14",
        "source_sret_local_register_multi_arg_return_43",
        "source_sret_local_stack_one_arg_return_49",
        "source_sret_local_stack_two_arg_return_57",
    }
    | {
        "imported_sret_one_arg_return_29",
        "imported_sret_register_multi_arg_return_43",
        "imported_sret_stack_two_arg_return_57",
    }
    | {
        "method_sret_receiver_only_return_24",
        "method_sret_receiver_register_args_return_43",
        "method_sret_receiver_stack_args_return_57",
    }
    | {
        "f64_cast_literal_to_i64_return_4",
        "f64_fractional_binop_cast_return_50",
        "f64_return_compare_exit_45",
        "f64_mixed_args_return_compare_exit_55",
        "f64_print_call_stdout_4_5",
    }
    | required_wide_source_differential_cases
    | set(required_generic_cases)
    | required_f128_arithmetic_differential_cases
    | required_f128_abi_differential_cases
    | required_f128_sret_differential_cases
    | required_f128_param_differential_cases
) - required_unavailable
if {case_id for case_id, row in differential_cases.items() if row.get("status") == "matched"} != required_matched_cases:
    raise SystemExit("differential receipt matched case set mismatch")
for case_id, row in differential_cases.items():
    if row.get("machine_module_legacy_fallback") is not False:
        raise SystemExit(f"{case_id} differential case used legacy fallback")
    if not row.get("machine_module_json_sha256") or not row.get("elf_sha256"):
        raise SystemExit(f"{case_id} differential case missing MachineModule or ELF hash")
    if row.get("native_v2_exit") != row.get("expected_exit"):
        raise SystemExit(f"{case_id} differential native-v2 exit mismatch")
    if row.get("status") == "matched" and row.get("lean_single_exit") != row.get("expected_exit"):
        raise SystemExit(f"{case_id} differential lean_single exit mismatch")
    if row.get("status") == "matched" and row.get("stdout_equal") is not True:
        raise SystemExit(f"{case_id} matched differential case must have equal stdout")
    if row.get("status") == "reference_unavailable":
        if row.get("lean_single_exit") == row.get("expected_exit") and row.get("stdout_equal") is True:
            raise SystemExit(f"{case_id} unavailable differential case unexpectedly matches")

if effect_receipt.get("schema") != "madaros.v2.s5.mir_effect_roundtrip/0.1":
    raise SystemExit("bad S5 MIR-effect receipt schema")
if effect_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing MIR-effect receipt")
if effect_receipt.get("s5_mir_effect_roundtrip_complete") is not True:
    raise SystemExit("program MIR/ABI gate requires completed MIR-effect roundtrip")
for false_field in [
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
    "real_program_mir_emitted",
    "real_abi_layout_emitted",
]:
    if effect_receipt.get(false_field) is not False:
        raise SystemExit(f"MIR-effect input must not overclaim {false_field}")

native_rows = list(effect_module.get("scalar_native_witnesses", []))
required_cases = {
    "scalar_i64_literal_return_42",
    "scalar_i64_direct_call_return_42",
    "scalar_bool_direct_call_return_1",
}
case_ids = [row.get("case_id") for row in native_rows]
if set(case_ids) != required_cases or len(case_ids) != len(required_cases):
    raise SystemExit(f"program MIR/ABI gate requires exact scalar cases {sorted(required_cases)}, got {sorted(case_ids)}")

specs: dict[str, dict[str, Any]] = {
    "scalar_i64_literal_return_42": {
        "expected_function_count": 1,
        "expected_internal_call_count": 1,
        "program_kind": "scalar_i64_literal_return",
        "abi_kind": "scalar_i64_return",
        "entry_function_legal_mir_ops": [
            "MOV_IMM",
            "STORE_STACK",
            "LOAD_STACK",
            "RET",
        ],
        "call_boundary_ops": ["RET"],
        "abi_signature": {
            "params": [],
            "return": {"type": "i64", "class": "scalar_i64", "register": "rax"},
            "arg_registers_used": [],
            "stack_arg_count": 0,
            "sret": False,
            "aggregate_layout": False,
        },
    },
    "scalar_i64_direct_call_return_42": {
        "expected_function_count": 2,
        "expected_internal_call_count": 2,
        "program_kind": "scalar_i64_direct_call_return",
        "abi_kind": "scalar_i64_direct_call_return",
        "entry_function_legal_mir_ops": [
            "LOAD_STACK",
            "ARG_MOVE",
            "CALL",
            "CAPTURE_RET",
            "STORE_STACK",
            "LOAD_STACK",
            "RET",
        ],
        "call_boundary_ops": ["ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "RET"],
        "abi_signature": {
            "params": [{"type": "i64", "class": "scalar_i64", "register": "rdi"}],
            "return": {"type": "i64", "class": "scalar_i64", "register": "rax"},
            "arg_registers_used": ["rdi"],
            "stack_arg_count": 0,
            "sret": False,
            "aggregate_layout": False,
        },
    },
    "scalar_bool_direct_call_return_1": {
        "expected_function_count": 2,
        "expected_internal_call_count": 2,
        "program_kind": "scalar_bool_direct_call_return",
        "abi_kind": "scalar_bool_direct_call_return",
        "entry_function_legal_mir_ops": [
            "LOAD_STACK",
            "ARG_MOVE",
            "CALL",
            "CAPTURE_RET",
            "STORE_STACK",
            "LOAD_STACK",
            "RET",
        ],
        "call_boundary_ops": ["ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "RET"],
        "abi_signature": {
            "params": [{"type": "i64", "class": "scalar_i64", "register": "rdi"}],
            "return": {"type": "bool", "class": "scalar_bool", "register": "rax", "canonical_values": [0, 1]},
            "arg_registers_used": ["rdi"],
            "stack_arg_count": 0,
            "sret": False,
            "aggregate_layout": False,
        },
    },
}

source_receipt_rows: dict[str, dict[str, Any]] = {}
receipt_lines = source_receipts_path.read_text(encoding="utf-8").splitlines()
if not receipt_lines:
    raise SystemExit("missing canonical S5 source receipt results")
receipt_header = receipt_lines[0].split("\t")
required_receipt_header = [
    "case_id",
    "program",
    "expected_exit",
    "s5_receipt_sha256",
    "s5_receipt_path",
    "status",
]
if receipt_header != required_receipt_header:
    raise SystemExit(f"bad canonical S5 receipt TSV header: {receipt_header}")
for line in receipt_lines[1:]:
    cols = line.split("\t")
    if len(cols) != len(receipt_header):
        raise SystemExit(f"bad canonical S5 receipt TSV row: {line!r}")
    row = dict(zip(receipt_header, cols, strict=True))
    case_id = row["case_id"]
    if case_id in source_receipt_rows:
        raise SystemExit(f"duplicate canonical S5 receipt case: {case_id}")
    if row["status"] != "ok":
        raise SystemExit(f"canonical S5 receipt row did not pass: {case_id}")
    receipt_json_path = Path(row["s5_receipt_path"])
    if not receipt_json_path.is_file():
        raise SystemExit(f"canonical S5 receipt path missing for {case_id}: {receipt_json_path}")
    digest = hashlib.sha256(receipt_json_path.read_bytes()).hexdigest()
    if digest != row["s5_receipt_sha256"]:
        raise SystemExit(f"canonical S5 receipt file hash mismatch for {case_id}")
    payload = load_json(receipt_json_path)
    if payload.get("schema") != "madaros.v2.s5.receipt/0.1":
        raise SystemExit(f"bad canonical S5 receipt schema for {case_id}")
    if payload.get("status") != "pass":
        raise SystemExit(f"canonical S5 receipt not passing for {case_id}")
    if payload.get("case_id") != case_id:
        raise SystemExit(f"canonical S5 receipt case_id mismatch for {case_id}")
    if payload.get("source") != row["program"]:
        raise SystemExit(f"canonical S5 receipt source mismatch for {case_id}")
    if int(payload.get("expected_exit")) != int(row["expected_exit"]):
        raise SystemExit(f"canonical S5 receipt expected_exit mismatch for {case_id}")
    if payload.get("actual_exit") != payload.get("expected_exit"):
        raise SystemExit(f"canonical S5 receipt actual_exit mismatch for {case_id}")
    for false_field in [
        "s5_mir_abi_boundary_complete",
        "s5_ready",
        "s5_implemented",
        "s5_full_complete",
        "real_abi_layout_emitted",
    ]:
        if payload.get(false_field) is not False:
            raise SystemExit(f"canonical S5 receipt must not overclaim {false_field} for {case_id}")
    for true_field in [
        "compiler_machine_module_exported",
        "real_program_mir_emitted",
        "s5_compiler_machine_module_export_slice_complete",
    ]:
        if payload.get(true_field) is not True:
            raise SystemExit(f"canonical S5 receipt must prove {true_field} for {case_id}")
    source_receipt_rows[case_id] = {
        "row": row,
        "payload": payload,
        "path": receipt_json_path,
        "stable_path": f"{receipt_json_path.parent.name}/{receipt_json_path.name}",
    }

if set(source_receipt_rows) != required_cases:
    raise SystemExit(
        "canonical S5 source receipts must cover exact scalar cases: "
        f"{sorted(required_cases)}, got {sorted(source_receipt_rows)}"
    )

programs: list[dict[str, Any]] = []
for row in sorted(native_rows, key=lambda item: item["case_id"]):
    case_id = row["case_id"]
    spec = specs[case_id]
    source_receipt = source_receipt_rows[case_id]["payload"]
    if row["abi_kind"] != spec["abi_kind"]:
        raise SystemExit(f"ABI kind mismatch for {case_id}: {row['abi_kind']} != {spec['abi_kind']}")
    if row["actual_exit"] != row["expected_exit"]:
        raise SystemExit(f"native exit mismatch for {case_id}")
    if source_receipt["program_kind"] != spec["program_kind"]:
        raise SystemExit(f"canonical S5 receipt program kind mismatch for {case_id}")
    if source_receipt["abi_kind"] != spec["abi_kind"]:
        raise SystemExit(f"canonical S5 receipt ABI kind mismatch for {case_id}")
    if source_receipt["actual_exit"] != int(row["actual_exit"]):
        raise SystemExit(f"canonical S5 receipt native exit mismatch for {case_id}")
    internal_calls = int(row["elf_internal_call_count"])
    if internal_calls != spec["expected_internal_call_count"]:
        raise SystemExit(
            f"internal call count mismatch for {case_id}: "
            f"{internal_calls} != {spec['expected_internal_call_count']}"
        )
    if int(source_receipt["native_v2_compile"]["elf_internal_call_count"]) != internal_calls:
        raise SystemExit(f"canonical S5 receipt internal-call evidence mismatch for {case_id}")
    if int(row["elf_ret_count"]) < 1 or int(row["elf_syscall_count"]) < 1:
        raise SystemExit(f"ELF ret/syscall evidence missing for {case_id}")
    compile_log = effect_dir / "native_scalar_witnesses" / f"{case_id}.compile.log"
    function_count = parse_function_count(compile_log)
    if function_count != spec["expected_function_count"]:
        raise SystemExit(f"Merged IR function count mismatch for {case_id}: {function_count}")
    if int(source_receipt["merged_ir_function_count"]) != function_count:
        raise SystemExit(f"canonical S5 receipt merged-IR function count mismatch for {case_id}")
    declared_ops = list(filter(None, row["required_machine_ops"].split(",")))
    if sorted(declared_ops) != sorted(spec["call_boundary_ops"]):
        raise SystemExit(f"machine op contract mismatch for {case_id}")

    program = {
        "schema": "madaros.v2.s5.program_mir_abi_program/0.1",
        "case_id": case_id,
        "program": row["program"],
        "program_kind": spec["program_kind"],
        "expected_exit": int(row["expected_exit"]),
        "actual_exit": int(row["actual_exit"]),
        "merged_ir_function_count": function_count,
        "program_mir_schema": "madaros.v2.s5.program_mir_shadow/0.1",
        "program_mir_source": "compiler_exported_machine_module_json",
        "compiler_machine_module_exported": True,
        "machine_module_schema": source_receipt["machine_module_schema"],
        "machine_module_path": source_receipt["machine_module_path"],
        "machine_module_json_sha256": source_receipt["machine_module_json_sha256"],
        "entry_function_legal_mir_ops": spec["entry_function_legal_mir_ops"],
        "call_boundary_ops": spec["call_boundary_ops"],
        "machine_ir_contract_source": "self-hosted/native/machine_ir.sio:native_v2_lower_legal_function_from_ir_ref",
        "codegen_contract_source": "self-hosted/native/codegen_x86_linux.sio:native_v2_emit_machine_instr",
        "abi_schema": "madaros.v2.s5.abi_scalar_call_return/0.1",
        "abi_signature": spec["abi_signature"],
        "native_v2_compile": {
            "elf_sha256": row["elf_sha256"],
            "compile_log_sha256": row["compile_log_sha256"],
            "stdout_sha256": row["stdout_sha256"],
            "stderr_sha256": row["stderr_sha256"],
            "elf_internal_call_count": internal_calls,
            "elf_ret_count": int(row["elf_ret_count"]),
            "elf_syscall_count": int(row["elf_syscall_count"]),
        },
        "canonical_s5_source_receipt": {
            "schema": source_receipt["schema"],
            "path": source_receipt_rows[case_id]["stable_path"],
            "file_sha256": source_receipt_rows[case_id]["row"]["s5_receipt_sha256"],
            "receipt_sha256": source_receipt["receipt_sha256"],
            "stage_contract_level": source_receipt["stage_contract_level"],
            "machine_module_json_sha256": source_receipt["machine_module_json_sha256"],
        },
    }
    program["program_shadow_sha256"] = sha256_text(stable_json(program))
    programs.append(program)

not_promoted = [
    {
        "surface": "f128_arbitrary_decimal_binary128_materialization",
        "status": "not_promoted_beyond_bounded_siglo_scale18_two_limb_scale0_to_18_and_explicit_truncated_value_contract_cases",
        "reason": "native-v2 now emits exact dyadic, bounded sig_hi=0/no-truncation/scale10<=18 rounded decimals, algorithmic two-limb no-truncation scale0..18 decimals, explicit truncated high-precision value-contract cases, and explicit subnormal/underflow/overflow anchors; large-scale arbitrary decimal-to-binary128 materialization remains fail-closed",
    },
    {
        "surface": "f128_generic_ieee_software_helper_semantics",
        "status": "not_promoted",
        "reason": "source-observable binary128 class-code helper is promoted for zero/subnormal/normal/infinity/NaN via a canonical quiet-NaN constructor; generic IEEE arithmetic, rounding-mode helper semantics, and differentials remain blockers",
    },
    {
        "surface": "f128_external_sysv_abi_and_sret",
        "status": "not_promoted",
        "reason": "internal native-v2 f128 direct call/return and SRET-arg-boundary receipts are promoted; external SysV f128 ABI/SRET compatibility remains outside the promoted S5 surface",
    },
    {
        "surface": "s4_negative_and_producer_dependent_rewrites",
        "status": "not_promoted",
        "reason": "S4 negative and producer-dependent blocked controls remain explicitly unselected by the S5 application plan",
    },
]

negative_and_blocked_controls = [
    {
        "case_id": "reject_distinct_symbolic_cmp_i64",
        "source": "tests/madaros/v2_s4/reject_distinct_symbolic_cmp_i64.sio",
        "class": "semantic_negative",
        "expected_status": "rejected_not_selected_not_promoted",
    },
    {
        "case_id": "reject_distinct_symbolic_sub_i64",
        "source": "tests/madaros/v2_s4/reject_distinct_symbolic_sub_i64.sio",
        "class": "semantic_negative",
        "expected_status": "rejected_not_selected_not_promoted",
    },
    {
        "case_id": "reject_div_self_zero",
        "source": "tests/madaros/v2_s4/reject_div_self_zero.sio",
        "class": "semantic_negative",
        "expected_status": "rejected_not_selected_not_promoted",
    },
    {
        "case_id": "reject_call_result_self_cmp_i64",
        "source": "tests/madaros/v2_s4/reject_call_result_self_cmp_i64.sio",
        "class": "producer_evaluation_blocker",
        "expected_status": "blocked_not_selected_not_promoted",
    },
    {
        "case_id": "reject_call_result_sub_self_i64",
        "source": "tests/madaros/v2_s4/reject_call_result_sub_self_i64.sio",
        "class": "producer_evaluation_blocker",
        "expected_status": "blocked_not_selected_not_promoted",
    },
]

module = {
    "schema": "madaros.v2.s5.program_mir_abi_module/0.1",
    "stage_contract_level": "S5_SCALAR_MACHINE_MODULE_EXPORT_WITH_ABI_SHADOW_NOT_FULL",
    "input_mir_effect_schema": effect_receipt["schema"],
    "input_mir_effect_sha256": effect_receipt["receipt_sha256"],
    "input_boundary_sha256": effect_receipt["input_boundary_sha256"],
    "input_applied_extraction_contract": effect_receipt["input_applied_extraction_contract"],
    "input_applied_extraction_sha256": effect_receipt["input_applied_extraction_sha256"],
    "s4_applied_extraction_consumed": True,
    "input_effect_count": effect_receipt["effect_count"],
    "input_selected_rewrite_count": effect_receipt["selected_rewrite_count"],
    "program_count": len(programs),
    "programs": programs,
    "program_kinds": sorted({program["program_kind"] for program in programs}),
    "abi_kinds": sorted({program["abi_signature"]["return"]["class"] for program in programs}),
    "canonical_s5_source_receipt_count": len(source_receipt_rows),
    "canonical_s5_source_receipts_present": True,
    "canonical_s5_source_receipts": [
        {
            "case_id": case_id,
            "source": source_receipt_rows[case_id]["row"]["program"],
            "path": source_receipt_rows[case_id]["stable_path"],
            "file_sha256": source_receipt_rows[case_id]["row"]["s5_receipt_sha256"],
            "receipt_sha256": source_receipt_rows[case_id]["payload"]["receipt_sha256"],
            "stage_contract_level": source_receipt_rows[case_id]["payload"]["stage_contract_level"],
        }
        for case_id in sorted(source_receipt_rows)
    ],
    "program_mir_shadow_serialized": True,
    "compiler_machine_module_exported": True,
    "real_program_mir_emitted": True,
    "real_abi_layout_emitted": True,
    "sret_abi_receipt": {
        "schema": sret_receipt["schema"],
        "path": f"{sret_receipt_path.parent.name}/{sret_receipt_path.name}",
        "receipt_sha256": sret_receipt["receipt_sha256"],
        "stage_contract_level": sret_receipt["stage_contract_level"],
        "case_id": sret_receipt["case_id"],
        "positive_exit": sret_receipt["positive"]["actual_exit"],
        "negative_plaincall_exit": sret_receipt["negative_plaincall"]["actual_exit"],
        "abi_signature": sret_receipt["abi_signature"],
    },
    "source_sret_receipt": {
        "schema": source_sret_receipt["schema"],
        "path": f"{source_sret_receipt_path.parent.name}/{source_sret_receipt_path.name}",
        "receipt_sha256": source_sret_receipt["receipt_sha256"],
        "stage_contract_level": source_sret_receipt["stage_contract_level"],
        "case_id": source_sret_receipt["case_id"],
        "case_count": source_sret_receipt["case_count"],
        "one_arg_case_id": source_sret_receipt["one_arg_case_id"],
        "one_arg_actual_exit": source_sret_receipt["one_arg_actual_exit"],
        "register_multi_arg_case_id": source_sret_receipt["register_multi_arg_case_id"],
        "register_multi_arg_actual_exit": source_sret_receipt["register_multi_arg_actual_exit"],
        "stack_one_arg_case_id": source_sret_receipt["stack_one_arg_case_id"],
        "stack_one_arg_actual_exit": source_sret_receipt["stack_one_arg_actual_exit"],
        "stack_two_arg_case_id": source_sret_receipt["stack_two_arg_case_id"],
        "stack_two_arg_actual_exit": source_sret_receipt["stack_two_arg_actual_exit"],
        "cases": source_sret_receipt["cases"],
    },
    "stack_call_receipt": {
        "schema": stack_call_receipt["schema"],
        "path": f"{stack_call_receipt_path.parent.name}/{stack_call_receipt_path.name}",
        "receipt_sha256": stack_call_receipt["receipt_sha256"],
        "stage_contract_level": stack_call_receipt["stage_contract_level"],
        "case_id": stack_call_receipt["case_id"],
        "case_count": stack_call_receipt["case_count"],
        "cases": stack_call_receipt["cases"],
    },
    "imported_sret_receipt": {
        "schema": imported_sret_receipt["schema"],
        "path": f"{imported_sret_receipt_path.parent.name}/{imported_sret_receipt_path.name}",
        "receipt_sha256": imported_sret_receipt["receipt_sha256"],
        "stage_contract_level": imported_sret_receipt["stage_contract_level"],
        "case_id": imported_sret_receipt["case_id"],
        "case_count": imported_sret_receipt["case_count"],
        "cases": imported_sret_receipt["cases"],
    },
    "method_sret_receipt": {
        "schema": method_sret_receipt["schema"],
        "path": f"{method_sret_receipt_path.parent.name}/{method_sret_receipt_path.name}",
        "receipt_sha256": method_sret_receipt["receipt_sha256"],
        "stage_contract_level": method_sret_receipt["stage_contract_level"],
        "case_id": method_sret_receipt["case_id"],
        "case_count": method_sret_receipt["case_count"],
        "cases": method_sret_receipt["cases"],
    },
    "f64_xmm0_receipt": {
        "schema": f64_xmm0_receipt["schema"],
        "path": f"{f64_xmm0_receipt_path.parent.name}/{f64_xmm0_receipt_path.name}",
        "receipt_sha256": f64_xmm0_receipt["receipt_sha256"],
        "stage_contract_level": f64_xmm0_receipt["stage_contract_level"],
        "case_id": f64_xmm0_receipt["case_id"],
        "case_count": f64_xmm0_receipt["case_count"],
        "cases": f64_xmm0_receipt["cases"],
    },
    "wide_int_receipt": {
        "schema": wide_int_receipt["schema"],
        "path": f"{wide_int_receipt_path.parent.name}/{wide_int_receipt_path.name}",
        "receipt_sha256": wide_int_receipt["receipt_sha256"],
        "stage_contract_level": wide_int_receipt["stage_contract_level"],
        "case_id": wide_int_receipt["case_id"],
        "case_count": wide_int_receipt["case_count"],
        "check_ok_case_count": wide_int_receipt["check_ok_case_count"],
        "check_reject_case_count": wide_int_receipt["check_reject_case_count"],
        "source_native_case_count": wide_int_receipt["source_native_case_count"],
        "native_emit_case_count": wide_int_receipt["native_emit_case_count"],
        "cases": wide_int_receipt["cases"],
    },
    "wide_machine_slot_metadata_receipt": {
        "schema": wide_machine_slot_receipt["schema"],
        "path": f"{wide_machine_slot_receipt_path.parent.name}/{wide_machine_slot_receipt_path.name}",
        "receipt_sha256": wide_machine_slot_receipt["receipt_sha256"],
        "stage_contract_level": wide_machine_slot_receipt["stage_contract_level"],
        "case_id": wide_machine_slot_receipt["case_id"],
        "case_count": wide_machine_slot_receipt["case_count"],
        "cases": wide_machine_slot_receipt["cases"],
        "wide_machine_slot_metadata_complete": wide_machine_slot_receipt["wide_machine_slot_metadata_complete"],
        "wide_i256_u256_machine_slots_promoted": wide_machine_slot_receipt["wide_i256_u256_machine_slots_promoted"],
    },
    "wide_abi_call_return_receipt": {
        "schema": wide_abi_call_return_receipt["schema"],
        "path": f"{wide_abi_call_return_receipt_path.parent.name}/{wide_abi_call_return_receipt_path.name}",
        "receipt_sha256": wide_abi_call_return_receipt["receipt_sha256"],
        "stage_contract_level": wide_abi_call_return_receipt["stage_contract_level"],
        "case_id": wide_abi_call_return_receipt["case_id"],
        "case_count": wide_abi_call_return_receipt["case_count"],
        "i256_case_count": wide_abi_call_return_receipt["i256_case_count"],
        "u256_case_count": wide_abi_call_return_receipt["u256_case_count"],
        "two_wide_arg_case_count": wide_abi_call_return_receipt["two_wide_arg_case_count"],
        "imported_module_case_count": wide_abi_call_return_receipt["imported_module_case_count"],
        "public_native_imported_case_count": wide_abi_call_return_receipt["public_native_imported_case_count"],
        "cases": wide_abi_call_return_receipt["cases"],
        "s5_wide_i256_u256_local_abi_call_return_complete": wide_abi_call_return_receipt["s5_wide_i256_u256_local_abi_call_return_complete"],
        "s5_wide_i256_u256_imported_abi_call_return_complete": wide_abi_call_return_receipt["s5_wide_i256_u256_imported_abi_call_return_complete"],
        "wide_i256_u256_local_abi_call_return_promoted": wide_abi_call_return_receipt["wide_i256_u256_local_abi_call_return_promoted"],
        "wide_i256_u256_imported_abi_call_return_promoted": wide_abi_call_return_receipt["wide_i256_u256_imported_abi_call_return_promoted"],
        "imported_module_wide_abi_promoted": wide_abi_call_return_receipt["imported_module_wide_abi_promoted"],
        "public_native_imported_route_checked": wide_abi_call_return_receipt["public_native_imported_route_checked"],
        "public_native_imported_route_uses_full_modular_native_v2": wide_abi_call_return_receipt["public_native_imported_route_uses_full_modular_native_v2"],
        "stale_compact_modular_ir_table_path_blocked": wide_abi_call_return_receipt["stale_compact_modular_ir_table_path_blocked"],
    },
    "generic_aggregate_sret_receipt": {
        "schema": generic_agg_receipt["schema"],
        "path": f"{generic_agg_receipt_path.parent.name}/{generic_agg_receipt_path.name}",
        "receipt_sha256": generic_agg_receipt["receipt_sha256"],
        "stage_contract_level": generic_agg_receipt["stage_contract_level"],
        "case_id": generic_agg_receipt["case_id"],
        "case_count": generic_agg_receipt["case_count"],
        "cases": generic_agg_receipt["cases"],
    },
    "diagnostics_receipt": {
        "schema": diagnostics_receipt["schema"],
        "path": f"{diagnostics_receipt_path.parent.name}/{diagnostics_receipt_path.name}",
        "receipt_sha256": diagnostics_receipt["receipt_sha256"],
        "stage_contract_level": diagnostics_receipt["stage_contract_level"],
        "case_id": diagnostics_receipt["case_id"],
        "case_count": diagnostics_receipt["case_count"],
        "negative_case_count": diagnostics_receipt["negative_case_count"],
        "positive_guard_case_count": diagnostics_receipt["positive_guard_case_count"],
        "cases": diagnostics_receipt["cases"],
    },
    "f128_literal_provenance_receipt": {
        "schema": f128_literal_provenance_receipt["schema"],
        "path": f"{f128_literal_provenance_receipt_path.parent.name}/{f128_literal_provenance_receipt_path.name}",
        "receipt_sha256": f128_literal_provenance_receipt["receipt_sha256"],
        "stage_contract_level": f128_literal_provenance_receipt["stage_contract_level"],
        "case_id": f128_literal_provenance_receipt["case_id"],
        "parser_source_sha256": f128_literal_provenance_receipt["parser_source_sha256"],
        "ast_source_sha256": f128_literal_provenance_receipt["ast_source_sha256"],
        "checker_source_sha256": f128_literal_provenance_receipt["checker_source_sha256"],
        "parse_float_literal_block_sha256": f128_literal_provenance_receipt["parse_float_literal_block_sha256"],
        "probe_source_sha256": f128_literal_provenance_receipt["probe_source_sha256"],
        "probe_check_rc": f128_literal_provenance_receipt["probe_check_rc"],
        "f128_literal_decimal_digit_count": f128_literal_provenance_receipt["f128_literal_decimal_digit_count"],
        "f128_literal_decimal_scale10": f128_literal_provenance_receipt["f128_literal_decimal_scale10"],
        "f128_literal_decimal_metadata_independent_from_f64": f128_literal_provenance_receipt["f128_literal_decimal_metadata_independent_from_f64"],
        "f128_type_system_awareness_promoted": f128_literal_provenance_receipt["f128_type_system_awareness_promoted"],
    },
    "f128_binary128_value_receipt": {
        "schema": f128_binary128_value_receipt["schema"],
        "path": f"{f128_binary128_value_receipt_path.parent.name}/{f128_binary128_value_receipt_path.name}",
        "receipt_sha256": f128_binary128_value_receipt["receipt_sha256"],
        "stage_contract_level": f128_binary128_value_receipt["stage_contract_level"],
        "case_id": f128_binary128_value_receipt["case_id"],
        "case_count": f128_binary128_value_receipt["case_count"],
        "rounding_mode": f128_binary128_value_receipt["rounding_mode"],
        "target_format": f128_binary128_value_receipt["target_format"],
        "cases": f128_binary128_value_receipt["cases"],
        "f128_binary128_value_contract_complete": f128_binary128_value_receipt["f128_binary128_value_contract_complete"],
    },
    "f128_literal_value_bridge_receipt": {
        "schema": f128_literal_value_bridge_receipt["schema"],
        "path": f"{f128_literal_value_bridge_receipt_path.parent.name}/{f128_literal_value_bridge_receipt_path.name}",
        "receipt_sha256": f128_literal_value_bridge_receipt["receipt_sha256"],
        "stage_contract_level": f128_literal_value_bridge_receipt["stage_contract_level"],
        "case_count": f128_literal_value_bridge_receipt["case_count"],
        "cases": f128_literal_value_bridge_receipt["cases"],
        "f128_literal_value_bridge_promoted": f128_literal_value_bridge_receipt["f128_literal_value_bridge_promoted"],
        "f128_literal_decimal_metadata_bridged_to_ir": f128_literal_value_bridge_receipt["f128_literal_decimal_metadata_bridged_to_ir"],
        "f128_literal_decimal_metadata_bridged_to_machine_ir": f128_literal_value_bridge_receipt["f128_literal_decimal_metadata_bridged_to_machine_ir"],
        "f128_literal_decimal_metadata_bridged_to_machine_module": f128_literal_value_bridge_receipt["f128_literal_decimal_metadata_bridged_to_machine_module"],
    },
    "machine_slot_metadata_receipt": {
        "schema": machine_slot_metadata_receipt["schema"],
        "path": f"{machine_slot_metadata_receipt_path.parent.name}/{machine_slot_metadata_receipt_path.name}",
        "receipt_sha256": machine_slot_metadata_receipt["receipt_sha256"],
        "stage_contract_level": machine_slot_metadata_receipt["stage_contract_level"],
        "case_id": machine_slot_metadata_receipt["case_id"],
        "case_count": machine_slot_metadata_receipt["case_count"],
        "slot_metadata_schema": machine_slot_metadata_receipt["slot_metadata_schema"],
        "slot_kinds_seen": machine_slot_metadata_receipt["slot_kinds_seen"],
        "f128_binary128_limb_count": machine_slot_metadata_receipt["f128_binary128_limb_count"],
        "f128_binary128_limb_bits": machine_slot_metadata_receipt["f128_binary128_limb_bits"],
        "cases": machine_slot_metadata_receipt["cases"],
    },
    "f128_abi_metadata_receipt": {
        "schema": f128_abi_metadata_receipt["schema"],
        "path": f"{f128_abi_metadata_receipt_path.parent.name}/{f128_abi_metadata_receipt_path.name}",
        "receipt_sha256": f128_abi_metadata_receipt["receipt_sha256"],
        "stage_contract_level": f128_abi_metadata_receipt["stage_contract_level"],
        "case_id": f128_abi_metadata_receipt["case_id"],
        "case_count": f128_abi_metadata_receipt["case_count"],
        "imported_module_case_count": f128_abi_metadata_receipt["imported_module_case_count"],
        "f128_binary128_slot_kind": f128_abi_metadata_receipt["f128_binary128_slot_kind"],
        "f128_binary128_width_words": f128_abi_metadata_receipt["f128_binary128_width_words"],
        "f128_sysv_classes": f128_abi_metadata_receipt["f128_sysv_classes"],
        "cases": f128_abi_metadata_receipt["cases"],
    },
    "f128_native_opaque_storage_receipt": {
        "schema": f128_native_opaque_storage_receipt["schema"],
        "path": f"{f128_native_opaque_storage_receipt_path.parent.name}/{f128_native_opaque_storage_receipt_path.name}",
        "receipt_sha256": f128_native_opaque_storage_receipt["receipt_sha256"],
        "stage_contract_level": f128_native_opaque_storage_receipt["stage_contract_level"],
        "claims": f128_native_opaque_storage_receipt["claims"],
        "cases": f128_native_opaque_storage_receipt["cases"],
    },
    "f128_opaque_call_return_abi_receipt": {
        "schema": f128_opaque_call_return_abi_receipt["schema"],
        "path": f"{f128_opaque_call_return_abi_receipt_path.parent.name}/{f128_opaque_call_return_abi_receipt_path.name}",
        "receipt_sha256": f128_opaque_call_return_abi_receipt["receipt_sha256"],
        "stage_contract_level": f128_opaque_call_return_abi_receipt["stage_contract_level"],
        "case_id": f128_opaque_call_return_abi_receipt["case_id"],
        "case_count": f128_opaque_call_return_abi_receipt["case_count"],
        "positive_case_count": f128_opaque_call_return_abi_receipt["positive_case_count"],
        "negative_case_count": f128_opaque_call_return_abi_receipt["negative_case_count"],
        "f128_opaque_imported_direct_call_return_abi_promoted": f128_opaque_call_return_abi_receipt["f128_opaque_imported_direct_call_return_abi_promoted"],
        "f128_native_internal_call_abi_promoted": f128_opaque_call_return_abi_receipt["f128_native_internal_call_abi_promoted"],
        "f128_native_internal_return_abi_promoted": f128_opaque_call_return_abi_receipt["f128_native_internal_return_abi_promoted"],
        "cases": f128_opaque_call_return_abi_receipt["cases"],
    },
    "f128_sret_internal_arg_boundary_receipt": {
        "schema": f128_sret_internal_arg_boundary_receipt["schema"],
        "path": f"{f128_sret_internal_arg_boundary_receipt_path.parent.name}/{f128_sret_internal_arg_boundary_receipt_path.name}",
        "receipt_sha256": f128_sret_internal_arg_boundary_receipt["receipt_sha256"],
        "stage_contract_level": f128_sret_internal_arg_boundary_receipt["stage_contract_level"],
        "case_id": f128_sret_internal_arg_boundary_receipt["case_id"],
        "case_count": f128_sret_internal_arg_boundary_receipt["case_count"],
        "direct_control_case_count": f128_sret_internal_arg_boundary_receipt["direct_control_case_count"],
        "sret_case_count": f128_sret_internal_arg_boundary_receipt["sret_case_count"],
        "sret_stack_case_count": f128_sret_internal_arg_boundary_receipt["sret_stack_case_count"],
        "f128_internal_sret_arg_boundary_promoted": f128_sret_internal_arg_boundary_receipt["f128_internal_sret_arg_boundary_promoted"],
        "f128_internal_sret_arg_stack_boundary_promoted": f128_sret_internal_arg_boundary_receipt["f128_internal_sret_arg_stack_boundary_promoted"],
        "f128_compact_vreg_classifier_base_only_promoted": f128_sret_internal_arg_boundary_receipt["f128_compact_vreg_classifier_base_only_promoted"],
        "cases": f128_sret_internal_arg_boundary_receipt["cases"],
    },
    "f128_binary128_native_anchor_receipt": {
        "schema": f128_binary128_native_anchor_receipt["schema"],
        "path": f"{f128_binary128_native_anchor_receipt_path.parent.name}/{f128_binary128_native_anchor_receipt_path.name}",
        "receipt_sha256": f128_binary128_native_anchor_receipt["receipt_sha256"],
        "stage_contract_level": f128_binary128_native_anchor_receipt["stage_contract_level"],
        "case_id": f128_binary128_native_anchor_receipt["case_id"],
        "case_count": f128_binary128_native_anchor_receipt["case_count"],
        "claims": f128_binary128_native_anchor_receipt["claims"],
        "cases": f128_binary128_native_anchor_receipt["cases"],
    },
    "f128_binary128_value_contract_native_receipt": {
        "schema": f128_binary128_value_contract_native_receipt["schema"],
        "path": f"{f128_binary128_value_contract_native_receipt_path.parent.name}/{f128_binary128_value_contract_native_receipt_path.name}",
        "receipt_sha256": f128_binary128_value_contract_native_receipt["receipt_sha256"],
        "stage_contract_level": f128_binary128_value_contract_native_receipt["stage_contract_level"],
        "case_id": f128_binary128_value_contract_native_receipt["case_id"],
        "case_count": f128_binary128_value_contract_native_receipt["case_count"],
        "negative_case_count": f128_binary128_value_contract_native_receipt["negative_case_count"],
        "claims": f128_binary128_value_contract_native_receipt["claims"],
        "cases": f128_binary128_value_contract_native_receipt["cases"],
        "negative_cases": f128_binary128_value_contract_native_receipt["negative_cases"],
    },
    "f128_arithmetic_value_contract_receipt": {
        "schema": f128_arithmetic_value_contract_receipt["schema"],
        "path": f"{f128_arithmetic_value_contract_receipt_path.parent.name}/{f128_arithmetic_value_contract_receipt_path.name}",
        "receipt_sha256": f128_arithmetic_value_contract_receipt["receipt_sha256"],
        "stage_contract_level": f128_arithmetic_value_contract_receipt["stage_contract_level"],
        "case_id": f128_arithmetic_value_contract_receipt["case_id"],
        "case_count": f128_arithmetic_value_contract_receipt["case_count"],
        "positive_case_count": f128_arithmetic_value_contract_receipt["positive_case_count"],
        "negative_case_count": f128_arithmetic_value_contract_receipt["negative_case_count"],
        "contract_scope": f128_arithmetic_value_contract_receipt["contract_scope"],
        "cases": f128_arithmetic_value_contract_receipt["cases"],
    },
    "f128_ieee_class_helper_receipt": {
        "schema": f128_ieee_class_helper_receipt["schema"],
        "path": f"{f128_ieee_class_helper_receipt_path.parent.name}/{f128_ieee_class_helper_receipt_path.name}",
        "receipt_sha256": f128_ieee_class_helper_receipt["receipt_sha256"],
        "stage_contract_level": f128_ieee_class_helper_receipt["stage_contract_level"],
        "case_id": f128_ieee_class_helper_receipt["case_id"],
        "case_count": f128_ieee_class_helper_receipt["case_count"],
        "negative_case_count": f128_ieee_class_helper_receipt["negative_case_count"],
        "class_code_contract": f128_ieee_class_helper_receipt["class_code_contract"],
        "claims": f128_ieee_class_helper_receipt["claims"],
        "cases": f128_ieee_class_helper_receipt["cases"],
        "negative_cases": f128_ieee_class_helper_receipt["negative_cases"],
    },
    "f128_ordered_compare_receipt": {
        "schema": f128_ordered_compare_receipt["schema"],
        "path": f"{f128_ordered_compare_receipt_path.parent.name}/{f128_ordered_compare_receipt_path.name}",
        "receipt_sha256": f128_ordered_compare_receipt["receipt_sha256"],
        "stage_contract_level": f128_ordered_compare_receipt["stage_contract_level"],
        "case_id": f128_ordered_compare_receipt["case_id"],
        "case_count": f128_ordered_compare_receipt["case_count"],
        "operators_covered": f128_ordered_compare_receipt["operators_covered"],
        "ordered_compare_contract": f128_ordered_compare_receipt["ordered_compare_contract"],
        "claims": f128_ordered_compare_receipt["claims"],
        "cases": f128_ordered_compare_receipt["cases"],
    },
    "f128_param_slot_layout_receipt": {
        "schema": f128_param_slot_layout_receipt["schema"],
        "path": f"{f128_param_slot_layout_receipt_path.parent.name}/{f128_param_slot_layout_receipt_path.name}",
        "receipt_sha256": f128_param_slot_layout_receipt["receipt_sha256"],
        "stage_contract_level": f128_param_slot_layout_receipt["stage_contract_level"],
        "case_id": f128_param_slot_layout_receipt["case_id"],
        "case_count": f128_param_slot_layout_receipt["case_count"],
        "f128_param_slot_layout_promoted": f128_param_slot_layout_receipt["f128_param_slot_layout_promoted"],
        "f128_param_slots_non_overlapping": f128_param_slot_layout_receipt["f128_param_slots_non_overlapping"],
        "f128_callee_add_sub_value_contract_helper_layout_promoted": f128_param_slot_layout_receipt["f128_callee_add_sub_value_contract_helper_layout_promoted"],
        "cases": f128_param_slot_layout_receipt["cases"],
    },
    "differential_receipt": {
        "schema": differential_receipt["schema"],
        "path": f"{differential_receipt_path.parent.name}/{differential_receipt_path.name}",
        "receipt_sha256": differential_receipt["receipt_sha256"],
        "stage_contract_level": differential_receipt["stage_contract_level"],
        "case_id": differential_receipt["case_id"],
        "case_count": differential_receipt["case_count"],
        "matched_case_count": differential_receipt["matched_case_count"],
        "reference_unavailable_case_count": differential_receipt["reference_unavailable_case_count"],
        "categories_compared": differential_receipt["categories_compared"],
        "cases": differential_receipt["cases"],
    },
    "scalar_abi_receipts": {
        "schema": "madaros.v2.s5.abi_scalar_call_return/0.1",
        "target": "x86_64-linux",
        "arg_register_order": ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        "return_register": "rax",
        "f64_return_register": "xmm0",
        "normal_call_stack_args_promoted": True,
        "source_sret_stack_args_promoted": True,
        "imported_sret_module_boundary_promoted": True,
        "method_sret_promoted": True,
        "sret_promoted": True,
        "source_sret_local_one_arg_promoted": True,
        "source_sret_local_register_multi_arg_promoted": True,
        "aggregate_layout_promoted": True,
        "generic_aggregate_return_promoted": True,
        "layout_derived_sret_alloc_promoted": True,
        "f64_xmm0_promoted": True,
        "wide_i128_i256_promoted": True,
        "wide_u128_u256_promoted": True,
        "wide_i256_u256_machine_slots_promoted": True,
        "wide_i256_u256_local_abi_call_return_promoted": True,
        "wide_i256_u256_imported_abi_call_return_promoted": True,
        "imported_module_wide_abi_promoted": True,
        "f128_literal_decimal_metadata_promoted": True,
        "f128_type_system_awareness_promoted": True,
        "f128_binary128_value_contract_promoted": True,
        "f128_literal_value_bridge_promoted": True,
        "machine_slot_metadata_promoted": True,
        "f128_abi_metadata_promoted": True,
        "f128_native_opaque_storage_promoted": True,
        "f128_opaque_direct_call_return_abi_promoted": True,
        "f128_opaque_direct_stack_call_abi_promoted": True,
        "f128_opaque_imported_direct_call_return_abi_promoted": True,
        "f128_native_internal_call_abi_promoted": True,
        "f128_native_internal_return_abi_promoted": True,
        "f128_machineir_return_high_word_capture_promoted": True,
        "f128_internal_sret_arg_boundary_promoted": True,
        "f128_internal_sret_arg_stack_boundary_promoted": True,
        "f128_compact_vreg_classifier_base_only_promoted": True,
        "f128_binary128_native_anchor_materialization_promoted": True,
    "f128_binary128_value_contract_native_materialization_promoted": True,
    "f128_native_exact_dyadic_decimal_binary128_materialization_promoted": True,
    "f128_native_bounded_rounded_decimal_binary128_materialization_promoted": True,
    "f128_native_general_bounded_decimal_siglo_scale18_materialization_promoted": True,
    "f128_native_two_limb_integer_decimal_binary128_materialization_promoted": True,
    "f128_native_two_limb_fractional_decimal_binary128_materialization_promoted": True,
    "f128_native_truncated_decimal_binary128_value_contract_promoted": True,
        "f128_native_subnormal_underflow_overflow_value_contract_promoted": True,
        "f128_arithmetic_value_contract_promoted": True,
        "f128_runtime_positive_rounded_tenths_add_helper_promoted": True,
        "f128_runtime_positive_rounded_decimal_add_matrix_promoted": True,
        "f128_runtime_callee_add_sub_mul_div_value_contract_promoted": True,
        "f128_native_ieee_class_code_helper_promoted": True,
        "f128_native_ieee_class_code_source_observable_zero_subnormal_normal_infinity_promoted": True,
        "f128_native_ieee_class_code_nan_source_surface_promoted": True,
        "f128_native_canonical_quiet_nan_constructor_promoted": True,
        "f128_ordered_binary128_compare_promoted": True,
        "f128_ordered_binary128_compare_source_observable_promoted": True,
        "f128_ordered_binary128_compare_nan_unordered_promoted": True,
        "f128_ordered_binary128_compare_signed_zero_promoted": True,
        "f128_ordered_binary128_compare_infinity_promoted": True,
        "f128_ordered_binary128_compare_subnormal_promoted": True,
        "f128_param_slot_layout_promoted": True,
        "f128_param_slots_non_overlapping": True,
        "f128_native_general_decimal_binary128_materialization_promoted": False,
        "f128_native_arbitrary_decimal_binary128_materialization_promoted": False,
        "f128_native_ieee_binary128_materialization_promoted": False,
        "f128_native_arithmetic_promoted": True,
        "f128_native_call_abi_promoted": False,
        "f128_native_return_abi_promoted": False,
        "unsupported_numeric_diagnostics_promoted": True,
        "unsupported_numeric_widths_fail_closed": True,
        "differential_native_v2_vs_lean_single_promoted": True,
        "f128_promoted": False,
    },
    "not_promoted_surfaces": not_promoted,
    "negative_and_blocked_controls": negative_and_blocked_controls,
    "roundtrip_contract": [
        "canonical_json_stable_after_parse_dump",
        "exact_three_scalar_program_witnesses",
        "program_shadow_hash_per_witness",
        "canonical_madaros_s5_receipt_per_witness",
        "compiler_exported_machine_module_json_per_witness",
        "merged_ir_function_count_matches_program_shape",
        "elf_internal_call_count_matches_program_shape",
        "scalar_abi_register_contract_recorded",
        "sret_hidden_dest_abi_discriminator_recorded",
        "source_sret_local_one_arg_receipt_recorded",
        "source_sret_local_register_multi_arg_receipt_recorded",
        "source_sret_local_stack_arg_receipt_recorded",
        "imported_sret_module_boundary_receipt_recorded",
        "method_sret_receipt_recorded",
        "f64_xmm0_call_return_receipt_recorded",
        "wide_int_i128_i256_receipt_recorded",
        "wide_int_machine_slot_metadata_receipt_recorded",
        "wide_i256_u256_local_abi_call_return_receipt_recorded",
        "wide_i256_u256_imported_abi_call_return_receipt_recorded",
        "generic_aggregate_sret_layout_receipt_recorded",
        "f128_literal_decimal_metadata_receipt_recorded",
        "f128_type_system_awareness_receipt_recorded",
        "f128_binary128_value_receipt_recorded",
        "f128_literal_value_bridge_receipt_recorded",
        "machine_slot_kind_width_metadata_receipt_recorded",
        "f128_abi_metadata_receipt_recorded",
        "f128_native_opaque_storage_receipt_recorded",
        "f128_opaque_direct_call_return_abi_receipt_recorded",
        "f128_binary128_native_anchor_receipt_recorded",
        "f128_binary128_value_contract_native_receipt_recorded",
        "f128_arithmetic_value_contract_receipt_recorded",
        "f128_ieee_class_code_helper_receipt_recorded",
        "f128_param_slot_layout_receipt_recorded",
        "unsupported_numeric_diagnostics_receipt_recorded",
        "differential_native_v2_vs_lean_single_receipt_recorded",
        "normal_call_stack_arg_receipt_recorded",
        "f128_binary128_native_anchor_materialization_promoted_for_exact_0_5_and_1_0_only",
        "f128_binary128_value_contract_native_materialization_promoted_for_current_case_set_including_truncated_high_precision_decimals",
        "f128_arithmetic_value_contract_promoted_for_finite_decimal_tenths_matrix_with_one_chain_and_callee_add_sub_mul_div_helper",
        "f128_runtime_positive_rounded_tenths_add_helper_promoted_for_0_1_plus_0_2_to_binary128_sum",
        "f128_runtime_positive_rounded_decimal_add_matrix_promoted_for_selected_binary128_source_sums",
        "f128_ieee_class_code_helper_promoted_for_source_observable_zero_subnormal_normal_infinity",
        "f128_ordered_compare_receipt_recorded",
        "f128_ordered_binary128_compare_promoted_for_finite_zero_subnormal_infinity_and_nan_unordered_cases",
        "f128_parameter_slots_non_overlapping_for_local_imported_and_mixed_shapes",
        "f128_opaque_direct_call_return_abi_promoted_for_local_and_imported_return_only_mixed_order_two_f128_direct_and_stack_shapes",
        "f128_arbitrary_decimal_binary128_materialization_not_promoted",
        "f128_generic_ieee_arithmetic_nan_source_and_external_abi_surfaces_not_promoted",
        "s4_negative_and_blocked_controls_not_promoted",
        "s4_applied_extraction_hash_propagates_to_program_receipt",
        "each_mir_effect_is_bound_to_a_source_s4_applied_effect_hash",
        "f128_promoted_surface_differentials_recorded_generic_ieee_and_external_abi_still_required_before_s5_ready",
    ],
}
canonical_module, module_sha = canonical_roundtrip(module)
module["program_mir_abi_module_sha256"] = module_sha
canonical_module_with_hash, module_with_hash_sha = canonical_roundtrip(module)
module_path.write_text(pretty_json(module), encoding="utf-8")

reloaded = load_json(module_path)
if stable_json(reloaded) != stable_json(module):
    raise SystemExit("pretty JSON program MIR/ABI module does not roundtrip")

receipt = {
    "schema": "madaros.v2.s5.program_mir_abi_scalar_shadow/0.1",
    "status": "pass",
    "stage_contract_level": "S5_SCALAR_MACHINE_MODULE_EXPORT_WITH_ABI_SHADOW_NOT_FULL",
    "s5_program_mir_abi_scalar_shadow_slice_complete": True,
    "s5_mir_effect_roundtrip_complete": True,
    "s5_mir_abi_input_boundary_complete": True,
    "s5_mir_abi_boundary_complete": False,
    "s5_ready": False,
    "s5_implemented": False,
    "s5_full_complete": False,
    "s_full_contract": "blocked_until_f128_software_helpers_codegen_and_execution_differentials_exist",
    "program_mir_shadow_serialized": True,
    "compiler_machine_module_exported": True,
    "real_program_mir_emitted": True,
    "real_abi_layout_emitted": True,
    "s5_sret_machine_module_abi_discriminator_complete": True,
    "s5_source_sret_local_one_arg_complete": True,
    "s5_source_sret_local_register_multi_arg_complete": True,
    "s5_source_sret_local_stack_arg_complete": True,
    "s5_imported_sret_module_boundary_complete": True,
    "s5_method_sret_complete": True,
    "s5_normal_call_stack_args_complete": True,
    "s5_f64_xmm0_call_return_complete": True,
    "s5_wide_int_i128_i256_complete": True,
    "s5_wide_int_machine_slot_metadata_complete": True,
    "s5_wide_i256_u256_local_abi_call_return_complete": True,
    "s5_wide_i256_u256_imported_abi_call_return_complete": True,
    "s5_generic_aggregate_sret_layout_complete": True,
    "s4_s5_f128_literal_decimal_metadata_complete": True,
    "s4_s5_f128_type_system_awareness_complete": True,
    "s5_f128_binary128_value_contract_complete": True,
    "s5_f128_literal_value_bridge_complete": True,
    "s5_machine_slot_kind_width_metadata_complete": True,
    "s5_f128_abi_metadata_complete": True,
    "s5_f128_native_opaque_storage_complete": True,
    "s5_f128_opaque_direct_call_return_abi_complete": True,
    "s5_f128_internal_sret_arg_boundary_complete": True,
    "s5_f128_binary128_native_anchor_materialization_complete": True,
    "s5_f128_binary128_value_contract_native_materialization_complete": True,
    "s5_f128_arithmetic_value_contract_complete": True,
    "s5_f128_ieee_class_code_helper_complete": True,
    "s5_f128_ordered_binary128_compare_complete": True,
    "s5_f128_param_slot_layout_complete": True,
    "s5_differential_native_v2_lean_single_complete": True,
    "source_frontend_lowers_local_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_local_register_multi_arg_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_local_stack_arg_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_imported_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_method_aggregate_return_to_IrCallSret": True,
    "source_frontend_dispatches_print_to_print_f64": True,
    "source_frontend_dispatches_println_f64_to_print_f64": True,
    "source_frontend_tracks_let_bound_f64_identifiers": True,
    "ir_lowers_f64_literals_to_IrLoadFloat": True,
    "native_v2_lowers_f64_to_i64_cast": True,
    "native_v2_lowers_fractional_f64_binops": True,
    "native_v2_materializes_print_f64_fraction_scale_without_rodata_relocation": True,
    "native_v2_bridges_print_f64_arg0_to_xmm0": True,
    "wide_i128_i256_promoted": True,
    "wide_u128_u256_promoted": True,
    "wide_i256_u256_machine_slots_promoted": True,
    "wide_i256_u256_local_abi_call_return_promoted": True,
    "wide_i256_u256_imported_abi_call_return_promoted": True,
    "imported_module_wide_abi_promoted": True,
    "generic_aggregate_return_promoted": True,
    "generic_aggregate_local_layout_promoted": True,
    "generic_aggregate_imported_layout_promoted": True,
    "generic_aggregate_method_layout_promoted": True,
    "layout_derived_sret_alloc_promoted": True,
    "wide9_sret_alloc_72_bytes_promoted": True,
    "source_level_wide_arithmetic_promoted": True,
    "native_v2_wide_limb_backend_promoted": True,
    "wide_type_identity_and_safety_promoted": True,
    "f128_binary128_value_contract_promoted": True,
    "f128_literal_value_bridge_promoted": True,
    "f128_machine_ir_opaque_literal_promoted": True,
    "f128_machine_ir_opaque_slot_promoted": True,
    "f128_machine_ir_local_metadata_copy_promoted": True,
    "f128_abi_metadata_promoted": True,
    "f128_native_opaque_storage_promoted": True,
    "f128_opaque_direct_call_return_abi_promoted": True,
    "f128_opaque_direct_stack_call_abi_promoted": True,
    "f128_opaque_imported_direct_call_return_abi_promoted": True,
    "f128_native_internal_call_abi_promoted": True,
    "f128_native_internal_return_abi_promoted": True,
    "f128_machineir_return_high_word_capture_promoted": True,
    "f128_internal_sret_arg_boundary_promoted": True,
    "f128_internal_sret_arg_stack_boundary_promoted": True,
    "f128_compact_vreg_classifier_base_only_promoted": True,
    "f128_binary128_native_anchor_materialization_promoted": True,
    "f128_binary128_value_contract_native_materialization_promoted": True,
    "f128_native_exact_dyadic_decimal_binary128_materialization_promoted": True,
    "f128_native_bounded_rounded_decimal_binary128_materialization_promoted": True,
    "f128_native_general_bounded_decimal_siglo_scale18_materialization_promoted": True,
    "f128_native_two_limb_integer_decimal_binary128_materialization_promoted": True,
    "f128_native_two_limb_fractional_decimal_binary128_materialization_promoted": True,
    "f128_native_truncated_decimal_binary128_value_contract_promoted": True,
    "f128_native_subnormal_underflow_overflow_value_contract_promoted": True,
    "f128_arithmetic_value_contract_promoted": True,
    "f128_runtime_positive_rounded_tenths_add_helper_promoted": True,
    "f128_runtime_positive_rounded_decimal_add_matrix_promoted": True,
    "f128_native_ieee_class_code_helper_promoted": True,
    "f128_native_ieee_class_code_source_observable_zero_subnormal_normal_infinity_promoted": True,
    "f128_native_ieee_class_code_nan_source_surface_promoted": True,
    "f128_native_canonical_quiet_nan_constructor_promoted": True,
    "f128_ordered_binary128_compare_promoted": True,
    "f128_ordered_binary128_compare_source_observable_promoted": True,
    "f128_ordered_binary128_compare_nan_unordered_promoted": True,
    "f128_ordered_binary128_compare_signed_zero_promoted": True,
    "f128_ordered_binary128_compare_infinity_promoted": True,
    "f128_ordered_binary128_compare_subnormal_promoted": True,
    "f128_param_slot_layout_promoted": True,
    "f128_param_slots_non_overlapping": True,
    "f128_native_general_decimal_binary128_materialization_promoted": False,
    "f128_native_arbitrary_decimal_binary128_materialization_promoted": False,
    "f128_native_ieee_binary128_materialization_promoted": False,
    "f128_native_arithmetic_promoted": True,
    "f128_native_call_abi_promoted": False,
    "f128_native_return_abi_promoted": False,
    "s5_diagnostics_unsupported_numeric_complete": True,
    "unsupported_numeric_widths_fail_closed": True,
    "differential_native_v2_vs_lean_single_promoted": True,
    "unsupported_widths_do_not_emit_elf": True,
    "front_half_unsupported_widths_do_not_emit_machine_module_json": True,
    "f128_blockers_emit_machine_module_json": True,
    "unsupported_widths_do_not_segfault": True,
    "legacy_fallback_for_unsupported_widths": False,
    "f128_full_execution_not_promoted": True,
    "i512_u512_rejected_not_promoted": True,
    "f128_promoted": False,
    "input_mir_effect_sha256": effect_receipt["receipt_sha256"],
    "input_boundary_sha256": effect_receipt["input_boundary_sha256"],
    "input_applied_extraction_contract": effect_receipt["input_applied_extraction_contract"],
    "input_applied_extraction_sha256": effect_receipt["input_applied_extraction_sha256"],
    "s4_applied_extraction_consumed": True,
    "sret_abi_receipt_sha256": sret_receipt["receipt_sha256"],
    "source_sret_receipt_sha256": source_sret_receipt["receipt_sha256"],
    "stack_call_receipt_sha256": stack_call_receipt["receipt_sha256"],
    "imported_sret_receipt_sha256": imported_sret_receipt["receipt_sha256"],
    "method_sret_receipt_sha256": method_sret_receipt["receipt_sha256"],
    "f64_xmm0_receipt_sha256": f64_xmm0_receipt["receipt_sha256"],
    "wide_int_receipt_sha256": wide_int_receipt["receipt_sha256"],
    "wide_machine_slot_metadata_receipt_sha256": wide_machine_slot_receipt["receipt_sha256"],
    "wide_abi_call_return_receipt_sha256": wide_abi_call_return_receipt["receipt_sha256"],
    "generic_aggregate_sret_receipt_sha256": generic_agg_receipt["receipt_sha256"],
    "f128_binary128_value_receipt_sha256": f128_binary128_value_receipt["receipt_sha256"],
    "f128_literal_value_bridge_receipt_sha256": f128_literal_value_bridge_receipt["receipt_sha256"],
    "machine_slot_metadata_receipt_sha256": machine_slot_metadata_receipt["receipt_sha256"],
    "f128_abi_metadata_receipt_sha256": f128_abi_metadata_receipt["receipt_sha256"],
    "f128_native_opaque_storage_receipt_sha256": f128_native_opaque_storage_receipt["receipt_sha256"],
    "f128_opaque_call_return_abi_receipt_sha256": f128_opaque_call_return_abi_receipt["receipt_sha256"],
    "f128_sret_internal_arg_boundary_receipt_sha256": f128_sret_internal_arg_boundary_receipt["receipt_sha256"],
    "f128_binary128_native_anchor_receipt_sha256": f128_binary128_native_anchor_receipt["receipt_sha256"],
    "f128_binary128_value_contract_native_receipt_sha256": f128_binary128_value_contract_native_receipt["receipt_sha256"],
    "f128_arithmetic_value_contract_receipt_sha256": f128_arithmetic_value_contract_receipt["receipt_sha256"],
    "f128_ieee_class_helper_receipt_sha256": f128_ieee_class_helper_receipt["receipt_sha256"],
    "f128_ordered_compare_receipt_sha256": f128_ordered_compare_receipt["receipt_sha256"],
    "f128_param_slot_layout_receipt_sha256": f128_param_slot_layout_receipt["receipt_sha256"],
    "diagnostics_receipt_sha256": diagnostics_receipt["receipt_sha256"],
    "differential_receipt_sha256": differential_receipt["receipt_sha256"],
    "program_mir_abi_module_path": module_path.name,
    "program_mir_abi_module_sha256": module_sha,
    "program_mir_abi_module_with_hash_sha256": module_with_hash_sha,
    "canonical_roundtrip_sha256": sha256_text(canonical_module_with_hash),
    "program_count": len(programs),
    "program_kinds": module["program_kinds"],
    "canonical_s5_source_receipt_count": module["canonical_s5_source_receipt_count"],
    "canonical_s5_source_receipts_present": module["canonical_s5_source_receipts_present"],
    "target": module["scalar_abi_receipts"]["target"],
    "arg_register_order": module["scalar_abi_receipts"]["arg_register_order"],
    "return_register": module["scalar_abi_receipts"]["return_register"],
    "sret_promoted": module["scalar_abi_receipts"]["sret_promoted"],
    "source_sret_local_one_arg_promoted": module["scalar_abi_receipts"]["source_sret_local_one_arg_promoted"],
    "source_sret_local_register_multi_arg_promoted": module["scalar_abi_receipts"]["source_sret_local_register_multi_arg_promoted"],
    "source_sret_stack_args_promoted": module["scalar_abi_receipts"]["source_sret_stack_args_promoted"],
    "imported_sret_module_boundary_promoted": module["scalar_abi_receipts"]["imported_sret_module_boundary_promoted"],
    "method_sret_promoted": module["scalar_abi_receipts"]["method_sret_promoted"],
    "normal_call_stack_args_promoted": module["scalar_abi_receipts"]["normal_call_stack_args_promoted"],
    "aggregate_layout_promoted": module["scalar_abi_receipts"]["aggregate_layout_promoted"],
    "f64_xmm0_promoted": module["scalar_abi_receipts"]["f64_xmm0_promoted"],
    "generic_aggregate_return_promoted": module["scalar_abi_receipts"]["generic_aggregate_return_promoted"],
    "layout_derived_sret_alloc_promoted": module["scalar_abi_receipts"]["layout_derived_sret_alloc_promoted"],
    "wide_i128_i256_promoted": module["scalar_abi_receipts"]["wide_i128_i256_promoted"],
    "wide_u128_u256_promoted": module["scalar_abi_receipts"]["wide_u128_u256_promoted"],
    "wide_i256_u256_machine_slots_promoted": module["scalar_abi_receipts"]["wide_i256_u256_machine_slots_promoted"],
    "wide_i256_u256_local_abi_call_return_promoted": module["scalar_abi_receipts"]["wide_i256_u256_local_abi_call_return_promoted"],
    "imported_module_wide_abi_promoted": module["scalar_abi_receipts"]["imported_module_wide_abi_promoted"],
    "wide_abi_call_return_case_count": wide_abi_call_return_receipt["case_count"],
    "wide_abi_call_return_two_wide_arg_case_count": wide_abi_call_return_receipt["two_wide_arg_case_count"],
    "f128_binary128_value_contract_promoted": module["scalar_abi_receipts"]["f128_binary128_value_contract_promoted"],
    "f128_literal_value_bridge_promoted": module["scalar_abi_receipts"]["f128_literal_value_bridge_promoted"],
    "machine_slot_metadata_promoted": module["scalar_abi_receipts"]["machine_slot_metadata_promoted"],
    "f128_abi_metadata_promoted": module["scalar_abi_receipts"]["f128_abi_metadata_promoted"],
    "f128_native_opaque_storage_promoted": module["scalar_abi_receipts"]["f128_native_opaque_storage_promoted"],
    "f128_opaque_direct_call_return_abi_promoted": module["scalar_abi_receipts"]["f128_opaque_direct_call_return_abi_promoted"],
    "f128_opaque_direct_stack_call_abi_promoted": module["scalar_abi_receipts"]["f128_opaque_direct_stack_call_abi_promoted"],
    "f128_opaque_imported_direct_call_return_abi_promoted": module["scalar_abi_receipts"]["f128_opaque_imported_direct_call_return_abi_promoted"],
    "f128_native_internal_call_abi_promoted": module["scalar_abi_receipts"]["f128_native_internal_call_abi_promoted"],
    "f128_native_internal_return_abi_promoted": module["scalar_abi_receipts"]["f128_native_internal_return_abi_promoted"],
    "f128_machineir_return_high_word_capture_promoted": module["scalar_abi_receipts"]["f128_machineir_return_high_word_capture_promoted"],
    "f128_internal_sret_arg_boundary_promoted": module["scalar_abi_receipts"]["f128_internal_sret_arg_boundary_promoted"],
    "f128_internal_sret_arg_stack_boundary_promoted": module["scalar_abi_receipts"]["f128_internal_sret_arg_stack_boundary_promoted"],
    "f128_compact_vreg_classifier_base_only_promoted": module["scalar_abi_receipts"]["f128_compact_vreg_classifier_base_only_promoted"],
    "f128_binary128_native_anchor_materialization_promoted": module["scalar_abi_receipts"]["f128_binary128_native_anchor_materialization_promoted"],
    "f128_binary128_value_contract_native_materialization_promoted": module["scalar_abi_receipts"]["f128_binary128_value_contract_native_materialization_promoted"],
    "f128_native_exact_dyadic_decimal_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_exact_dyadic_decimal_binary128_materialization_promoted"],
    "f128_native_bounded_rounded_decimal_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_bounded_rounded_decimal_binary128_materialization_promoted"],
    "f128_native_general_bounded_decimal_siglo_scale18_materialization_promoted": module["scalar_abi_receipts"]["f128_native_general_bounded_decimal_siglo_scale18_materialization_promoted"],
    "f128_native_two_limb_integer_decimal_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_two_limb_integer_decimal_binary128_materialization_promoted"],
    "f128_native_two_limb_fractional_decimal_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_two_limb_fractional_decimal_binary128_materialization_promoted"],
    "f128_native_truncated_decimal_binary128_value_contract_promoted": module["scalar_abi_receipts"]["f128_native_truncated_decimal_binary128_value_contract_promoted"],
    "f128_native_subnormal_underflow_overflow_value_contract_promoted": module["scalar_abi_receipts"]["f128_native_subnormal_underflow_overflow_value_contract_promoted"],
    "f128_arithmetic_value_contract_promoted": module["scalar_abi_receipts"]["f128_arithmetic_value_contract_promoted"],
    "f128_runtime_positive_rounded_tenths_add_helper_promoted": module["scalar_abi_receipts"]["f128_runtime_positive_rounded_tenths_add_helper_promoted"],
    "f128_runtime_positive_rounded_decimal_add_matrix_promoted": module["scalar_abi_receipts"]["f128_runtime_positive_rounded_decimal_add_matrix_promoted"],
    "f128_native_ieee_class_code_helper_promoted": module["scalar_abi_receipts"]["f128_native_ieee_class_code_helper_promoted"],
    "f128_native_ieee_class_code_source_observable_zero_subnormal_normal_infinity_promoted": module["scalar_abi_receipts"]["f128_native_ieee_class_code_source_observable_zero_subnormal_normal_infinity_promoted"],
    "f128_native_ieee_class_code_nan_source_surface_promoted": module["scalar_abi_receipts"]["f128_native_ieee_class_code_nan_source_surface_promoted"],
    "f128_native_canonical_quiet_nan_constructor_promoted": module["scalar_abi_receipts"]["f128_native_canonical_quiet_nan_constructor_promoted"],
    "f128_ordered_binary128_compare_promoted": module["scalar_abi_receipts"]["f128_ordered_binary128_compare_promoted"],
    "f128_ordered_binary128_compare_source_observable_promoted": module["scalar_abi_receipts"]["f128_ordered_binary128_compare_source_observable_promoted"],
    "f128_ordered_binary128_compare_nan_unordered_promoted": module["scalar_abi_receipts"]["f128_ordered_binary128_compare_nan_unordered_promoted"],
    "f128_ordered_binary128_compare_signed_zero_promoted": module["scalar_abi_receipts"]["f128_ordered_binary128_compare_signed_zero_promoted"],
    "f128_ordered_binary128_compare_infinity_promoted": module["scalar_abi_receipts"]["f128_ordered_binary128_compare_infinity_promoted"],
    "f128_ordered_binary128_compare_subnormal_promoted": module["scalar_abi_receipts"]["f128_ordered_binary128_compare_subnormal_promoted"],
    "f128_param_slot_layout_promoted": module["scalar_abi_receipts"]["f128_param_slot_layout_promoted"],
    "f128_param_slots_non_overlapping": module["scalar_abi_receipts"]["f128_param_slots_non_overlapping"],
    "f128_native_general_decimal_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_general_decimal_binary128_materialization_promoted"],
    "f128_native_arbitrary_decimal_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_arbitrary_decimal_binary128_materialization_promoted"],
    "f128_native_ieee_binary128_materialization_promoted": module["scalar_abi_receipts"]["f128_native_ieee_binary128_materialization_promoted"],
    "f128_native_arithmetic_promoted": module["scalar_abi_receipts"]["f128_native_arithmetic_promoted"],
    "f128_native_call_abi_promoted": module["scalar_abi_receipts"]["f128_native_call_abi_promoted"],
    "f128_native_return_abi_promoted": module["scalar_abi_receipts"]["f128_native_return_abi_promoted"],
    "unsupported_numeric_diagnostics_promoted": module["scalar_abi_receipts"]["unsupported_numeric_diagnostics_promoted"],
    "unsupported_numeric_widths_fail_closed": module["scalar_abi_receipts"]["unsupported_numeric_widths_fail_closed"],
    "differential_compared_case_count": differential_receipt["matched_case_count"],
    "differential_unavailable_case_count": differential_receipt["reference_unavailable_case_count"],
    "differential_categories_compared": differential_receipt["categories_compared"],
    "differential_native_v2_vs_lean_single_promoted": module["scalar_abi_receipts"]["differential_native_v2_vs_lean_single_promoted"],
    "f128_promoted": module["scalar_abi_receipts"]["f128_promoted"],
    "not_promoted_surfaces": [item["surface"] for item in not_promoted],
    "negative_and_blocked_controls": [
        f"{item['case_id']}:{item['expected_status']}" for item in negative_and_blocked_controls
    ],
    "gate_invariants": module["roundtrip_contract"],
    "missing_full_obligations": [
        "f128 arbitrary decimal-to-binary128 native materialization beyond the current finite value-contract case set",
        "f128 generic IEEE arithmetic helpers beyond the class-code and ordered-comparison helpers, including rounding-mode-sensitive operations",
        "f128 arithmetic beyond the finite signed decimal-tenths plus quarter value-contract matrix, the bounded rounded-decimal add matrix, direct literal/parameter-return call propagation, callee-side exact add/sub/mul/div runtime helper, external SysV f128 ABI/SRET, and full IEEE helper differentials",
    ],
}
receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
print(
    f"[madaros-v2-s5-program-mir-abi] ok programs={receipt['program_count']} "
    f"target={receipt['target']} sha={receipt['receipt_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-program-mir-abi] PASS: scalar i64/bool + SRET + f64/XMM0 + wide-int + local+imported i256/u256 wide ABI call-return + generic aggregate + f128 internal native-v2 call-return/SRET-arg-boundary/value-contract binary128 compiler MachineModule ABI receipts are deterministic without claiming S5 FULL"
echo "[madaros-v2-s5-program-mir-abi] PASS: i512/u512 fail closed before MachineModule export; f128 emits supported opaque MachineIR metadata/literal bridge/ABI metadata, exact-dyadic, bounded-rounded, algorithmic two-limb scale0..18 decimal, and truncated high-precision value-contract binary128 materialization, finite signed decimal-tenths and quarter value-contract arithmetic with direct literal/parameter-return propagation, bounded rounded-decimal add matrix for selected binary128 source sums, callee-side add/sub/mul/div runtime helper execution, source-observable IEEE class-code helper classification for zero/subnormal/normal/infinity/NaN via canonical quiet-NaN constructor, and ordered binary128 comparisons for finite/zero/subnormal/infinity/NaN-unordered cases, while unsupported f128 surfaces fail closed either before ELF emission or through explicit runtime rc=12 helper traps, without segfault or fallback"
echo "[madaros-v2-s5-program-mir-abi] PASS: native-v2 vs lean_single differential receipt covers promoted comparable S5 surfaces including f128 value-contract/local ABI/SRET-boundary/layout/classifier/ordered-comparison cases; generic IEEE arithmetic and external ABI differentials remain explicit full blockers"
echo "[madaros-v2-s5-program-mir-abi] module=$MODULE"
echo "[madaros-v2-s5-program-mir-abi] receipt=$RECEIPT"
