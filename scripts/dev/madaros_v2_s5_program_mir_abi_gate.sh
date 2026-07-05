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
GENERIC_AGG_RECEIPT_DIR="$OUT_DIR/generic_aggregate_sret_receipt"
F128_LITERAL_PROVENANCE_RECEIPT_DIR="$OUT_DIR/f128_literal_provenance_receipt"
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
GENERIC_AGG_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_generic_aggregate_sret_receipt.py"
F128_LITERAL_PROVENANCE_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_f128_literal_provenance_receipt.py"
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
GENERIC_AGG_RECEIPT="$GENERIC_AGG_RECEIPT_DIR/madaros_v2_s5_generic_aggregate_sret.receipt.json"
F128_LITERAL_PROVENANCE_RECEIPT="$F128_LITERAL_PROVENANCE_RECEIPT_DIR/madaros_v2_f128_literal_provenance.receipt.json"
DIAGNOSTICS_RECEIPT="$DIAGNOSTICS_RECEIPT_DIR/madaros_v2_s5_diagnostics.receipt.json"
DIFFERENTIAL_RECEIPT="$DIFFERENTIAL_RECEIPT_DIR/madaros_v2_s5_differential.receipt.json"

mkdir -p "$EFFECT_DIR" "$S5_RECEIPT_DIR" "$SRET_RECEIPT_DIR" "$SOURCE_SRET_RECEIPT_DIR" "$STACK_CALL_RECEIPT_DIR" "$IMPORTED_SRET_RECEIPT_DIR" "$METHOD_SRET_RECEIPT_DIR" "$F64_XMM0_RECEIPT_DIR" "$WIDE_INT_RECEIPT_DIR" "$GENERIC_AGG_RECEIPT_DIR" "$F128_LITERAL_PROVENANCE_RECEIPT_DIR" "$DIAGNOSTICS_RECEIPT_DIR" "$DIFFERENTIAL_RECEIPT_DIR"

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

python3 - "$EFFECT_DIR" "$S5_RECEIPT_RESULTS" "$SRET_RECEIPT" "$SOURCE_SRET_RECEIPT" "$STACK_CALL_RECEIPT" "$IMPORTED_SRET_RECEIPT" "$METHOD_SRET_RECEIPT" "$F64_XMM0_RECEIPT" "$WIDE_INT_RECEIPT" "$GENERIC_AGG_RECEIPT" "$F128_LITERAL_PROVENANCE_RECEIPT" "$DIAGNOSTICS_RECEIPT" "$DIFFERENTIAL_RECEIPT" "$MODULE" "$RECEIPT" <<'PY'
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
generic_agg_receipt_path = Path(sys.argv[10])
f128_literal_provenance_receipt_path = Path(sys.argv[11])
diagnostics_receipt_path = Path(sys.argv[12])
differential_receipt_path = Path(sys.argv[13])
module_path = Path(sys.argv[14])
receipt_path = Path(sys.argv[15])

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
generic_agg_receipt = load_json(generic_agg_receipt_path)
f128_literal_provenance_receipt = load_json(f128_literal_provenance_receipt_path)
diagnostics_receipt = load_json(diagnostics_receipt_path)
differential_receipt = load_json(differential_receipt_path)

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

if f128_literal_provenance_receipt.get("schema") != "madaros.v2.f128_literal_provenance_receipt/0.1":
    raise SystemExit("bad f128 literal provenance receipt schema")
if f128_literal_provenance_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing f128 literal provenance receipt")
if f128_literal_provenance_receipt.get("stage_contract_level") != "S4_S5_F128_LITERAL_PROVENANCE_PROMOTED_NOT_F128_EXECUTION":
    raise SystemExit("f128 literal provenance receipt must declare parser provenance stage contract")
for field in [
    "raw_literal_capture_before_advance",
    "float_literal_ast_name_preserved",
    "float_literal_f64_value_still_preserved",
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

if diagnostics_receipt.get("schema") != "madaros.v2.s5.diagnostics_receipt/0.1":
    raise SystemExit("bad S5 diagnostics receipt schema")
if diagnostics_receipt.get("status") != "pass":
    raise SystemExit("program MIR/ABI gate requires passing diagnostics receipt")
if diagnostics_receipt.get("stage_contract_level") != "S5_UNSUPPORTED_NUMERIC_DIAGNOSTICS_PROMOTED_NOT_F128":
    raise SystemExit("diagnostics receipt must declare unsupported numeric diagnostic stage contract")
if diagnostics_receipt.get("case_count") != 5:
    raise SystemExit("diagnostics receipt must contain exact five cases")
if diagnostics_receipt.get("negative_case_count") != 4:
    raise SystemExit("diagnostics receipt must contain exact four negative cases")
if diagnostics_receipt.get("positive_guard_case_count") != 1:
    raise SystemExit("diagnostics receipt must contain exact one positive guard case")
required_diagnostics_true_flags = [
    "s5_diagnostics_unsupported_numeric_complete",
    "unsupported_numeric_widths_fail_closed",
    "unsupported_widths_do_not_emit_elf",
    "unsupported_widths_do_not_emit_machine_module_json",
    "unsupported_widths_do_not_segfault",
    "f128_rejected_not_promoted",
    "i512_u512_rejected_not_promoted",
    "promoted_i256_width_preserved",
]
for field in required_diagnostics_true_flags:
    if diagnostics_receipt.get(field) is not True:
        raise SystemExit(f"diagnostics receipt missing required true flag: {field}")
for field in [
    "legacy_fallback_for_unsupported_widths",
    "f128_promoted",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
]:
    if diagnostics_receipt.get(field) is not False:
        raise SystemExit(f"diagnostics receipt must not overclaim {field}")
diagnostic_cases = {row.get("case_id"): row for row in diagnostics_receipt.get("cases", [])}
required_diagnostic_negative = {
    "reject_f128_let_annotation_native_v2": {"width": "f128", "detail": "let annotation"},
    "reject_f128_cast_native_v2": {"width": "f128", "detail": "cast"},
    "reject_i512_let_annotation_native_v2": {"width": "i512", "detail": "let annotation"},
    "reject_u512_cast_native_v2": {"width": "u512", "detail": "cast"},
}
required_diagnostic_positive = {
    "preserve_i256_promoted_width_native_v2": {"width": "i256", "exit": 7},
}
if set(diagnostic_cases) != set(required_diagnostic_negative) | set(required_diagnostic_positive):
    raise SystemExit(f"diagnostics receipt cases mismatch: {sorted(diagnostic_cases)}")
for case_id, expected in required_diagnostic_negative.items():
    row = diagnostic_cases[case_id]
    if row.get("status") != "fail_closed":
        raise SystemExit(f"{case_id} must fail closed")
    if row.get("unsupported_width") != expected["width"]:
        raise SystemExit(f"{case_id} expected unsupported width {expected['width']}, got {row.get('unsupported_width')}")
    if row.get("expected_detail") != expected["detail"]:
        raise SystemExit(f"{case_id} expected detail {expected['detail']}, got {row.get('expected_detail')}")
    if row.get("native_v2_compile_rc") == 0:
        raise SystemExit(f"{case_id} unexpectedly has rc=0")
    if row.get("diagnostic_fragment") != "native-v2 S5 unsupported numeric width":
        raise SystemExit(f"{case_id} missing stable diagnostic fragment")
    if row.get("elf_emitted") is not False:
        raise SystemExit(f"{case_id} must not emit an ELF")
    if row.get("machine_module_json_emitted") is not False:
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
if differential_receipt.get("stage_contract_level") != "S5_NATIVE_V2_LEAN_SINGLE_DIFFERENTIAL_PROMOTED_NOT_F128":
    raise SystemExit("differential receipt must declare promoted native-v2/lean_single stage contract")
if differential_receipt.get("case_count") != 33:
    raise SystemExit("differential receipt must contain exact 33 cases")
if differential_receipt.get("matched_case_count") != 31:
    raise SystemExit("differential receipt must contain exact 31 matched comparable cases")
if differential_receipt.get("reference_unavailable_case_count") != 2:
    raise SystemExit("differential receipt must contain exact two reference-unavailable cases")
required_differential_flags = [
    "native_v2_vs_lean_single_differential_complete",
    "s5_differential_native_v2_lean_single_complete",
    "differential_native_v2_vs_lean_single_promoted",
    "all_reference_available_cases_match_exit_and_stdout",
    "all_native_v2_cases_compile_without_legacy_fallback",
    "all_native_v2_cases_return_expected_exit",
    "all_lean_single_cases_return_expected_exit",
    "known_reference_unavailable_cases_recorded",
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
}
if set(differential_receipt.get("categories_compared", [])) != required_differential_categories:
    raise SystemExit("differential receipt categories mismatch")
differential_cases = {row.get("case_id"): row for row in differential_receipt.get("cases", [])}
required_unavailable = {
    "f64_println_call_stdout_4_5",
    "f64_let_bound_println_stdout_4_5",
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
    if row.get("lean_single_exit") != row.get("expected_exit"):
        raise SystemExit(f"{case_id} differential lean_single exit mismatch")
    if row.get("status") == "matched" and row.get("stdout_equal") is not True:
        raise SystemExit(f"{case_id} matched differential case must have equal stdout")
    if row.get("status") == "reference_unavailable" and row.get("stdout_equal") is not False:
        raise SystemExit(f"{case_id} unavailable differential case must record stdout mismatch")

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
        "surface": "normal_call_stack_args_gt6",
        "status": "promoted_by_normal_stack_call_receipt",
        "reason": "normal scalar calls with one and two outgoing stack args now have source-to-MachineModule receipts",
    },
    {
        "surface": "aggregate_return",
        "status": "promoted_by_local_imported_method_and_generic_layout_receipts",
        "reason": "local, imported, method, non-Big field names, Pair/Quad small shapes, and Wide9 72-byte SRET layouts now have source-to-MachineModule receipts",
    },
    {
        "surface": "imported_call",
        "status": "promoted_for_imported_aggregate_sret_module_boundary",
        "reason": "imported aggregate-return calls now export MachineModule JSON and execute one-arg, register multi-arg, and stack-arg SRET witnesses",
    },
    {
        "surface": "f128_numeric_width",
        "status": "not_promoted_by_this_slice",
        "reason": "i128/i256/u128/u256 wide integers are promoted by receipt; f128 source literal provenance is preserved for future binary128 lowering and native-v2 f128 execution still fails closed with stable diagnostics until IR/MIR/ABI/software-helper receipts exist",
    },
    {
        "surface": "f128_literal_provenance",
        "status": "promoted_by_parser_provenance_receipt_not_execution",
        "reason": "ExprFloatLit now preserves original source spelling in Expr.name before rounding to f64 compatibility value, enabling a future binary128 parser without claiming f128 execution",
    },
    {
        "surface": "unsupported_numeric_diagnostics",
        "status": "promoted_by_diagnostics_receipt",
        "reason": "unsupported f128/i512/u512 native-v2 numeric widths now fail closed without ELF, MachineModule JSON, segfault, or legacy fallback",
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
        "parse_float_literal_block_sha256": f128_literal_provenance_receipt["parse_float_literal_block_sha256"],
        "probe_source_sha256": f128_literal_provenance_receipt["probe_source_sha256"],
        "probe_check_rc": f128_literal_provenance_receipt["probe_check_rc"],
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
        "f128_literal_provenance_promoted": True,
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
        "generic_aggregate_sret_layout_receipt_recorded",
        "f128_literal_provenance_receipt_recorded",
        "unsupported_numeric_diagnostics_receipt_recorded",
        "differential_native_v2_vs_lean_single_receipt_recorded",
        "normal_call_stack_arg_receipt_recorded",
        "f128_execution_surfaces_not_promoted",
        "s4_negative_and_blocked_controls_not_promoted",
        "f128_numeric_tower_still_required_before_s5_ready",
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
    "s_full_contract": "blocked_until_f128_numeric_tower_exists",
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
    "s5_generic_aggregate_sret_layout_complete": True,
    "s4_s5_f128_literal_provenance_complete": True,
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
    "generic_aggregate_return_promoted": True,
    "generic_aggregate_local_layout_promoted": True,
    "generic_aggregate_imported_layout_promoted": True,
    "generic_aggregate_method_layout_promoted": True,
    "layout_derived_sret_alloc_promoted": True,
    "wide9_sret_alloc_72_bytes_promoted": True,
    "source_level_wide_arithmetic_promoted": True,
    "native_v2_wide_limb_backend_promoted": True,
    "wide_type_identity_and_safety_promoted": True,
    "s5_diagnostics_unsupported_numeric_complete": True,
    "unsupported_numeric_widths_fail_closed": True,
    "differential_native_v2_vs_lean_single_promoted": True,
    "unsupported_widths_do_not_emit_elf": True,
    "unsupported_widths_do_not_emit_machine_module_json": True,
    "unsupported_widths_do_not_segfault": True,
    "legacy_fallback_for_unsupported_widths": False,
    "f128_rejected_not_promoted": True,
    "i512_u512_rejected_not_promoted": True,
    "f128_promoted": False,
    "input_mir_effect_sha256": effect_receipt["receipt_sha256"],
    "sret_abi_receipt_sha256": sret_receipt["receipt_sha256"],
    "source_sret_receipt_sha256": source_sret_receipt["receipt_sha256"],
    "stack_call_receipt_sha256": stack_call_receipt["receipt_sha256"],
    "imported_sret_receipt_sha256": imported_sret_receipt["receipt_sha256"],
    "method_sret_receipt_sha256": method_sret_receipt["receipt_sha256"],
    "f64_xmm0_receipt_sha256": f64_xmm0_receipt["receipt_sha256"],
    "wide_int_receipt_sha256": wide_int_receipt["receipt_sha256"],
    "generic_aggregate_sret_receipt_sha256": generic_agg_receipt["receipt_sha256"],
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
        "f128 numeric tower width receipts",
    ],
}
receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
print(
    f"[madaros-v2-s5-program-mir-abi] ok programs={receipt['program_count']} "
    f"target={receipt['target']} sha={receipt['receipt_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-program-mir-abi] PASS: scalar i64/bool + SRET + f64/XMM0 + wide-int + generic aggregate compiler MachineModule ABI receipts are deterministic without claiming S5 FULL"
echo "[madaros-v2-s5-program-mir-abi] PASS: unsupported f128/i512/u512 native-v2 numeric widths fail closed without ELF, MachineModule JSON, segfault, or fallback"
echo "[madaros-v2-s5-program-mir-abi] PASS: native-v2 vs lean_single differential receipt covers promoted comparable S5 surfaces; f128 remains the explicit full blocker"
echo "[madaros-v2-s5-program-mir-abi] module=$MODULE"
echo "[madaros-v2-s5-program-mir-abi] receipt=$RECEIPT"
