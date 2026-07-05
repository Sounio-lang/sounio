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
EFFECT_GATE="${ROOT_DIR}/scripts/dev/madaros_v2_s5_mir_effect_gate.sh"
SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_sret_abi_receipt.py"
SOURCE_SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_source_sret_receipt.py"
STACK_CALL_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_stack_call_receipt.py"
IMPORTED_SRET_RECEIPT_TOOL="${ROOT_DIR}/scripts/dev/madaros_v2_s5_imported_sret_receipt.py"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
MANIFEST="${SOUNIO_MADAROS_V2_S5_SCALAR_MANIFEST:-tests/madaros/v2_s5/scalar_mir_abi_manifest.tsv}"
MODULE="$OUT_DIR/madaros_v2_s5_program_mir_abi.module.json"
RECEIPT="$OUT_DIR/madaros_v2_s5_program_mir_abi.receipt.json"
S5_RECEIPT_RESULTS="$OUT_DIR/madaros_v2_s5_source_receipts.tsv"
SRET_RECEIPT="$SRET_RECEIPT_DIR/madaros_v2_s5_sret_abi.receipt.json"
SOURCE_SRET_RECEIPT="$SOURCE_SRET_RECEIPT_DIR/madaros_v2_s5_source_sret.receipt.json"
STACK_CALL_RECEIPT="$STACK_CALL_RECEIPT_DIR/madaros_v2_s5_stack_call.receipt.json"
IMPORTED_SRET_RECEIPT="$IMPORTED_SRET_RECEIPT_DIR/madaros_v2_s5_imported_sret.receipt.json"

mkdir -p "$EFFECT_DIR" "$S5_RECEIPT_DIR" "$SRET_RECEIPT_DIR" "$SOURCE_SRET_RECEIPT_DIR" "$STACK_CALL_RECEIPT_DIR" "$IMPORTED_SRET_RECEIPT_DIR"

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

python3 - "$EFFECT_DIR" "$S5_RECEIPT_RESULTS" "$SRET_RECEIPT" "$SOURCE_SRET_RECEIPT" "$STACK_CALL_RECEIPT" "$IMPORTED_SRET_RECEIPT" "$MODULE" "$RECEIPT" <<'PY'
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
module_path = Path(sys.argv[7])
receipt_path = Path(sys.argv[8])

effect_receipt_path = effect_dir / "madaros_v2_s5_mir_effect.receipt.json"
effect_module_path = effect_dir / "madaros_v2_s5_mir_effect.module.json"
effect_receipt = load_json(effect_receipt_path)
effect_module = load_json(effect_module_path)
sret_receipt = load_json(sret_receipt_path)
source_sret_receipt = load_json(source_sret_receipt_path)
stack_call_receipt = load_json(stack_call_receipt_path)
imported_sret_receipt = load_json(imported_sret_receipt_path)

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
        "status": "partially_promoted_by_source_sret_local_and_imported_receipts",
        "reason": "local and imported source aggregate returns now have source-to-IrCallSret receipts across one-arg, register multi-arg, and stack-arg call shapes; method/generic aggregate surfaces remain",
    },
    {
        "surface": "imported_call",
        "status": "promoted_for_imported_aggregate_sret_module_boundary",
        "reason": "imported aggregate-return calls now export MachineModule JSON and execute one-arg, register multi-arg, and stack-arg SRET witnesses",
    },
    {
        "surface": "f64_call_return",
        "status": "not_promoted_by_this_slice",
        "reason": "must close XMM0 call-return receipt before f128 promotion",
    },
    {
        "surface": "f128_i256",
        "status": "not_promoted_by_this_slice",
        "reason": "requires layout, operations, ABI, diagnostics, and fallback semantics",
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
    "scalar_abi_receipts": {
        "schema": "madaros.v2.s5.abi_scalar_call_return/0.1",
        "target": "x86_64-linux",
        "arg_register_order": ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        "return_register": "rax",
        "normal_call_stack_args_promoted": True,
        "source_sret_stack_args_promoted": True,
        "imported_sret_module_boundary_promoted": True,
        "sret_promoted": True,
        "source_sret_local_one_arg_promoted": True,
        "source_sret_local_register_multi_arg_promoted": True,
        "aggregate_layout_promoted": True,
        "f64_xmm0_promoted": False,
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
        "normal_call_stack_arg_receipt_recorded",
        "method_generic_and_f64_surfaces_not_promoted",
        "s4_negative_and_blocked_controls_not_promoted",
        "full_abi_numeric_differential_gates_still_required_before_s5_ready",
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
    "s_full_contract": "blocked_until_full_abi_numeric_differential_gates_exist",
    "program_mir_shadow_serialized": True,
    "compiler_machine_module_exported": True,
    "real_program_mir_emitted": True,
    "real_abi_layout_emitted": True,
    "s5_sret_machine_module_abi_discriminator_complete": True,
    "s5_source_sret_local_one_arg_complete": True,
    "s5_source_sret_local_register_multi_arg_complete": True,
    "s5_source_sret_local_stack_arg_complete": True,
    "s5_imported_sret_module_boundary_complete": True,
    "s5_normal_call_stack_args_complete": True,
    "source_frontend_lowers_local_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_local_register_multi_arg_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_local_stack_arg_aggregate_return_to_IrCallSret": True,
    "source_frontend_lowers_imported_aggregate_return_to_IrCallSret": True,
    "input_mir_effect_sha256": effect_receipt["receipt_sha256"],
    "sret_abi_receipt_sha256": sret_receipt["receipt_sha256"],
    "source_sret_receipt_sha256": source_sret_receipt["receipt_sha256"],
    "stack_call_receipt_sha256": stack_call_receipt["receipt_sha256"],
    "imported_sret_receipt_sha256": imported_sret_receipt["receipt_sha256"],
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
    "normal_call_stack_args_promoted": module["scalar_abi_receipts"]["normal_call_stack_args_promoted"],
    "aggregate_layout_promoted": module["scalar_abi_receipts"]["aggregate_layout_promoted"],
    "not_promoted_surfaces": [item["surface"] for item in not_promoted],
    "negative_and_blocked_controls": [
        f"{item['case_id']}:{item['expected_status']}" for item in negative_and_blocked_controls
    ],
    "gate_invariants": module["roundtrip_contract"],
    "missing_full_obligations": [
        "method/generic/module-boundary aggregate return coverage",
        "f64 XMM0 call/return witnesses before f128 promotion",
        "numeric tower width receipts for f128/i256",
        "diagnostics and fallback semantics for unsupported layouts and numeric widths",
        "differential native-v2 vs interpreter/lean_single validation where available",
    ],
}
receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
print(
    f"[madaros-v2-s5-program-mir-abi] ok programs={receipt['program_count']} "
    f"target={receipt['target']} sha={receipt['receipt_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-program-mir-abi] PASS: scalar i64/bool + SRET compiler MachineModule ABI receipts are deterministic without claiming S5 FULL"
echo "[madaros-v2-s5-program-mir-abi] module=$MODULE"
echo "[madaros-v2-s5-program-mir-abi] receipt=$RECEIPT"
