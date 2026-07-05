#!/usr/bin/env bash
# Madaros v2 S5 MIR-effect gate: serialize the current S4 selected rewrite
# subset into deterministic MIR-effect records and prove JSON roundtrip
# stability. This is not full program MIR, does not mutate IR, and does not
# claim S5 FULL completion.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S5_MIR_EFFECT_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s5-mir-effect.XXXXXX)}"
A_DIR="$OUT_DIR/a"
NATIVE_DIR="$OUT_DIR/native_scalar_witnesses"
MIR_ABI_GATE="${ROOT_DIR}/scripts/dev/madaros_v2_s5_mir_abi_gate.sh"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
MANIFEST="${SOUNIO_MADAROS_V2_S5_SCALAR_MANIFEST:-tests/madaros/v2_s5/scalar_mir_abi_manifest.tsv}"
MODULE="$OUT_DIR/madaros_v2_s5_mir_effect.module.json"
RECEIPT="$OUT_DIR/madaros_v2_s5_mir_effect.receipt.json"
NATIVE_RESULTS="$OUT_DIR/madaros_v2_s5_scalar_native_results.tsv"

mkdir -p "$A_DIR" "$NATIVE_DIR"

echo "[madaros-v2-s5-mir-effect] START"
echo "[madaros-v2-s5-mir-effect] out=$OUT_DIR"

SOUNIO_MADAROS_V2_S5_MIR_ABI_DIR="$A_DIR" "$MIR_ABI_GATE"

if [[ ! -f "$MANIFEST" ]]; then
  echo "[madaros-v2-s5-mir-effect] FAIL: missing scalar ABI manifest: $MANIFEST" >&2
  exit 1
fi

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_sha256_stdin() {
  sha256sum 2>/dev/null | awk '{print $1}' || shasum -a 256 | awk '{print $1}'
}

elf_exec_metrics() {
  python3 - "$1" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
blob = path.read_bytes()
if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
    raise SystemExit(f"not an ELF64 little-endian binary: {path}")

phoff = struct.unpack_from("<Q", blob, 32)[0]
phentsize = struct.unpack_from("<H", blob, 54)[0]
phnum = struct.unpack_from("<H", blob, 56)[0]
internal_call_count = 0
ret_count = 0
syscall_count = 0

for index in range(phnum):
    off = phoff + index * phentsize
    if off + phentsize > len(blob):
        raise SystemExit(f"bad program header table in {path}")
    p_type, p_flags = struct.unpack_from("<II", blob, off)
    p_offset, p_vaddr, _p_paddr, p_filesz, _p_memsz, _p_align = struct.unpack_from("<QQQQQQ", blob, off + 8)
    if p_type != 1 or not (p_flags & 1):
        continue
    segment = blob[p_offset:p_offset + p_filesz]
    if len(segment) != p_filesz:
        raise SystemExit(f"truncated executable segment in {path}")
    ret_count += segment.count(b"\xc3")
    syscall_count += segment.count(b"\x0f\x05")
    for cursor in range(0, max(0, len(segment) - 4)):
        if segment[cursor] != 0xE8:
            continue
        rel = struct.unpack_from("<i", segment, cursor + 1)[0]
        target = p_vaddr + cursor + 5 + rel
        if p_vaddr <= target < p_vaddr + p_filesz:
            internal_call_count += 1

print(f"{internal_call_count}\t{ret_count}\t{syscall_count}")
PY
}

printf 'case_id\tprogram\texpected_exit\tactual_exit\tabi_kind\trequired_machine_ops\telf_sha256\tcompile_log_sha256\tstdout_sha256\tstderr_sha256\telf_internal_call_count\telf_ret_count\telf_syscall_count\tstatus\n' >"$NATIVE_RESULTS"

while IFS=$'\t' read -r case_id program expected_exit abi_kind required_machine_ops; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi
  if [[ ! -f "$program" ]]; then
    echo "[madaros-v2-s5-mir-effect] FAIL: missing scalar witness program for $case_id: $program" >&2
    exit 1
  fi
  bin_path="$NATIVE_DIR/$case_id.native_v2"
  compile_log="$NATIVE_DIR/$case_id.compile.log"
  stdout_log="$NATIVE_DIR/$case_id.stdout"
  stderr_log="$NATIVE_DIR/$case_id.stderr"
  if ! "$COMPILER" --native-v2-compile "$program" -o "$bin_path" >"$compile_log" 2>&1; then
    echo "[madaros-v2-s5-mir-effect] FAIL: native-v2 compile failed for $case_id" >&2
    tail -n 80 "$compile_log" >&2 || true
    exit 1
  fi
  chmod +x "$bin_path" 2>/dev/null || true
  set +e
  "$bin_path" >"$stdout_log" 2>"$stderr_log"
  actual_exit=$?
  set -e
  if [[ "$actual_exit" != "$expected_exit" ]]; then
    echo "[madaros-v2-s5-mir-effect] FAIL: $case_id expected_exit=$expected_exit actual_exit=$actual_exit" >&2
    exit 1
  fi
  IFS=$'\t' read -r elf_internal_call_count elf_ret_count elf_syscall_count < <(elf_exec_metrics "$bin_path")
  status="ok"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$program" "$expected_exit" "$actual_exit" "$abi_kind" "$required_machine_ops" \
    "$(portable_sha256 "$bin_path")" "$(sed "s|$OUT_DIR|<OUT_DIR>|g" "$compile_log" | portable_sha256_stdin)" \
    "$(portable_sha256 "$stdout_log")" "$(portable_sha256 "$stderr_log")" \
    "$elf_internal_call_count" "$elf_ret_count" "$elf_syscall_count" "$status" \
    >>"$NATIVE_RESULTS"
done <"$MANIFEST"

python3 - "$A_DIR/madaros_v2_s5_mir_abi_input_boundary.receipt.json" "$NATIVE_RESULTS" "$MODULE" "$RECEIPT" <<'PY'
import hashlib
import json
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


def classify_mir_effect(witness: dict[str, Any]) -> dict[str, Any]:
    value_kind = witness["mir_value_kind"]
    lowering = witness["lowering_effect"]
    rewrite_kind = witness["rewrite_kind"]
    width = witness["numeric_width_bits"]

    if value_kind == "const_i64":
        opcode = "mir.const.i64"
        result_type = "i64"
        materialization = {
            "kind": "const_i64",
            "value_source": "rewrite_result_const",
            "value_semantics": "exact_twos_complement_i64",
        }
    elif value_kind == "const_bool":
        opcode = "mir.const.bool"
        result_type = "bool"
        materialization = {
            "kind": "const_bool",
            "value_source": "rewrite_result_const",
            "value_semantics": "canonical_bool_0_or_1",
        }
    elif value_kind == "existing_value_ref_i64":
        opcode = "mir.alias.i64"
        result_type = "i64"
        materialization = {
            "kind": "existing_value_ref_i64",
            "value_source": "same_hlir_value_ref",
            "value_semantics": "no_new_computation_alias_existing_ssa_value",
        }
    else:
        raise SystemExit(f"unsupported MIR value kind for effect serialization: {value_kind}")

    allowed_lowerings = {
        "constant_fold_i64": {
            "replace_binary_constant_expr_with_const",
        },
        "symbolic_identity_i64": {
            "replace_binary_identity_expr_with_existing_value",
        },
        "symbolic_reflexive_cmp_i64": {
            "replace_binary_predicate_expr_with_const_bool",
            "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated",
        },
        "symbolic_sub_self_i64": {
            "replace_binary_sub_self_expr_with_const_i64_zero",
            "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated",
        },
    }
    if lowering not in allowed_lowerings.get(rewrite_kind, set()):
        raise SystemExit(f"unsupported lowering for MIR-effect serialization: {rewrite_kind} {lowering}")

    if width not in {1, 64}:
        raise SystemExit(f"MIR-effect slice only supports bool/i64 widths, got {width}")

    return {
        "opcode": opcode,
        "result_type": result_type,
        "materialization": materialization,
    }


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    parsed = json.loads(first)
    second = stable_json(parsed)
    if first != second:
        raise SystemExit("MIR-effect canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


input_receipt_path = Path(sys.argv[1])
native_results_path = Path(sys.argv[2])
module_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])

boundary = load_json(input_receipt_path)
if boundary.get("schema") != "madaros.v2.s5.mir_abi_input_boundary/0.1":
    raise SystemExit("bad S5 MIR/ABI input-boundary schema")
if boundary.get("status") != "pass":
    raise SystemExit("S5 MIR-effect gate requires a passing input-boundary receipt")
if boundary.get("s5_mir_abi_input_boundary_complete") is not True:
    raise SystemExit("S5 MIR-effect gate requires completed input-boundary receipt")
for false_field in [
    "s5_mir_abi_boundary_complete",
    "s5_ready",
    "s5_implemented",
    "s5_full_complete",
    "real_mir_emitted",
    "real_abi_layout_emitted",
]:
    if boundary.get(false_field) is not False:
        raise SystemExit(f"input-boundary must not overclaim {false_field}")

witnesses = list(boundary.get("rewrite_witnesses", []))
if len(witnesses) != boundary.get("selected_rewrite_count"):
    raise SystemExit("input-boundary witness count mismatch")

native_rows: list[dict[str, Any]] = []
lines = native_results_path.read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("missing scalar native witness results")
header = lines[0].split("\t")
required_header = [
    "case_id",
    "program",
    "expected_exit",
    "actual_exit",
    "abi_kind",
    "required_machine_ops",
    "elf_sha256",
    "compile_log_sha256",
    "stdout_sha256",
    "stderr_sha256",
    "elf_internal_call_count",
    "elf_ret_count",
    "elf_syscall_count",
    "status",
]
if header != required_header:
    raise SystemExit("bad scalar native witness header")
for line in lines[1:]:
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) != len(header):
        raise SystemExit(f"bad scalar native witness row: {line}")
    row = dict(zip(header, parts))
    if row["status"] != "ok":
        raise SystemExit(f"scalar native witness did not pass: {row['case_id']}")
    if row["actual_exit"] != row["expected_exit"]:
        raise SystemExit(f"scalar native witness exit mismatch: {row['case_id']}")
    if not row["elf_sha256"] or len(row["elf_sha256"]) != 64:
        raise SystemExit(f"scalar native witness missing ELF sha: {row['case_id']}")
    for int_field in ["elf_internal_call_count", "elf_ret_count", "elf_syscall_count"]:
        try:
            value = int(row[int_field])
        except ValueError as exc:
            raise SystemExit(f"scalar native witness has non-integer {int_field}: {row['case_id']}") from exc
        if value < 0:
            raise SystemExit(f"scalar native witness has negative {int_field}: {row['case_id']}")
    native_rows.append(row)

required_case_ids = {
    "scalar_i64_literal_return_42",
    "scalar_i64_direct_call_return_42",
    "scalar_bool_direct_call_return_1",
}
case_ids = [row["case_id"] for row in native_rows]
if len(native_rows) != len(required_case_ids):
    raise SystemExit(f"scalar native witness manifest must contain exactly {len(required_case_ids)} rows, got {len(native_rows)}")
if set(case_ids) != required_case_ids or len(case_ids) != len(set(case_ids)):
    raise SystemExit(f"scalar native witness manifest must contain exactly {sorted(required_case_ids)}, got {sorted(case_ids)}")

required_abi_kinds = {
    "scalar_i64_return",
    "scalar_i64_direct_call_return",
    "scalar_bool_direct_call_return",
}
seen_abi_kinds = {row["abi_kind"] for row in native_rows}
if seen_abi_kinds != required_abi_kinds:
    raise SystemExit(f"scalar native witnesses must cover {sorted(required_abi_kinds)}, got {sorted(seen_abi_kinds)}")
for row in native_rows:
    ops = set(filter(None, row["required_machine_ops"].split(",")))
    if row["abi_kind"] == "scalar_i64_return" and ops != {"RET"}:
        raise SystemExit("literal scalar i64 witness should require only RET")
    internal_calls = int(row["elf_internal_call_count"])
    ret_count = int(row["elf_ret_count"])
    syscall_count = int(row["elf_syscall_count"])
    if ret_count < 1 or syscall_count < 1:
        raise SystemExit(f"scalar native witness must contain return and syscall instructions: {row['case_id']}")
    if row["abi_kind"] == "scalar_i64_return" and internal_calls != 1:
        raise SystemExit(f"literal scalar i64 witness should have exactly one runtime-to-main internal call, got {internal_calls}")
    if row["abi_kind"] in {"scalar_i64_direct_call_return", "scalar_bool_direct_call_return"}:
        required = {"ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "RET"}
        if ops != required:
            raise SystemExit(f"direct-call scalar witness missing machine op contract: {row['case_id']}")
        if internal_calls < 2:
            raise SystemExit(f"direct-call scalar witness must contain an additional internal call in the ELF: {row['case_id']}")

effects: list[dict[str, Any]] = []
seen_ids: set[str] = set()
for index, witness in enumerate(witnesses):
    rid = witness["rewrite_id"]
    if rid in seen_ids:
        raise SystemExit(f"duplicate rewrite id in S5 MIR-effect input: {rid}")
    seen_ids.add(rid)
    if witness.get("selected_for_s5_input_boundary") is not True:
        raise SystemExit("MIR-effect serialization only accepts selected S5 input witnesses")
    if witness.get("mir_abi_safe") is not True:
        raise SystemExit("MIR-effect serialization requires MIR/ABI safe witness")
    if witness.get("abi_impact") != "none":
        raise SystemExit("MIR-effect serialization rejects ABI-impacting witnesses")
    for no_effect_field in [
        "call_signature_effect",
        "stack_effect",
        "sret_effect",
        "aggregate_layout_effect",
    ]:
        if witness.get(no_effect_field) != "none":
            raise SystemExit(f"MIR-effect serialization rejects {no_effect_field}")
    if witness.get("applied_to_mir") is not False or witness.get("applied_to_abi") is not False:
        raise SystemExit("MIR-effect serialization must start from non-mutating input-boundary witnesses")
    if not witness.get("exact_fallback_expr_sha256") or not witness.get("validator_log_sha256"):
        raise SystemExit("MIR-effect serialization requires fallback and validator hashes")

    effect_kind = classify_mir_effect(witness)
    effect = {
        "schema": "madaros.v2.s5.mir_effect/0.1",
        "effect_index": index,
        "case_id": witness["case_id"],
        "source": witness["source"],
        "rewrite_id": rid,
        "rewrite_kind": witness["rewrite_kind"],
        "proposal_kind": witness["proposal_kind"],
        "mir_opcode": effect_kind["opcode"],
        "mir_result_type": effect_kind["result_type"],
        "mir_materialization": effect_kind["materialization"],
        "input_hlir_sha256": witness["input_hlir_sha256"],
        "input_egraph_sha256": witness["input_egraph_sha256"],
        "input_extraction_sha256": witness["input_extraction_sha256"],
        "original_enode_sha256": witness["original_enode_sha256"],
        "rewritten_enode_sha256": witness["rewritten_enode_sha256"],
        "lowering_effect": witness["lowering_effect"],
        "lowering_contract": witness["lowering_contract"],
        "numeric_width_bits": witness["numeric_width_bits"],
        "abi_class": witness["abi_class"],
        "register_class": witness["register_class"],
        "producer_evaluation_preservation": witness["producer_evaluation_preservation"],
        "producer_evaluation_policy": witness["producer_evaluation_policy"],
        "exact_fallback_expr_sha256": witness["exact_fallback_expr_sha256"],
        "validator_log_sha256": witness["validator_log_sha256"],
        "source_witness_sha256": witness["witness_sha256"],
        "abi_impact": "none",
        "program_ir_mutation": False,
        "full_program_mir_emitted": False,
    }
    effect["effect_sha256"] = sha256_text(stable_json(effect))
    effects.append(effect)

if len(effects) != boundary.get("selected_rewrite_count"):
    raise SystemExit("MIR-effect count does not match selected rewrite count")

module = {
    "schema": "madaros.v2.s5.mir_effect_module/0.1",
    "stage_contract_level": "S5_MIR_EFFECT_ROUNDTRIP_NOT_FULL",
    "input_boundary_schema": boundary["schema"],
    "input_boundary_sha256": boundary["boundary_sha256"],
    "input_preflight_sha256": boundary["input_preflight_sha256"],
    "input_s4_gate_sha256": boundary["input_s4_gate_sha256"],
    "effect_count": len(effects),
    "semantic_rejected_rewrite_count": boundary["semantic_rejected_rewrite_count"],
    "blocked_rewrite_count": boundary["blocked_rewrite_count"],
    "mir_effects": effects,
    "mir_opcodes": sorted({effect["mir_opcode"] for effect in effects}),
    "mir_result_types": sorted({effect["mir_result_type"] for effect in effects}),
    "abi_classes": boundary["abi_classes"],
    "register_classes": boundary["register_classes"],
    "scalar_native_witnesses": native_rows,
    "scalar_native_witness_count": len(native_rows),
    "scalar_native_abi_kinds": sorted(seen_abi_kinds),
    "scalar_native_machine_op_contract": {
        "source": "self-hosted/native/machine_ir.sio native_v2_lower_legal_function_from_ir_ref",
        "literal_return": ["RET"],
        "direct_call_return": ["ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "RET"],
        "abi_arg_registers": ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        "abi_return_register": "rax",
        "stack_args_promoted": False,
        "sret_promoted": False,
        "aggregate_return_promoted": False,
        "f64_call_return_promoted": False,
    },
    "roundtrip_contract": [
        "canonical_json_stable_after_parse_dump",
        "effect_count_equals_selected_s4_rewrites",
        "semantic_rejected_and_blocked_rewrites_excluded",
        "scalar_native_witnesses_exit_with_expected_codes",
        "scalar_native_elf_contains_expected_internal_call_shape",
        "scalar_i64_bool_direct_call_return_machine_op_contract_recorded",
        "no_program_ir_mutation",
        "no_abi_layout_or_call_signature_effect",
    ],
    "program_mir_complete": False,
    "program_ir_mutation": False,
}
canonical_module, module_sha = canonical_roundtrip(module)
module["mir_effect_module_sha256"] = module_sha
canonical_module_with_hash, module_with_hash_sha = canonical_roundtrip(module)
module_path.write_text(pretty_json(module), encoding="utf-8")

reloaded_module = load_json(module_path)
if stable_json(reloaded_module) != stable_json(module):
    raise SystemExit("pretty JSON MIR-effect module does not roundtrip to canonical module")

receipt = {
    "schema": "madaros.v2.s5.mir_effect_roundtrip/0.1",
    "status": "pass",
    "stage_contract_level": "S5_MIR_EFFECT_ROUNDTRIP_NOT_FULL",
    "s5_mir_effect_roundtrip_complete": True,
    "s5_scalar_i64_bool_direct_call_return_slice_complete": True,
    "s5_mir_abi_input_boundary_complete": True,
    "s5_mir_abi_boundary_complete": False,
    "s5_ready": False,
    "s5_implemented": False,
    "s5_full_complete": False,
    "s_full_contract": "blocked_until_full_program_mir_abi_layout_numeric_and_differential_gates_exist",
    "real_mir_effects_serialized": True,
    "real_program_mir_emitted": False,
    "real_abi_layout_emitted": False,
    "program_ir_mutation": False,
    "input_boundary_sha256": boundary["boundary_sha256"],
    "mir_effect_module_path": module_path.name,
    "mir_effect_module_sha256": module_sha,
    "mir_effect_module_with_hash_sha256": module_with_hash_sha,
    "canonical_roundtrip_sha256": sha256_text(canonical_module_with_hash),
    "effect_count": len(effects),
    "scalar_native_witness_count": len(native_rows),
    "scalar_native_abi_kinds": sorted(seen_abi_kinds),
    "selected_rewrite_count": boundary["selected_rewrite_count"],
    "semantic_rejected_rewrite_count": boundary["semantic_rejected_rewrite_count"],
    "blocked_rewrite_count": boundary["blocked_rewrite_count"],
    "mir_opcodes": module["mir_opcodes"],
    "mir_result_types": module["mir_result_types"],
    "abi_classes": module["abi_classes"],
    "register_classes": module["register_classes"],
    "gate_invariants": [
        "input_s4_gate_double_emits_each_case",
        "canonical_json_roundtrip_stable",
        "effect_count_equals_selected_rewrite_count",
        "scalar_native_witnesses_exit_with_expected_codes",
        "scalar_native_elf_contains_expected_internal_call_shape",
        "scalar_direct_call_return_contract_requires_arg_move_call_capture_ret_store_ret",
        "rejected_and_blocked_rewrites_excluded_from_mir_effects",
        "no_program_ir_mutation",
        "no_call_stack_sret_or_aggregate_layout_effects",
    ],
    "missing_full_obligations": [
        "full program MIR serialization and hash receipts",
        "ABI layout receipts for scalar, aggregate, SRET, imported call, and return paths",
        "native-v2 codegen application of the MIR-effect module",
        "f128 IR/MIR/ABI/software-helper receipts before f128 promotion",
        "f128 IR/MIR/ABI/software-helper receipts",
        "diagnostics and fallback semantics for unsupported MIR/ABI effects",
        "differential native-v2 vs interpreter/lean_single validation where available",
    ],
}
receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
print(
    f"[madaros-v2-s5-mir-effect] ok effects={receipt['effect_count']} "
    f"native_witnesses={receipt['scalar_native_witness_count']} "
    f"opcodes={','.join(receipt['mir_opcodes'])} sha={receipt['receipt_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-mir-effect] PASS: S5 MIR-effect module roundtrips for the current selected subset without claiming full program MIR/ABI or S5 FULL"
echo "[madaros-v2-s5-mir-effect] module=$MODULE"
echo "[madaros-v2-s5-mir-effect] receipt=$RECEIPT"
