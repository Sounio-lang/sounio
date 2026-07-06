#!/usr/bin/env bash
# Madaros v2 S5 MIR/ABI input-boundary gate: turn the current S4 selected
# extraction subset into deterministic MIR/ABI input classification receipts.
# This does not emit real MIR, does not mutate IR, and does not claim S5 FULL
# completion.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S5_MIR_ABI_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s5-mir-abi.XXXXXX)}"
PREFLIGHT_DIR="$OUT_DIR/preflight"
PREFLIGHT_GATE="${ROOT_DIR}/scripts/dev/madaros_v2_s5_preflight_gate.sh"
RECEIPT="$OUT_DIR/madaros_v2_s5_mir_abi_input_boundary.receipt.json"

mkdir -p "$PREFLIGHT_DIR"

echo "[madaros-v2-s5-mir-abi] START"
echo "[madaros-v2-s5-mir-abi] out=$OUT_DIR"

SOUNIO_MADAROS_V2_S5_PREFLIGHT_DIR="$PREFLIGHT_DIR" "$PREFLIGHT_GATE"

python3 - "$PREFLIGHT_DIR" "$RECEIPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def stable_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify_rewrite_abi(rewrite, decision):
    kind = rewrite.get("rewrite_kind")
    lowering = decision.get("lowering_effect")
    result_const = list(rewrite.get("result_const", []))
    if kind == "constant_fold_i64":
        result_kind = result_const[0] if result_const else "int"
        if result_kind == "bool":
            return {
                "mir_value_kind": "const_bool",
                "numeric_width_bits": 1,
                "abi_class": "scalar_bool",
                "register_class": "gpr_predicate",
                "layout_effect": "none",
                "lowering_contract": "constant expression replaced by exact bool constant",
            }
        return {
            "mir_value_kind": "const_i64",
            "numeric_width_bits": 64,
            "abi_class": "scalar_i64",
            "register_class": "gpr_i64",
            "layout_effect": "none",
            "lowering_contract": "constant expression replaced by exact i64 constant",
        }
    if kind == "symbolic_identity_i64":
        if lowering != "replace_binary_identity_expr_with_existing_value":
            raise SystemExit("symbolic identity MIR/ABI receipt requires value-ref lowering")
        return {
            "mir_value_kind": "existing_value_ref_i64",
            "numeric_width_bits": 64,
            "abi_class": "scalar_i64",
            "register_class": "gpr_i64",
            "layout_effect": "none",
            "lowering_contract": "binary identity replaced by existing SSA value",
        }
    if kind == "symbolic_reflexive_cmp_i64":
        if lowering not in {
            "replace_binary_predicate_expr_with_const_bool",
            "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated",
        }:
            raise SystemExit("reflexive comparison MIR/ABI receipt requires bool const lowering")
        return {
            "mir_value_kind": "const_bool",
            "numeric_width_bits": 1,
            "abi_class": "scalar_bool",
            "register_class": "gpr_predicate",
            "layout_effect": "none",
            "lowering_contract": "same-SSA comparison replaced by exact bool constant",
        }
    if kind == "symbolic_sub_self_i64":
        if lowering not in {
            "replace_binary_sub_self_expr_with_const_i64_zero",
            "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated",
        }:
            raise SystemExit("sub-self MIR/ABI receipt requires i64 zero lowering")
        return {
            "mir_value_kind": "const_i64",
            "numeric_width_bits": 64,
            "abi_class": "scalar_i64",
            "register_class": "gpr_i64",
            "layout_effect": "none",
            "lowering_contract": "same-SSA subtraction replaced by exact i64 zero",
        }
    raise SystemExit(f"unsupported S5 MIR/ABI rewrite kind: {kind}")


preflight_dir = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
s4_dir = preflight_dir / "s4"
preflight_path = preflight_dir / "madaros_v2_s5_preflight.receipt.json"
s4_summary_path = s4_dir / "madaros_v2_s4_gate.receipt.json"

preflight = load_json(preflight_path)
s4_summary = load_json(s4_summary_path)

if preflight.get("schema") != "madaros.v2.s5.preflight/0.1":
    raise SystemExit("bad S5 preflight schema")
if preflight.get("status") != "pass":
    raise SystemExit("S5 MIR/ABI boundary requires passing S5 preflight")
if preflight.get("stage_contract_level") != "S5_PREFLIGHT_NOT_FULL":
    raise SystemExit("S5 preflight must not claim FULL stage completion")
if preflight.get("s5_input_contract_ready") is not True:
    raise SystemExit("S5 preflight input contract is not ready")
if preflight.get("s5_full_complete") is not False:
    raise SystemExit("S5 preflight must not claim S5 FULL completion")
if preflight.get("s4_applied_extraction_consumed") is not True:
    raise SystemExit("S5 MIR/ABI boundary requires preflight to consume S4 applied extraction")
if preflight.get("input_applied_extraction_contract") != "madaros.v2.s4.applied_extraction/0.1":
    raise SystemExit("S5 MIR/ABI boundary requires the S4 applied-extraction contract")
if s4_summary.get("schema") != "madaros.v2.s4.gate/0.1" or s4_summary.get("status") != "pass":
    raise SystemExit("missing passing S4 gate summary")
applied_path = s4_dir / s4_summary.get("s4_applied_extraction_path", "")
applied_extraction = load_json(applied_path)
if applied_extraction.get("schema") != "madaros.v2.s4.applied_extraction/0.1":
    raise SystemExit("bad S4 applied extraction schema for S5 MIR/ABI boundary")
if applied_extraction.get("applied_extraction_sha256") != preflight.get("input_applied_extraction_sha256"):
    raise SystemExit("S5 MIR/ABI boundary applied-extraction hash mismatch")
if applied_extraction.get("application_applied_to_s5_input") is not True:
    raise SystemExit("S5 MIR/ABI boundary requires applied S5-input materialization")
if applied_extraction.get("application_applied_to_compiler_ir") is not False:
    raise SystemExit("S5 MIR/ABI boundary rejects compiler-IR mutation from S4")

applied_effects = {}
for case in applied_extraction.get("cases", []):
    for effect in case.get("selected_effects", []):
        rid = effect.get("rewrite_id")
        if rid in applied_effects:
            raise SystemExit(f"duplicate S4 applied effect in MIR/ABI boundary: {rid}")
        applied_effects[rid] = effect

rewrite_witnesses = []
rejected_ids = set()
blocked_ids = set()
for receipt_file in sorted(s4_dir.glob("*/*/*.s4.receipt.json")):
    if "/b/" in receipt_file.as_posix():
        continue
    receipt = load_json(receipt_file)
    rewrites = load_json(receipt_file.parent / receipt["rewrites_path"])
    extraction = load_json(receipt_file.parent / receipt["extraction_path"])
    decisions = {d["rewrite_id"]: d for d in extraction.get("decisions", [])}
    for rewrite in rewrites:
        rid = rewrite["proposed_rewrite_id"]
        decision = decisions.get(rid)
        if decision is None:
            raise SystemExit(f"missing extraction decision for rewrite {rid}")
        if rewrite.get("accepted") is not True:
            if rewrite.get("blocked") is True:
                blocked_ids.add(rid)
            else:
                rejected_ids.add(rid)
            continue
        if decision.get("decision") != "select" or decision.get("selected") is not True:
            raise SystemExit(f"accepted rewrite not selected by extraction: {rid}")
        if decision.get("mir_abi_safe") is not True:
            raise SystemExit(f"selected rewrite is not MIR/ABI safe: {rid}")
        if decision.get("abi_impact") != "none":
            raise SystemExit(f"selected rewrite has ABI impact: {rid}")
        if decision.get("extraction_applied_to_ir") is not False:
            raise SystemExit(f"S5 boundary rejects already-applied extraction: {rid}")
        if rewrite.get("validator") != "translation-validation" or rewrite.get("error_bound") != "0":
            raise SystemExit(f"S5 boundary requires exact translation validation: {rid}")
        applied_effect = applied_effects.get(rid)
        if applied_effect is None:
            raise SystemExit(f"S5 MIR/ABI boundary missing applied S5-input effect: {rid}")
        if applied_effect.get("application_applied_to_s5_input") is not True:
            raise SystemExit(f"S5 MIR/ABI boundary requires materialized applied effect: {rid}")
        if applied_effect.get("application_applied_to_compiler_ir") is not False:
            raise SystemExit(f"S5 MIR/ABI boundary rejects compiler-IR-mutating applied effect: {rid}")
        if applied_effect.get("mir_abi_safe") is not True or applied_effect.get("abi_impact") != "none":
            raise SystemExit(f"S5 MIR/ABI boundary requires ABI-safe applied effect: {rid}")
        if applied_effect.get("output_enode_sha256") != rewrite.get("rewritten_enode_sha256"):
            raise SystemExit(f"S5 MIR/ABI boundary applied effect output mismatch: {rid}")
        if not applied_effect.get("applied_effect_sha256"):
            raise SystemExit(f"S5 MIR/ABI boundary missing applied effect hash: {rid}")
        abi = classify_rewrite_abi(rewrite, decision)
        keep_producer = str(decision.get("lowering_effect", "")).endswith("_keep_producer_evaluated")
        producer_policy = rewrite.get("producer_evaluation_policy", "not-required")
        if keep_producer and producer_policy != "direct_call_leaf_pure_keep_producer_evaluated":
            raise SystemExit(f"keep-producer lowering missing producer policy: {rid}")
        witness = {
            "schema": "madaros.v2.s5.mir_abi_input_witness/0.1",
            "case_id": receipt["case_id"],
            "source": receipt["source"],
            "rewrite_id": rid,
            "rewrite_kind": rewrite["rewrite_kind"],
            "proposal_kind": rewrite["proposal_kind"],
            "input_hlir_sha256": receipt["input_hlir_sha256"],
            "input_egraph_sha256": receipt["egraph_sha256"],
            "input_extraction_sha256": receipt["extraction_sha256"],
            "input_applied_extraction_sha256": applied_extraction["applied_extraction_sha256"],
            "source_applied_effect_sha256": applied_effect["applied_effect_sha256"],
            "post_apply_selected_enode_sha256": applied_effect["post_apply_selected_enode_sha256"],
            "post_apply_s5_input_hlir_sha256": applied_effect["post_mutation_hlir_sha256"],
            "post_apply_s5_input_egraph_sha256": applied_effect["post_mutation_egraph_sha256"],
            "original_enode_sha256": rewrite["original_enode_sha256"],
            "rewritten_enode_sha256": rewrite["rewritten_enode_sha256"],
            "lowering_effect": decision["lowering_effect"],
            "lowering_contract": abi["lowering_contract"],
            "mir_value_kind": abi["mir_value_kind"],
            "numeric_width_bits": abi["numeric_width_bits"],
            "abi_class": abi["abi_class"],
            "register_class": abi["register_class"],
            "layout_effect": abi["layout_effect"],
            "call_signature_effect": "none",
            "stack_effect": "none",
            "sret_effect": "none",
            "aggregate_layout_effect": "none",
            "producer_evaluation_preservation": (
                "required_keep_original_producer_evaluated"
                if keep_producer
                else "not-required-for-this-rewrite"
            ),
            "producer_evaluation_policy": producer_policy,
            "abi_impact": decision["abi_impact"],
            "mir_abi_safe": True,
            "exact_fallback_expr_sha256": rewrite["exact_fallback_expr_sha256"],
            "validator_log_sha256": rewrite["validator_log_sha256"],
            "selected_for_s5_input_boundary": True,
            "applied_to_mir": False,
            "applied_to_abi": False,
        }
        witness["witness_sha256"] = sha256_text(stable_json(witness))
        rewrite_witnesses.append(witness)

if len(rewrite_witnesses) != preflight.get("selected_rewrite_count"):
    raise SystemExit("S5 MIR/ABI witness count does not match preflight selected rewrite count")
if len(rejected_ids) != preflight.get("semantic_rejected_rewrite_count"):
    raise SystemExit("S5 MIR/ABI rejected count mismatch")
if len(blocked_ids) != preflight.get("blocked_rewrite_count"):
    raise SystemExit("S5 MIR/ABI blocked count mismatch")

receipt = {
    "schema": "madaros.v2.s5.mir_abi_input_boundary/0.1",
    "status": "pass",
    "stage_contract_level": "S5_MIR_ABI_INPUT_BOUNDARY_NOT_FULL",
    "s5_input_contract_ready": True,
    "s5_mir_abi_input_boundary_complete": True,
    "s5_mir_abi_boundary_complete": False,
    "s5_ready": False,
    "s5_implemented": False,
    "s5_full_complete": False,
    "s_full_contract": "blocked_until_real_mir_serialization_abi_layout_numeric_and_differential_gates_exist",
    "real_mir_emitted": False,
    "real_abi_layout_emitted": False,
    "mir_schema": "not-emitted: input-boundary classification only",
    "abi_schema": "madaros.v2.s5.abi_input_classification/0.1",
    "input_preflight_sha256": preflight["preflight_sha256"],
    "input_s4_gate_sha256": s4_summary["gate_sha256"],
    "input_applied_extraction_contract": applied_extraction["schema"],
    "input_applied_extraction_sha256": applied_extraction["applied_extraction_sha256"],
    "s4_applied_extraction_consumed": True,
    "input_contract": preflight["schema"],
    "selected_rewrite_count": len(rewrite_witnesses),
    "semantic_rejected_rewrite_count": len(rejected_ids),
    "blocked_rewrite_count": len(blocked_ids),
    "mir_abi_safe_subset": sorted({w["rewrite_kind"] for w in rewrite_witnesses}),
    "lowering_effects": sorted({w["lowering_effect"] for w in rewrite_witnesses}),
    "abi_classes": sorted({w["abi_class"] for w in rewrite_witnesses}),
    "register_classes": sorted({w["register_class"] for w in rewrite_witnesses}),
    "producer_evaluation_preservation_modes": sorted({w["producer_evaluation_preservation"] for w in rewrite_witnesses}),
    "input_boundary_invariants": [
        "selected_ids_equal_s4_accepted_selected_ids",
        "each_witness_carries_s4_applied_effect_hash",
        "input_applied_extraction_hash_matches_s5_preflight",
        "abi_impact_none_for_every_selected_rewrite",
        "no_call_signature_effect",
        "no_stack_effect",
        "no_sret_effect",
        "no_aggregate_layout_effect",
        "no_mir_or_abi_mutation_applied",
        "keep_producer_evaluated_for_local_leaf_call_rewrites",
    ],
    "rewrite_witnesses": rewrite_witnesses,
    "missing_full_obligations": [
        "real MIR serialization and hash receipts",
        "ABI layout receipts for scalar, aggregate, SRET, imported call, and return paths",
        "f128 IR/MIR/ABI/software-helper receipts before f128 promotion",
        "f128 IR/MIR/ABI/software-helper receipts",
        "diagnostics and fallback semantics for unsupported layouts and numeric widths",
        "differential native-v2 vs interpreter/lean_single validation where available",
    ],
}
receipt["boundary_sha256"] = sha256_text(stable_json(receipt))
receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(
    f"[madaros-v2-s5-mir-abi] ok rewrites={receipt['selected_rewrite_count']} "
    f"blocked={receipt['blocked_rewrite_count']} abi_classes={','.join(receipt['abi_classes'])} "
    f"sha={receipt['boundary_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-mir-abi] PASS: S5 MIR/ABI input-boundary receipts classify the current S4 selected subset without claiming real MIR/ABI or S5 FULL"
echo "[madaros-v2-s5-mir-abi] receipt=$RECEIPT"
