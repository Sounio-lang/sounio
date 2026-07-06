#!/usr/bin/env bash
# Madaros v2 S5 preflight: prove the current S4 boundary receipts are
# MIR/ABI-safe inputs. This does not implement S5 and does not claim S5 is
# globally ready; it only certifies the exact, extraction-selected S4 subset
# consumed here.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S5_PREFLIGHT_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s5-preflight.XXXXXX)}"
S4_DIR="$OUT_DIR/s4"
S4_GATE="${ROOT_DIR}/scripts/dev/madaros_v2_s4_gate.sh"

mkdir -p "$S4_DIR"

echo "[madaros-v2-s5-preflight] START"
echo "[madaros-v2-s5-preflight] out=$OUT_DIR"

SOUNIO_MADAROS_V2_S4_GATE_DIR="$S4_DIR" "$S4_GATE"

python3 - "$S4_DIR" "$OUT_DIR/madaros_v2_s5_preflight.receipt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

s4_dir = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
summary_path = s4_dir / "madaros_v2_s4_gate.receipt.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("schema") != "madaros.v2.s4.gate/0.1":
    raise SystemExit("missing S4 gate summary")
if summary.get("status") != "pass":
    raise SystemExit("S4 gate did not pass")
if summary.get("stage_contract_level") != "S4_BOUNDARY_NOT_FULL":
    raise SystemExit("S5 preflight requires S4 boundary-not-full classification")
if summary.get("s4_full_complete") is not False:
    raise SystemExit("S5 preflight requires S4 full-complete=false")
if summary.get("s4_to_s5_application_plan_complete") is not True:
    raise SystemExit("S5 preflight requires explicit S4->S5 application plan")
if summary.get("s4_applied_extraction_complete") is not True:
    raise SystemExit("S5 preflight requires explicit S4 applied extraction artifact")
plan_path = s4_dir / summary.get("s4_to_s5_application_plan_path", "")
if not plan_path.is_file():
    raise SystemExit("missing S4->S5 application plan")
application_plan = json.loads(plan_path.read_text(encoding="utf-8"))
if application_plan.get("schema") != "madaros.v2.s4.to_s5_application_plan/0.1":
    raise SystemExit("bad S4->S5 application plan schema")
if application_plan.get("status") != "pass":
    raise SystemExit("S4->S5 application plan did not pass")
if application_plan.get("stage_contract_level") != "S4_TO_S5_EXACT_APPLICATION_PLAN_NOT_MUTATING":
    raise SystemExit("S4->S5 application plan must be non-mutating exact boundary")
if application_plan.get("s4_to_s5_application_plan_complete") is not True:
    raise SystemExit("S4->S5 application plan not complete")
if application_plan.get("ir_mutation_allowed") is not False or application_plan.get("application_applied_to_ir") is not False:
    raise SystemExit("S5 preflight rejects mutating S4->S5 application plans")
plan_hash = application_plan.get("application_plan_sha256")
if not plan_hash or plan_hash != summary.get("s4_to_s5_application_plan_sha256"):
    raise SystemExit("S4->S5 application plan hash mismatch with S4 summary")
plan_for_hash = dict(application_plan)
plan_for_hash.pop("application_plan_sha256", None)
computed_plan_hash = hashlib.sha256(
    json.dumps(plan_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest()
if computed_plan_hash != plan_hash:
    raise SystemExit("S4->S5 application plan canonical hash mismatch")

applied_path = s4_dir / summary.get("s4_applied_extraction_path", "")
if not applied_path.is_file():
    raise SystemExit("missing S4 applied extraction artifact")
applied_extraction = json.loads(applied_path.read_text(encoding="utf-8"))
if applied_extraction.get("schema") != "madaros.v2.s4.applied_extraction/0.1":
    raise SystemExit("bad S4 applied extraction schema")
if applied_extraction.get("status") != "pass":
    raise SystemExit("S4 applied extraction did not pass")
if applied_extraction.get("stage_contract_level") != "S4_EXACT_EXTRACTION_APPLIED_TO_S5_INPUT_NOT_COMPILER_IR":
    raise SystemExit("S4 applied extraction has wrong stage contract")
if applied_extraction.get("s4_downstream_integration_slice_complete") is not True:
    raise SystemExit("S4 applied extraction must complete the downstream integration slice")
if applied_extraction.get("s4_full_complete") is not False:
    raise SystemExit("S4 applied extraction must not claim S4 FULL")
if applied_extraction.get("input_application_plan_sha256") != application_plan["application_plan_sha256"]:
    raise SystemExit("S4 applied extraction input plan hash mismatch")
if applied_extraction.get("application_applied_to_s5_input") is not True:
    raise SystemExit("S4 applied extraction must materialize S5 input effects")
if applied_extraction.get("application_applied_to_compiler_ir") is not False:
    raise SystemExit("S4 applied extraction must not claim compiler IR mutation")
if applied_extraction.get("ir_mutation_allowed") is not False:
    raise SystemExit("S4 applied extraction must remain non-mutating")
applied_hash = applied_extraction.get("applied_extraction_sha256")
if not applied_hash or applied_hash != summary.get("s4_applied_extraction_sha256"):
    raise SystemExit("S4 applied extraction hash mismatch with S4 summary")
applied_for_hash = dict(applied_extraction)
applied_for_hash.pop("applied_extraction_sha256", None)
computed_applied_hash = hashlib.sha256(
    json.dumps(applied_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest()
if computed_applied_hash != applied_hash:
    raise SystemExit("S4 applied extraction canonical hash mismatch")
plan_selected_ids = set(application_plan.get("selected_rewrite_ids", []))
plan_rejected_ids = set(application_plan.get("rejected_rewrite_ids", []))
plan_blocked_ids = set(application_plan.get("blocked_rewrite_ids", []))
if plan_selected_ids & plan_rejected_ids or plan_selected_ids & plan_blocked_ids or plan_rejected_ids & plan_blocked_ids:
    raise SystemExit("S4->S5 application plan action buckets overlap")
plan_actions = {}
for case in application_plan.get("cases", []):
    for key, expected_action in (
        ("selected_actions", "apply_to_s5_input"),
        ("rejected_actions", "reject_before_s5_input"),
        ("blocked_actions", "block_before_s5_input"),
    ):
        rows = case.get(key, [])
        count_key = key.replace("actions", "action_count")
        if len(rows) != case.get(count_key):
            raise SystemExit(f"S4->S5 application plan count mismatch for {case.get('case_id')} {key}")
        for row in rows:
            rid = row.get("rewrite_id")
            if rid in plan_actions:
                raise SystemExit(f"duplicate rewrite id in S4->S5 application plan: {rid}")
            if row.get("action") != expected_action:
                raise SystemExit(f"bad S4->S5 application action for {rid}")
            if row.get("extraction_applied_to_ir") is not False or row.get("ir_mutation_allowed") is not False:
                raise SystemExit(f"S4->S5 application action must be non-mutating: {rid}")
            plan_actions[rid] = row

applied_selected_ids = set(applied_extraction.get("selected_rewrite_ids", []))
applied_rejected_ids = set(applied_extraction.get("rejected_rewrite_ids", []))
applied_blocked_ids = set(applied_extraction.get("blocked_rewrite_ids", []))
if applied_selected_ids != plan_selected_ids:
    raise SystemExit("S4 applied extraction selected IDs do not match application plan")
if applied_rejected_ids != plan_rejected_ids:
    raise SystemExit("S4 applied extraction rejected IDs do not match application plan")
if applied_blocked_ids != plan_blocked_ids:
    raise SystemExit("S4 applied extraction blocked IDs do not match application plan")
if applied_selected_ids & applied_rejected_ids or applied_selected_ids & applied_blocked_ids or applied_rejected_ids & applied_blocked_ids:
    raise SystemExit("S4 applied extraction action buckets overlap")

applied_effects = {}
applied_rejections = {}
applied_blockers = {}
for case in applied_extraction.get("cases", []):
    if len(case.get("selected_effects", [])) != case.get("selected_effect_count"):
        raise SystemExit(f"S4 applied extraction selected count mismatch for {case.get('case_id')}")
    if len(case.get("rejected_effects", [])) != case.get("rejected_effect_count"):
        raise SystemExit(f"S4 applied extraction rejected count mismatch for {case.get('case_id')}")
    if len(case.get("blocked_effects", [])) != case.get("blocked_effect_count"):
        raise SystemExit(f"S4 applied extraction blocked count mismatch for {case.get('case_id')}")
    for effect in case.get("selected_effects", []):
        rid = effect.get("rewrite_id")
        if rid in applied_effects:
            raise SystemExit(f"duplicate selected applied effect: {rid}")
        if effect.get("application_applied_to_s5_input") is not True:
            raise SystemExit(f"selected applied effect must be materialized for S5 input: {rid}")
        if effect.get("application_applied_to_compiler_ir") is not False or effect.get("ir_mutation_allowed") is not False:
            raise SystemExit(f"selected applied effect must not mutate compiler IR: {rid}")
        if effect.get("mir_abi_safe") is not True or effect.get("abi_impact") != "none":
            raise SystemExit(f"selected applied effect must be MIR/ABI safe: {rid}")
        if not effect.get("post_apply_selected_enode_sha256"):
            raise SystemExit(f"selected applied effect missing post-apply selected enode hash: {rid}")
        if effect.get("post_mutation_hlir_sha256") != effect.get("post_apply_selected_enode_sha256"):
            raise SystemExit(f"selected applied effect post HLIR hash must be the deterministic S5-input post-state hash: {rid}")
        if not effect.get("post_mutation_egraph_sha256"):
            raise SystemExit(f"selected applied effect missing post egraph hash: {rid}")
        effect_hash = effect.get("applied_effect_sha256")
        if not effect_hash:
            raise SystemExit(f"selected applied effect missing hash: {rid}")
        effect_for_hash = dict(effect)
        effect_for_hash.pop("applied_effect_sha256", None)
        computed_effect_hash = hashlib.sha256(
            json.dumps(effect_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        if computed_effect_hash != effect_hash:
            raise SystemExit(f"selected applied effect canonical hash mismatch: {rid}")
        applied_effects[rid] = effect
    for effect in case.get("rejected_effects", []):
        rid = effect.get("rewrite_id")
        if rid in applied_rejections:
            raise SystemExit(f"duplicate rejected applied effect: {rid}")
        if effect.get("application_applied_to_s5_input") is not False or effect.get("selected_for_s5") is not False:
            raise SystemExit(f"rejected applied effect must not enter S5 input: {rid}")
        if not effect.get("counterexample_set_sha256"):
            raise SystemExit(f"rejected applied effect missing counterexample set hash: {rid}")
        applied_rejections[rid] = effect
    for effect in case.get("blocked_effects", []):
        rid = effect.get("rewrite_id")
        if rid in applied_blockers:
            raise SystemExit(f"duplicate blocked applied effect: {rid}")
        if effect.get("application_applied_to_s5_input") is not False or effect.get("selected_for_s5") is not False:
            raise SystemExit(f"blocked applied effect must not enter S5 input: {rid}")
        if not effect.get("proof_obligation"):
            raise SystemExit(f"blocked applied effect missing proof obligation: {rid}")
        applied_blockers[rid] = effect

if set(applied_effects) != applied_selected_ids:
    raise SystemExit("S4 applied extraction selected effect coverage mismatch")
if set(applied_rejections) != applied_rejected_ids:
    raise SystemExit("S4 applied extraction rejected effect coverage mismatch")
if set(applied_blockers) != applied_blocked_ids:
    raise SystemExit("S4 applied extraction blocked effect coverage mismatch")

allowed_rewrites = {"constant_fold_i64", "symbolic_identity_i64", "symbolic_reflexive_cmp_i64", "symbolic_sub_self_i64"}
allowed_basis = {"exact_symbolic"}
allowed_validators = {"translation-validation"}
consumed = []
all_rewrites = []
blocked_rewrites = []
semantic_rejections = []

for receipt_file in sorted(s4_dir.glob("*/*/*.s4.receipt.json")):
    if "/b/" in receipt_file.as_posix():
        continue
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("schema") != "madaros.v2.s4.receipt/0.1":
        raise SystemExit(f"bad S4 receipt schema: {receipt_file}")
    if receipt.get("s4_boundary_complete") is not True:
        raise SystemExit("S4 boundary receipt not complete")
    if receipt.get("s4_extraction_boundary_complete") is not True:
        raise SystemExit("S4 extraction boundary receipt not complete")
    if receipt.get("s4_complete") is not False:
        raise SystemExit("S4 receipt must not claim global completion")
    if receipt.get("s4_full_complete") is not False:
        raise SystemExit("S4 receipt must not claim S4 FULL completion")
    if receipt.get("stage_contract_level") != "S4_BOUNDARY_NOT_FULL":
        raise SystemExit("S4 receipt must be classified as boundary-not-full")
    rewrites_path = receipt_file.parent / receipt["rewrites_path"]
    extraction_path = receipt_file.parent / receipt["extraction_path"]
    rewrites = json.loads(rewrites_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if len(rewrites) != receipt["rewrite_count"]:
        raise SystemExit("S4 rewrites count mismatch")
    if extraction.get("schema") != "madaros.v2.s4.extraction/0.1":
        raise SystemExit("bad S4 extraction schema")
    if extraction.get("s4_extraction_boundary_complete") is not True:
        raise SystemExit("S4 extraction boundary is not complete")
    if extraction.get("s4_extraction_complete") is not False:
        raise SystemExit("S4 extraction must not claim global completion")
    if extraction.get("mutation_plan") != "none: receipt-only extractor":
        raise SystemExit("S5 preflight only accepts receipt-only S4 extraction")
    if extraction.get("ir_mutation_allowed") is not False:
        raise SystemExit("S5 preflight rejects mutating S4 extraction")
    if extraction.get("extraction_sha256") != receipt.get("extraction_sha256"):
        raise SystemExit("S4 extraction hash mismatch")
    selected_ids = set(extraction.get("selected_rewrite_ids", []))
    rejected_ids = set(extraction.get("rejected_rewrite_ids", []))
    blocked_ids = set(extraction.get("blocked_rewrite_ids", []))
    decisions = {decision["rewrite_id"]: decision for decision in extraction.get("decisions", [])}
    if len(decisions) != extraction.get("input_rewrite_count"):
        raise SystemExit("S4 extraction decision coverage mismatch")
    for rewrite in rewrites:
        rid = rewrite["proposed_rewrite_id"]
        plan_action = plan_actions.get(rid)
        if plan_action is None:
            raise SystemExit(f"S5 preflight missing S4->S5 application action for rewrite {rid}")
        if rid not in decisions:
            raise SystemExit("missing extraction decision for S4 rewrite")
        decision = decisions[rid]
        if decision.get("extraction_applied_to_ir") is not False:
            raise SystemExit("S5 preflight rejects already-mutating S4 extraction")
        if rewrite.get("accepted") is not True:
            if rewrite.get("selected_for_extraction") is not False:
                raise SystemExit("S5 preflight rejects extracted non-accepted S4 rewrites")
            if rewrite.get("ir_mutation_allowed") is not False:
                raise SystemExit("S5 preflight rejects mutating non-accepted S4 rewrites")
            if rewrite.get("blocked") is True:
                if rid not in plan_blocked_ids or plan_action.get("action") != "block_before_s5_input":
                    raise SystemExit("blocked S4 rewrite must be blocked by the S4->S5 plan")
                if rid not in blocked_ids or decision.get("selected") is not False:
                    raise SystemExit("S5 preflight requires blocked rewrites to be excluded by extraction")
                if decision.get("rejection_reason_code") not in {"operand_provenance_ambiguous", "producer_evaluation_not_proven"}:
                    raise SystemExit("S5 preflight requires accepted blocker evidence")
                blocked_rewrites.append(rewrite)
            else:
                if rid not in plan_rejected_ids or plan_action.get("action") != "reject_before_s5_input":
                    raise SystemExit("rejected S4 rewrite must be rejected by the S4->S5 plan")
                if rid not in rejected_ids or decision.get("selected") is not False:
                    raise SystemExit("S5 preflight requires rejected rewrites to be blocked by extraction")
                if not decision.get("counterexample_set_sha256"):
                    raise SystemExit("S5 preflight requires rejected extraction counterexample evidence")
                semantic_rejections.append(rewrite)
            continue
        if rid not in plan_selected_ids or plan_action.get("action") != "apply_to_s5_input":
            raise SystemExit("accepted S4 rewrite must be selected by the S4->S5 plan")
        applied_effect = applied_effects.get(rid)
        if applied_effect is None:
            raise SystemExit(f"accepted S4 rewrite missing applied S5 input effect: {rid}")
        if rid not in selected_ids or decision.get("selected") is not True:
            raise SystemExit("S5 preflight consumes only extraction-selected accepted rewrites")
        if rewrite.get("rewrite_kind") not in allowed_rewrites:
            raise SystemExit(f"S5 preflight rejects ABI-risk rewrite: {rewrite.get('rewrite_kind')}")
        if rewrite.get("basis_family") not in allowed_basis:
            raise SystemExit("S5 preflight rejects approximate/non-exact basis")
        if rewrite.get("validator") not in allowed_validators:
            raise SystemExit("S5 preflight rejects rewrite without translation validation")
        if rewrite.get("error_bound") != "0":
            raise SystemExit("S5 preflight rejects nonzero-error rewrite")
        if not rewrite.get("exact_fallback_expr_sha256"):
            raise SystemExit("S5 preflight requires exact fallback hash")
        if not rewrite.get("original_enode_sha256") or not rewrite.get("rewritten_enode_sha256"):
            raise SystemExit("S5 preflight requires enode hashes")
        if decision.get("mir_abi_safe") is not True:
            raise SystemExit("S5 preflight requires extraction MIR/ABI safe decision")
        if decision.get("abi_impact") != "none":
            raise SystemExit("S5 preflight rejects S4 extraction with ABI impact")
        for field in (
            "lowering_effect",
            "exact_fallback_expr_sha256",
            "rewritten_enode_sha256",
            "validator_log_sha256",
            "mir_abi_safe",
            "abi_impact",
        ):
            source_value = decision.get(field, rewrite.get(field))
            if plan_action.get(field) != source_value:
                raise SystemExit(f"S4->S5 application action {field} mismatch for {rid}")
        if rewrite.get("rewrite_kind") == "symbolic_identity_i64":
            if decision.get("lowering_effect") != "replace_binary_identity_expr_with_existing_value":
                raise SystemExit("S5 preflight rejects symbolic identity without value-ref lowering effect")
        if rewrite.get("rewrite_kind") == "symbolic_reflexive_cmp_i64":
            if decision.get("lowering_effect") not in {
                "replace_binary_predicate_expr_with_const_bool",
                "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated",
            }:
                raise SystemExit("S5 preflight rejects reflexive comparison without bool-const lowering effect")
            if decision.get("lowering_effect") == "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated":
                if rewrite.get("producer_evaluation_policy") != "direct_call_leaf_pure_keep_producer_evaluated":
                    raise SystemExit("S5 preflight requires call producer evaluation policy for keep-producer lowering")
                producer = rewrite.get("symbolic_producer", {})
                if producer.get("producer_kind") != "call_direct" or producer.get("call_leaf_pure") is not True:
                    raise SystemExit("S5 preflight requires local leaf call producer for keep-producer lowering")
        if rewrite.get("rewrite_kind") == "symbolic_sub_self_i64":
            if decision.get("lowering_effect") not in {
                "replace_binary_sub_self_expr_with_const_i64_zero",
                "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated",
            }:
                raise SystemExit("S5 preflight rejects sub-self without const-zero lowering effect")
            if decision.get("lowering_effect") == "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated":
                if rewrite.get("producer_evaluation_policy") != "direct_call_leaf_pure_keep_producer_evaluated":
                    raise SystemExit("S5 preflight requires call producer evaluation policy for sub-self keep-producer lowering")
                producer = rewrite.get("symbolic_producer", {})
                if producer.get("producer_kind") != "call_direct" or producer.get("call_leaf_pure") is not True:
                    raise SystemExit("S5 preflight requires local leaf call producer for sub-self keep-producer lowering")
        if decision.get("selected_enode_sha256") != rewrite.get("rewritten_enode_sha256"):
            raise SystemExit("S5 preflight selected enode must match rewrite")
        for field in (
            "rewrite_kind",
            "proposal_kind",
            "function",
            "block",
            "instruction_result",
            "lowering_effect",
            "exact_fallback_expr_sha256",
            "validator_log_sha256",
            "coefficient_sha256",
            "basis_family",
            "error_bound",
            "domain",
            "domain_bounds",
            "mir_abi_safe",
            "abi_impact",
        ):
            expected_value = decision.get(field, rewrite.get(field))
            if applied_effect.get(field) != expected_value:
                raise SystemExit(f"S4 applied extraction field {field} mismatch for {rid}")
        if applied_effect.get("output_enode_sha256") != rewrite.get("rewritten_enode_sha256"):
            raise SystemExit("S4 applied extraction output enode must match rewrite")
        if applied_effect.get("application_applied_to_s5_input") is not True:
            raise SystemExit("S4 applied extraction must be applied to S5 input")
        if not applied_effect.get("post_apply_selected_enode_sha256"):
            raise SystemExit("S4 applied extraction must carry post-apply selected enode hash")
        all_rewrites.append(rewrite)
    consumed.append({
        "case_id": receipt["case_id"],
        "source": receipt["source"],
        "input_hlir_sha256": receipt["input_hlir_sha256"],
        "egraph_sha256": receipt["egraph_sha256"],
        "extraction_sha256": receipt["extraction_sha256"],
        "accepted_rewrite_count": receipt["accepted_rewrite_count"],
        "selected_rewrite_count": extraction["selected_rewrite_count"],
        "rejected_from_extraction_count": extraction["rejected_rewrite_count"],
        "blocked_from_extraction_count": extraction.get("blocked_rewrite_count", 0),
    })

input_ready = len(all_rewrites) > 0
if {rewrite["proposed_rewrite_id"] for rewrite in all_rewrites} != plan_selected_ids:
    raise SystemExit("S5 preflight selected rewrites do not match S4->S5 application plan")
if {rewrite["proposed_rewrite_id"] for rewrite in semantic_rejections} != plan_rejected_ids:
    raise SystemExit("S5 preflight semantic rejections do not match S4->S5 application plan")
if {rewrite["proposed_rewrite_id"] for rewrite in blocked_rewrites} != plan_blocked_ids:
    raise SystemExit("S5 preflight blocked rewrites do not match S4->S5 application plan")
preflight = {
    "schema": "madaros.v2.s5.preflight/0.1",
    "status": "pass" if input_ready else "blocked",
    "stage_contract_level": "S5_PREFLIGHT_NOT_FULL",
    "s5_ready": False,
    "s5_input_contract_ready": input_ready,
    "s5_implemented": False,
    "s5_full_complete": False,
    "s_full_contract": "blocked_until_mir_abi_numeric_and_differential_gates_exist",
    "input_contract": "madaros.v2.s4.gate/0.1",
    "input_extraction_contract": "madaros.v2.s4.extraction/0.1",
    "input_application_plan_contract": application_plan["schema"],
    "input_application_plan_sha256": application_plan["application_plan_sha256"],
    "s4_to_s5_application_plan_consumed": True,
    "input_applied_extraction_contract": applied_extraction["schema"],
    "input_applied_extraction_sha256": applied_extraction["applied_extraction_sha256"],
    "s4_applied_extraction_consumed": True,
    "s4_downstream_integration_slice_complete": True,
    "mir_abi_safe_subset": sorted(allowed_rewrites),
    "abi_impact": "none: S5 consumes only extraction-selected accepted rewrites",
    "numeric_semantics": "i64 exact rewrites only when zero-error translation-validation receipts survive operand-provenance guards",
    "consumed_s4_cases": consumed,
    "accepted_rewrite_count": len(all_rewrites),
    "selected_rewrite_count": len(all_rewrites),
    "semantic_rejected_rewrite_count": len(semantic_rejections),
    "blocked_rewrite_count": len(blocked_rewrites),
    "blocking_reason": "" if input_ready else "S4 has no accepted extraction-selected rewrites after safety guards",
    "required_next_lane": [
        "repair/prove S3 HLIR binary operand provenance before symbolic/algebraic S4 extraction",
        "full S4 equality saturation and E-KAN rejection/proposal receipts",
        "MIR hash receipt",
        "ABI layout/call/return receipt",
        "f128 IR/MIR/ABI/software-helper receipts before f128 promotion",
        "diagnostics and fallback semantics for ABI/numeric-width failures",
        "differential native-v2 vs interpreter/lean_single gate where available",
    ],
    "missing_full_obligations": [
        "MIR serialization and hash receipts",
        "ABI layout receipts for scalar, aggregate, SRET, imported call, and return paths",
        "f128 IR/MIR/ABI/software-helper receipts before f128 promotion",
        "f128 IR/MIR/ABI/software-helper receipts",
        "diagnostics and fallback semantics for unsupported layouts and numeric widths",
        "differential native-v2 vs interpreter/lean_single validation where available",
    ],
}
payload = json.dumps(preflight, sort_keys=True, indent=2) + "\n"
preflight["preflight_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
payload = json.dumps(preflight, sort_keys=True, indent=2) + "\n"
receipt_path.write_text(payload, encoding="utf-8")
print(
    f"[madaros-v2-s5-preflight] ok cases={len(consumed)} "
    f"rewrites={len(all_rewrites)} blocked={len(blocked_rewrites)} status={preflight['status']} "
    f"sha={preflight['preflight_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-preflight] PASS: S5 preflight classified current S4 extraction input without overclaiming readiness"
echo "[madaros-v2-s5-preflight] receipt=$OUT_DIR/madaros_v2_s5_preflight.receipt.json"
