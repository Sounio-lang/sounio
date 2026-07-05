#!/usr/bin/env python3
"""Emit Madaros v2 S4 e-graph/E-KAN rewrite receipts from S3 HLIR.

This is intentionally conservative. It does not mutate compiler output and it
does not accept approximate rewrites. It consumes compiler-native S3 HLIR JSON,
builds a persistent equality/e-graph receipt, and accepts only rewrites that
carry a translation-validation witness and an exact fallback expression hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s4.receipt/0.1"
EGRAPH_SCHEMA = "madaros.v2.s4.egraph/0.1"
REWRITE_SCHEMA = "madaros.v2.ekan.rewrite/0.1"
EXTRACTION_SCHEMA = "madaros.v2.s4.extraction/0.1"
S3_SCHEMA = "madaros.v2.s3.receipt/0.1"

BIN_OPS = {
    0: "add",
    1: "sub",
    2: "mul",
    3: "sdiv",
    5: "srem",
    18: "eq",
    19: "ne",
    20: "slt",
    21: "sle",
    22: "sgt",
    23: "sge",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_s3_receipt(compiler: Path, source: Path, root: Path, out_dir: Path, case_id: str, timeout_s: int) -> tuple[Path, Path]:
    cmd = [
        str(compiler),
        "s3-receipt",
        relpath(source, root),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"s3-receipt failed for {source} rc={proc.returncode}\n{proc.stdout}")
    default_stem = source.stem
    receipt = out_dir / f"{default_stem}.s3.receipt.json"
    hlir = out_dir / f"{default_stem}.s3.hlir.json"
    if case_id != default_stem:
        renamed_receipt = out_dir / f"{case_id}.s3.receipt.json"
        renamed_hlir = out_dir / f"{case_id}.s3.hlir.json"
        receipt.replace(renamed_receipt)
        hlir.replace(renamed_hlir)
        receipt = renamed_receipt
        hlir = renamed_hlir
    if not receipt.is_file() or not hlir.is_file():
        raise SystemExit(f"missing S3 artifacts for {source}")
    return receipt, hlir


def const_value(instr: dict[str, Any]) -> tuple[str, int | bool] | None:
    c = instr.get("constant", {})
    if c.get("kind") == "int":
        return ("int", int(c.get("int_val", 0)))
    if c.get("kind") == "bool":
        return ("bool", bool(c.get("bool_val", False)))
    return None


def eval_bin(op: int, lhs: int | bool, rhs: int | bool) -> tuple[str, int | bool] | None:
    if not isinstance(lhs, int) or not isinstance(rhs, int):
        return None
    if op == 0:
        return ("int", lhs + rhs)
    if op == 1:
        return ("int", lhs - rhs)
    if op == 2:
        return ("int", lhs * rhs)
    if op == 3 and rhs != 0:
        return ("int", int(lhs / rhs))
    if op == 5 and rhs != 0:
        return ("int", lhs % rhs)
    if op == 18:
        return ("bool", lhs == rhs)
    if op == 19:
        return ("bool", lhs != rhs)
    if op == 20:
        return ("bool", lhs < rhs)
    if op == 21:
        return ("bool", lhs <= rhs)
    if op == 22:
        return ("bool", lhs > rhs)
    if op == 23:
        return ("bool", lhs >= rhs)
    return None


def make_const_enode(kind: str, value: int | bool) -> dict[str, Any]:
    return {
        "op": "const",
        "constant": {
            "kind": kind,
            "int_val": int(value) if kind == "int" else 0,
            "bool_val": bool(value) if kind == "bool" else False,
        },
    }


def rewrite_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    lhs_const: tuple[str, int | bool],
    rhs_const: tuple[str, int | bool],
    result_const: tuple[str, int | bool],
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr["bin_op"]), f"op{instr['bin_op']}")
    fallback = {
        "op": op_name,
        "lhs": lhs_const,
        "rhs": rhs_const,
        "result": result_const,
    }
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["constant"],
        "constant": result_const[1],
    }
    validator_log = {
        "validator": "translation-validation",
        "method": "constant-evaluation-over-s3-hlir",
        "accepted": True,
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "op": op_name,
        "lhs": lhs_const,
        "rhs": rhs_const,
        "result_const": result_const,
    }
    rid = "s4-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "constant_fold_i64",
        "proposal_kind": "exact_constant_fold",
        "proposal_origin": "madaros_v2_s4_receipt.constant-evaluation-over-s3-hlir",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "constant_fold_i64", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_exact_symbolic_constant",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(make_const_enode(*result_const))),
        "rewritten_enode_sha256": sha256_text(stable_json(make_const_enode(*result_const))),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "exact-symbolic-no-training"})),
        "domain": "singleton-known-s3-hlir-constant-operands",
        "domain_bounds": {
            "kind": "singleton-known-s3-hlir-constant-operands",
            "lhs": lhs_const,
            "rhs": rhs_const,
        },
        "error_bound": "0",
        "error_bound_method": "exact-evaluation",
        "gum_covariance_assumptions": "not-applicable: exact constant rewrite",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "translation-validation",
        "validator_attempted": ["translation-validation"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "selected_for_extraction": True,
        "ir_mutation_allowed": False,
        "accepted": True,
    }


def rejected_div_self_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
) -> dict[str, Any]:
    fallback = {
        "op": "sdiv",
        "lhs": ["symbolic_value", int(instr["lhs"])],
        "rhs": ["symbolic_value", int(instr["rhs"])],
    }
    rewritten = make_const_enode("int", 1)
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["constant"],
        "constant": 1,
        "proposal": "x_div_x_to_one",
    }
    counterexample = {
        "symbolic_value": int(instr["lhs"]),
        "value": 0,
        "original_behavior": "division_by_zero_trap",
        "rewritten_behavior": "returns_1",
    }
    validator_log = {
        "validator": "rejected",
        "method": "counterexample-guided-translation-validation",
        "accepted": False,
        "rejection_reason": "counterexample_division_by_zero",
        "counterexamples": [counterexample],
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "proposal": "x_div_x_to_one",
        "counterexample": counterexample,
    }
    rid = "s4-reject-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "x_div_x_to_one",
        "proposal_kind": "algebraic_identity",
        "proposal_origin": "madaros_v2_s4_receipt.counterexample-guided-negative-lane",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "reject_x_div_x_to_one", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_rejected_counterexample",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(rewritten)),
        "rewritten_enode_sha256": sha256_text(stable_json(rewritten)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "hand-authored-algebraic-proposal"})),
        "domain": "all-i64-values",
        "domain_bounds": {
            "kind": "all-i64-values",
            "symbolic_value": int(instr["lhs"]),
            "preconditions": [],
        },
        "error_bound": "unbounded: proposal rejected",
        "error_bound_method": "counterexample",
        "gum_covariance_assumptions": "not-applicable: rejected exact symbolic proposal",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "rejected",
        "validator_attempted": ["translation-validation", "counterexample"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "rejection_reason_code": "counterexample_found",
        "rejection_reason": "counterexample_division_by_zero",
        "counterexample_sha256": sha256_text(stable_json(counterexample)),
        "counterexample_set_sha256": sha256_text(stable_json([counterexample])),
        "counterexample_count": 1,
        "counterexamples": [counterexample],
        "selected_for_extraction": False,
        "ir_mutation_allowed": False,
        "accepted": False,
    }


def blocked_operand_provenance_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    proposed: dict[str, Any],
    reason_detail: dict[str, Any],
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr.get("bin_op", -1)), f"op{instr.get('bin_op', -1)}")
    fallback = {
        "op": op_name,
        "lhs": ["hlir_value", int(instr.get("lhs", -1))],
        "rhs": ["hlir_value", int(instr.get("rhs", -1))],
        "blocked_reason": reason_detail,
    }
    validator_log = {
        "validator": "blocked",
        "method": "operand-provenance-fidelity-guard",
        "accepted": False,
        "blocked": True,
        "rejection_reason": "operand_provenance_ambiguous",
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "proposal": "blocked_operand_provenance",
        "reason": reason_detail,
    }
    rid = "s4-blocked-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "operand_provenance_blocked",
        "proposal_kind": "blocked_optimization_candidate",
        "proposal_origin": "madaros_v2_s4_receipt.operand-provenance-fidelity-guard",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "operand_provenance_guard", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_blocked_operand_provenance",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(proposed)),
        "rewritten_enode_sha256": sha256_text(stable_json(proposed)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json({"basis_family": "exact_symbolic", "blocked": True})),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "blocked-no-training", "reason": reason_detail})),
        "domain": "blocked: operand provenance is ambiguous in S3 HLIR",
        "domain_bounds": {
            "kind": "blocked_operand_provenance",
            "reason": reason_detail,
        },
        "error_bound": "unproven: operand provenance ambiguous",
        "error_bound_method": "blocked-before-validation",
        "gum_covariance_assumptions": "not-applicable: blocked before optimization",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "blocked",
        "validator_attempted": ["operand-provenance-fidelity-guard"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "blocked": True,
        "rejection_reason_code": "operand_provenance_ambiguous",
        "rejection_reason": "s3_hlir_binary_operands_do_not_prove_source_operands",
        "selected_for_extraction": False,
        "ir_mutation_allowed": False,
        "accepted": False,
    }


def s4_cost_model_payload() -> dict[str, Any]:
    return {
        "schema": "madaros.v2.s4.cost_model/0.1",
        "name": "madaros-v2-s4-exact-boundary-cost-model",
        "version": "0.1",
        "objective": [
            "preserve exact semantics",
            "prefer translation-validated exact constants over original binary operations",
            "reject every counterexample-backed or unvalidated proposal",
            "do not mutate downstream IR in this boundary lane",
        ],
        "weights": {
            "const": 1,
            "binary": 3,
            "call": 5,
            "unknown": 10,
            "unvalidated_penalty": 1000000,
        },
    }


def enode_cost_from_hash(egraph: dict[str, Any], eclass_id: str, enode_hash: str) -> int:
    model = s4_cost_model_payload()["weights"]
    for eclass in egraph["eclasses"]:
        if eclass["eclass_id"] != eclass_id:
            continue
        for enode in eclass["enodes"]:
            if enode.get("sha256") != enode_hash:
                continue
            op = enode.get("op")
            if op == "const":
                return int(model["const"])
            if op == "binary":
                return int(model["binary"])
            if op == "call":
                return int(model["call"])
            return int(model["unknown"])
    raise SystemExit(f"missing enode hash in egraph extraction: {eclass_id} {enode_hash}")


def build_extraction(case_id: str, source: str, hlir_sha: str, egraph: dict[str, Any], rewrites: list[dict[str, Any]]) -> dict[str, Any]:
    cost_model = s4_cost_model_payload()
    cost_model_sha = sha256_text(stable_json(cost_model))
    rewrites_sha = sha256_text(stable_json(rewrites))
    decisions: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    rejected_ids: list[str] = []
    blocked_ids: list[str] = []
    for rewrite in rewrites:
        rid = rewrite["proposed_rewrite_id"]
        common = {
            "rewrite_id": rid,
            "eclass_id": rewrite["eclass_id"],
            "rewrite_kind": rewrite["rewrite_kind"],
            "proposal_kind": rewrite["proposal_kind"],
            "function": rewrite["function"],
            "block": rewrite["block"],
            "instruction_result": rewrite["instruction_result"],
            "original_enode_sha256": rewrite["original_enode_sha256"],
            "proposed_enode_sha256": rewrite["proposed_enode_sha256"],
            "validator": rewrite["validator"],
            "validator_log_sha256": rewrite["validator_log_sha256"],
            "cost_model": cost_model["name"],
            "cost_model_sha256": cost_model_sha,
            "cost_model_config_sha256": cost_model_sha,
            "ir_mutation_allowed": False,
            "extraction_applied_to_ir": False,
        }
        original_cost = enode_cost_from_hash(egraph, rewrite["eclass_id"], rewrite["original_enode_sha256"])
        proposed_cost = enode_cost_from_hash(egraph, rewrite["eclass_id"], rewrite["proposed_enode_sha256"])
        if rewrite["accepted"] is True:
            if rewrite.get("selected_for_extraction") is not True:
                raise SystemExit(f"accepted rewrite not marked selectable: {rid}")
            decision = {
                **common,
                "decision": "select",
                "selected": True,
                "selected_enode_sha256": rewrite["proposed_enode_sha256"],
                "replacement_enode_sha256": rewrite["rewritten_enode_sha256"],
                "cost_before": original_cost,
                "cost_after": proposed_cost,
                "cost_delta": original_cost - proposed_cost,
                "cost_components": {
                    "original": {"base": original_cost, "children": 0, "total": original_cost},
                    "selected": {"base": proposed_cost, "children": 0, "total": proposed_cost},
                },
                "selection_reason": "accepted_translation_validated_exact_lower_cost",
                "proof_obligation": "translation-validation already accepted with zero error bound",
                "exact_fallback_expr_sha256": rewrite["exact_fallback_expr_sha256"],
                "coefficient_sha256": rewrite["coefficient_sha256"],
                "basis_family": rewrite["basis_family"],
                "error_bound": rewrite["error_bound"],
                "domain": rewrite["domain"],
                "domain_bounds": rewrite["domain_bounds"],
                "lowering_effect": "replace_binary_constant_expr_with_const",
                "abi_impact": "none",
                "mir_abi_safe": True,
            }
            if decision["cost_after"] > decision["cost_before"]:
                raise SystemExit(f"accepted rewrite increases extraction cost: {rid}")
            selected_ids.append(rid)
        elif rewrite.get("blocked") is True:
            if rewrite.get("selected_for_extraction") is not False:
                raise SystemExit(f"blocked rewrite marked selectable: {rid}")
            decision = {
                **common,
                "decision": "block",
                "selected": False,
                "selected_enode_sha256": rewrite["original_enode_sha256"],
                "replacement_enode_sha256": "",
                "cost_before": original_cost,
                "cost_after": original_cost + int(cost_model["weights"]["unvalidated_penalty"]),
                "cost_delta": -int(cost_model["weights"]["unvalidated_penalty"]),
                "cost_components": {
                    "original": {"base": original_cost, "children": 0, "total": original_cost},
                    "blocked_candidate": {
                        "base": proposed_cost,
                        "children": 0,
                        "penalty": int(cost_model["weights"]["unvalidated_penalty"]),
                        "total": original_cost + int(cost_model["weights"]["unvalidated_penalty"]),
                    },
                },
                "selection_reason": "blocked_by_operand_provenance_guard",
                "rejection_reason_code": rewrite["rejection_reason_code"],
                "proof_obligation": "prove S3 HLIR operand provenance before extraction",
                "exact_fallback_expr_sha256": rewrite["exact_fallback_expr_sha256"],
                "coefficient_sha256": rewrite["coefficient_sha256"],
                "basis_family": rewrite["basis_family"],
                "error_bound": rewrite["error_bound"],
                "domain": rewrite["domain"],
                "domain_bounds": rewrite["domain_bounds"],
                "lowering_effect": "none: blocked candidate",
                "abi_impact": "none: blocked candidate",
                "mir_abi_safe": False,
            }
            blocked_ids.append(rid)
        else:
            if rewrite.get("selected_for_extraction") is not False:
                raise SystemExit(f"rejected rewrite marked selectable: {rid}")
            decision = {
                **common,
                "decision": "reject",
                "selected": False,
                "selected_enode_sha256": rewrite["original_enode_sha256"],
                "replacement_enode_sha256": "",
                "cost_before": original_cost,
                "cost_after": original_cost + int(cost_model["weights"]["unvalidated_penalty"]),
                "cost_delta": -int(cost_model["weights"]["unvalidated_penalty"]),
                "cost_components": {
                    "original": {"base": original_cost, "children": 0, "total": original_cost},
                    "blocked_candidate": {
                        "base": proposed_cost,
                        "children": 0,
                        "penalty": int(cost_model["weights"]["unvalidated_penalty"]),
                        "total": original_cost + int(cost_model["weights"]["unvalidated_penalty"]),
                    },
                },
                "selection_reason": "rejected_by_validator_counterexample",
                "rejection_reason_code": rewrite["rejection_reason_code"],
                "counterexample_set_sha256": rewrite["counterexample_set_sha256"],
                "proof_obligation": "counterexample blocks extraction",
                "exact_fallback_expr_sha256": rewrite["exact_fallback_expr_sha256"],
                "coefficient_sha256": rewrite["coefficient_sha256"],
                "basis_family": rewrite["basis_family"],
                "error_bound": rewrite["error_bound"],
                "domain": rewrite["domain"],
                "domain_bounds": rewrite["domain_bounds"],
                "lowering_effect": "none: rejected proposal",
                "abi_impact": "none: rejected proposal",
                "mir_abi_safe": False,
            }
            rejected_ids.append(rid)
        decisions.append(decision)

    extraction = {
        "schema": EXTRACTION_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_hlir_sha256": hlir_sha,
        "input_egraph_sha256": egraph["egraph_sha256"],
        "input_rewrites_sha256": rewrites_sha,
        "input_rewrite_count": len(rewrites),
        "cost_model_schema": cost_model["schema"],
        "cost_model_name": cost_model["name"],
        "cost_model": cost_model,
        "cost_model_sha256": cost_model_sha,
        "cost_model_config_sha256": cost_model_sha,
        "extraction_policy": "accepted_translation_validated_cheapest_enode",
        "deterministic_extraction": True,
        "extractor": "madaros-v2-s4-receipt-only-extractor",
        "extractor_version": "0.1",
        "mutation_plan": "none: receipt-only extractor",
        "ir_mutation_allowed": False,
        "candidate_eclass_count": len({r["eclass_id"] for r in rewrites}),
        "selected_eclass_count": len({r["eclass_id"] for r in rewrites if r["accepted"] is True}),
        "selected_rewrite_ids": selected_ids,
        "rejected_rewrite_ids": rejected_ids,
        "blocked_rewrite_ids": blocked_ids,
        "selected_rewrite_count": len(selected_ids),
        "rejected_rewrite_count": len(rejected_ids),
        "blocked_rewrite_count": len(blocked_ids),
        "gate_invariants": [
            "deterministic_double_emit",
            "selected_ids_equal_accepted_ids",
            "rejected_ids_blocked_from_extraction",
            "cost_model_hash_present",
            "accepted_translation_validation_zero_error",
            "receipt_only_no_ir_mutation",
        ],
        "s4_extraction_complete": False,
        "s4_extraction_boundary_complete": True,
        "decisions": decisions,
    }
    extraction["extraction_sha256"] = sha256_text(stable_json(extraction))
    return extraction


def operand_provenance_ambiguous(instr: dict[str, Any], lhs: tuple[str, int | bool] | None, rhs: tuple[str, int | bool] | None) -> dict[str, Any] | None:
    if lhs is None or rhs is None:
        return None
    if instr.get("lhs") == instr.get("rhs"):
        return {
            "kind": "duplicate_hlir_operand_id",
            "lhs": int(instr.get("lhs", -1)),
            "rhs": int(instr.get("rhs", -1)),
            "bin_op": int(instr.get("bin_op", -1)),
            "observed_lhs_const": lhs,
            "observed_rhs_const": rhs,
            "impact": "constant fold would be a proof about duplicated HLIR IDs, not source operands",
        }
    return None


def analyze_hlir(case_id: str, source: str, hlir_text: str, hlir_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(hlir_text)
    module = data["module"]
    eclasses: list[dict[str, Any]] = []
    rewrites: list[dict[str, Any]] = []
    for func in module["functions"]:
        value_consts: dict[int, tuple[str, int | bool]] = {}
        for block in func["blocks"]:
            for instr in block["instrs"]:
                result = int(instr["result"])
                enodes = [{
                    "kind": "hlir-original",
                    "sha256": sha256_text(stable_json(instr)),
                    "op": instr.get("op"),
                }]
                cv = const_value(instr)
                if cv is not None:
                    value_consts[result] = cv
                if instr.get("op") == "binary":
                    lhs = value_consts.get(int(instr.get("lhs", -1)))
                    rhs = value_consts.get(int(instr.get("rhs", -1)))
                    if int(instr.get("bin_op", -1)) == 3 and instr.get("lhs") == instr.get("rhs"):
                        receipt = rejected_div_self_receipt(
                            case_id,
                            source,
                            hlir_sha,
                            func["name"],
                            block["label"],
                            instr,
                        )
                        rewrites.append(receipt)
                        enodes.append({
                            "kind": "s4-rejected-rewrite",
                            "rewrite_id": receipt["proposed_rewrite_id"],
                            "sha256": receipt["rewritten_enode_sha256"],
                            "op": "const",
                            "rejection_reason": receipt["rejection_reason"],
                        })
                    if lhs is not None and rhs is not None:
                        folded = eval_bin(int(instr.get("bin_op", -1)), lhs[1], rhs[1])
                        if folded is not None:
                            ambiguity = operand_provenance_ambiguous(instr, lhs, rhs)
                            if ambiguity is not None:
                                proposed = make_const_enode(*folded)
                                receipt = blocked_operand_provenance_receipt(
                                    case_id,
                                    source,
                                    hlir_sha,
                                    func["name"],
                                    block["label"],
                                    instr,
                                    proposed,
                                    ambiguity,
                                )
                                rewrites.append(receipt)
                                enodes.append({
                                    "kind": "s4-blocked-rewrite",
                                    "rewrite_id": receipt["proposed_rewrite_id"],
                                    "sha256": receipt["rewritten_enode_sha256"],
                                    "op": "const",
                                    "rejection_reason": receipt["rejection_reason"],
                                })
                            else:
                                value_consts[result] = folded
                                receipt = rewrite_receipt(
                                    case_id,
                                    source,
                                    hlir_sha,
                                    func["name"],
                                    block["label"],
                                    instr,
                                    lhs,
                                    rhs,
                                    folded,
                                )
                                rewrites.append(receipt)
                                enodes.append({
                                    "kind": "s4-rewrite",
                                    "rewrite_id": receipt["proposed_rewrite_id"],
                                    "sha256": receipt["rewritten_enode_sha256"],
                                    "op": "const",
                                })
                eclasses.append({
                    "eclass_id": f"{func['name']}:{block['label']}:{result}",
                    "function": func["name"],
                    "block": block["label"],
                    "result": result,
                    "type": instr.get("ty", {}),
                    "enodes": enodes,
                })
    egraph = {
        "schema": EGRAPH_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_hlir_sha256": hlir_sha,
        "eclass_count": len(eclasses),
        "rewrite_count": len(rewrites),
        "eclasses": eclasses,
    }
    egraph["egraph_sha256"] = sha256_text(stable_json(egraph))
    return egraph, rewrites


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = args.case_id or source.stem
    source_rel = relpath(source, root)

    s3_dir = out_dir / "s3"
    s3_dir.mkdir(parents=True, exist_ok=True)
    s3_receipt_path, hlir_path = run_s3_receipt(compiler, source, root, s3_dir, case_id, args.timeout)
    s3_receipt = json.loads(s3_receipt_path.read_text(encoding="utf-8"))
    if s3_receipt.get("schema") != S3_SCHEMA:
        raise SystemExit(f"bad S3 receipt schema: {s3_receipt.get('schema')}")
    s3_facts = s3_receipt.get("facts", {})
    if s3_facts.get("binary_operand_integrity") is not True:
        raise SystemExit("S4 requires S3 binary_operand_integrity=true before consuming HLIR")
    hlir_text = hlir_path.read_text(encoding="utf-8")
    hlir_sha = sha256_text(hlir_text)
    if hlir_sha != s3_receipt.get("hlir_byte_sha256"):
        raise SystemExit("S3 HLIR hash mismatch")

    egraph, rewrites = analyze_hlir(case_id, source_rel, hlir_text, hlir_sha)
    accepted = [r for r in rewrites if r["accepted"]]
    blocked = [r for r in rewrites if r.get("blocked") is True]
    rejected = [r for r in rewrites if not r["accepted"] and r.get("blocked") is not True]
    extraction = build_extraction(case_id, source_rel, hlir_sha, egraph, rewrites)
    egraph_path = out_dir / f"{case_id}.s4.egraph.json"
    rewrites_path = out_dir / f"{case_id}.s4.rewrites.json"
    extraction_path = out_dir / f"{case_id}.s4.extraction.json"
    receipt_path = out_dir / f"{case_id}.s4.receipt.json"
    egraph_path.write_text(json.dumps(egraph, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rewrites_path.write_text(json.dumps(rewrites, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    extraction_path.write_text(json.dumps(extraction, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": SCHEMA,
        "case_id": case_id,
        "source": source_rel,
        "input_s3_receipt_sha256": sha256_bytes(s3_receipt_path.read_bytes()),
        "input_hlir_sha256": hlir_sha,
        "egraph_schema": EGRAPH_SCHEMA,
        "egraph_path": egraph_path.name,
        "egraph_sha256": egraph["egraph_sha256"],
        "rewrite_schema": REWRITE_SCHEMA,
        "rewrites_path": rewrites_path.name,
        "rewrite_count": len(rewrites),
        "accepted_rewrite_count": len(accepted),
        "rejected_rewrite_count": len(rejected),
        "blocked_rewrite_count": len(blocked),
        "accepted_rewrite_ids": [r["proposed_rewrite_id"] for r in accepted],
        "rejected_rewrite_ids": [r["proposed_rewrite_id"] for r in rejected],
        "blocked_rewrite_ids": [r["proposed_rewrite_id"] for r in blocked],
        "extraction_schema": EXTRACTION_SCHEMA,
        "extraction_path": extraction_path.name,
        "extraction_sha256": extraction["extraction_sha256"],
        "input_rewrites_sha256": extraction["input_rewrites_sha256"],
        "cost_model_sha256": extraction["cost_model_sha256"],
        "cost_model_config_sha256": extraction["cost_model_config_sha256"],
        "selected_rewrite_count": extraction["selected_rewrite_count"],
        "selected_rewrite_ids": extraction["selected_rewrite_ids"],
        "rejected_from_extraction_count": extraction["rejected_rewrite_count"],
        "blocked_from_extraction_count": extraction["blocked_rewrite_count"],
        "validators": sorted({r["validator"] for r in rewrites}),
        "basis_families": sorted({r["basis_family"] for r in rewrites}),
        "s4_complete": False,
        "s4_boundary_complete": True,
        "s4_extraction_boundary_complete": True,
        "s4_claim": "conservative_egraph_ekan_receipt_boundary_with_operand_provenance_guard",
        "s4_remaining": [
            "S3 HLIR operand provenance repair/proof",
            "multi-rule equality saturation",
            "non-constant algebraic identities",
            "downstream optimizer integration beyond receipt-only extraction",
        ],
    }
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    receipt["receipt_sha256"] = sha256_text(payload)
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    receipt_path.write_text(payload, encoding="utf-8")
    print(
        f"madaros-v2-s4: case={case_id} accepted={len(accepted)} blocked={len(blocked)} "
        f"egraph_sha={egraph['egraph_sha256'][:12]} receipt={receipt_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--source", required=True)
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--compiler", default=str(repo_root_from_script() / "bin" / "madaros"))
    emit_p.add_argument("--case-id", default="")
    emit_p.add_argument("--root", default=str(repo_root_from_script()))
    emit_p.add_argument("--timeout", type=int, default=120)
    emit_p.set_defaults(func=emit)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
