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


def make_value_ref_enode(value_id: int, producer: dict[str, Any]) -> dict[str, Any]:
    return {
        "op": "value_ref",
        "value_id": int(value_id),
        "producer_kind": producer.get("producer_kind", "unknown"),
        "producer_label": producer.get("producer_label", producer.get("label", "")),
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


def symbolic_identity_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    identity_kind: str,
    symbolic_value_id: int,
    symbolic_producer: dict[str, Any],
    neutral_side: str,
    neutral_const: tuple[str, int | bool],
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr["bin_op"]), f"op{instr['bin_op']}")
    proposed = make_value_ref_enode(symbolic_value_id, symbolic_producer)
    fallback = {
        "op": op_name,
        "lhs": ["hlir_value", int(instr["lhs"])],
        "rhs": ["hlir_value", int(instr["rhs"])],
        "identity_kind": identity_kind,
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "neutral_side": neutral_side,
        "neutral_const": neutral_const,
    }
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["identity"],
        "identity_kind": identity_kind,
        "symbolic_value": symbolic_value_id,
        "neutral_side": neutral_side,
        "neutral_const": neutral_const,
    }
    domain_bounds = {
        "kind": "all-i64-values-with-neutral-element",
        "identity_kind": identity_kind,
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "neutral_side": neutral_side,
        "neutral_const": neutral_const,
        "preconditions": [
            "symbolic operand is already evaluated as an S3 HLIR SSA value",
            "neutral operand is an exact i64 constant",
            "rewrite selects the existing symbolic value and does not remove its producer",
        ],
    }
    validator_log = {
        "validator": "translation-validation",
        "method": "neutral-element-symbolic-identity-over-s3-hlir",
        "accepted": True,
        "identity_kind": identity_kind,
        "domain_bounds": domain_bounds,
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "identity_kind": identity_kind,
        "symbolic_value": symbolic_value_id,
        "neutral_side": neutral_side,
        "neutral_const": neutral_const,
    }
    rid = "s4-ident-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "symbolic_identity_i64",
        "identity_kind": identity_kind,
        "proposal_kind": "exact_symbolic_identity",
        "proposal_origin": "madaros_v2_s4_receipt.neutral-element-symbolic-identity",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "symbolic_identity_i64", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_exact_symbolic_identity",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_lhs": int(instr["lhs"]),
        "original_rhs": int(instr["rhs"]),
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "neutral_side": neutral_side,
        "neutral_const": neutral_const,
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(proposed)),
        "rewritten_enode_sha256": sha256_text(stable_json(proposed)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "exact-symbolic-identity-no-training"})),
        "domain": "all-i64-values-with-neutral-element",
        "domain_bounds": domain_bounds,
        "error_bound": "0",
        "error_bound_method": "exact-neutral-element-identity",
        "gum_covariance_assumptions": "not-applicable: exact symbolic identity",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "translation-validation",
        "validator_attempted": ["translation-validation", "neutral-element-proof"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "selected_for_extraction": True,
        "ir_mutation_allowed": False,
        "accepted": True,
    }


def symbolic_reflexive_cmp_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    comparison_kind: str,
    symbolic_value_id: int,
    symbolic_producer: dict[str, Any],
    result_const: tuple[str, bool],
    producer_evaluation_policy: str,
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr["bin_op"]), f"op{instr['bin_op']}")
    proposed = make_const_enode(*result_const)
    fallback = {
        "op": op_name,
        "lhs": ["hlir_value", int(instr["lhs"])],
        "rhs": ["hlir_value", int(instr["rhs"])],
        "comparison_kind": comparison_kind,
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "result_const": result_const,
    }
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["reflexive_comparison"],
        "comparison_kind": comparison_kind,
        "symbolic_value": symbolic_value_id,
        "result_const": result_const,
    }
    domain_bounds = {
        "kind": "all-i64-values-with-reflexive-equality-and-order",
        "comparison_kind": comparison_kind,
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "result_const": result_const,
        "preconditions": [
            "both operands are the same S3 HLIR SSA value",
            "symbolic operand is not a constant-fold duplicate",
            "operand type is exact i64, not f64 or an approximate numeric domain",
            "comparison has no side effects",
            "producer evaluation is preserved according to producer_evaluation_policy",
        ],
    }
    validator_log = {
        "validator": "translation-validation",
        "method": "reflexive-comparison-symbolic-proof-over-s3-hlir",
        "accepted": True,
        "comparison_kind": comparison_kind,
        "domain_bounds": domain_bounds,
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "comparison_kind": comparison_kind,
        "symbolic_value": symbolic_value_id,
        "result_const": result_const,
    }
    rid = "s4-reflexive-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "symbolic_reflexive_cmp_i64",
        "comparison_kind": comparison_kind,
        "proposal_kind": "exact_symbolic_reflexive_comparison",
        "proposal_origin": "madaros_v2_s4_receipt.reflexive-comparison-symbolic-proof",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "symbolic_reflexive_cmp_i64", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_exact_symbolic_predicate",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_lhs": int(instr["lhs"]),
        "original_rhs": int(instr["rhs"]),
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "same_operand_id": True,
        "producer_evaluation_policy": producer_evaluation_policy,
        "result_const": result_const,
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(proposed)),
        "rewritten_enode_sha256": sha256_text(stable_json(proposed)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "exact-symbolic-reflexive-predicate-no-training"})),
        "domain": "all-i64-values-with-reflexive-equality-and-order",
        "domain_bounds": domain_bounds,
        "error_bound": "0",
        "error_bound_method": "exact-reflexive-comparison",
        "gum_covariance_assumptions": "not-applicable: exact i64 symbolic predicate",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "translation-validation",
        "validator_attempted": ["translation-validation", "reflexive-comparison-proof", "producer-evaluation-preservation-proof"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "selected_for_extraction": True,
        "ir_mutation_allowed": False,
        "accepted": True,
    }


def symbolic_sub_self_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    symbolic_value_id: int,
    symbolic_producer: dict[str, Any],
    producer_evaluation_policy: str,
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr["bin_op"]), f"op{instr['bin_op']}")
    result_const = ("int", 0)
    proposed = make_const_enode(*result_const)
    fallback = {
        "op": op_name,
        "lhs": ["hlir_value", int(instr["lhs"])],
        "rhs": ["hlir_value", int(instr["rhs"])],
        "subtraction_kind": "sub_self_zero",
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "result_const": result_const,
    }
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["same_ssa_subtraction"],
        "subtraction_kind": "sub_self_zero",
        "symbolic_value": symbolic_value_id,
        "result_const": result_const,
    }
    domain_bounds = {
        "kind": "all-i64-values-with-same-ssa-subtraction",
        "subtraction_kind": "sub_self_zero",
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "result_const": result_const,
        "preconditions": [
            "both operands are the same S3 HLIR SSA value",
            "symbolic operand is not a constant-fold duplicate",
            "operand type is exact i64, not f64 or an approximate numeric domain",
            "subtraction has no side effects",
            "producer evaluation is preserved according to producer_evaluation_policy",
        ],
    }
    validator_log = {
        "validator": "translation-validation",
        "method": "same-ssa-subtraction-symbolic-proof-over-s3-hlir",
        "accepted": True,
        "subtraction_kind": "sub_self_zero",
        "domain_bounds": domain_bounds,
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "subtraction_kind": "sub_self_zero",
        "symbolic_value": symbolic_value_id,
        "result_const": result_const,
    }
    rid = "s4-sub-self-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "symbolic_sub_self_i64",
        "subtraction_kind": "sub_self_zero",
        "proposal_kind": "exact_symbolic_sub_self",
        "proposal_origin": "madaros_v2_s4_receipt.sub-self-symbolic-proof",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "symbolic_sub_self_i64", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_exact_symbolic_arithmetic",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_lhs": int(instr["lhs"]),
        "original_rhs": int(instr["rhs"]),
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "same_operand_id": True,
        "producer_evaluation_policy": producer_evaluation_policy,
        "result_const": result_const,
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(proposed)),
        "rewritten_enode_sha256": sha256_text(stable_json(proposed)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "exact-symbolic-sub-self-no-training"})),
        "domain": "all-i64-values-with-same-ssa-subtraction",
        "domain_bounds": domain_bounds,
        "error_bound": "0",
        "error_bound_method": "exact-same-ssa-subtraction",
        "gum_covariance_assumptions": "not-applicable: exact i64 symbolic arithmetic",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "translation-validation",
        "validator_attempted": ["translation-validation", "same-ssa-subtraction-proof", "producer-evaluation-preservation-proof"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "selected_for_extraction": True,
        "ir_mutation_allowed": False,
        "accepted": True,
    }


def blocked_producer_evaluation_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    rewrite_kind: str,
    blocked_kind: str,
    symbolic_value_id: int,
    symbolic_producer: dict[str, Any],
    result_const: tuple[str, int | bool],
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr.get("bin_op", -1)), f"op{instr.get('bin_op', -1)}")
    proposed = make_const_enode(*result_const)
    if rewrite_kind == "symbolic_reflexive_cmp_i64":
        proposal_kind = "blocked_symbolic_reflexive_comparison"
        config_pass = "symbolic_reflexive_cmp_i64_eval_guard"
        reason_noun = "predicate"
        receipt_family = "reflexive_comparison"
    elif rewrite_kind == "symbolic_sub_self_i64":
        proposal_kind = "blocked_symbolic_sub_self"
        config_pass = "symbolic_sub_self_i64_eval_guard"
        reason_noun = "arithmetic rewrite"
        receipt_family = "same_ssa_subtraction"
    else:
        raise SystemExit(f"unsupported producer-evaluation blocker rewrite kind: {rewrite_kind}")
    reason_detail = {
        "kind": "producer_evaluation_not_proven",
        "blocked_kind": blocked_kind,
        "receipt_family": receipt_family,
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "result_const": result_const,
        "impact": f"rewriting the {reason_noun} before S5/DCE proves producer evaluation could erase an effectful producer",
    }
    fallback = {
        "op": op_name,
        "lhs": ["hlir_value", int(instr.get("lhs", -1))],
        "rhs": ["hlir_value", int(instr.get("rhs", -1))],
        "blocked_reason": reason_detail,
    }
    validator_log = {
        "validator": "blocked",
        "method": "producer-evaluation-preservation-guard",
        "accepted": False,
        "blocked": True,
        "rejection_reason": "producer_evaluation_not_proven",
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "proposal": "blocked_producer_evaluation",
        "reason": reason_detail,
    }
    rid = "s4-blocked-eval-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": rewrite_kind,
        "proposal_kind": proposal_kind,
        "proposal_origin": "madaros_v2_s4_receipt.producer-evaluation-preservation-guard",
        "proposal_config_sha256": sha256_text(stable_json({"pass": config_pass, "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_blocked_producer_evaluation",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_lhs": int(instr["lhs"]),
        "original_rhs": int(instr["rhs"]),
        "symbolic_value": symbolic_value_id,
        "symbolic_producer": symbolic_producer,
        "same_operand_id": True,
        "blocked_kind": blocked_kind,
        "comparison_kind": blocked_kind if rewrite_kind == "symbolic_reflexive_cmp_i64" else None,
        "producer_evaluation_policy": "blocked: producer evaluation is not proven",
        "result_const": result_const,
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(proposed)),
        "rewritten_enode_sha256": sha256_text(stable_json(proposed)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json({"basis_family": "exact_symbolic", "blocked": True, "rewrite_kind": rewrite_kind, "blocked_kind": blocked_kind})),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "blocked-no-training", "reason": reason_detail})),
        "domain": "blocked: producer evaluation is not proven",
        "domain_bounds": {
            "kind": "blocked_producer_evaluation",
            "reason": reason_detail,
        },
        "error_bound": "unproven: producer evaluation not proven",
        "error_bound_method": "blocked-before-validation",
        "gum_covariance_assumptions": "not-applicable: blocked before optimization",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "blocked",
        "validator_attempted": ["producer-evaluation-preservation-guard"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "blocked": True,
        "rejection_reason_code": "producer_evaluation_not_proven",
        "rejection_reason": f"producer evaluation must be preserved before replacing the {reason_noun}",
        "selected_for_extraction": False,
        "ir_mutation_allowed": False,
        "accepted": False,
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


def rejected_distinct_sub_self_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    lhs_info: dict[str, Any],
    rhs_info: dict[str, Any],
) -> dict[str, Any]:
    fallback = {
        "op": "sub",
        "lhs": ["symbolic_value", int(instr["lhs"])],
        "rhs": ["symbolic_value", int(instr["rhs"])],
        "proposal": "distinct_symbolic_sub_to_zero",
    }
    rewritten = make_const_enode("int", 0)
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["constant"],
        "constant": 0,
        "proposal": "distinct_symbolic_sub_to_zero",
    }
    counterexample = {
        "lhs_symbolic_value": int(instr["lhs"]),
        "rhs_symbolic_value": int(instr["rhs"]),
        "lhs_value": 1,
        "rhs_value": 2,
        "original_behavior": "returns_-1",
        "rewritten_behavior": "returns_0",
    }
    validator_log = {
        "validator": "rejected",
        "method": "counterexample-guided-translation-validation",
        "accepted": False,
        "rejection_reason": "counterexample_distinct_symbolic_subtraction",
        "counterexamples": [counterexample],
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "proposal": "distinct_symbolic_sub_to_zero",
        "counterexample": counterexample,
    }
    rid = "s4-reject-sub-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "symbolic_sub_self_i64",
        "subtraction_kind": "distinct_symbolic_sub_not_zero",
        "proposal_kind": "rejected_symbolic_sub_self",
        "proposal_origin": "madaros_v2_s4_receipt.counterexample-guided-sub-self-negative-lane",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "reject_symbolic_sub_self_distinct_operands", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_rejected_counterexample",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_lhs": int(instr["lhs"]),
        "original_rhs": int(instr["rhs"]),
        "lhs_symbolic_producer": lhs_info,
        "rhs_symbolic_producer": rhs_info,
        "same_operand_id": False,
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(rewritten)),
        "rewritten_enode_sha256": sha256_text(stable_json(rewritten)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "hand-authored-symbolic-sub-negative-proposal"})),
        "domain": "all-i64-values-with-distinct-ssa-subtraction",
        "domain_bounds": {
            "kind": "all-i64-values-with-distinct-ssa-subtraction",
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "preconditions": ["operands are distinct S3 HLIR SSA values"],
        },
        "error_bound": "unbounded: proposal rejected",
        "error_bound_method": "counterexample",
        "gum_covariance_assumptions": "not-applicable: rejected exact symbolic proposal",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "rejected",
        "validator_attempted": ["translation-validation", "counterexample"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "rejection_reason_code": "counterexample_found",
        "rejection_reason": "counterexample_distinct_symbolic_subtraction",
        "counterexample_sha256": sha256_text(stable_json(counterexample)),
        "counterexample_set_sha256": sha256_text(stable_json([counterexample])),
        "counterexample_count": 1,
        "counterexamples": [counterexample],
        "selected_for_extraction": False,
        "ir_mutation_allowed": False,
        "accepted": False,
    }


def rejected_distinct_reflexive_cmp_receipt(
    case_id: str,
    source: str,
    hlir_sha: str,
    func_name: str,
    block_label: str,
    instr: dict[str, Any],
    lhs_info: dict[str, Any],
    rhs_info: dict[str, Any],
    comparison_kind: str,
    proposed_const: tuple[str, bool],
) -> dict[str, Any]:
    op_name = BIN_OPS.get(int(instr["bin_op"]), f"op{instr['bin_op']}")
    fallback = {
        "op": op_name,
        "lhs": ["symbolic_value", int(instr["lhs"])],
        "rhs": ["symbolic_value", int(instr["rhs"])],
        "proposal": "distinct_symbolic_comparison_as_reflexive",
        "comparison_kind": comparison_kind,
    }
    rewritten = make_const_enode(*proposed_const)
    coefficients = {
        "basis_family": "exact_symbolic",
        "basis": ["reflexive_comparison"],
        "comparison_kind": comparison_kind,
        "proposal": "distinct_symbolic_comparison_as_reflexive",
        "constant": proposed_const[1],
    }
    if comparison_kind == "eq_self_true":
        counterexample = {
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "lhs_value": 1,
            "rhs_value": 2,
            "original_behavior": "returns_false",
            "rewritten_behavior": "returns_true",
        }
    elif comparison_kind == "ne_self_false":
        counterexample = {
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "lhs_value": 1,
            "rhs_value": 2,
            "original_behavior": "returns_true",
            "rewritten_behavior": "returns_false",
        }
    elif comparison_kind == "le_self_true":
        counterexample = {
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "lhs_value": 2,
            "rhs_value": 1,
            "original_behavior": "returns_false",
            "rewritten_behavior": "returns_true",
        }
    elif comparison_kind == "ge_self_true":
        counterexample = {
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "lhs_value": 1,
            "rhs_value": 2,
            "original_behavior": "returns_false",
            "rewritten_behavior": "returns_true",
        }
    elif comparison_kind == "lt_self_false":
        counterexample = {
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "lhs_value": 1,
            "rhs_value": 2,
            "original_behavior": "returns_true",
            "rewritten_behavior": "returns_false",
        }
    elif comparison_kind == "gt_self_false":
        counterexample = {
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "lhs_value": 2,
            "rhs_value": 1,
            "original_behavior": "returns_true",
            "rewritten_behavior": "returns_false",
        }
    else:
        raise SystemExit(f"unsupported distinct comparison rejection kind: {comparison_kind}")
    validator_log = {
        "validator": "rejected",
        "method": "counterexample-guided-translation-validation",
        "accepted": False,
        "rejection_reason": "counterexample_distinct_symbolic_comparison_not_reflexive",
        "comparison_kind": comparison_kind,
        "counterexamples": [counterexample],
        "fallback": fallback,
    }
    rid_payload = {
        "case_id": case_id,
        "func": func_name,
        "block": block_label,
        "result": instr["result"],
        "proposal": "distinct_symbolic_comparison_as_reflexive",
        "comparison_kind": comparison_kind,
        "counterexample": counterexample,
    }
    rid = "s4-reject-cmp-" + sha256_text(stable_json(rid_payload))[:16]
    eclass_id = f"{func_name}:{block_label}:{instr['result']}"
    return {
        "schema_version": REWRITE_SCHEMA,
        "case_id": case_id,
        "source": source,
        "input_ir_sha256": hlir_sha,
        "eclass_id": eclass_id,
        "proposed_rewrite_id": rid,
        "rewrite_kind": "symbolic_reflexive_cmp_i64",
        "comparison_kind": comparison_kind,
        "proposal_kind": "rejected_symbolic_reflexive_comparison",
        "proposal_origin": "madaros_v2_s4_receipt.counterexample-guided-distinct-comparison-negative-lane",
        "proposal_config_sha256": sha256_text(stable_json({"pass": "reject_symbolic_reflexive_cmp_distinct_operands", "schema": REWRITE_SCHEMA})),
        "ekan_receipt_kind": "ekan_rejected_counterexample",
        "function": func_name,
        "block": block_label,
        "instruction_result": instr["result"],
        "original_lhs": int(instr["lhs"]),
        "original_rhs": int(instr["rhs"]),
        "lhs_symbolic_producer": lhs_info,
        "rhs_symbolic_producer": rhs_info,
        "same_operand_id": False,
        "result_const": proposed_const,
        "original_enode_sha256": sha256_text(stable_json(instr)),
        "proposed_enode_sha256": sha256_text(stable_json(rewritten)),
        "rewritten_enode_sha256": sha256_text(stable_json(rewritten)),
        "basis_family": "exact_symbolic",
        "coefficient_sha256": sha256_text(stable_json(coefficients)),
        "training_or_provenance_sha256": sha256_text(stable_json({"provenance": "hand-authored-symbolic-comparison-negative-proposal"})),
        "domain": "all-i64-values-with-distinct-ssa-comparison",
        "domain_bounds": {
            "kind": "all-i64-values-with-distinct-ssa-comparison",
            "lhs_symbolic_value": int(instr["lhs"]),
            "rhs_symbolic_value": int(instr["rhs"]),
            "comparison_kind": comparison_kind,
            "preconditions": ["operands are distinct S3 HLIR SSA values"],
        },
        "error_bound": "unbounded: proposal rejected",
        "error_bound_method": "counterexample",
        "gum_covariance_assumptions": "not-applicable: rejected exact symbolic proposal",
        "exact_fallback_expr_sha256": sha256_text(stable_json(fallback)),
        "validator": "rejected",
        "validator_attempted": ["translation-validation", "counterexample"],
        "validator_log_sha256": sha256_text(stable_json(validator_log)),
        "rejection_reason_code": "counterexample_found",
        "rejection_reason": "counterexample_distinct_symbolic_comparison_not_reflexive",
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
            "value_ref": 1,
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
            if op == "value_ref":
                return int(model["value_ref"])
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
                "proof_obligation": (
                    "translation-validation already accepted with zero error bound; S5 must keep the local leaf call producer evaluated"
                    if rewrite.get("producer_evaluation_policy") == "direct_call_leaf_pure_keep_producer_evaluated"
                    else "translation-validation already accepted with zero error bound"
                ),
                "exact_fallback_expr_sha256": rewrite["exact_fallback_expr_sha256"],
                "coefficient_sha256": rewrite["coefficient_sha256"],
                "basis_family": rewrite["basis_family"],
                "error_bound": rewrite["error_bound"],
                "domain": rewrite["domain"],
                "domain_bounds": rewrite["domain_bounds"],
                "lowering_effect": (
                    "replace_binary_identity_expr_with_existing_value"
                    if rewrite["rewrite_kind"] == "symbolic_identity_i64"
                    else "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated"
                    if rewrite["rewrite_kind"] == "symbolic_sub_self_i64"
                    and rewrite.get("producer_evaluation_policy") == "direct_call_leaf_pure_keep_producer_evaluated"
                    else "replace_binary_sub_self_expr_with_const_i64_zero"
                    if rewrite["rewrite_kind"] == "symbolic_sub_self_i64"
                    else "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated"
                    if rewrite["rewrite_kind"] == "symbolic_reflexive_cmp_i64"
                    and rewrite.get("producer_evaluation_policy") == "direct_call_leaf_pure_keep_producer_evaluated"
                    else "replace_binary_predicate_expr_with_const_bool"
                    if rewrite["rewrite_kind"] == "symbolic_reflexive_cmp_i64"
                    else "replace_binary_constant_expr_with_const"
                ),
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
                "selection_reason": (
                    "blocked_by_producer_evaluation_guard"
                    if rewrite.get("rejection_reason_code") == "producer_evaluation_not_proven"
                    else "blocked_by_operand_provenance_guard"
                ),
                "rejection_reason_code": rewrite["rejection_reason_code"],
                "proof_obligation": (
                    "prove producer evaluation preservation before extraction"
                    if rewrite.get("rejection_reason_code") == "producer_evaluation_not_proven"
                    else "prove S3 HLIR operand provenance before extraction"
                ),
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


def const_from_info(info: dict[str, Any] | None) -> tuple[str, int | bool] | None:
    if not info or info.get("producer_kind") != "const":
        return None
    return (str(info.get("const_kind")), info.get("value", 0))


def is_i64_const(info: dict[str, Any] | None, value: int) -> bool:
    cv = const_from_info(info)
    return cv == ("int", value)


def is_symbolic_value(info: dict[str, Any] | None) -> bool:
    return bool(info) and info.get("producer_kind") != "const"


def param_info(param: dict[str, Any], func_name: str) -> dict[str, Any]:
    label = f"param:{param['name']}"
    return {
        "producer_kind": "param",
        "label": label,
        "producer_label": label,
        "function": func_name,
        "name": param["name"],
        "value_id": int(param["value_id"]),
    }


def block_param_info(param: dict[str, Any], func_name: str, block_label: str) -> dict[str, Any]:
    label = f"block_param:{block_label}:{param['value_id']}"
    return {
        "producer_kind": "block_param",
        "label": label,
        "producer_label": label,
        "function": func_name,
        "block": block_label,
        "value_id": int(param["value_id"]),
    }


def const_info(instr: dict[str, Any], cv: tuple[str, int | bool], func_name: str, block_label: str) -> dict[str, Any]:
    label = f"const:{cv[0]}:{cv[1]}"
    return {
        "producer_kind": "const",
        "label": label,
        "producer_label": label,
        "function": func_name,
        "block": block_label,
        "value_id": int(instr["result"]),
        "const_kind": cv[0],
        "value": cv[1],
    }


def instr_value_info(instr: dict[str, Any], func_name: str, block_label: str, function_summaries: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    op = instr.get("op")
    function_summaries = function_summaries or {}
    call_summary: dict[str, Any] = {}
    if op == "call_direct":
        call_name = instr.get("call_name", "")
        call_summary = dict(function_summaries.get(call_name, {}))
        label = f"call:{call_name}"
    else:
        label = f"{op}:{instr.get('result')}"
    return {
        "producer_kind": op,
        "label": label,
        "producer_label": label,
        "function": func_name,
        "block": block_label,
        "value_id": int(instr["result"]),
        "call_name": instr.get("call_name", ""),
        "call_summary": call_summary,
        "call_leaf_pure": bool(call_summary.get("leaf_pure", False)),
        "call_purity_reason": call_summary.get("purity_reason", "not-a-call"),
    }


def identity_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[str, int, dict[str, Any], str, tuple[str, int | bool]] | None:
    op = int(instr.get("bin_op", -1))
    lhs_id = int(instr.get("lhs", -1))
    rhs_id = int(instr.get("rhs", -1))
    if op == 0 and is_symbolic_value(lhs_info) and is_i64_const(rhs_info, 0):
        return ("add_zero_rhs", lhs_id, lhs_info or {}, "rhs", ("int", 0))
    if op == 0 and is_i64_const(lhs_info, 0) and is_symbolic_value(rhs_info):
        return ("add_zero_lhs", rhs_id, rhs_info or {}, "lhs", ("int", 0))
    if op == 2 and is_symbolic_value(lhs_info) and is_i64_const(rhs_info, 1):
        return ("mul_one_rhs", lhs_id, lhs_info or {}, "rhs", ("int", 1))
    if op == 2 and is_i64_const(lhs_info, 1) and is_symbolic_value(rhs_info):
        return ("mul_one_lhs", rhs_id, rhs_info or {}, "lhs", ("int", 1))
    if op == 1 and is_symbolic_value(lhs_info) and is_i64_const(rhs_info, 0):
        return ("sub_zero_rhs", lhs_id, lhs_info or {}, "rhs", ("int", 0))
    return None


def reflexive_cmp_result(
    op: int,
) -> tuple[str, tuple[str, bool]] | None:
    if op == 18:
        return ("eq_self_true", ("bool", True))
    if op == 19:
        return ("ne_self_false", ("bool", False))
    if op == 21:
        return ("le_self_true", ("bool", True))
    if op == 23:
        return ("ge_self_true", ("bool", True))
    if op == 20:
        return ("lt_self_false", ("bool", False))
    if op == 22:
        return ("gt_self_false", ("bool", False))
    return None


def producer_evaluation_policy(info: dict[str, Any]) -> str | None:
    if info.get("producer_kind") in {"param", "block_param"}:
        return "producer_is_param_or_block_param_no_effectful_eval"
    if info.get("producer_kind") == "call_direct" and info.get("call_leaf_pure") is True:
        return "direct_call_leaf_pure_keep_producer_evaluated"
    return None


def reflexive_cmp_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[str, int, dict[str, Any], tuple[str, bool], str] | None:
    if int(instr.get("lhs", -1)) != int(instr.get("rhs", -2)):
        return None
    if not is_symbolic_value(lhs_info) or not rhs_info:
        return None
    if lhs_info.get("value_id") != rhs_info.get("value_id"):
        return None
    if instr.get("ty", {}).get("kind") != "bool":
        return None
    op = int(instr.get("bin_op", -1))
    result = reflexive_cmp_result(op)
    if result is None:
        return None
    policy = producer_evaluation_policy(lhs_info)
    if policy is None:
        return None
    lhs_id = int(instr.get("lhs", -1))
    comparison_kind, result_const = result
    return (comparison_kind, lhs_id, lhs_info, result_const, policy)


def blocked_reflexive_cmp_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[str, int, dict[str, Any], tuple[str, bool]] | None:
    if int(instr.get("lhs", -1)) != int(instr.get("rhs", -2)):
        return None
    if not is_symbolic_value(lhs_info) or not rhs_info:
        return None
    if lhs_info.get("value_id") != rhs_info.get("value_id"):
        return None
    if instr.get("ty", {}).get("kind") != "bool":
        return None
    result = reflexive_cmp_result(int(instr.get("bin_op", -1)))
    if result is None:
        return None
    if producer_evaluation_policy(lhs_info) is not None:
        return None
    comparison_kind, result_const = result
    return (comparison_kind, int(instr.get("lhs", -1)), lhs_info, result_const)


def sub_self_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[int, dict[str, Any], str] | None:
    if int(instr.get("bin_op", -1)) != 1:
        return None
    if int(instr.get("lhs", -1)) != int(instr.get("rhs", -2)):
        return None
    if not is_symbolic_value(lhs_info) or not rhs_info:
        return None
    if lhs_info.get("value_id") != rhs_info.get("value_id"):
        return None
    if instr.get("ty", {}).get("kind") != "i64":
        return None
    policy = producer_evaluation_policy(lhs_info)
    if policy is None:
        return None
    return (int(instr.get("lhs", -1)), lhs_info, policy)


def blocked_sub_self_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]] | None:
    if int(instr.get("bin_op", -1)) != 1:
        return None
    if int(instr.get("lhs", -1)) != int(instr.get("rhs", -2)):
        return None
    if not is_symbolic_value(lhs_info) or not rhs_info:
        return None
    if lhs_info.get("value_id") != rhs_info.get("value_id"):
        return None
    if instr.get("ty", {}).get("kind") != "i64":
        return None
    if producer_evaluation_policy(lhs_info) is not None:
        return None
    return (int(instr.get("lhs", -1)), lhs_info)


def rejected_distinct_sub_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if int(instr.get("bin_op", -1)) != 1:
        return None
    if int(instr.get("lhs", -1)) == int(instr.get("rhs", -2)):
        return None
    if not is_symbolic_value(lhs_info) or not is_symbolic_value(rhs_info):
        return None
    if lhs_info.get("value_id") == rhs_info.get("value_id"):
        return None
    if instr.get("ty", {}).get("kind") != "i64":
        return None
    return (lhs_info, rhs_info)


def rejected_distinct_reflexive_cmp_candidate(
    instr: dict[str, Any],
    lhs_info: dict[str, Any] | None,
    rhs_info: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], str, tuple[str, bool]] | None:
    if int(instr.get("lhs", -1)) == int(instr.get("rhs", -2)):
        return None
    if not is_symbolic_value(lhs_info) or not is_symbolic_value(rhs_info):
        return None
    if lhs_info.get("value_id") == rhs_info.get("value_id"):
        return None
    if instr.get("ty", {}).get("kind") != "bool":
        return None
    result = reflexive_cmp_result(int(instr.get("bin_op", -1)))
    if result is None:
        return None
    comparison_kind, proposed_const = result
    return (lhs_info, rhs_info, comparison_kind, proposed_const)


def analyze_hlir(case_id: str, source: str, hlir_text: str, hlir_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(hlir_text)
    module = data["module"]
    eclasses: list[dict[str, Any]] = []
    rewrites: list[dict[str, Any]] = []
    function_summaries: dict[str, dict[str, Any]] = {}
    for summary_func in module["functions"]:
        call_count = 0
        op_kinds: set[str] = set()
        for block in summary_func.get("blocks", []):
            for instr in block.get("instrs", []):
                op_kinds.add(str(instr.get("op", "")))
                if instr.get("op") == "call_direct":
                    call_count += 1
        function_summaries[summary_func["name"]] = {
            "leaf_pure": call_count == 0,
            "purity_reason": "local_leaf_no_call_direct" if call_count == 0 else "contains_call_direct",
            "call_direct_count": call_count,
            "op_kinds": sorted(op_kinds),
        }
    for func in module["functions"]:
        value_info: dict[int, dict[str, Any]] = {}
        for param in func.get("params", []):
            value_info[int(param["value_id"])] = param_info(param, func["name"])
        for block in func["blocks"]:
            for param in block.get("params", []):
                value_info[int(param["value_id"])] = block_param_info(param, func["name"], block["label"])
            for instr in block["instrs"]:
                result = int(instr["result"])
                enodes = [{
                    "kind": "hlir-original",
                    "sha256": sha256_text(stable_json(instr)),
                    "op": instr.get("op"),
                }]
                cv = const_value(instr)
                if cv is not None:
                    value_info[result] = const_info(instr, cv, func["name"], block["label"])
                if instr.get("op") == "binary":
                    lhs_info = value_info.get(int(instr.get("lhs", -1)))
                    rhs_info = value_info.get(int(instr.get("rhs", -1)))
                    lhs = const_from_info(lhs_info)
                    rhs = const_from_info(rhs_info)
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
                                value_info[result] = const_info(
                                    {
                                        "result": result,
                                        "constant": {
                                            "kind": folded[0],
                                            "int_val": int(folded[1]) if folded[0] == "int" else 0,
                                            "bool_val": bool(folded[1]) if folded[0] == "bool" else False,
                                        },
                                    },
                                    folded,
                                    func["name"],
                                    block["label"],
                                )
                    else:
                        sub_self = sub_self_candidate(instr, lhs_info, rhs_info)
                        if sub_self is not None:
                            symbolic_value_id, symbolic_producer, eval_policy = sub_self
                            result_const = ("int", 0)
                            receipt = symbolic_sub_self_receipt(
                                case_id,
                                source,
                                hlir_sha,
                                func["name"],
                                block["label"],
                                instr,
                                symbolic_value_id,
                                symbolic_producer,
                                eval_policy,
                            )
                            rewrites.append(receipt)
                            enodes.append({
                                "kind": "s4-rewrite",
                                "rewrite_id": receipt["proposed_rewrite_id"],
                                "sha256": receipt["rewritten_enode_sha256"],
                                "op": "const",
                            })
                            value_info[result] = const_info(
                                {
                                    "result": result,
                                    "constant": {
                                        "kind": result_const[0],
                                        "int_val": int(result_const[1]),
                                        "bool_val": False,
                                    },
                                },
                                result_const,
                                func["name"],
                                block["label"],
                            )
                        else:
                            blocked_sub = blocked_sub_self_candidate(instr, lhs_info, rhs_info)
                            if blocked_sub is not None:
                                symbolic_value_id, symbolic_producer = blocked_sub
                                receipt = blocked_producer_evaluation_receipt(
                                    case_id,
                                    source,
                                    hlir_sha,
                                    func["name"],
                                    block["label"],
                                    instr,
                                    "symbolic_sub_self_i64",
                                    "sub_self_zero",
                                    symbolic_value_id,
                                    symbolic_producer,
                                    ("int", 0),
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
                                rejected_sub = rejected_distinct_sub_candidate(instr, lhs_info, rhs_info)
                                if rejected_sub is not None:
                                    distinct_lhs_info, distinct_rhs_info = rejected_sub
                                    receipt = rejected_distinct_sub_self_receipt(
                                        case_id,
                                        source,
                                        hlir_sha,
                                        func["name"],
                                        block["label"],
                                        instr,
                                        distinct_lhs_info,
                                        distinct_rhs_info,
                                    )
                                    rewrites.append(receipt)
                                    enodes.append({
                                        "kind": "s4-rejected-rewrite",
                                        "rewrite_id": receipt["proposed_rewrite_id"],
                                        "sha256": receipt["rewritten_enode_sha256"],
                                        "op": "const",
                                        "rejection_reason": receipt["rejection_reason"],
                                    })
                        reflexive_cmp = reflexive_cmp_candidate(instr, lhs_info, rhs_info)
                        if reflexive_cmp is not None:
                            comparison_kind, symbolic_value_id, symbolic_producer, result_const, eval_policy = reflexive_cmp
                            receipt = symbolic_reflexive_cmp_receipt(
                                case_id,
                                source,
                                hlir_sha,
                                func["name"],
                                block["label"],
                                instr,
                                comparison_kind,
                                symbolic_value_id,
                                symbolic_producer,
                                result_const,
                                eval_policy,
                            )
                            rewrites.append(receipt)
                            enodes.append({
                                "kind": "s4-rewrite",
                                "rewrite_id": receipt["proposed_rewrite_id"],
                                "sha256": receipt["rewritten_enode_sha256"],
                                "op": "const",
                            })
                            value_info[result] = const_info(
                                {
                                    "result": result,
                                    "constant": {
                                        "kind": result_const[0],
                                        "int_val": 0,
                                        "bool_val": bool(result_const[1]),
                                    },
                                },
                                result_const,
                                func["name"],
                                block["label"],
                            )
                        else:
                            blocked_reflexive = blocked_reflexive_cmp_candidate(instr, lhs_info, rhs_info)
                            if blocked_reflexive is not None:
                                comparison_kind, symbolic_value_id, symbolic_producer, result_const = blocked_reflexive
                                receipt = blocked_producer_evaluation_receipt(
                                    case_id,
                                    source,
                                    hlir_sha,
                                    func["name"],
                                    block["label"],
                                    instr,
                                    "symbolic_reflexive_cmp_i64",
                                    comparison_kind,
                                    symbolic_value_id,
                                    symbolic_producer,
                                    result_const,
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
                                rejected_cmp = rejected_distinct_reflexive_cmp_candidate(instr, lhs_info, rhs_info)
                                if rejected_cmp is not None:
                                    distinct_lhs_info, distinct_rhs_info, comparison_kind, proposed_const = rejected_cmp
                                    receipt = rejected_distinct_reflexive_cmp_receipt(
                                        case_id,
                                        source,
                                        hlir_sha,
                                        func["name"],
                                        block["label"],
                                        instr,
                                        distinct_lhs_info,
                                        distinct_rhs_info,
                                        comparison_kind,
                                        proposed_const,
                                    )
                                    rewrites.append(receipt)
                                    enodes.append({
                                        "kind": "s4-rejected-rewrite",
                                        "rewrite_id": receipt["proposed_rewrite_id"],
                                        "sha256": receipt["rewritten_enode_sha256"],
                                        "op": "const",
                                        "rejection_reason": receipt["rejection_reason"],
                                    })
                        identity = identity_candidate(instr, lhs_info, rhs_info)
                        if identity is not None:
                            identity_kind, symbolic_value_id, symbolic_producer, neutral_side, neutral_const = identity
                            receipt = symbolic_identity_receipt(
                                case_id,
                                source,
                                hlir_sha,
                                func["name"],
                                block["label"],
                                instr,
                                identity_kind,
                                symbolic_value_id,
                                symbolic_producer,
                                neutral_side,
                                neutral_const,
                            )
                            rewrites.append(receipt)
                            enodes.append({
                                "kind": "s4-rewrite",
                                "rewrite_id": receipt["proposed_rewrite_id"],
                                "sha256": receipt["rewritten_enode_sha256"],
                                "op": "value_ref",
                                "value_id": symbolic_value_id,
                            })
                            value_info[result] = {
                                **symbolic_producer,
                                "label": f"identity:{identity_kind}->{symbolic_producer.get('label', symbolic_value_id)}",
                                "producer_label": f"identity:{identity_kind}->{symbolic_producer.get('producer_label', symbolic_value_id)}",
                                "identity_source_result": result,
                            }
                        if result not in value_info:
                            value_info[result] = instr_value_info(instr, func["name"], block["label"], function_summaries)
                elif cv is None and result >= 0:
                    value_info[result] = instr_value_info(instr, func["name"], block["label"], function_summaries)
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
        "s4_full_complete": False,
        "stage_contract_level": "S4_BOUNDARY_NOT_FULL",
        "s_full_contract": "blocked_until_full_s4_obligations_are_gated",
        "s4_claim": "conservative_egraph_ekan_receipt_boundary_with_operand_provenance_guard",
        "s4_remaining": [
            "multi-rule equality saturation",
            "learned or approximate E-KAN proposals with declared domains and fallback expressions",
            "broad counterexample search over accepted and tempting sibling rewrites",
            "producer purity and evaluation-preservation beyond the current local leaf subset",
            "broader non-constant algebraic identities beyond neutral-element, reflexive-comparison, and same-SSA subtraction identities",
            "downstream optimizer integration beyond receipt-only extraction",
            "full-domain translation validation for every selected rewrite family",
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
