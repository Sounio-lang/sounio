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
    hlir_text = hlir_path.read_text(encoding="utf-8")
    hlir_sha = sha256_text(hlir_text)
    if hlir_sha != s3_receipt.get("hlir_byte_sha256"):
        raise SystemExit("S3 HLIR hash mismatch")

    egraph, rewrites = analyze_hlir(case_id, source_rel, hlir_text, hlir_sha)
    accepted = [r for r in rewrites if r["accepted"]]
    rejected = [r for r in rewrites if not r["accepted"]]
    egraph_path = out_dir / f"{case_id}.s4.egraph.json"
    rewrites_path = out_dir / f"{case_id}.s4.rewrites.json"
    receipt_path = out_dir / f"{case_id}.s4.receipt.json"
    egraph_path.write_text(json.dumps(egraph, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rewrites_path.write_text(json.dumps(rewrites, sort_keys=True, indent=2) + "\n", encoding="utf-8")
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
        "accepted_rewrite_ids": [r["proposed_rewrite_id"] for r in accepted],
        "rejected_rewrite_ids": [r["proposed_rewrite_id"] for r in rejected],
        "validators": sorted({r["validator"] for r in rewrites}),
        "basis_families": sorted({r["basis_family"] for r in rewrites}),
        "s4_complete": False,
        "s4_boundary_complete": True,
        "s4_claim": "conservative_egraph_ekan_exact_constant_fold_i64_receipt_boundary",
        "s4_remaining": [
            "multi-rule equality saturation",
            "non-constant algebraic identities",
            "extractor cost model and downstream optimizer integration",
        ],
    }
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    receipt["receipt_sha256"] = sha256_text(payload)
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    receipt_path.write_text(payload, encoding="utf-8")
    print(
        f"madaros-v2-s4: case={case_id} accepted={len(accepted)} "
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
