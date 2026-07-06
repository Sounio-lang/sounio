#!/usr/bin/env bash
# Madaros v2 S4 gate: conservative e-graph/E-KAN rewrite receipts over S3 HLIR.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_MADAROS_V2_GATE_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT_DIR="${SOUNIO_MADAROS_V2_S4_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s4.XXXXXX)}"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
MANIFEST="${ROOT_DIR}/tests/madaros/v2_s4/manifest.tsv"
PY="${ROOT_DIR}/scripts/dev/madaros_v2_s4_receipt.py"

receipt_ok() {
  local elf="$1"
  local receipt="$elf.gate-receipt"
  [[ -f "$receipt" ]] || return 1
  local want
  want="$(sha256sum "$elf" 2>/dev/null | cut -d' ' -f1)"
  [[ -n "$want" ]] || return 1
  grep -Fq "$want" "$receipt" 2>/dev/null || return 1
  grep -Fxq "smt_skip=0" "$receipt" 2>/dev/null
}

ensure_s4_raw_artifact() {
  if [[ -n "${MADAROS_RAW_BIN:-}" ]]; then
    return 0
  fi
  local artifact="${ROOT_DIR}/artifacts/self-hosted/madaros"
  if [[ ! -x "$artifact" ]]; then
    echo "[madaros-v2-s4] FAIL: missing current Madaros artifact: $artifact" >&2
    return 1
  fi
  if ! receipt_ok "$artifact"; then
    echo "[madaros-v2-s4] proving current artifact with madaros_full_gate.sh before S4 gate"
    MADAROS_RAW_BIN="$artifact" bash "${ROOT_DIR}/scripts/ci/madaros_full_gate.sh" >/dev/null
  fi
  export MADAROS_RAW_BIN="$artifact"
}

ensure_s4_raw_artifact

mkdir -p "$OUT_DIR"

echo "[madaros-v2-s4] START"
echo "[madaros-v2-s4] out=$OUT_DIR"
echo "[madaros-v2-s4] compiler=$COMPILER"

run_case() {
  local case_id="$1"
  local source="$2"
  local min_accepted="$3"
  local min_rejected="$4"
  local min_blocked="$5"
  local max_accepted="$6"
  local max_rejected="$7"
  local max_blocked="$8"
  local required_accepted="$9"
  local required_rejected="${10}"
  local required_blocked="${11}"
  local a_dir="$OUT_DIR/$case_id/a"
  local b_dir="$OUT_DIR/$case_id/b"
  mkdir -p "$a_dir" "$b_dir"

  echo "[madaros-v2-s4] case=$case_id source=$source"
  "$PY" emit --source "$source" --out-dir "$a_dir" --case-id "$case_id" --compiler "$COMPILER"
  "$PY" emit --source "$source" --out-dir "$b_dir" --case-id "$case_id" --compiler "$COMPILER"

  cmp "$a_dir/$case_id.s4.receipt.json" "$b_dir/$case_id.s4.receipt.json" >/dev/null
  cmp "$a_dir/$case_id.s4.egraph.json" "$b_dir/$case_id.s4.egraph.json" >/dev/null
  cmp "$a_dir/$case_id.s4.rewrites.json" "$b_dir/$case_id.s4.rewrites.json" >/dev/null
  cmp "$a_dir/$case_id.s4.extraction.json" "$b_dir/$case_id.s4.extraction.json" >/dev/null

  python3 - "$a_dir/$case_id.s4.receipt.json" "$a_dir/$case_id.s4.egraph.json" "$a_dir/$case_id.s4.rewrites.json" "$a_dir/$case_id.s4.extraction.json" "$min_accepted" "$min_rejected" "$min_blocked" "$max_accepted" "$max_rejected" "$max_blocked" "$required_accepted" "$required_rejected" "$required_blocked" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

IDENTITY_KINDS = {
    "add_zero_rhs": {"neutral_side": "rhs", "neutral_const": ["int", 0]},
    "add_zero_lhs": {"neutral_side": "lhs", "neutral_const": ["int", 0]},
    "mul_one_rhs": {"neutral_side": "rhs", "neutral_const": ["int", 1]},
    "mul_one_lhs": {"neutral_side": "lhs", "neutral_const": ["int", 1]},
    "sub_zero_rhs": {"neutral_side": "rhs", "neutral_const": ["int", 0]},
}
REFLEXIVE_CMP_KINDS = {
    "eq_self_true": ["bool", True],
    "ne_self_false": ["bool", False],
    "le_self_true": ["bool", True],
    "ge_self_true": ["bool", True],
    "lt_self_false": ["bool", False],
    "gt_self_false": ["bool", False],
}
SUB_SELF_KINDS = {
    "sub_self_zero": ["int", 0],
}

def stable_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def value_ref_hash(value_id, producer):
    payload = {
        "op": "value_ref",
        "value_id": int(value_id),
        "producer_kind": producer.get("producer_kind", "unknown"),
        "producer_label": producer.get("producer_label", producer.get("label", "")),
    }
    return sha256_text(stable_json(payload))

def const_hash(kind, value):
    payload = {
        "op": "const",
        "constant": {
            "kind": kind,
            "int_val": int(value) if kind == "int" else 0,
            "bool_val": bool(value) if kind == "bool" else False,
        },
    }
    return sha256_text(stable_json(payload))

def assert_symbolic_producer_policy(rewrite, family):
    eval_policy = rewrite.get("producer_evaluation_policy")
    producer = rewrite.get("symbolic_producer", {})
    if producer.get("producer_kind") in {"param", "block_param"}:
        if eval_policy != "producer_is_param_or_block_param_no_effectful_eval":
            raise SystemExit(f"{family} param/block_param evaluation policy mismatch")
    elif producer.get("producer_kind") == "call_direct":
        if eval_policy != "direct_call_leaf_pure_keep_producer_evaluated":
            raise SystemExit(f"{family} call evaluation policy mismatch")
        if producer.get("call_leaf_pure") is not True:
            raise SystemExit(f"{family} accepted call must be local leaf pure")
        summary = producer.get("call_summary", {})
        if summary.get("purity_reason") != "local_leaf_no_call_direct":
            raise SystemExit(f"{family} accepted call must prove local leaf purity")
    else:
        raise SystemExit(f"{family} accepted producer must be param/block_param or local leaf call_direct")
    if not producer.get("producer_label") and not producer.get("label"):
        raise SystemExit(f"{family} missing stable producer label")
    return producer

receipt_path, egraph_path, rewrites_path, extraction_path, min_accepted, min_rejected, min_blocked, max_accepted, max_rejected, max_blocked, required_accepted, required_rejected, required_blocked = sys.argv[1:14]
receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
egraph = json.loads(Path(egraph_path).read_text(encoding="utf-8"))
rewrites = json.loads(Path(rewrites_path).read_text(encoding="utf-8"))
extraction = json.loads(Path(extraction_path).read_text(encoding="utf-8"))
if receipt["schema"] != "madaros.v2.s4.receipt/0.1":
    raise SystemExit("bad S4 receipt schema")
if receipt["egraph_schema"] != "madaros.v2.s4.egraph/0.1":
    raise SystemExit("bad egraph schema")
if receipt["rewrite_schema"] != "madaros.v2.ekan.rewrite/0.1":
    raise SystemExit("bad rewrite schema")
if receipt["extraction_schema"] != "madaros.v2.s4.extraction/0.1":
    raise SystemExit("bad extraction schema in S4 receipt")
if receipt["s4_boundary_complete"] is not True:
    raise SystemExit("S4 boundary receipt must be complete")
if receipt.get("s4_extraction_boundary_complete") is not True:
    raise SystemExit("S4 extraction boundary receipt must be complete")
if receipt["s4_complete"] is not False:
    raise SystemExit("S4 gate must not claim global S4 completion")
if receipt.get("s4_full_complete") is not False:
    raise SystemExit("S4 gate must not claim S4 FULL completion")
if receipt.get("stage_contract_level") != "S4_BOUNDARY_NOT_FULL":
    raise SystemExit("S4 receipt must classify this as a boundary, not FULL S4")
if receipt.get("s_full_contract") != "blocked_until_full_s4_obligations_are_gated":
    raise SystemExit("S4 receipt missing S-FULL blocked contract")
remaining = receipt.get("s4_remaining", [])
required_remaining = {
    "multi-rule equality saturation",
    "learned or approximate E-KAN proposals with declared domains and fallback expressions",
    "broad counterexample search over accepted and tempting sibling rewrites",
    "producer purity and evaluation-preservation beyond the current local leaf subset",
    "downstream optimizer integration beyond receipt-only extraction",
    "full-domain translation validation for every selected rewrite family",
}
if not required_remaining.issubset(set(remaining)):
    raise SystemExit("S4 receipt must list missing FULL obligations")
egraph_for_hash = dict(egraph)
egraph_for_hash.pop("egraph_sha256", None)
egraph_canonical = json.dumps(egraph_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
if hashlib.sha256(egraph_canonical.encode()).hexdigest() != receipt["egraph_sha256"]:
    raise SystemExit("egraph hash mismatch")
extraction_for_hash = dict(extraction)
extraction_for_hash.pop("extraction_sha256", None)
extraction_canonical = json.dumps(extraction_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
if hashlib.sha256(extraction_canonical.encode()).hexdigest() != receipt["extraction_sha256"]:
    raise SystemExit("extraction hash mismatch")
if extraction["schema"] != "madaros.v2.s4.extraction/0.1":
    raise SystemExit("bad extraction schema")
if extraction["input_egraph_sha256"] != receipt["egraph_sha256"]:
    raise SystemExit("extraction/egraph hash mismatch")
if extraction["input_rewrites_sha256"] != receipt["input_rewrites_sha256"]:
    raise SystemExit("extraction/rewrites hash mismatch")
if extraction["input_rewrite_count"] != len(rewrites):
    raise SystemExit("extraction rewrite input count mismatch")
rewrites_canonical = json.dumps(rewrites, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
if hashlib.sha256(rewrites_canonical.encode()).hexdigest() != extraction["input_rewrites_sha256"]:
    raise SystemExit("rewrites canonical hash mismatch")
if extraction["ir_mutation_allowed"] is not False:
    raise SystemExit("S4 extraction boundary must be non-mutating")
if extraction["mutation_plan"] != "none: receipt-only extractor":
    raise SystemExit("unexpected S4 mutation plan")
if extraction.get("deterministic_extraction") is not True:
    raise SystemExit("extraction must be deterministic")
if extraction.get("s4_extraction_boundary_complete") is not True:
    raise SystemExit("extraction boundary must be complete")
if extraction.get("s4_extraction_complete") is not False:
    raise SystemExit("extraction must not claim global S4 completion")
if not extraction.get("cost_model_sha256") or not extraction.get("cost_model_config_sha256"):
    raise SystemExit("missing extraction cost model hash")
if extraction["cost_model_sha256"] != extraction["cost_model_config_sha256"]:
    raise SystemExit("cost model/config hash mismatch")
if receipt["cost_model_sha256"] != extraction["cost_model_sha256"]:
    raise SystemExit("receipt/extraction cost model hash mismatch")
for invariant in [
    "deterministic_double_emit",
    "selected_ids_equal_accepted_ids",
    "rejected_ids_blocked_from_extraction",
    "accepted_translation_validation_zero_error",
    "receipt_only_no_ir_mutation",
]:
    if invariant not in extraction.get("gate_invariants", []):
        raise SystemExit(f"missing extraction invariant: {invariant}")
if receipt["rewrite_count"] != len(rewrites):
    raise SystemExit("rewrite count mismatch")
accepted = [rewrite for rewrite in rewrites if rewrite.get("accepted") is True]
blocked = [rewrite for rewrite in rewrites if rewrite.get("blocked") is True]
rejected = [rewrite for rewrite in rewrites if rewrite.get("accepted") is False and rewrite.get("blocked") is not True]
if receipt["accepted_rewrite_count"] != len(accepted):
    raise SystemExit("accepted rewrite count mismatch")
if receipt["rejected_rewrite_count"] != len(rejected):
    raise SystemExit("rejected rewrite count mismatch")
if receipt["blocked_rewrite_count"] != len(blocked):
    raise SystemExit("blocked rewrite count mismatch")
if receipt["rewrite_count"] != receipt["accepted_rewrite_count"] + receipt["rejected_rewrite_count"] + receipt["blocked_rewrite_count"]:
    raise SystemExit("rewrite total must equal accepted + rejected + blocked")
if receipt["accepted_rewrite_count"] < int(min_accepted):
    raise SystemExit(f"too few accepted rewrites: {receipt['accepted_rewrite_count']} < {min_accepted}")
if receipt["rejected_rewrite_count"] < int(min_rejected):
    raise SystemExit(f"too few rejected rewrites: {receipt['rejected_rewrite_count']} < {min_rejected}")
if receipt["blocked_rewrite_count"] < int(min_blocked):
    raise SystemExit(f"too few blocked rewrites: {receipt['blocked_rewrite_count']} < {min_blocked}")
if max_accepted != "-" and receipt["accepted_rewrite_count"] > int(max_accepted):
    raise SystemExit(f"too many accepted rewrites: {receipt['accepted_rewrite_count']} > {max_accepted}")
if max_rejected != "-" and receipt["rejected_rewrite_count"] > int(max_rejected):
    raise SystemExit(f"too many rejected rewrites: {receipt['rejected_rewrite_count']} > {max_rejected}")
if max_blocked != "-" and receipt["blocked_rewrite_count"] > int(max_blocked):
    raise SystemExit(f"too many blocked rewrites: {receipt['blocked_rewrite_count']} > {max_blocked}")
accepted_ids = set(receipt.get("accepted_rewrite_ids", []))
rejected_ids = set(receipt.get("rejected_rewrite_ids", []))
blocked_ids = set(receipt.get("blocked_rewrite_ids", []))
if accepted_ids & rejected_ids or accepted_ids & blocked_ids or rejected_ids & blocked_ids:
    raise SystemExit("rewrite ids cannot appear in multiple status buckets")
if set(extraction["selected_rewrite_ids"]) != accepted_ids:
    raise SystemExit("extraction selected ids must equal accepted rewrite ids")
if set(extraction["rejected_rewrite_ids"]) != rejected_ids:
    raise SystemExit("extraction rejected ids must equal rejected rewrite ids")
if set(extraction.get("blocked_rewrite_ids", [])) != blocked_ids:
    raise SystemExit("extraction blocked ids must equal blocked rewrite ids")
if receipt["selected_rewrite_count"] != len(accepted_ids):
    raise SystemExit("receipt selected rewrite count mismatch")
if receipt["rejected_from_extraction_count"] != len(rejected_ids):
    raise SystemExit("receipt rejected-from-extraction count mismatch")
if receipt["blocked_from_extraction_count"] != len(blocked_ids):
    raise SystemExit("receipt blocked-from-extraction count mismatch")
observed_accepted = set()
observed_rejected = set()
observed_blocked = set()
accepted_reflexive_cmp_kinds = set()
rejected_reflexive_cmp_kinds = set()
blocked_reflexive_cmp_kinds = set()
for rewrite in rewrites:
    if rewrite["schema_version"] != "madaros.v2.ekan.rewrite/0.1":
        raise SystemExit("bad rewrite receipt schema")
    for required in [
        "eclass_id",
        "proposed_enode_sha256",
        "proposal_kind",
        "proposal_origin",
        "proposal_config_sha256",
        "training_or_provenance_sha256",
        "domain_bounds",
        "error_bound_method",
        "validator_attempted",
    ]:
        if required not in rewrite:
            raise SystemExit(f"rewrite missing required S4 provenance field: {required}")
    if not rewrite["exact_fallback_expr_sha256"]:
        raise SystemExit("missing exact fallback hash")
    if not rewrite["coefficient_sha256"]:
        raise SystemExit("missing E-KAN coefficient receipt")
    if rewrite["accepted"] is True:
        if rewrite["validator"] != "translation-validation":
            raise SystemExit("accepted rewrite missing translation-validation")
        if rewrite["error_bound"] != "0":
            raise SystemExit("accepted exact rewrite must have zero error bound")
        if rewrite["basis_family"] != "exact_symbolic":
            raise SystemExit("accepted rewrite must use exact symbolic basis")
        if rewrite.get("selected_for_extraction") is not True:
            raise SystemExit("accepted rewrite must be selected for extraction")
        if rewrite.get("ir_mutation_allowed") is not False:
            raise SystemExit("S4 receipt lane must remain non-mutating")
        if rewrite["rewrite_kind"] == "symbolic_identity_i64":
            identity_kind = rewrite.get("identity_kind")
            if identity_kind not in IDENTITY_KINDS:
                raise SystemExit(f"unexpected symbolic identity kind: {identity_kind}")
            expected = IDENTITY_KINDS[identity_kind]
            if rewrite.get("proposal_kind") != "exact_symbolic_identity":
                raise SystemExit("symbolic identity must use exact_symbolic_identity proposal")
            if rewrite.get("ekan_receipt_kind") != "ekan_exact_symbolic_identity":
                raise SystemExit("symbolic identity must use E-KAN exact identity receipt")
            if rewrite.get("neutral_side") != expected["neutral_side"]:
                raise SystemExit("symbolic identity neutral side mismatch")
            if list(rewrite.get("neutral_const", [])) != expected["neutral_const"]:
                raise SystemExit("symbolic identity neutral const mismatch")
            symbolic_value = int(rewrite.get("symbolic_value", -1))
            if rewrite["neutral_side"] == "rhs" and symbolic_value != int(rewrite.get("original_lhs", -2)):
                raise SystemExit("symbolic identity rhs-neutral value must be original lhs")
            if rewrite["neutral_side"] == "lhs" and symbolic_value != int(rewrite.get("original_rhs", -2)):
                raise SystemExit("symbolic identity lhs-neutral value must be original rhs")
            producer = rewrite.get("symbolic_producer", {})
            if producer.get("producer_kind") == "const":
                raise SystemExit("symbolic identity must not duplicate constant-fold path")
            if not producer.get("producer_label") and not producer.get("label"):
                raise SystemExit("symbolic identity missing stable producer label")
            if rewrite.get("domain") != "all-i64-values-with-neutral-element":
                raise SystemExit("symbolic identity domain mismatch")
            bounds = rewrite.get("domain_bounds", {})
            if bounds.get("kind") != "all-i64-values-with-neutral-element":
                raise SystemExit("symbolic identity domain bounds mismatch")
            if "neutral-element-proof" not in rewrite.get("validator_attempted", []):
                raise SystemExit("symbolic identity missing neutral-element proof marker")
            expected_hash = value_ref_hash(symbolic_value, producer)
            if rewrite.get("proposed_enode_sha256") != expected_hash:
                raise SystemExit("symbolic identity proposed enode is not value_ref(symbolic_value)")
            if rewrite.get("rewritten_enode_sha256") != expected_hash:
                raise SystemExit("symbolic identity rewritten enode is not value_ref(symbolic_value)")
        if rewrite["rewrite_kind"] == "symbolic_reflexive_cmp_i64":
            comparison_kind = rewrite.get("comparison_kind")
            if comparison_kind not in REFLEXIVE_CMP_KINDS:
                raise SystemExit(f"unexpected reflexive comparison kind: {comparison_kind}")
            if rewrite.get("proposal_kind") != "exact_symbolic_reflexive_comparison":
                raise SystemExit("reflexive comparison must use exact symbolic proposal")
            if rewrite.get("ekan_receipt_kind") != "ekan_exact_symbolic_predicate":
                raise SystemExit("reflexive comparison must use exact predicate E-KAN receipt")
            symbolic_value = int(rewrite.get("symbolic_value", -1))
            if symbolic_value != int(rewrite.get("original_lhs", -2)):
                raise SystemExit("reflexive comparison symbolic value must equal original lhs")
            if symbolic_value != int(rewrite.get("original_rhs", -3)):
                raise SystemExit("reflexive comparison symbolic value must equal original rhs")
            if rewrite.get("same_operand_id") is not True:
                raise SystemExit("reflexive comparison must assert same_operand_id")
            assert_symbolic_producer_policy(rewrite, "reflexive comparison")
            if rewrite.get("domain") != "all-i64-values-with-reflexive-equality-and-order":
                raise SystemExit("reflexive comparison domain mismatch")
            bounds = rewrite.get("domain_bounds", {})
            if bounds.get("kind") != "all-i64-values-with-reflexive-equality-and-order":
                raise SystemExit("reflexive comparison domain bounds mismatch")
            if "reflexive-comparison-proof" not in rewrite.get("validator_attempted", []):
                raise SystemExit("reflexive comparison missing proof marker")
            if "producer-evaluation-preservation-proof" not in rewrite.get("validator_attempted", []):
                raise SystemExit("reflexive comparison missing producer evaluation proof marker")
            expected_const = REFLEXIVE_CMP_KINDS[comparison_kind]
            if list(rewrite.get("result_const", [])) != expected_const:
                raise SystemExit("reflexive comparison result const mismatch")
            expected_hash = const_hash(expected_const[0], expected_const[1])
            if rewrite.get("proposed_enode_sha256") != expected_hash:
                raise SystemExit("reflexive comparison proposed enode is not expected bool const")
            if rewrite.get("rewritten_enode_sha256") != expected_hash:
                raise SystemExit("reflexive comparison rewritten enode is not expected bool const")
            accepted_reflexive_cmp_kinds.add(comparison_kind)
        if rewrite["rewrite_kind"] == "symbolic_sub_self_i64":
            subtraction_kind = rewrite.get("subtraction_kind")
            if subtraction_kind not in SUB_SELF_KINDS:
                raise SystemExit(f"unexpected sub-self kind: {subtraction_kind}")
            if rewrite.get("proposal_kind") != "exact_symbolic_sub_self":
                raise SystemExit("sub-self must use exact symbolic proposal")
            if rewrite.get("ekan_receipt_kind") != "ekan_exact_symbolic_arithmetic":
                raise SystemExit("sub-self must use exact arithmetic E-KAN receipt")
            symbolic_value = int(rewrite.get("symbolic_value", -1))
            if symbolic_value != int(rewrite.get("original_lhs", -2)):
                raise SystemExit("sub-self symbolic value must equal original lhs")
            if symbolic_value != int(rewrite.get("original_rhs", -3)):
                raise SystemExit("sub-self symbolic value must equal original rhs")
            if rewrite.get("same_operand_id") is not True:
                raise SystemExit("sub-self must assert same_operand_id")
            assert_symbolic_producer_policy(rewrite, "sub-self")
            if rewrite.get("domain") != "all-i64-values-with-same-ssa-subtraction":
                raise SystemExit("sub-self domain mismatch")
            bounds = rewrite.get("domain_bounds", {})
            if bounds.get("kind") != "all-i64-values-with-same-ssa-subtraction":
                raise SystemExit("sub-self domain bounds mismatch")
            if "same-ssa-subtraction-proof" not in rewrite.get("validator_attempted", []):
                raise SystemExit("sub-self missing same-SSA subtraction proof marker")
            if "producer-evaluation-preservation-proof" not in rewrite.get("validator_attempted", []):
                raise SystemExit("sub-self missing producer evaluation proof marker")
            expected_const = SUB_SELF_KINDS[subtraction_kind]
            if list(rewrite.get("result_const", [])) != expected_const:
                raise SystemExit("sub-self result const mismatch")
            expected_hash = const_hash(expected_const[0], expected_const[1])
            if rewrite.get("proposed_enode_sha256") != expected_hash:
                raise SystemExit("sub-self proposed enode is not expected int zero")
            if rewrite.get("rewritten_enode_sha256") != expected_hash:
                raise SystemExit("sub-self rewritten enode is not expected int zero")
        observed_accepted.add(rewrite["rewrite_kind"])
        observed_accepted.add(rewrite["ekan_receipt_kind"])
    elif rewrite.get("blocked") is True:
        if rewrite["validator"] != "blocked":
            raise SystemExit("blocked rewrite must use validator=blocked")
        if rewrite.get("selected_for_extraction") is not False:
            raise SystemExit("blocked rewrite must not be selected for extraction")
        if rewrite.get("ir_mutation_allowed") is not False:
            raise SystemExit("blocked rewrite must not allow IR mutation")
        if rewrite.get("rejection_reason_code") not in {"operand_provenance_ambiguous", "producer_evaluation_not_proven"}:
            raise SystemExit("blocked rewrite missing accepted blocker reason")
        if rewrite.get("rejection_reason_code") == "producer_evaluation_not_proven":
            if rewrite.get("rewrite_kind") not in {"symbolic_reflexive_cmp_i64", "symbolic_sub_self_i64"}:
                raise SystemExit("producer evaluation blocker must target an evaluation-sensitive symbolic rewrite")
            if rewrite.get("ekan_receipt_kind") != "ekan_blocked_producer_evaluation":
                raise SystemExit("producer evaluation blocker missing E-KAN blocked receipt kind")
            if rewrite.get("producer_evaluation_policy") != "blocked: producer evaluation is not proven":
                raise SystemExit("producer evaluation blocker policy mismatch")
            if rewrite.get("same_operand_id") is not True:
                raise SystemExit("producer evaluation blocker must assert same_operand_id")
            if rewrite.get("rewrite_kind") == "symbolic_reflexive_cmp_i64":
                comparison_kind = rewrite.get("comparison_kind")
                if comparison_kind not in REFLEXIVE_CMP_KINDS:
                    raise SystemExit(f"unexpected blocked reflexive comparison kind: {comparison_kind}")
                blocked_reflexive_cmp_kinds.add(comparison_kind)
        observed_blocked.add(rewrite["rewrite_kind"])
        observed_blocked.add(rewrite["ekan_receipt_kind"])
        observed_blocked.add(rewrite["rejection_reason_code"])
    else:
        if rewrite["validator"] != "rejected":
            raise SystemExit("rejected rewrite must use validator=rejected")
        if rewrite.get("selected_for_extraction") is not False:
            raise SystemExit("rejected rewrite must not be selected for extraction")
        if rewrite.get("ir_mutation_allowed") is not False:
            raise SystemExit("rejected rewrite must not allow IR mutation")
        if not rewrite.get("rejection_reason_code") or not rewrite.get("rejection_reason"):
            raise SystemExit("rejected rewrite missing rejection reason")
        if rewrite.get("counterexample_count", 0) <= 0:
            raise SystemExit("semantic rejected rewrite must carry a counterexample")
        if not rewrite.get("counterexample_set_sha256") or not rewrite.get("counterexample_sha256"):
            raise SystemExit("rejected rewrite missing counterexample hashes")
        if rewrite.get("rewrite_kind") == "symbolic_reflexive_cmp_i64":
            comparison_kind = rewrite.get("comparison_kind")
            if comparison_kind not in REFLEXIVE_CMP_KINDS:
                raise SystemExit(f"unexpected rejected reflexive comparison kind: {comparison_kind}")
            rejected_reflexive_cmp_kinds.add(comparison_kind)
        observed_rejected.add(rewrite["rewrite_kind"])
        observed_rejected.add(rewrite["ekan_receipt_kind"])
        observed_rejected.add(rewrite["rejection_reason_code"])

decision_ids = set()
for decision in extraction["decisions"]:
    rid = decision["rewrite_id"]
    decision_ids.add(rid)
    for required in [
        "eclass_id",
        "original_enode_sha256",
        "proposed_enode_sha256",
        "selected_enode_sha256",
        "validator_log_sha256",
        "cost_model_sha256",
        "cost_model_config_sha256",
        "cost_before",
        "cost_after",
        "cost_delta",
        "cost_components",
        "proof_obligation",
        "selection_reason",
        "extraction_applied_to_ir",
    ]:
        if required not in decision:
            raise SystemExit(f"extraction decision missing {required}")
    if decision["cost_model_sha256"] != extraction["cost_model_sha256"]:
        raise SystemExit("decision cost model hash mismatch")
    if decision["cost_model_config_sha256"] != extraction["cost_model_config_sha256"]:
        raise SystemExit("decision cost model config hash mismatch")
    if decision.get("ir_mutation_allowed") is not False:
        raise SystemExit("extraction decision must be non-mutating")
    if decision.get("extraction_applied_to_ir") is not False:
        raise SystemExit("S4 boundary extraction must not apply to IR")
    if rid in accepted_ids:
        if decision["decision"] != "select" or decision["selected"] is not True:
            raise SystemExit("accepted rewrite must be selected by extractor")
        if decision["selected_enode_sha256"] != decision["proposed_enode_sha256"]:
            raise SystemExit("accepted extraction must select proposed enode")
        if decision["cost_after"] > decision["cost_before"]:
            raise SystemExit("accepted extraction cannot increase cost")
        if "translation-validation" not in decision["proof_obligation"]:
            raise SystemExit("accepted extraction missing validation proof obligation")
        if decision.get("basis_family") != "exact_symbolic":
            raise SystemExit("accepted extraction must use exact symbolic basis")
        if decision.get("error_bound") != "0":
            raise SystemExit("accepted extraction must carry zero error bound")
        if decision.get("mir_abi_safe") is not True:
            raise SystemExit("accepted extraction must be MIR/ABI safe")
        if decision.get("rewrite_kind") == "symbolic_identity_i64":
            if decision.get("lowering_effect") != "replace_binary_identity_expr_with_existing_value":
                raise SystemExit("symbolic identity extraction has wrong lowering effect")
            if decision.get("abi_impact") != "none":
                raise SystemExit("symbolic identity extraction must have no ABI impact")
        if decision.get("rewrite_kind") == "symbolic_reflexive_cmp_i64":
            if decision.get("lowering_effect") not in {
                "replace_binary_predicate_expr_with_const_bool",
                "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated",
            }:
                raise SystemExit("reflexive comparison extraction has wrong lowering effect")
            if decision.get("abi_impact") != "none":
                raise SystemExit("reflexive comparison extraction must have no ABI impact")
        if decision.get("rewrite_kind") == "symbolic_sub_self_i64":
            if decision.get("lowering_effect") not in {
                "replace_binary_sub_self_expr_with_const_i64_zero",
                "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated",
            }:
                raise SystemExit("sub-self extraction has wrong lowering effect")
            if decision.get("abi_impact") != "none":
                raise SystemExit("sub-self extraction must have no ABI impact")
    elif rid in rejected_ids:
        if decision["decision"] != "reject" or decision["selected"] is not False:
            raise SystemExit("rejected rewrite must not be selected by extractor")
        if decision["selected_enode_sha256"] != decision["original_enode_sha256"]:
            raise SystemExit("rejected extraction must keep original enode")
        if not decision.get("counterexample_set_sha256"):
            raise SystemExit("rejected extraction missing counterexample set hash")
        if decision.get("rejection_reason_code") != "counterexample_found":
            raise SystemExit("rejected extraction missing counterexample reason code")
    elif rid in blocked_ids:
        if decision["decision"] != "block" or decision["selected"] is not False:
            raise SystemExit("blocked rewrite must not be selected by extractor")
        if decision["selected_enode_sha256"] != decision["original_enode_sha256"]:
            raise SystemExit("blocked extraction must keep original enode")
        if decision.get("rejection_reason_code") not in {"operand_provenance_ambiguous", "producer_evaluation_not_proven"}:
            raise SystemExit("blocked extraction missing accepted reason code")
        if decision.get("rejection_reason_code") == "operand_provenance_ambiguous" and "operand provenance" not in decision.get("proof_obligation", ""):
            raise SystemExit("blocked extraction missing provenance proof obligation")
        if decision.get("rejection_reason_code") == "producer_evaluation_not_proven" and "producer evaluation" not in decision.get("proof_obligation", ""):
            raise SystemExit("blocked extraction missing producer evaluation proof obligation")
    else:
        raise SystemExit(f"extraction decision references unknown rewrite id: {rid}")
if decision_ids != accepted_ids | rejected_ids | blocked_ids:
    raise SystemExit("extraction decisions must cover every rewrite exactly once")

missing_accepted = [item for item in required_accepted.split(",") if item and item != "-" and item not in observed_accepted]
if missing_accepted:
    raise SystemExit(f"missing required accepted markers: {missing_accepted}; observed={sorted(observed_accepted)}")
missing_rejected = [item for item in required_rejected.split(",") if item and item != "-" and item not in observed_rejected]
if missing_rejected:
    raise SystemExit(f"missing required rejected markers: {missing_rejected}; observed={sorted(observed_rejected)}")
missing_blocked = [item for item in required_blocked.split(",") if item and item != "-" and item not in observed_blocked]
if missing_blocked:
    raise SystemExit(f"missing required blocked markers: {missing_blocked}; observed={sorted(observed_blocked)}")
required_reflexive_cmp_kinds = set(REFLEXIVE_CMP_KINDS)
case_id = receipt.get("case_id")
if case_id in {"symbolic_reflexive_cmp_i64", "symbolic_reflexive_cmp_pure_call_i64"}:
    if accepted_reflexive_cmp_kinds != required_reflexive_cmp_kinds:
        raise SystemExit(
            f"{case_id} must accept the complete reflexive comparison matrix; "
            f"observed={sorted(accepted_reflexive_cmp_kinds)}"
        )
if case_id == "reject_distinct_symbolic_cmp_i64":
    if rejected_reflexive_cmp_kinds != required_reflexive_cmp_kinds:
        raise SystemExit(
            f"{case_id} must reject the complete distinct comparison matrix; "
            f"observed={sorted(rejected_reflexive_cmp_kinds)}"
        )
if case_id == "reject_call_result_self_cmp_i64":
    if blocked_reflexive_cmp_kinds != required_reflexive_cmp_kinds:
        raise SystemExit(
            f"{case_id} must block the complete effectful-call comparison matrix; "
            f"observed={sorted(blocked_reflexive_cmp_kinds)}"
        )
print(
    f"[madaros-v2-s4] ok receipt={Path(receipt_path).name} "
    f"accepted={receipt['accepted_rewrite_count']} rejected={receipt['rejected_rewrite_count']} blocked={receipt['blocked_rewrite_count']} "
    f"selected={receipt['selected_rewrite_count']} egraph_sha={receipt['egraph_sha256'][:12]} "
    f"extraction_sha={receipt['extraction_sha256'][:12]}"
)
PY
}

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r case_id source min_accepted min_rejected min_blocked max_accepted max_rejected max_blocked required_accepted required_rejected required_blocked; do
  run_case "$case_id" "$source" "$min_accepted" "$min_rejected" "$min_blocked" "$max_accepted" "$max_rejected" "$max_blocked" "$required_accepted" "$required_rejected" "$required_blocked"
done

python3 - "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
receipts = []
for path in sorted(out.glob("*/*/*.s4.receipt.json")):
    if "/b/" in path.as_posix():
        continue
    receipts.append(json.loads(path.read_text(encoding="utf-8")))
if not receipts:
    raise SystemExit("no S4 receipts")

allowed_to_s5_rewrites = {
    "constant_fold_i64",
    "symbolic_identity_i64",
    "symbolic_reflexive_cmp_i64",
    "symbolic_sub_self_i64",
}
allowed_lowering_effects = {
    "replace_binary_constant_expr_with_const",
    "replace_binary_identity_expr_with_existing_value",
    "replace_binary_predicate_expr_with_const_bool",
    "replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated",
    "replace_binary_sub_self_expr_with_const_i64_zero",
    "replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated",
}
application_cases = []
selected_application_ids = []
rejected_application_ids = []
blocked_application_ids = []
for receipt in receipts:
    case_dir = out / receipt["case_id"] / "a"
    rewrites_path = case_dir / receipt["rewrites_path"]
    extraction_path = case_dir / receipt["extraction_path"]
    if not rewrites_path.is_file() or not extraction_path.is_file():
        raise SystemExit(f"missing S4 case artifacts for application plan: {receipt['case_id']}")
    rewrites = json.loads(rewrites_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    decisions = {decision["rewrite_id"]: decision for decision in extraction.get("decisions", [])}
    selected_actions = []
    rejected_actions = []
    blocked_actions = []
    for rewrite in rewrites:
        rid = rewrite["proposed_rewrite_id"]
        decision = decisions.get(rid)
        if decision is None:
            raise SystemExit(f"missing extraction decision in application plan for {rid}")
        common = {
            "rewrite_id": rid,
            "rewrite_kind": rewrite["rewrite_kind"],
            "proposal_kind": rewrite["proposal_kind"],
            "eclass_id": rewrite["eclass_id"],
            "function": rewrite["function"],
            "block": rewrite["block"],
            "instruction_result": rewrite["instruction_result"],
            "original_enode_sha256": rewrite["original_enode_sha256"],
            "proposed_enode_sha256": rewrite["proposed_enode_sha256"],
            "rewritten_enode_sha256": rewrite["rewritten_enode_sha256"],
            "validator": rewrite["validator"],
            "validator_log_sha256": rewrite["validator_log_sha256"],
            "exact_fallback_expr_sha256": rewrite["exact_fallback_expr_sha256"],
            "coefficient_sha256": rewrite["coefficient_sha256"],
            "basis_family": rewrite["basis_family"],
            "domain": rewrite["domain"],
            "domain_bounds": rewrite["domain_bounds"],
            "error_bound": rewrite["error_bound"],
            "decision": decision["decision"],
            "selection_reason": decision["selection_reason"],
            "cost_before": decision["cost_before"],
            "cost_after": decision["cost_after"],
            "cost_delta": decision["cost_delta"],
            "cost_model_sha256": decision["cost_model_sha256"],
            "proof_obligation": decision["proof_obligation"],
            "extraction_applied_to_ir": decision["extraction_applied_to_ir"],
            "ir_mutation_allowed": decision["ir_mutation_allowed"],
        }
        if rewrite.get("accepted") is True:
            if rewrite["rewrite_kind"] not in allowed_to_s5_rewrites:
                raise SystemExit(f"application plan rejects unsupported selected rewrite kind: {rewrite['rewrite_kind']}")
            if decision.get("selected") is not True or decision.get("decision") != "select":
                raise SystemExit(f"accepted rewrite must be selected in S4 application plan: {rid}")
            if decision.get("lowering_effect") not in allowed_lowering_effects:
                raise SystemExit(f"bad S4->S5 lowering effect for {rid}: {decision.get('lowering_effect')}")
            if decision.get("mir_abi_safe") is not True:
                raise SystemExit(f"S4->S5 application plan requires mir_abi_safe for {rid}")
            if decision.get("abi_impact") != "none":
                raise SystemExit(f"S4->S5 application plan rejects ABI impact for {rid}")
            if rewrite.get("validator") != "translation-validation" or rewrite.get("error_bound") != "0":
                raise SystemExit(f"S4->S5 application plan requires exact translation validation for {rid}")
            if rewrite.get("basis_family") != "exact_symbolic":
                raise SystemExit(f"S4->S5 application plan only accepts exact symbolic basis for {rid}")
            keep_producer = str(decision.get("lowering_effect", "")).endswith("_keep_producer_evaluated")
            producer_policy = rewrite.get("producer_evaluation_policy", "not-required-for-this-rewrite")
            if keep_producer and producer_policy != "direct_call_leaf_pure_keep_producer_evaluated":
                raise SystemExit(f"keep-producer selected rewrite lacks policy for {rid}")
            action = {
                **common,
                "action": "apply_to_s5_input",
                "lowering_effect": decision["lowering_effect"],
                "lowering_effect_schema": "madaros.v2.s4.to_s5.lowering_effect/0.1",
                "selected_enode_sha256": decision["selected_enode_sha256"],
                "replacement_enode_sha256": decision["replacement_enode_sha256"],
                "mir_abi_safe": True,
                "abi_impact": "none",
                "call_signature_effect": "none",
                "stack_effect": "none",
                "sret_effect": "none",
                "aggregate_layout_effect": "none",
                "producer_evaluation_preservation": (
                    "required_keep_original_producer_evaluated" if keep_producer else "not-required-for-this-rewrite"
                ),
                "producer_evaluation_policy": producer_policy,
                "selected_for_s5": True,
            }
            selected_actions.append(action)
            selected_application_ids.append(rid)
        elif rewrite.get("blocked") is True:
            action = {
                **common,
                "action": "block_before_s5_input",
                "selected_for_s5": False,
                "rejection_reason_code": rewrite["rejection_reason_code"],
                "rejection_reason": rewrite["rejection_reason"],
                "mir_abi_safe": False,
            }
            blocked_actions.append(action)
            blocked_application_ids.append(rid)
        else:
            action = {
                **common,
                "action": "reject_before_s5_input",
                "selected_for_s5": False,
                "rejection_reason_code": rewrite["rejection_reason_code"],
                "rejection_reason": rewrite["rejection_reason"],
                "counterexample_set_sha256": rewrite["counterexample_set_sha256"],
                "mir_abi_safe": False,
            }
            rejected_actions.append(action)
            rejected_application_ids.append(rid)
    if len(selected_actions) != receipt["selected_rewrite_count"]:
        raise SystemExit(f"S4->S5 selected action count mismatch for {receipt['case_id']}")
    if len(rejected_actions) != receipt["rejected_from_extraction_count"]:
        raise SystemExit(f"S4->S5 rejected action count mismatch for {receipt['case_id']}")
    if len(blocked_actions) != receipt["blocked_from_extraction_count"]:
        raise SystemExit(f"S4->S5 blocked action count mismatch for {receipt['case_id']}")
    application_cases.append({
        "case_id": receipt["case_id"],
        "source": receipt["source"],
        "input_hlir_sha256": receipt["input_hlir_sha256"],
        "egraph_sha256": receipt["egraph_sha256"],
        "extraction_sha256": receipt["extraction_sha256"],
        "selected_action_count": len(selected_actions),
        "rejected_action_count": len(rejected_actions),
        "blocked_action_count": len(blocked_actions),
        "selected_actions": selected_actions,
        "rejected_actions": rejected_actions,
        "blocked_actions": blocked_actions,
    })

application_plan = {
    "schema": "madaros.v2.s4.to_s5_application_plan/0.1",
    "status": "pass",
    "stage_contract_level": "S4_TO_S5_EXACT_APPLICATION_PLAN_NOT_MUTATING",
    "s4_to_s5_application_plan_complete": True,
    "s4_full_complete": False,
    "s5_input_contract_ready": bool(selected_application_ids),
    "input_contract": "madaros.v2.s4.extraction/0.1",
    "output_contract": "madaros.v2.s5.preflight/0.1",
    "mutation_plan": "none: S4 emits a deterministic application plan for S5 consumers",
    "ir_mutation_allowed": False,
    "application_applied_to_ir": False,
    "accepted_application_count": len(selected_application_ids),
    "rejected_application_count": len(rejected_application_ids),
    "blocked_application_count": len(blocked_application_ids),
    "selected_rewrite_ids": selected_application_ids,
    "rejected_rewrite_ids": rejected_application_ids,
    "blocked_rewrite_ids": blocked_application_ids,
    "allowed_rewrite_kinds": sorted(allowed_to_s5_rewrites),
    "allowed_lowering_effects": sorted(allowed_lowering_effects),
    "cross_stage_invariants": [
        "selected_actions_equal_s4_accepted_selected_ids",
        "rejected_and_blocked_actions_are_never_selected_for_s5",
        "every_selected_action_has_exact_fallback_hash",
        "every_selected_action_has_translation_validation_zero_error",
        "every_selected_action_has_mir_abi_safe_true_and_abi_impact_none",
        "keep_producer_actions_carry_producer_evaluation_policy",
        "application_plan_is_non_mutating",
    ],
    "cases": application_cases,
}
application_plan["application_plan_sha256"] = hashlib.sha256(
    json.dumps(application_plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest()
application_plan_path = out / "madaros_v2_s4_to_s5_application_plan.json"
application_plan_path.write_text(json.dumps(application_plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")

applied_cases = []
applied_selected_ids = []
applied_rejected_ids = []
applied_blocked_ids = []
applied_effect_hashes = []
for case in application_plan["cases"]:
    selected_effects = []
    rejected_effects = []
    blocked_effects = []
    for action in case["selected_actions"]:
        rid = action["rewrite_id"]
        if action.get("action") != "apply_to_s5_input":
            raise SystemExit(f"selected applied action has wrong action: {rid}")
        if action.get("selected_for_s5") is not True:
            raise SystemExit(f"selected applied action must be selected_for_s5: {rid}")
        if action.get("mir_abi_safe") is not True or action.get("abi_impact") != "none":
            raise SystemExit(f"selected applied action must be MIR/ABI safe with no ABI impact: {rid}")
        if action.get("extraction_applied_to_ir") is not False or action.get("ir_mutation_allowed") is not False:
            raise SystemExit(f"selected applied action must not mutate compiler IR yet: {rid}")
        effect = {
            "rewrite_id": rid,
            "case_id": case["case_id"],
            "source": case["source"],
            "pre_mutation_hlir_sha256": case["input_hlir_sha256"],
            "pre_mutation_egraph_sha256": case["egraph_sha256"],
            "function": action["function"],
            "block": action["block"],
            "instruction_result": action["instruction_result"],
            "rewrite_kind": action["rewrite_kind"],
            "proposal_kind": action["proposal_kind"],
            "lowering_effect": action["lowering_effect"],
            "input_enode_sha256": action["original_enode_sha256"],
            "output_enode_sha256": action["rewritten_enode_sha256"],
            "selected_enode_sha256": action["selected_enode_sha256"],
            "replacement_enode_sha256": action["replacement_enode_sha256"],
            "exact_fallback_expr_sha256": action["exact_fallback_expr_sha256"],
            "validator_log_sha256": action["validator_log_sha256"],
            "coefficient_sha256": action["coefficient_sha256"],
            "basis_family": action["basis_family"],
            "error_bound": action["error_bound"],
            "domain": action["domain"],
            "domain_bounds": action["domain_bounds"],
            "cost_before": action["cost_before"],
            "cost_after": action["cost_after"],
            "cost_delta": action["cost_delta"],
            "producer_evaluation_preservation": action["producer_evaluation_preservation"],
            "producer_evaluation_policy": action["producer_evaluation_policy"],
            "mir_abi_safe": True,
            "abi_impact": "none",
            "call_signature_effect": action["call_signature_effect"],
            "stack_effect": action["stack_effect"],
            "sret_effect": action["sret_effect"],
            "aggregate_layout_effect": action["aggregate_layout_effect"],
            "s5_input_materialization": "applied_s4_exact_rewrite_effect",
            "application_applied_to_s5_input": True,
            "application_applied_to_compiler_ir": False,
            "ir_mutation_allowed": False,
        }
        effect["post_apply_selected_enode_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "schema": "madaros.v2.s4.post_apply_selected_enode/0.1",
                    "rewrite_id": rid,
                    "input_hlir_sha256": case["input_hlir_sha256"],
                    "input_egraph_sha256": case["egraph_sha256"],
                    "output_enode_sha256": effect["output_enode_sha256"],
                    "lowering_effect": effect["lowering_effect"],
                    "application_applied_to_s5_input": True,
                    "application_applied_to_compiler_ir": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        effect["post_mutation_hlir_sha256"] = effect["post_apply_selected_enode_sha256"]
        effect["post_mutation_egraph_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "schema": "madaros.v2.s4.post_apply_egraph/0.1",
                    "rewrite_id": rid,
                    "input_egraph_sha256": case["egraph_sha256"],
                    "post_apply_selected_enode_sha256": effect["post_apply_selected_enode_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        effect["applied_effect_sha256"] = hashlib.sha256(
            json.dumps(effect, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        selected_effects.append(effect)
        applied_selected_ids.append(rid)
        applied_effect_hashes.append(effect["applied_effect_sha256"])
    for action in case["rejected_actions"]:
        rid = action["rewrite_id"]
        if action.get("action") != "reject_before_s5_input" or action.get("selected_for_s5") is not False:
            raise SystemExit(f"rejected applied action must stay out of S5 input: {rid}")
        if not action.get("counterexample_set_sha256"):
            raise SystemExit(f"rejected applied action must carry counterexample set hash: {rid}")
        rejected_effects.append({
            "rewrite_id": rid,
            "case_id": case["case_id"],
            "source": case["source"],
            "rewrite_kind": action["rewrite_kind"],
            "proposal_kind": action["proposal_kind"],
            "rejection_reason_code": action["rejection_reason_code"],
            "counterexample_set_sha256": action["counterexample_set_sha256"],
            "application_applied_to_s5_input": False,
            "selected_for_s5": False,
            "mir_abi_safe": False,
        })
        applied_rejected_ids.append(rid)
    for action in case["blocked_actions"]:
        rid = action["rewrite_id"]
        if action.get("action") != "block_before_s5_input" or action.get("selected_for_s5") is not False:
            raise SystemExit(f"blocked applied action must stay out of S5 input: {rid}")
        blocked_effects.append({
            "rewrite_id": rid,
            "case_id": case["case_id"],
            "source": case["source"],
            "rewrite_kind": action["rewrite_kind"],
            "proposal_kind": action["proposal_kind"],
            "rejection_reason_code": action["rejection_reason_code"],
            "proof_obligation": action["proof_obligation"],
            "application_applied_to_s5_input": False,
            "selected_for_s5": False,
            "mir_abi_safe": False,
        })
        applied_blocked_ids.append(rid)
    if len(selected_effects) != case["selected_action_count"]:
        raise SystemExit(f"applied selected count mismatch for {case['case_id']}")
    if len(rejected_effects) != case["rejected_action_count"]:
        raise SystemExit(f"applied rejected count mismatch for {case['case_id']}")
    if len(blocked_effects) != case["blocked_action_count"]:
        raise SystemExit(f"applied blocked count mismatch for {case['case_id']}")
    applied_cases.append({
        "case_id": case["case_id"],
        "source": case["source"],
        "input_hlir_sha256": case["input_hlir_sha256"],
        "input_egraph_sha256": case["egraph_sha256"],
        "input_extraction_sha256": case["extraction_sha256"],
        "post_apply_selected_enode_sha256": [effect["post_apply_selected_enode_sha256"] for effect in selected_effects],
        "post_mutation_hlir_sha256": [effect["post_mutation_hlir_sha256"] for effect in selected_effects],
        "post_mutation_egraph_sha256": [effect["post_mutation_egraph_sha256"] for effect in selected_effects],
        "selected_effect_count": len(selected_effects),
        "rejected_effect_count": len(rejected_effects),
        "blocked_effect_count": len(blocked_effects),
        "selected_effects": selected_effects,
        "rejected_effects": rejected_effects,
        "blocked_effects": blocked_effects,
    })

if set(applied_selected_ids) != set(selected_application_ids):
    raise SystemExit("applied selected ids do not match S4->S5 selected application ids")
if set(applied_rejected_ids) != set(rejected_application_ids):
    raise SystemExit("applied rejected ids do not match S4->S5 rejected application ids")
if set(applied_blocked_ids) != set(blocked_application_ids):
    raise SystemExit("applied blocked ids do not match S4->S5 blocked application ids")
if set(applied_selected_ids) & set(applied_rejected_ids) or set(applied_selected_ids) & set(applied_blocked_ids) or set(applied_rejected_ids) & set(applied_blocked_ids):
    raise SystemExit("applied extraction action buckets overlap")

applied_extraction = {
    "schema": "madaros.v2.s4.applied_extraction/0.1",
    "status": "pass",
    "stage_contract_level": "S4_EXACT_EXTRACTION_APPLIED_TO_S5_INPUT_NOT_COMPILER_IR",
    "input_application_plan_schema": application_plan["schema"],
    "input_application_plan_sha256": application_plan["application_plan_sha256"],
    "s4_downstream_integration_slice_complete": True,
    "s4_full_complete": False,
    "application_applied_to_s5_input": True,
    "application_applied_to_compiler_ir": False,
    "ir_mutation_allowed": False,
    "mutation_plan": "materialize deterministic S5 input effects without mutating compiler IR",
    "s5_input_contract_ready": bool(applied_selected_ids),
    "selected_effect_count": len(applied_selected_ids),
    "rejected_effect_count": len(applied_rejected_ids),
    "blocked_effect_count": len(applied_blocked_ids),
    "selected_rewrite_ids": applied_selected_ids,
    "rejected_rewrite_ids": applied_rejected_ids,
    "blocked_rewrite_ids": applied_blocked_ids,
    "selected_effect_sha256": applied_effect_hashes,
    "post_apply_selected_enode_sha256": [
        effect["post_apply_selected_enode_sha256"]
        for case in applied_cases
        for effect in case["selected_effects"]
    ],
    "cross_stage_invariants": [
        "applied_selected_ids_equal_application_plan_selected_ids",
        "applied_rejected_ids_equal_application_plan_rejected_ids",
        "applied_blocked_ids_equal_application_plan_blocked_ids",
        "selected_effects_are_mir_abi_safe_and_have_no_abi_impact",
        "selected_effects_materialize_exact_zero_error_translation_validated_rewrites",
        "rejected_and_blocked_effects_are_never_materialized_for_s5",
        "application_is_to_s5_input_not_compiler_ir",
    ],
    "cases": applied_cases,
}
applied_extraction["applied_extraction_sha256"] = hashlib.sha256(
    json.dumps(applied_extraction, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest()
applied_extraction_path = out / "madaros_v2_s4_applied_extraction.json"
applied_extraction_path.write_text(json.dumps(applied_extraction, sort_keys=True, indent=2) + "\n", encoding="utf-8")

summary = {
    "schema": "madaros.v2.s4.gate/0.1",
    "status": "pass",
    "stage_contract_level": "S4_BOUNDARY_NOT_FULL",
    "s4_boundary_complete": True,
    "s4_to_s5_application_plan_complete": True,
    "s4_to_s5_application_plan_path": application_plan_path.name,
    "s4_to_s5_application_plan_sha256": application_plan["application_plan_sha256"],
    "s4_applied_extraction_complete": True,
    "s4_applied_extraction_path": applied_extraction_path.name,
    "s4_applied_extraction_sha256": applied_extraction["applied_extraction_sha256"],
    "s4_downstream_integration_slice_complete": True,
    "s4_full_complete": False,
    "s_full_contract": "blocked_until_full_s4_obligations_are_gated",
    "missing_full_obligations": [
        "multi-rule equality saturation",
        "learned or approximate E-KAN proposals with declared domains and fallback expressions",
        "broad counterexample search over accepted and tempting sibling rewrites",
        "producer purity and evaluation-preservation beyond the current local leaf subset",
        "broader non-constant algebraic identities beyond neutral-element, reflexive-comparison, and same-SSA subtraction identities",
        "downstream compiler IR mutation beyond applied S4->S5 input materialization",
        "full-domain translation validation for every selected rewrite family",
    ],
    "case_count": len(receipts),
    "accepted_rewrite_count": sum(r["accepted_rewrite_count"] for r in receipts),
    "rejected_rewrite_count": sum(r["rejected_rewrite_count"] for r in receipts),
    "blocked_rewrite_count": sum(r["blocked_rewrite_count"] for r in receipts),
    "selected_rewrite_count": sum(r["selected_rewrite_count"] for r in receipts),
    "rejected_from_extraction_count": sum(r["rejected_from_extraction_count"] for r in receipts),
    "blocked_from_extraction_count": sum(r["blocked_from_extraction_count"] for r in receipts),
    "s4_to_s5_accepted_application_count": application_plan["accepted_application_count"],
    "s4_to_s5_rejected_application_count": application_plan["rejected_application_count"],
    "s4_to_s5_blocked_application_count": application_plan["blocked_application_count"],
    "s4_applied_selected_effect_count": applied_extraction["selected_effect_count"],
    "s4_applied_rejected_effect_count": applied_extraction["rejected_effect_count"],
    "s4_applied_blocked_effect_count": applied_extraction["blocked_effect_count"],
    "input_hlir_sha256": [r["input_hlir_sha256"] for r in receipts],
    "receipt_sha256": [r["receipt_sha256"] for r in receipts],
    "extraction_sha256": [r["extraction_sha256"] for r in receipts],
    "cases": receipts,
}
reason_counts = {}
for receipt in receipts:
    for case_dir in out.glob(f"{receipt['case_id']}/*"):
        rewrites_path = case_dir / receipt["rewrites_path"]
        if not rewrites_path.is_file() or case_dir.name == "b":
            continue
        for rewrite in json.loads(rewrites_path.read_text(encoding="utf-8")):
            if rewrite.get("blocked") is True:
                reason = rewrite.get("rejection_reason_code", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            elif rewrite.get("accepted") is False:
                reason = rewrite.get("rejection_reason_code", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
summary["validator_rejection_reason_counts"] = dict(sorted(reason_counts.items()))
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
summary["gate_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
(out / "madaros_v2_s4_gate.receipt.json").write_text(payload, encoding="utf-8")
print(
    f"[madaros-v2-s4] summary_sha={summary['gate_sha256'][:12]} "
    f"accepted={summary['accepted_rewrite_count']} rejected={summary['rejected_rewrite_count']} blocked={summary['blocked_rewrite_count']} "
    f"selected={summary['selected_rewrite_count']} app_plan={application_plan['application_plan_sha256'][:12]} "
    f"applied={applied_extraction['applied_extraction_sha256'][:12]}"
)
PY

echo "[madaros-v2-s4] PASS: S4 boundary receipts are deterministic and validated (S4 FULL remains blocked by listed obligations)"
echo "[madaros-v2-s4] PASS: S4->S5 application plan emitted for selected exact rewrites without mutating IR"
echo "[madaros-v2-s4] PASS: S4 applied extraction materialized as deterministic S5 input effects without mutating compiler IR"
echo "[madaros-v2-s4] receipts=$OUT_DIR"
