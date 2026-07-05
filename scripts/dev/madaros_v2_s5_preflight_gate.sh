#!/usr/bin/env bash
# Madaros v2 S5 preflight: prove the current S4 boundary receipts are
# MIR/ABI-safe inputs. This does not implement S5 and does not claim S5 is
# globally ready; it only certifies the exact constant-fold subset consumed here.
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

allowed_rewrites = {"constant_fold_i64", "symbolic_identity_i64", "symbolic_reflexive_cmp_i64"}
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
                if rid not in blocked_ids or decision.get("selected") is not False:
                    raise SystemExit("S5 preflight requires blocked rewrites to be excluded by extraction")
                if decision.get("rejection_reason_code") != "operand_provenance_ambiguous":
                    raise SystemExit("S5 preflight requires operand provenance blocker evidence")
                blocked_rewrites.append(rewrite)
            else:
                if rid not in rejected_ids or decision.get("selected") is not False:
                    raise SystemExit("S5 preflight requires rejected rewrites to be blocked by extraction")
                if not decision.get("counterexample_set_sha256"):
                    raise SystemExit("S5 preflight requires rejected extraction counterexample evidence")
                semantic_rejections.append(rewrite)
            continue
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
        if rewrite.get("rewrite_kind") == "symbolic_identity_i64":
            if decision.get("lowering_effect") != "replace_binary_identity_expr_with_existing_value":
                raise SystemExit("S5 preflight rejects symbolic identity without value-ref lowering effect")
        if rewrite.get("rewrite_kind") == "symbolic_reflexive_cmp_i64":
            if decision.get("lowering_effect") != "replace_binary_predicate_expr_with_const_bool":
                raise SystemExit("S5 preflight rejects reflexive comparison without bool-const lowering effect")
        if decision.get("selected_enode_sha256") != rewrite.get("rewritten_enode_sha256"):
            raise SystemExit("S5 preflight selected enode must match rewrite")
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

input_ready = len(all_rewrites) > 0 and len(blocked_rewrites) == 0
preflight = {
    "schema": "madaros.v2.s5.preflight/0.1",
    "status": "pass" if input_ready else "blocked",
    "s5_ready": False,
    "s5_input_contract_ready": input_ready,
    "s5_implemented": False,
    "input_contract": "madaros.v2.s4.gate/0.1",
    "input_extraction_contract": "madaros.v2.s4.extraction/0.1",
    "mir_abi_safe_subset": sorted(allowed_rewrites),
    "abi_impact": "none: S5 consumes only extraction-selected accepted rewrites",
    "numeric_semantics": "i64 exact rewrites only when zero-error translation-validation receipts survive operand-provenance guards",
    "consumed_s4_cases": consumed,
    "accepted_rewrite_count": len(all_rewrites),
    "selected_rewrite_count": len(all_rewrites),
    "semantic_rejected_rewrite_count": len(semantic_rejections),
    "blocked_rewrite_count": len(blocked_rewrites),
    "blocking_reason": "" if input_ready else "S4 has no accepted extraction-selected rewrites after operand-provenance guard",
    "required_next_lane": [
        "repair/prove S3 HLIR binary operand provenance before symbolic/algebraic S4 extraction",
        "full S4 equality saturation and E-KAN rejection/proposal receipts",
        "MIR hash receipt",
        "ABI layout/call/return receipt",
        "numeric tower width receipts for f128/i256 before promotion",
        "differential native-v2 vs interpreter/lean_single gate where available",
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
