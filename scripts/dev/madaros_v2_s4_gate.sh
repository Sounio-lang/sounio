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
  local required_accepted="$6"
  local required_rejected="$7"
  local required_blocked="$8"
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

  python3 - "$a_dir/$case_id.s4.receipt.json" "$a_dir/$case_id.s4.egraph.json" "$a_dir/$case_id.s4.rewrites.json" "$a_dir/$case_id.s4.extraction.json" "$min_accepted" "$min_rejected" "$min_blocked" "$required_accepted" "$required_rejected" "$required_blocked" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, egraph_path, rewrites_path, extraction_path, min_accepted, min_rejected, min_blocked, required_accepted, required_rejected, required_blocked = sys.argv[1:11]
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
        observed_accepted.add(rewrite["rewrite_kind"])
        observed_accepted.add(rewrite["ekan_receipt_kind"])
    elif rewrite.get("blocked") is True:
        if rewrite["validator"] != "blocked":
            raise SystemExit("blocked rewrite must use validator=blocked")
        if rewrite.get("selected_for_extraction") is not False:
            raise SystemExit("blocked rewrite must not be selected for extraction")
        if rewrite.get("ir_mutation_allowed") is not False:
            raise SystemExit("blocked rewrite must not allow IR mutation")
        if rewrite.get("rejection_reason_code") != "operand_provenance_ambiguous":
            raise SystemExit("blocked rewrite missing operand provenance reason")
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
        if decision.get("rejection_reason_code") != "operand_provenance_ambiguous":
            raise SystemExit("blocked extraction missing operand provenance reason code")
        if "operand provenance" not in decision.get("proof_obligation", ""):
            raise SystemExit("blocked extraction missing provenance proof obligation")
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
print(
    f"[madaros-v2-s4] ok receipt={Path(receipt_path).name} "
    f"accepted={receipt['accepted_rewrite_count']} rejected={receipt['rejected_rewrite_count']} blocked={receipt['blocked_rewrite_count']} "
    f"selected={receipt['selected_rewrite_count']} egraph_sha={receipt['egraph_sha256'][:12]} "
    f"extraction_sha={receipt['extraction_sha256'][:12]}"
)
PY
}

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r case_id source min_accepted min_rejected min_blocked required_accepted required_rejected required_blocked; do
  run_case "$case_id" "$source" "$min_accepted" "$min_rejected" "$min_blocked" "$required_accepted" "$required_rejected" "$required_blocked"
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
summary = {
    "schema": "madaros.v2.s4.gate/0.1",
    "status": "pass",
    "case_count": len(receipts),
    "accepted_rewrite_count": sum(r["accepted_rewrite_count"] for r in receipts),
    "rejected_rewrite_count": sum(r["rejected_rewrite_count"] for r in receipts),
    "blocked_rewrite_count": sum(r["blocked_rewrite_count"] for r in receipts),
    "selected_rewrite_count": sum(r["selected_rewrite_count"] for r in receipts),
    "rejected_from_extraction_count": sum(r["rejected_from_extraction_count"] for r in receipts),
    "blocked_from_extraction_count": sum(r["blocked_from_extraction_count"] for r in receipts),
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
    f"selected={summary['selected_rewrite_count']}"
)
PY

echo "[madaros-v2-s4] PASS: conservative e-graph/E-KAN rewrite and extraction receipts are deterministic and validated"
echo "[madaros-v2-s4] receipts=$OUT_DIR"
