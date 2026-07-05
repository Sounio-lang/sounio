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
  local required_accepted="$5"
  local required_rejected="$6"
  local a_dir="$OUT_DIR/$case_id/a"
  local b_dir="$OUT_DIR/$case_id/b"
  mkdir -p "$a_dir" "$b_dir"

  echo "[madaros-v2-s4] case=$case_id source=$source"
  "$PY" emit --source "$source" --out-dir "$a_dir" --case-id "$case_id" --compiler "$COMPILER"
  "$PY" emit --source "$source" --out-dir "$b_dir" --case-id "$case_id" --compiler "$COMPILER"

  cmp "$a_dir/$case_id.s4.receipt.json" "$b_dir/$case_id.s4.receipt.json" >/dev/null
  cmp "$a_dir/$case_id.s4.egraph.json" "$b_dir/$case_id.s4.egraph.json" >/dev/null
  cmp "$a_dir/$case_id.s4.rewrites.json" "$b_dir/$case_id.s4.rewrites.json" >/dev/null

  python3 - "$a_dir/$case_id.s4.receipt.json" "$a_dir/$case_id.s4.egraph.json" "$a_dir/$case_id.s4.rewrites.json" "$min_accepted" "$min_rejected" "$required_accepted" "$required_rejected" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, egraph_path, rewrites_path, min_accepted, min_rejected, required_accepted, required_rejected = sys.argv[1:8]
receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
egraph = json.loads(Path(egraph_path).read_text(encoding="utf-8"))
rewrites = json.loads(Path(rewrites_path).read_text(encoding="utf-8"))
if receipt["schema"] != "madaros.v2.s4.receipt/0.1":
    raise SystemExit("bad S4 receipt schema")
if receipt["egraph_schema"] != "madaros.v2.s4.egraph/0.1":
    raise SystemExit("bad egraph schema")
if receipt["rewrite_schema"] != "madaros.v2.ekan.rewrite/0.1":
    raise SystemExit("bad rewrite schema")
if receipt["s4_boundary_complete"] is not True:
    raise SystemExit("S4 boundary receipt must be complete")
if receipt["s4_complete"] is not False:
    raise SystemExit("S4 gate must not claim global S4 completion")
egraph_for_hash = dict(egraph)
egraph_for_hash.pop("egraph_sha256", None)
egraph_canonical = json.dumps(egraph_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
if hashlib.sha256(egraph_canonical.encode()).hexdigest() != receipt["egraph_sha256"]:
    raise SystemExit("egraph hash mismatch")
if receipt["rewrite_count"] != len(rewrites):
    raise SystemExit("rewrite count mismatch")
accepted = [rewrite for rewrite in rewrites if rewrite.get("accepted") is True]
rejected = [rewrite for rewrite in rewrites if rewrite.get("accepted") is False]
if receipt["accepted_rewrite_count"] != len(accepted):
    raise SystemExit("accepted rewrite count mismatch")
if receipt["rejected_rewrite_count"] != len(rejected):
    raise SystemExit("rejected rewrite count mismatch")
if receipt["rewrite_count"] != receipt["accepted_rewrite_count"] + receipt["rejected_rewrite_count"]:
    raise SystemExit("rewrite total must equal accepted + rejected")
if receipt["accepted_rewrite_count"] < int(min_accepted):
    raise SystemExit(f"too few accepted rewrites: {receipt['accepted_rewrite_count']} < {min_accepted}")
if receipt["rejected_rewrite_count"] < int(min_rejected):
    raise SystemExit(f"too few rejected rewrites: {receipt['rejected_rewrite_count']} < {min_rejected}")
accepted_ids = set(receipt.get("accepted_rewrite_ids", []))
rejected_ids = set(receipt.get("rejected_rewrite_ids", []))
if accepted_ids & rejected_ids:
    raise SystemExit("rewrite ids cannot be both accepted and rejected")
observed_accepted = set()
observed_rejected = set()
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

missing_accepted = [item for item in required_accepted.split(",") if item and item != "-" and item not in observed_accepted]
if missing_accepted:
    raise SystemExit(f"missing required accepted markers: {missing_accepted}; observed={sorted(observed_accepted)}")
missing_rejected = [item for item in required_rejected.split(",") if item and item != "-" and item not in observed_rejected]
if missing_rejected:
    raise SystemExit(f"missing required rejected markers: {missing_rejected}; observed={sorted(observed_rejected)}")
print(
    f"[madaros-v2-s4] ok receipt={Path(receipt_path).name} "
    f"accepted={receipt['accepted_rewrite_count']} rejected={receipt['rejected_rewrite_count']} "
    f"egraph_sha={receipt['egraph_sha256'][:12]}"
)
PY
}

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r case_id source min_accepted min_rejected required_accepted required_rejected; do
  run_case "$case_id" "$source" "$min_accepted" "$min_rejected" "$required_accepted" "$required_rejected"
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
    "input_hlir_sha256": [r["input_hlir_sha256"] for r in receipts],
    "receipt_sha256": [r["receipt_sha256"] for r in receipts],
    "cases": receipts,
}
reason_counts = {}
for receipt in receipts:
    for case_dir in out.glob(f"{receipt['case_id']}/*"):
        rewrites_path = case_dir / receipt["rewrites_path"]
        if not rewrites_path.is_file() or case_dir.name == "b":
            continue
        for rewrite in json.loads(rewrites_path.read_text(encoding="utf-8")):
            if rewrite.get("accepted") is False:
                reason = rewrite.get("rejection_reason_code", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
summary["validator_rejection_reason_counts"] = dict(sorted(reason_counts.items()))
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
summary["gate_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
(out / "madaros_v2_s4_gate.receipt.json").write_text(payload, encoding="utf-8")
print(
    f"[madaros-v2-s4] summary_sha={summary['gate_sha256'][:12]} "
    f"accepted={summary['accepted_rewrite_count']} rejected={summary['rejected_rewrite_count']}"
)
PY

echo "[madaros-v2-s4] PASS: conservative e-graph/E-KAN accepted and rejected rewrite receipts are deterministic and validated"
echo "[madaros-v2-s4] receipts=$OUT_DIR"
