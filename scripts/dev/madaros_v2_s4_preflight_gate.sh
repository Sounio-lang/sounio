#!/usr/bin/env bash
# Madaros v2 S4 preflight: prove S3 HLIR receipts are consumable by the next
# e-graph/E-KAN optimizer lane without claiming S4 optimization is implemented.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S4_PREFLIGHT_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s4-preflight.XXXXXX)}"
S3_DIR="$OUT_DIR/s3"
S3_GATE="${ROOT_DIR}/scripts/dev/madaros_v2_s3_gate.sh"

mkdir -p "$S3_DIR"

echo "[madaros-v2-s4-preflight] START"
echo "[madaros-v2-s4-preflight] out=$OUT_DIR"

SOUNIO_MADAROS_V2_S3_GATE_DIR="$S3_DIR" "$S3_GATE"

python3 - "$S3_DIR" "$OUT_DIR/madaros_v2_s4_preflight.receipt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

s3_dir = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
summary_path = s3_dir / "madaros_v2_s3_gate.receipt.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("schema") != "madaros.v2.s3.gate/0.1":
    raise SystemExit("missing S3 gate summary")
if summary.get("status") != "pass":
    raise SystemExit("S3 gate summary did not pass")

required_receipt_fields = {
    "hlir_schema",
    "source_to_hlir",
    "normalized_ids",
    "hlir_byte_sha256",
    "hlir_canonical_roundtrip_sha256",
    "ops",
    "terminators",
}
required_input_ops = {"const", "call_direct", "binary"}
required_input_terms = {"return", "branch", "cond_branch"}
observed_ops: set[str] = set()
observed_terms: set[str] = set()
observed_calls: set[str] = set()
consumed = []

for case in summary.get("cases", []):
    missing = sorted(required_receipt_fields - set(case))
    if missing:
        raise SystemExit(f"S3 receipt missing S4 input fields for {case.get('case_id')}: {missing}")
    if case["hlir_schema"] != "madaros.hlir.module/0.2":
        raise SystemExit(f"unsupported HLIR schema for S4: {case['hlir_schema']}")
    if case["source_to_hlir"] != "compiler_native_hlir_lower_module":
        raise SystemExit("S4 preflight requires compiler-native HLIR")
    if case["normalized_ids"] is not True:
        raise SystemExit("S4 preflight requires normalized_ids=true")
    if not case["hlir_byte_sha256"] or not case["hlir_canonical_roundtrip_sha256"]:
        raise SystemExit("missing S3 HLIR hashes")
    observed_ops.update(case.get("ops", []))
    observed_terms.update(case.get("terminators", []))
    observed_calls.update(case.get("calls", []))

    hlir_path = s3_dir / f"{case['case_id']}.a.hlir.json"
    text = hlir_path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != case["hlir_byte_sha256"]:
        raise SystemExit(f"HLIR byte hash mismatch for {case['case_id']}")
    data = json.loads(text)
    module = data["module"]
    for func in module["functions"]:
        for field in ("id", "name", "params", "return_type", "effects", "blocks", "compile_strategy"):
            if field not in func:
                raise SystemExit(f"S4 input missing function field {field} in {case['case_id']}")
        if func["block_count"] != len(func["blocks"]):
            raise SystemExit("S4 input block_count mismatch")
        for block in func["blocks"]:
            if "terminator" not in block:
                raise SystemExit("S4 input block missing terminator")
            for instr in block["instrs"]:
                for field in ("result", "op", "ty", "constant"):
                    if field not in instr:
                        raise SystemExit(f"S4 input instruction missing {field}")
    consumed.append({
        "case_id": case["case_id"],
        "source": case["source"],
        "hlir_byte_sha256": case["hlir_byte_sha256"],
        "hlir_canonical_roundtrip_sha256": case["hlir_canonical_roundtrip_sha256"],
        "function_count": case["function_count"],
        "instruction_count": case["instruction_count"],
    })

missing_ops = sorted(required_input_ops - observed_ops)
missing_terms = sorted(required_input_terms - observed_terms)
if missing_ops:
    raise SystemExit(f"S4 preflight missing required input ops: {missing_ops}")
if missing_terms:
    raise SystemExit(f"S4 preflight missing required input terminators: {missing_terms}")
if "println" not in observed_calls or "fact" not in observed_calls:
    raise SystemExit(f"S4 preflight missing representative direct calls: {sorted(observed_calls)}")

preflight = {
    "schema": "madaros.v2.s4.preflight/0.1",
    "status": "pass",
    "s4_implemented": False,
    "s4_ready": True,
    "input_contract": "madaros.v2.s3.gate/0.1",
    "optimizer_contract_target": "madaros.v2.ekan.rewrite/0.1",
    "consumed_s3_cases": consumed,
    "observed_ops": sorted(observed_ops),
    "observed_terminators": sorted(observed_terms),
    "observed_calls": sorted(observed_calls),
    "required_next_lane": [
        "persistent e-graph/equality receipt",
        "E-KAN proposal receipt",
        "domain-bounded error proof",
        "exact fallback expression hash",
        "translation-validation or SMT acceptance gate",
    ],
}
payload = json.dumps(preflight, sort_keys=True, indent=2) + "\n"
preflight["preflight_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
payload = json.dumps(preflight, sort_keys=True, indent=2) + "\n"
receipt_path.write_text(payload, encoding="utf-8")
print(
    f"[madaros-v2-s4-preflight] ok cases={len(consumed)} "
    f"ops={','.join(sorted(observed_ops))} sha={preflight['preflight_sha256'][:12]}"
)
PY

echo "[madaros-v2-s4-preflight] PASS: S3 HLIR receipts are consumable by S4; S4 optimizer remains future work"
echo "[madaros-v2-s4-preflight] receipt=$OUT_DIR/madaros_v2_s4_preflight.receipt.json"
