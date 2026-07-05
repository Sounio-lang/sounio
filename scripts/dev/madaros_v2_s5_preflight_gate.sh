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
if summary.get("accepted_rewrite_count", 0) <= 0:
    raise SystemExit("S5 preflight needs at least one accepted S4 rewrite")

allowed_rewrites = {"constant_fold_i64"}
allowed_basis = {"exact_symbolic"}
allowed_validators = {"translation-validation"}
consumed = []
all_rewrites = []

for receipt_file in sorted(s4_dir.glob("*/*/*.s4.receipt.json")):
    if "/b/" in receipt_file.as_posix():
        continue
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("schema") != "madaros.v2.s4.receipt/0.1":
        raise SystemExit(f"bad S4 receipt schema: {receipt_file}")
    if receipt.get("s4_boundary_complete") is not True:
        raise SystemExit("S4 boundary receipt not complete")
    if receipt.get("s4_complete") is not False:
        raise SystemExit("S4 receipt must not claim global completion")
    rewrites_path = receipt_file.parent / receipt["rewrites_path"]
    rewrites = json.loads(rewrites_path.read_text(encoding="utf-8"))
    if len(rewrites) != receipt["rewrite_count"]:
        raise SystemExit("S4 rewrites count mismatch")
    for rewrite in rewrites:
        if rewrite.get("accepted") is not True:
            if rewrite.get("selected_for_extraction") is not False:
                raise SystemExit("S5 preflight rejects extracted rejected S4 rewrites")
            if rewrite.get("ir_mutation_allowed") is not False:
                raise SystemExit("S5 preflight rejects mutating rejected S4 rewrites")
            continue
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
        all_rewrites.append(rewrite)
    consumed.append({
        "case_id": receipt["case_id"],
        "source": receipt["source"],
        "input_hlir_sha256": receipt["input_hlir_sha256"],
        "egraph_sha256": receipt["egraph_sha256"],
        "accepted_rewrite_count": receipt["accepted_rewrite_count"],
    })

preflight = {
    "schema": "madaros.v2.s5.preflight/0.1",
    "status": "pass",
    "s5_ready": False,
    "s5_input_contract_ready": True,
    "s5_implemented": False,
    "input_contract": "madaros.v2.s4.gate/0.1",
    "mir_abi_safe_subset": sorted(allowed_rewrites),
    "abi_impact": "none: accepted rewrites replace binary constant expressions with exact constants only",
    "numeric_semantics": "i64 constant-folding with zero-error translation-validation receipts",
    "consumed_s4_cases": consumed,
    "accepted_rewrite_count": len(all_rewrites),
    "required_next_lane": [
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
    f"rewrites={len(all_rewrites)} sha={preflight['preflight_sha256'][:12]}"
)
PY

echo "[madaros-v2-s5-preflight] PASS: current S4 boundary receipts are MIR/ABI-safe S5 inputs; S4 global completion and S5 remain future work"
echo "[madaros-v2-s5-preflight] receipt=$OUT_DIR/madaros_v2_s5_preflight.receipt.json"
