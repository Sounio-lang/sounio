#!/usr/bin/env bash
# Madaros v2 S2 gate: deterministic contract/audit scaffold witnesses.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S2_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s2.XXXXXX)}"
PY="${ROOT_DIR}/scripts/dev/madaros_v2_s2_receipt.py"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
PARSER_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p "$OUT_DIR"

echo "[madaros-v2-s2] START"
echo "[madaros-v2-s2] out=$OUT_DIR"
echo "[madaros-v2-s2] compiler=$COMPILER"

run_case() {
  local case_id="$1"
  local source="$2"
  local a_dir="$OUT_DIR/$case_id/a"
  local b_dir="$OUT_DIR/$case_id/b"
  mkdir -p "$a_dir" "$b_dir"

  echo "[madaros-v2-s2] case=$case_id source=$source"
  "$PY" emit --source "$source" --out-dir "$a_dir" --case-id "$case_id" \
    --compiler "$COMPILER" --parser-sha "$PARSER_SHA" --deterministic-time
  "$PY" emit --source "$source" --out-dir "$b_dir" --case-id "$case_id" \
    --compiler "$COMPILER" --parser-sha "$PARSER_SHA" --deterministic-time

  cmp "$a_dir/$case_id.s2.receipt.json" "$b_dir/$case_id.s2.receipt.json" >/dev/null
  cmp "$a_dir/$case_id.s2.public_symbols.tsv" "$b_dir/$case_id.s2.public_symbols.tsv" >/dev/null
  cmp "$a_dir/$case_id.s2.import_audit.tsv" "$b_dir/$case_id.s2.import_audit.tsv" >/dev/null
  cmp "$a_dir/$case_id.s2.effects.tsv" "$b_dir/$case_id.s2.effects.tsv" >/dev/null
  cmp "$a_dir/$case_id.s2.refinements.tsv" "$b_dir/$case_id.s2.refinements.tsv" >/dev/null
  cmp "$a_dir/$case_id.s2.epistemic_decls.tsv" "$b_dir/$case_id.s2.epistemic_decls.tsv" >/dev/null
  cmp "$a_dir/$case_id.s2.diagnostics.json" "$b_dir/$case_id.s2.diagnostics.json" >/dev/null

  python3 - "$a_dir/$case_id.s2.receipt.json" "$a_dir/$case_id.s2.diagnostics.json" <<'PY'
import hashlib
import json
import sys

receipt_path, diagnostics_path = sys.argv[1:3]
with open(receipt_path, encoding="utf-8") as fh:
    data = json.load(fh)

required = [
    "schema_version",
    "source_path",
    "source_sha256",
    "parser_sha",
    "compiler_route",
    "compiler_route_kind",
    "claim_level",
    "s2_complete",
    "s2_status",
    "upstream_s1",
    "typed_hir_sha256",
    "typed_hir_status",
    "typed_hir_roundtrip_status",
    "effect_table_sha256",
    "import_audit_table_sha256",
    "public_symbol_table_sha256",
    "refinement_table_sha256",
    "epistemic_declaration_table_sha256",
    "diagnostic_table_sha256",
    "structured_diagnostics",
    "visibility_audit",
    "phase_caps",
    "receipt_sha256",
]
missing = [key for key in required if key not in data]
if missing:
    raise SystemExit(f"missing fields: {missing}")
if data["schema_version"] != "madaros.v2.s2.receipt/0.1":
    raise SystemExit("bad schema_version")
if data["claim_level"] != "s2_contract_scaffold" or data["s2_complete"] is not False:
    raise SystemExit("S2 receipt must stay an explicit scaffold")
if data["s2_status"] != "no_current_madaros_typed_hir_serializer":
    raise SystemExit("unexpected s2_status")
if data["typed_hir_sha256"] is not None:
    raise SystemExit("typed_hir_sha256 must be null until native typed HIR exists")
if data["typed_hir_status"] != "not_emitted_by_current_madaros":
    raise SystemExit("unexpected typed_hir_status")
if data["typed_hir_roundtrip_status"] != "not_available":
    raise SystemExit("unexpected typed_hir_roundtrip_status")
with open(diagnostics_path, "rb") as fh:
    diagnostics_sha = hashlib.sha256(fh.read()).hexdigest()
if data["diagnostic_table_sha256"] != diagnostics_sha:
    raise SystemExit("diagnostic_table_sha256 does not match sidecar")
diag = data["structured_diagnostics"]
if diag["compiler_check_rc"] != 0 or diag["compiler_check_status"] != "check_ok":
    raise SystemExit(f"compiler check failed: {diag}")
bad_bulk = (
    diag["by_code"].get("E175", 0) >= 50
    or diag["by_code"].get("E177", 0) >= 10
    or diag["by_code"].get("E046", 0) >= 5
)
if bad_bulk:
    raise SystemExit(f"bulk diagnostic spew detected: {diag['by_code']}")
if diag["by_severity"].get("panic", 0) or diag["by_severity"].get("sigsegv", 0) or diag["by_severity"].get("segfault", 0):
    raise SystemExit(f"panic/segfault diagnostics detected: {diag['by_severity']}")
if data["phase_caps"]["observed_modules"] < 1:
    raise SystemExit("empty observed module set")
PY
}

run_case hello examples/hello.sio
run_case smt_basic tests/stdlib/theorem/test_smt_solver_basic.sio
run_case selfhost_s2_contract self-hosted/compiler/madaros_v2_s2_receipt.sio
run_case gpu_ptx_combo tests/madaros/v2_s1/gpu_ptx_combo.sio

"$COMPILER" check self-hosted/compiler/madaros_v2_s2_receipt.sio >"$OUT_DIR/s2_contract.check.log" 2>&1

echo "[madaros-v2-s2] PASS: deterministic S2 contract scaffold; native typed HIR not yet emitted"
echo "[madaros-v2-s2] receipts=$OUT_DIR"
