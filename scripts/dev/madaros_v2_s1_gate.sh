#!/usr/bin/env bash
# Madaros v2 S1 gate: deterministic compiler-native AST/source/module receipt witnesses.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S1_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s1.XXXXXX)}"
PY="${ROOT_DIR}/scripts/dev/madaros_v2_s1_receipt.py"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
PARSER_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p "$OUT_DIR"

echo "[madaros-v2-s1] START"
echo "[madaros-v2-s1] out=$OUT_DIR"
echo "[madaros-v2-s1] compiler=$COMPILER"

run_case() {
  local case_id="$1"
  local source="$2"
  local a_dir="$OUT_DIR/$case_id/a"
  local b_dir="$OUT_DIR/$case_id/b"
  mkdir -p "$a_dir" "$b_dir"

  echo "[madaros-v2-s1] case=$case_id source=$source"
  "$PY" emit --source "$source" --out-dir "$a_dir" --case-id "$case_id" \
    --compiler "$COMPILER" --parser-sha "$PARSER_SHA" --deterministic-time
  "$PY" emit --source "$source" --out-dir "$b_dir" --case-id "$case_id" \
    --compiler "$COMPILER" --parser-sha "$PARSER_SHA" --deterministic-time

  cmp "$a_dir/$case_id.s1.receipt.json" "$b_dir/$case_id.s1.receipt.json" >/dev/null
  cmp "$a_dir/$case_id.s1.module_edges.tsv" "$b_dir/$case_id.s1.module_edges.tsv" >/dev/null
  cmp "$a_dir/$case_id.s1.ast.json" "$b_dir/$case_id.s1.ast.json" >/dev/null

  python3 - "$a_dir/$case_id.s1.receipt.json" "$a_dir/$case_id.s1.ast.json" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
ast_path = sys.argv[2]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

required = [
    "schema_version",
    "source_path",
    "source_sha256",
    "parser_sha",
    "compiler_route",
    "compiler_route_kind",
    "module_graph",
    "module_graph_sha256",
    "canonical_ast_sha256",
    "canonical_ast_relpath",
    "canonical_ast_status",
    "ast_surface_kind",
    "ast_boundary",
    "ast_serializer_version",
    "canonical_source_graph_sha256",
    "canonical_source_graph_status",
    "diagnostic_count",
    "diagnostics_sha256",
    "phase_caps",
    "ast_emit",
    "compiler_check",
    "receipt_sha256",
]
missing = [key for key in required if key not in data]
if missing:
    raise SystemExit(f"missing fields: {missing}")
if data["schema_version"] != "madaros.v2.s1.receipt/0.2":
    raise SystemExit("bad schema_version")
with open(ast_path, "rb") as fh:
    ast_sha = hashlib.sha256(fh.read()).hexdigest()
if data["canonical_ast_sha256"] != ast_sha:
    raise SystemExit("canonical_ast_sha256 does not match AST sidecar")
if data["canonical_ast_sha256"] == hashlib.sha256(b"").hexdigest():
    raise SystemExit("canonical_ast_sha256 must not be the empty hash")
if not data["canonical_ast_relpath"].endswith(".s1.ast.json"):
    raise SystemExit("unexpected AST sidecar path")
if data["canonical_ast_status"] != "stable_stage1_ast_serializer":
    raise SystemExit("unexpected AST status")
if data["ast_surface_kind"] != "compiler_native_top_level_ast_json":
    raise SystemExit("unexpected AST surface kind")
if data["ast_serializer_version"] != "madaros.stage1.ast/0.1":
    raise SystemExit("unexpected AST serializer version")
if data["canonical_source_graph_status"] != "stable_l1_source_import_public_symbol_surrogate":
    raise SystemExit("unexpected source-graph boundary")
if data["ast_emit"]["ast_emit_rc"] != 0 or data["ast_emit"]["ast_emit_status"] != "emit_ast_ok":
    raise SystemExit(f"AST emit failed: {data['ast_emit']}")
if not data["module_graph"]:
    raise SystemExit("empty module graph")
if data["compiler_check"]["compiler_check_rc"] != 0:
    raise SystemExit(f"compiler check failed: {data['compiler_check']}")
if data["phase_caps"]["observed_modules"] != len(data["module_graph"]):
    raise SystemExit("phase cap/module count mismatch")
PY
}

run_case hello examples/hello.sio
run_case smt_basic tests/stdlib/theorem/test_smt_solver_basic.sio
run_case selfhost_s1_contract self-hosted/compiler/madaros_v2_s1_receipt.sio
run_case gpu_ptx_combo tests/madaros/v2_s1/gpu_ptx_combo.sio

"$COMPILER" check self-hosted/compiler/madaros_v2_s1_receipt.sio >"$OUT_DIR/s1_contract.check.log" 2>&1

echo "[madaros-v2-s1] PASS: 4 receipts deterministic + contract module checks"
echo "[madaros-v2-s1] receipts=$OUT_DIR"
