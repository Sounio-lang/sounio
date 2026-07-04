#!/usr/bin/env bash
# Madaros v2 S3 gate: compiler-native HLIR JSON, deterministic hash, roundtrip.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MADAROS_V2_S3_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-v2-s3.XXXXXX)}"
COMPILER="${MADAROS_BIN:-${ROOT_DIR}/bin/madaros}"
MANIFEST="${ROOT_DIR}/tests/madaros/v2_s3/manifest.tsv"
READINESS="${ROOT_DIR}/scripts/dev/madaros_v2_s3_readiness_gate.sh"

mkdir -p "$OUT_DIR"

echo "[madaros-v2-s3] START"
echo "[madaros-v2-s3] out=$OUT_DIR"
echo "[madaros-v2-s3] compiler=$COMPILER"

"$READINESS"

run_case() {
  local case_id="$1"
  local source="$2"
  local min_functions="$3"
  local min_instrs="$4"
  local required_ops="$5"
  local required_terms="$6"
  local required_calls="$7"
  local required_const_kinds="$8"
  local a_json="$OUT_DIR/$case_id.a.hlir.json"
  local b_json="$OUT_DIR/$case_id.b.hlir.json"
  local err="$OUT_DIR/$case_id.stderr.log"
  local receipt="$OUT_DIR/$case_id.s3.receipt.json"

  echo "[madaros-v2-s3] case=$case_id source=$source"
  "$COMPILER" --emit-hlir "$source" >"$a_json" 2>"$err"
  "$COMPILER" --emit-hlir "$source" >"$b_json" 2>>"$err"
  cmp "$a_json" "$b_json" >/dev/null
  if [[ -s "$err" ]]; then
    echo "[madaros-v2-s3] stderr was not empty for $case_id" >&2
    cat "$err" >&2
    return 1
  fi

  python3 - "$case_id" "$source" "$a_json" "$receipt" "$min_functions" "$min_instrs" \
    "$required_ops" "$required_terms" "$required_calls" "$required_const_kinds" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    case_id,
    source,
    json_path,
    receipt_path,
    min_functions,
    min_instrs,
    required_ops,
    required_terms,
    required_calls,
    required_const_kinds,
) = sys.argv[1:11]

text = Path(json_path).read_text(encoding="utf-8")
if not text.startswith('{"schema":"madaros.hlir.module/0.2"'):
    raise SystemExit("HLIR output is not clean JSON; banner or diagnostics leaked to stdout")
if "\n" in text.rstrip("\n"):
    raise SystemExit("HLIR output must be one canonical JSON object line")

data = json.loads(text)
if data.get("schema") != "madaros.hlir.module/0.2":
    raise SystemExit("bad schema")
if data.get("stage") != "S3":
    raise SystemExit("bad stage")
if data.get("source") != source:
    raise SystemExit(f"bad source field: {data.get('source')!r}")
if data.get("source_to_hlir") != "compiler_native_hlir_lower_module":
    raise SystemExit("HLIR was not produced by compiler_native_hlir_lower_module")
if data.get("ownership_effect_normalization") != "hlir_lower_module_v0":
    raise SystemExit("missing ownership/effect normalization marker")
if data.get("normalized_ids") is not True:
    raise SystemExit("normalized_ids must be true")

module = data.get("module")
if not isinstance(module, dict):
    raise SystemExit("module must be an object")
functions = module.get("functions")
globals_ = module.get("globals")
typedefs = module.get("typedefs")
if module.get("function_count") != len(functions):
    raise SystemExit("function_count mismatch")
if module.get("global_count") != len(globals_):
    raise SystemExit("global_count mismatch")
if module.get("typedef_count") != len(typedefs):
    raise SystemExit("typedef_count mismatch")
if len(functions) < int(min_functions):
    raise SystemExit(f"too few functions: {len(functions)} < {min_functions}")

ops = set()
terms = set()
calls = set()
const_kinds = set()
instr_total = 0
for func in functions:
    if func["param_count"] != len(func["params"]):
        raise SystemExit(f"param_count mismatch in {func['name']}")
    if func["effect_count"] != len(func["effects"]):
        raise SystemExit(f"effect_count mismatch in {func['name']}")
    if func["block_count"] != len(func["blocks"]):
        raise SystemExit(f"block_count mismatch in {func['name']}")
    if "return_type" not in func or "compile_strategy" not in func:
        raise SystemExit(f"function missing S4-consumable metadata: {func['name']}")
    for block in func["blocks"]:
        if block["param_count"] != len(block["params"]):
            raise SystemExit(f"block param_count mismatch in {func['name']}")
        if block["instr_count"] != len(block["instrs"]):
            raise SystemExit(f"instr_count mismatch in {func['name']}/{block['label']}")
        instr_total += block["instr_count"]
        term = block.get("terminator")
        if not isinstance(term, dict):
            raise SystemExit(f"missing terminator in {func['name']}/{block['label']}")
        terms.add(term.get("kind"))
        for instr in block["instrs"]:
            ops.add(instr.get("op"))
            calls.add(instr.get("call_name", ""))
            const = instr.get("constant", {})
            const_kinds.add(const.get("kind"))
            if "ty" not in instr or "result" not in instr:
                raise SystemExit("instruction missing result/type fields")

if instr_total < int(min_instrs):
    raise SystemExit(f"too few instructions: {instr_total} < {min_instrs}")

def require_all(label, observed, csv):
    wanted = [item for item in csv.split(",") if item and item != "-"]
    missing = [item for item in wanted if item not in observed]
    if missing:
        raise SystemExit(f"missing {label}: {missing}; observed={sorted(observed)}")

require_all("ops", ops, required_ops)
require_all("terminators", terms, required_terms)
require_all("calls", calls, required_calls)
require_all("constant kinds", const_kinds, required_const_kinds)

canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
receipt = {
    "schema": "madaros.v2.s3.receipt/0.1",
    "case_id": case_id,
    "source": source,
    "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
    "hlir_schema": data["schema"],
    "source_to_hlir": data["source_to_hlir"],
    "normalized_ids": data["normalized_ids"],
    "function_count": len(functions),
    "instruction_count": instr_total,
    "ops": sorted(op for op in ops if op),
    "terminators": sorted(term for term in terms if term),
    "calls": sorted(call for call in calls if call),
    "const_kinds": sorted(kind for kind in const_kinds if kind),
    "hlir_byte_sha256": hashlib.sha256(text.encode()).hexdigest(),
    "hlir_canonical_roundtrip_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
}
payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
Path(receipt_path).write_text(payload, encoding="utf-8")
print(
    f"[madaros-v2-s3] ok case={case_id} fns={len(functions)} "
    f"instrs={instr_total} hlir_sha={receipt['hlir_byte_sha256'][:12]}"
)
PY
}

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r case_id source min_functions min_instrs required_ops required_terms required_calls required_const_kinds; do
  run_case "$case_id" "$source" "$min_functions" "$min_instrs" "$required_ops" "$required_terms" "$required_calls" "$required_const_kinds"
done

python3 - "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
receipts = []
for path in sorted(out.glob("*.s3.receipt.json")):
    receipts.append(json.loads(path.read_text(encoding="utf-8")))
if not receipts:
    raise SystemExit("no S3 receipts produced")
summary = {
    "schema": "madaros.v2.s3.gate/0.1",
    "status": "pass",
    "case_count": len(receipts),
    "cases": receipts,
}
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
summary["gate_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
(out / "madaros_v2_s3_gate.receipt.json").write_text(payload, encoding="utf-8")
print(f"[madaros-v2-s3] summary_sha={summary['gate_sha256'][:12]} cases={len(receipts)}")
PY

echo "[madaros-v2-s3] PASS: native HLIR JSON deterministic, parseable, roundtrippable, S4-ready"
echo "[madaros-v2-s3] receipts=$OUT_DIR"
