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
  grep -Fq "$want" "$receipt" || return 1
  grep -Fxq "smt_skip=0" "$receipt"
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
  local min_rewrites="$3"
  local required_rewrites="$4"
  local a_dir="$OUT_DIR/$case_id/a"
  local b_dir="$OUT_DIR/$case_id/b"
  mkdir -p "$a_dir" "$b_dir"

  echo "[madaros-v2-s4] case=$case_id source=$source"
  "$PY" emit --source "$source" --out-dir "$a_dir" --case-id "$case_id" --compiler "$COMPILER"
  "$PY" emit --source "$source" --out-dir "$b_dir" --case-id "$case_id" --compiler "$COMPILER"

  cmp "$a_dir/$case_id.s4.receipt.json" "$b_dir/$case_id.s4.receipt.json" >/dev/null
  cmp "$a_dir/$case_id.s4.egraph.json" "$b_dir/$case_id.s4.egraph.json" >/dev/null
  cmp "$a_dir/$case_id.s4.rewrites.json" "$b_dir/$case_id.s4.rewrites.json" >/dev/null

  python3 - "$a_dir/$case_id.s4.receipt.json" "$a_dir/$case_id.s4.egraph.json" "$a_dir/$case_id.s4.rewrites.json" "$min_rewrites" "$required_rewrites" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, egraph_path, rewrites_path, min_rewrites, required_rewrites = sys.argv[1:6]
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
if receipt["accepted_rewrite_count"] < int(min_rewrites):
    raise SystemExit(f"too few accepted rewrites: {receipt['accepted_rewrite_count']} < {min_rewrites}")
if receipt["rejected_rewrite_count"] != 0:
    raise SystemExit("current S4 gate should not emit rejected rewrites")
observed = set()
for rewrite in rewrites:
    if rewrite["schema_version"] != "madaros.v2.ekan.rewrite/0.1":
        raise SystemExit("bad rewrite receipt schema")
    if rewrite["accepted"] is not True:
        raise SystemExit("unaccepted rewrite in S4 gate")
    if rewrite["validator"] != "translation-validation":
        raise SystemExit("accepted rewrite missing translation-validation")
    if rewrite["error_bound"] != "0":
        raise SystemExit("accepted exact rewrite must have zero error bound")
    if not rewrite["exact_fallback_expr_sha256"]:
        raise SystemExit("missing exact fallback hash")
    if not rewrite["coefficient_sha256"] or rewrite["basis_family"] != "exact_symbolic":
        raise SystemExit("missing exact symbolic E-KAN coefficient receipt")
    observed.add(rewrite["rewrite_kind"])
    observed.add(rewrite["ekan_receipt_kind"])
missing = [item for item in required_rewrites.split(",") if item and item not in observed]
if missing:
    raise SystemExit(f"missing required rewrite markers: {missing}; observed={sorted(observed)}")
print(f"[madaros-v2-s4] ok receipt={Path(receipt_path).name} accepted={receipt['accepted_rewrite_count']} egraph_sha={receipt['egraph_sha256'][:12]}")
PY
}

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r case_id source min_rewrites required_rewrites; do
  run_case "$case_id" "$source" "$min_rewrites" "$required_rewrites"
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
    "input_hlir_sha256": [r["input_hlir_sha256"] for r in receipts],
    "receipt_sha256": [r["receipt_sha256"] for r in receipts],
    "cases": receipts,
}
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
summary["gate_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
(out / "madaros_v2_s4_gate.receipt.json").write_text(payload, encoding="utf-8")
print(f"[madaros-v2-s4] summary_sha={summary['gate_sha256'][:12]} accepted={summary['accepted_rewrite_count']}")
PY

echo "[madaros-v2-s4] PASS: conservative e-graph/E-KAN rewrite receipts are deterministic and validated"
echo "[madaros-v2-s4] receipts=$OUT_DIR"
