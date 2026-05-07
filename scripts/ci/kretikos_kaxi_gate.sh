#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_KAXI_GATE_DIR:-$(mktemp -d /tmp/kretikos-kaxi.XXXXXX)}"
mkdir -p "$OUT_DIR"

SELF_CHECK_LOG="$OUT_DIR/kaxi_self_check.out"
GATE_JSON="$OUT_DIR/kretikos_kaxi_gate.v1.json"

echo "kretikos_kaxi_gate: shell"
bash -n bin/kretikos

echo "kretikos_kaxi_gate: self-check"
./bin/kretikos emit-kaxi self-check >"$SELF_CHECK_LOG" 2>&1
cat "$SELF_CHECK_LOG"
grep -q "PASS kaxi_self_check" "$SELF_CHECK_LOG"

patterns=(
  exit_only
  vec_add
  epistemic_elementwise_f32
  epistemic_dual_output_f32
)

for pattern in "${patterns[@]}"; do
  out="$OUT_DIR/${pattern}.kaxi"
  log="$OUT_DIR/${pattern}.emit.out"
  echo "kretikos_kaxi_gate: emit pattern=$pattern out=$out"
  ./bin/kretikos emit-kaxi "$pattern" -o "$out" >"$log" 2>&1
  cat "$log"
  grep -q "K-AXI epistemic kernel assembly" "$out"
  grep -q "ret seq=" "$out"
done

grep -q "add r" "$OUT_DIR/vec_add.kaxi"
grep -q "fma r" "$OUT_DIR/epistemic_elementwise_f32.kaxi"
grep -q "store_global r" "$OUT_DIR/epistemic_elementwise_f32.kaxi"
grep -q "add r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "mul r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "store_global r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"

python3 - "$GATE_JSON" "$OUT_DIR" "$SELF_CHECK_LOG" "${patterns[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

gate_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
self_check_log = Path(sys.argv[3])
patterns = sys.argv[4:]

cases = []
for pattern in patterns:
    path = out_dir / f"{pattern}.kaxi"
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith(";")]
    cases.append({
        "pattern": pattern,
        "artifact": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "instruction_count": len(lines),
        "has_add": "add r" in text,
        "has_mul": "mul r" in text,
        "has_fma": "fma r" in text,
        "has_store_global": "store_global r" in text,
        "has_variance_annotation": "var=+" in text or "var=0%" in text,
    })

payload = {
    "schema": "sounio.kretikos.kaxi-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "pass",
    "self_check_log": self_check_log.name,
    "cases": cases,
    "boundaries": [
        "kaxi_is_sounio_owned_epistemic_gpu_assembly",
        "gate_proves_self_hosted_emitter_compiles_and_emits_structural_artifacts",
        "gate_proves_profile_level_lowering_for_explicit_kretikos_epistemic_profiles",
        "gate_does_not_claim_arbitrary_sounio_gpu_lowering",
        "gate_does_not_replace_slurm_cuda_runtime_authority",
    ],
}

gate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "kretikos_kaxi_gate: PASS out=$OUT_DIR"
