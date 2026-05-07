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
  witness="$OUT_DIR/${pattern}.kaxi-witness.json"
  log="$OUT_DIR/${pattern}.emit.out"
  echo "kretikos_kaxi_gate: witness pattern=$pattern out=$out witness=$witness"
  ./bin/kretikos kaxi-witness "$pattern" -o "$witness" --asm-output "$out" >"$log" 2>&1
  cat "$log"
  grep -q "K-AXI epistemic kernel assembly" "$out"
  grep -q "ret seq=" "$out"
  python3 - "$witness" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if obj.get("status") != "pass":
    raise SystemExit(f"witness_status_not_pass:{obj.get('status')}")
if not obj.get("assembly", {}).get("seq_dense_zero_based"):
    raise SystemExit("witness_sequence_not_dense_zero_based")
if obj.get("failures"):
    raise SystemExit(f"witness_failures:{obj['failures']}")
PY
done

grep -q "add r" "$OUT_DIR/vec_add.kaxi"
grep -q "fma r" "$OUT_DIR/epistemic_elementwise_f32.kaxi"
grep -q "store_global r" "$OUT_DIR/epistemic_elementwise_f32.kaxi"
grep -q "add r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "mul r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "store_global r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"

source_witnesses=(
  "epistemic_elementwise_f32:examples/kretikos/real_epistemic_elementwise.sio"
  "epistemic_dual_output_f32:examples/kretikos/real_epistemic_dual_output.sio"
)

for item in "${source_witnesses[@]}"; do
  label="${item%%:*}"
  src="${item#*:}"
  source_witness="$OUT_DIR/${label}.kaxi-source-witness.json"
  source_log="$OUT_DIR/${label}.source-witness.out"
  echo "kretikos_kaxi_gate: source-witness label=$label source=$src witness=$source_witness"
  ./bin/kretikos kaxi-source-witness "$src" -o "$source_witness" >"$source_log" 2>&1
  cat "$source_log"
  python3 - "$source_witness" "$label" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
label = sys.argv[2]
if obj.get("status") != "pass":
    raise SystemExit(f"source_witness_status_not_pass:{obj.get('status')}")
if obj.get("profile", {}).get("name") != label:
    raise SystemExit(f"source_witness_profile_mismatch:{obj.get('profile', {}).get('name')} != {label}")
if obj.get("failures"):
    raise SystemExit(f"source_witness_failures:{obj['failures']}")
PY
done

for item in "${source_witnesses[@]}"; do
  label="${item%%:*}"
  src="${item#*:}"
  certificate="$OUT_DIR/${label}.kaxi-certificate.json"
  cert_work="$OUT_DIR/${label}.certificate.d"
  cert_log="$OUT_DIR/${label}.certificate.out"
  echo "kretikos_kaxi_gate: certificate label=$label source=$src certificate=$certificate"
  ./bin/kretikos kaxi-certificate "$src" -o "$certificate" --work-dir "$cert_work" --force >"$cert_log" 2>&1
  cat "$cert_log"
  python3 - "$certificate" "$label" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
label = sys.argv[2]
if obj.get("status") != "pass":
    raise SystemExit(f"certificate_status_not_pass:{obj.get('status')}")
if obj.get("profile", {}).get("name") != label:
    raise SystemExit(f"certificate_profile_mismatch:{obj.get('profile', {}).get('name')} != {label}")
if obj.get("failures"):
    raise SystemExit(f"certificate_failures:{obj['failures']}")
runtime = obj.get("runtime", {}).get("runtime_validation", {})
if runtime.get("rung") != label:
    raise SystemExit(f"certificate_runtime_rung_mismatch:{runtime.get('rung')} != {label}")
PY
done

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
    witness_path = out_dir / f"{pattern}.kaxi-witness.json"
    text = path.read_text(encoding="utf-8")
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    lines = [line for line in text.splitlines() if line and not line.startswith(";")]
    cases.append({
        "pattern": pattern,
        "artifact": path.name,
        "witness": witness_path.name,
        "witness_sha256": hashlib.sha256(witness_path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "instruction_count": len(lines),
        "semantic_profile": witness.get("semantic_profile"),
        "seq_dense_zero_based": witness.get("assembly", {}).get("seq_dense_zero_based"),
        "store_global_count": witness.get("epistemic_lanes", {}).get("store_global_count"),
        "has_add": "add r" in text,
        "has_mul": "mul r" in text,
        "has_fma": "fma r" in text,
        "has_store_global": "store_global r" in text,
        "has_variance_annotation": "var=+" in text or "var=0%" in text,
    })

source_cases = []
for source_path in sorted(out_dir.glob("*.kaxi-source-witness.json")):
    obj = json.loads(source_path.read_text(encoding="utf-8"))
    source_cases.append({
        "witness": source_path.name,
        "witness_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source": obj.get("source", {}).get("path"),
        "source_sha256": obj.get("source", {}).get("sha256"),
        "profile": obj.get("profile", {}).get("name"),
        "kaxi_pattern": obj.get("kaxi", {}).get("pattern"),
        "semantic_profile": obj.get("kaxi", {}).get("semantic_profile"),
        "status": obj.get("status"),
    })

certificate_cases = []
for cert_path in sorted(out_dir.glob("*.kaxi-certificate.json")):
    obj = json.loads(cert_path.read_text(encoding="utf-8"))
    runtime = obj.get("runtime", {}).get("runtime_validation", {})
    certificate_cases.append({
        "certificate": cert_path.name,
        "certificate_sha256": hashlib.sha256(cert_path.read_bytes()).hexdigest(),
        "source": obj.get("source", {}).get("path"),
        "source_sha256": obj.get("source", {}).get("sha256"),
        "profile": obj.get("profile", {}).get("name"),
        "kaxi_pattern": obj.get("kaxi", {}).get("pattern"),
        "semantic_profile": obj.get("kaxi", {}).get("semantic_profile"),
        "runtime_status": runtime.get("status"),
        "runtime_reason": runtime.get("reason"),
        "runtime_rung": runtime.get("rung"),
        "runtime_kernel": runtime.get("kernel"),
        "status": obj.get("status"),
    })

payload = {
    "schema": "sounio.kretikos.kaxi-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "pass",
    "self_check_log": self_check_log.name,
    "cases": cases,
    "source_cases": source_cases,
    "certificate_cases": certificate_cases,
    "boundaries": [
        "kaxi_is_sounio_owned_epistemic_gpu_assembly",
        "gate_proves_self_hosted_emitter_compiles_and_emits_structural_artifacts",
        "gate_proves_kaxi_witness_json_for_each_profile",
        "gate_proves_checked_source_to_kaxi_witness_link_for_epistemic_profiles",
        "gate_proves_kaxi_certificate_braids_source_kaxi_and_runtime_bundle_evidence",
        "gate_proves_profile_level_lowering_for_explicit_kretikos_epistemic_profiles",
        "gate_does_not_claim_arbitrary_sounio_gpu_lowering",
        "gate_does_not_replace_slurm_cuda_runtime_authority",
    ],
}

gate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "kretikos_kaxi_gate: PASS out=$OUT_DIR"
