#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_KAXI_LOWERING_GATE_DIR:-$(mktemp -d /tmp/kretikos-kaxi-lowering.XXXXXX)}"
mkdir -p "$OUT_DIR"

PROFILE_SOURCE="examples/kretikos/real_vec_add.sio"
NEGATIVE_STDOUT="$OUT_DIR/profile_directive_negative.stdout"
NEGATIVE_STDERR="$OUT_DIR/profile_directive_negative.stderr"
GATE_JSON="$OUT_DIR/kretikos_kaxi_lowering_gate.v1.json"

lowering_cases=(
  "vec_add_f32|examples/kretikos/lower_vec_add_f32.sio|source_vec_add_f32|indexed_f32_vector_add|add"
  "vec_sub_f32|examples/kretikos/lower_vec_sub_f32.sio|source_vec_sub_f32|indexed_f32_vector_sub|sub"
  "vec_mul_f32|examples/kretikos/lower_vec_mul_f32.sio|source_vec_mul_f32|indexed_f32_vector_mul|mul"
  "vec_div_f32|examples/kretikos/lower_vec_div_f32.sio|source_vec_div_f32|indexed_f32_vector_div|div"
  "fma_f32|examples/kretikos/lower_fma_f32.sio|source_fma_f32|indexed_f32_affine_mad|mul,add"
  "epistemic_dual_output_f32|examples/kretikos/lower_epistemic_dual_output_f32.sio|source_epistemic_dual_output_f32|indexed_f32_epistemic_dual_output|add,mul"
)

echo "kretikos_kaxi_lowering_gate: shell"
bash -n bin/kretikos "$0"

for item in "${lowering_cases[@]}"; do
  IFS='|' read -r label source kaxi_pattern recognized_pattern opcodes <<<"$item"
  asm_out="$OUT_DIR/${label}.kaxi"
  witness_out="$OUT_DIR/${label}.kaxi-witness.json"
  lowering_out="$OUT_DIR/${label}.kaxi-lowering.json"

  echo "kretikos_kaxi_lowering_gate: source-check label=$label source=$source"
  ./bin/kretikos check "$source"

  if grep -Eq 'kretikos:[[:space:]]*profile[[:space:]]*=' "$source"; then
    echo "kretikos_kaxi_lowering_gate: source unexpectedly contains a profile directive: $source" >&2
    exit 1
  fi

  echo "kretikos_kaxi_lowering_gate: kaxi-witness label=$label pattern=$kaxi_pattern"
  ./bin/kretikos kaxi-witness "$kaxi_pattern" -o "$witness_out" --asm-output "$asm_out"
  grep -q "get_tid r" "$asm_out"
  grep -q "load_global r" "$asm_out"
  grep -q "store_global r" "$asm_out"
  grep -q "ret seq=" "$asm_out"
  IFS=',' read -ra opcode_list <<<"$opcodes"
  for opcode in "${opcode_list[@]}"; do
    grep -q "$opcode r" "$asm_out"
  done

  echo "kretikos_kaxi_lowering_gate: source-lowering label=$label source=$source"
  ./bin/kretikos kaxi-lower-source "$source" -o "$lowering_out" --asm-output "$asm_out" --kaxi-witness-output "$witness_out"

  python3 - "$lowering_out" "$label" "$kaxi_pattern" "$recognized_pattern" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
label, kaxi_pattern, recognized_pattern = sys.argv[2:]
if obj.get("status") != "pass":
    raise SystemExit(f"lowering_status_not_pass:{label}:{obj.get('status')}")
lowering = obj.get("lowering", {})
if lowering.get("profile_hint") != "none":
    raise SystemExit(f"profile_hint_not_none:{label}:{lowering.get('profile_hint')}")
if lowering.get("fallback_path") != "none":
    raise SystemExit(f"fallback_path_not_none:{label}:{lowering.get('fallback_path')}")
if lowering.get("kaxi_pattern") != kaxi_pattern:
    raise SystemExit(f"kaxi_pattern_mismatch:{label}:{lowering.get('kaxi_pattern')} != {kaxi_pattern}")
if lowering.get("recognized_pattern") != recognized_pattern:
    raise SystemExit(f"recognized_pattern_mismatch:{label}:{lowering.get('recognized_pattern')} != {recognized_pattern}")
if not lowering.get("source_lowered_to_kaxi"):
    raise SystemExit(f"source_lowered_to_kaxi_not_true:{label}")
store_count = obj.get("kaxi", {}).get("epistemic_lanes", {}).get("store_global_count", 0)
required_stores = 2 if label == "epistemic_dual_output_f32" else 1
if store_count < required_stores:
    raise SystemExit(f"store_global_count_missing:{label}")
PY
done

echo "kretikos_kaxi_lowering_gate: negative profile-directive rejection source=$PROFILE_SOURCE"
if ./bin/kretikos kaxi-lower-source "$PROFILE_SOURCE" -o "$OUT_DIR/profile_directive_negative.json" >"$NEGATIVE_STDOUT" 2>"$NEGATIVE_STDERR"; then
  cat "$NEGATIVE_STDOUT"
  cat "$NEGATIVE_STDERR" >&2
  echo "kretikos_kaxi_lowering_gate: profile directive source unexpectedly lowered" >&2
  exit 1
fi
grep -q "profile_directive_present" "$NEGATIVE_STDERR"

python3 - "$GATE_JSON" "$OUT_DIR" "$NEGATIVE_STDERR" "${lowering_cases[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

gate_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
negative_stderr = Path(sys.argv[3])
case_specs = sys.argv[4:]

failures = []
cases = []
for spec in case_specs:
    label, source, kaxi_pattern, recognized_pattern, opcodes = spec.split("|")
    source_path = Path(source)
    asm_path = out_dir / f"{label}.kaxi"
    witness_path = out_dir / f"{label}.kaxi-witness.json"
    lowering_path = out_dir / f"{label}.kaxi-lowering.json"
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    lowering = json.loads(lowering_path.read_text(encoding="utf-8"))
    lowering_contract = lowering.get("lowering", {})
    assembly_text = asm_path.read_text(encoding="utf-8")

    if witness.get("status") != "pass":
        failures.append({"kind": "witness_status_not_pass", "label": label, "status": witness.get("status")})
    if witness.get("pattern") != kaxi_pattern:
        failures.append({"kind": "witness_pattern_mismatch", "label": label, "pattern": witness.get("pattern")})
    if lowering.get("status") != "pass":
        failures.append({"kind": "lowering_status_not_pass", "label": label, "status": lowering.get("status")})
    if lowering_contract.get("profile_hint") != "none":
        failures.append({"kind": "profile_hint_not_none", "label": label, "profile_hint": lowering_contract.get("profile_hint")})
    if lowering_contract.get("fallback_path") != "none":
        failures.append({"kind": "fallback_path_not_none", "label": label, "fallback_path": lowering_contract.get("fallback_path")})
    if lowering_contract.get("recognized_pattern") != recognized_pattern:
        failures.append({"kind": "recognized_pattern_mismatch", "label": label, "recognized_pattern": lowering_contract.get("recognized_pattern")})
    if lowering_contract.get("kaxi_pattern") != kaxi_pattern:
        failures.append({"kind": "kaxi_pattern_mismatch", "label": label, "kaxi_pattern": lowering_contract.get("kaxi_pattern")})
    if not lowering_contract.get("source_lowered_to_kaxi"):
        failures.append({"kind": "source_lowered_to_kaxi_not_true", "label": label})
    store_count = witness.get("epistemic_lanes", {}).get("store_global_count", 0)
    required_stores = 2 if label == "epistemic_dual_output_f32" else 1
    if store_count < required_stores:
        failures.append({
            "kind": "store_global_count_below_required",
            "label": label,
            "store_global_count": store_count,
            "required": required_stores,
        })
    for opcode in ("get_tid", "load_global", "store_global", "ret"):
        if opcode not in assembly_text:
            failures.append({"kind": "missing_lowered_opcode", "label": label, "opcode": opcode})
    for opcode in opcodes.split(","):
        if opcode and opcode not in assembly_text:
            failures.append({"kind": "missing_arithmetic_opcode", "label": label, "opcode": opcode})

    cases.append({
        "label": label,
        "source": source,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "profile_hint": lowering_contract.get("profile_hint"),
        "fallback_path": lowering_contract.get("fallback_path"),
        "recognized_pattern": lowering_contract.get("recognized_pattern"),
        "kaxi_pattern": lowering_contract.get("kaxi_pattern"),
        "source_lowered_to_kaxi": lowering_contract.get("source_lowered_to_kaxi"),
        "semantic_ops": lowering_contract.get("semantic_ops", []),
        "assembly": asm_path.name,
        "assembly_sha256": hashlib.sha256(asm_path.read_bytes()).hexdigest(),
        "witness": witness_path.name,
        "witness_sha256": hashlib.sha256(witness_path.read_bytes()).hexdigest(),
        "opcodes": witness.get("assembly", {}).get("opcodes", []),
        "seq_dense_zero_based": witness.get("assembly", {}).get("seq_dense_zero_based"),
        "store_global_count": store_count,
        "required_store_global_count": required_stores,
    })

if "profile_directive_present" not in negative_stderr.read_text(encoding="utf-8"):
    failures.append({"kind": "negative_profile_directive_rejection_missing"})

payload = {
    "schema": "sounio.kretikos.kaxi-source-lowering-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "pass" if not failures else "fail",
    "out_dir": str(out_dir),
    "case_count": len(cases),
    "cases": cases,
    "negative_cases": [
        {
            "name": "profile_directive_rejected",
            "source": "examples/kretikos/real_vec_add.sio",
            "stderr": negative_stderr.name,
            "status": "pass",
        }
    ],
    "failures": failures,
    "boundaries": [
        "gate_proves_checked_source_to_kaxi_lowering_for_f32_arithmetic_corpus",
        "gate_requires_profile_hint_none",
        "gate_requires_fallback_path_none",
        "gate_rejects_legacy_profile_directive_path",
        "gate_does_not_claim_arbitrary_sounio_gpu_lowering",
        "gate_does_not_replace_slurm_cuda_runtime_authority",
    ],
}

gate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if failures:
    raise SystemExit(json.dumps(failures))
PY

echo "kretikos_kaxi_lowering_gate: PASS out=$OUT_DIR"
