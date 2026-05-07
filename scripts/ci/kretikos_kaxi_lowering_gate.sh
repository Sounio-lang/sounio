#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_KAXI_LOWERING_GATE_DIR:-$(mktemp -d /tmp/kretikos-kaxi-lowering.XXXXXX)}"
mkdir -p "$OUT_DIR"

SOURCE="examples/kretikos/lower_vec_add_f32.sio"
PROFILE_SOURCE="examples/kretikos/real_vec_add.sio"
ASM_OUT="$OUT_DIR/source_vec_add_f32.kaxi"
WITNESS_OUT="$OUT_DIR/source_vec_add_f32.kaxi-witness.json"
LOWERING_OUT="$OUT_DIR/source_vec_add_f32.kaxi-lowering.json"
NEGATIVE_STDOUT="$OUT_DIR/profile_directive_negative.stdout"
NEGATIVE_STDERR="$OUT_DIR/profile_directive_negative.stderr"
GATE_JSON="$OUT_DIR/kretikos_kaxi_lowering_gate.v1.json"

echo "kretikos_kaxi_lowering_gate: shell"
bash -n bin/kretikos "$0"

echo "kretikos_kaxi_lowering_gate: source-check source=$SOURCE"
./bin/kretikos check "$SOURCE"

if grep -Eq 'kretikos:[[:space:]]*profile[[:space:]]*=' "$SOURCE"; then
  echo "kretikos_kaxi_lowering_gate: source unexpectedly contains a profile directive" >&2
  exit 1
fi

echo "kretikos_kaxi_lowering_gate: kaxi-witness pattern=source_vec_add_f32"
./bin/kretikos kaxi-witness source_vec_add_f32 -o "$WITNESS_OUT" --asm-output "$ASM_OUT"
grep -q "get_tid r" "$ASM_OUT"
grep -q "load_global r" "$ASM_OUT"
grep -q "add r" "$ASM_OUT"
grep -q "store_global r" "$ASM_OUT"
grep -q "ret seq=" "$ASM_OUT"

echo "kretikos_kaxi_lowering_gate: source-lowering source=$SOURCE"
./bin/kretikos kaxi-lower-source "$SOURCE" -o "$LOWERING_OUT" --asm-output "$ASM_OUT" --kaxi-witness-output "$WITNESS_OUT"

echo "kretikos_kaxi_lowering_gate: negative profile-directive rejection source=$PROFILE_SOURCE"
if ./bin/kretikos kaxi-lower-source "$PROFILE_SOURCE" -o "$OUT_DIR/profile_directive_negative.json" >"$NEGATIVE_STDOUT" 2>"$NEGATIVE_STDERR"; then
  cat "$NEGATIVE_STDOUT"
  cat "$NEGATIVE_STDERR" >&2
  echo "kretikos_kaxi_lowering_gate: profile directive source unexpectedly lowered" >&2
  exit 1
fi
grep -q "profile_directive_present" "$NEGATIVE_STDERR"

python3 - "$GATE_JSON" "$OUT_DIR" "$SOURCE" "$ASM_OUT" "$WITNESS_OUT" "$LOWERING_OUT" "$NEGATIVE_STDERR" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

gate_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
source = Path(sys.argv[3])
asm_path = Path(sys.argv[4])
witness_path = Path(sys.argv[5])
lowering_path = Path(sys.argv[6])
negative_stderr = Path(sys.argv[7])

witness = json.loads(witness_path.read_text(encoding="utf-8"))
lowering = json.loads(lowering_path.read_text(encoding="utf-8"))
assembly_text = asm_path.read_text(encoding="utf-8")

failures = []
if witness.get("status") != "pass":
    failures.append({"kind": "witness_status_not_pass", "status": witness.get("status")})
if witness.get("pattern") != "source_vec_add_f32":
    failures.append({"kind": "witness_pattern_mismatch", "pattern": witness.get("pattern")})
if lowering.get("status") != "pass":
    failures.append({"kind": "lowering_status_not_pass", "status": lowering.get("status")})
lowering_contract = lowering.get("lowering", {})
if lowering_contract.get("profile_hint") != "none":
    failures.append({"kind": "profile_hint_not_none", "profile_hint": lowering_contract.get("profile_hint")})
if lowering_contract.get("fallback_path") != "none":
    failures.append({"kind": "fallback_path_not_none", "fallback_path": lowering_contract.get("fallback_path")})
if lowering_contract.get("recognized_pattern") != "indexed_f32_vector_add":
    failures.append({"kind": "recognized_pattern_mismatch", "recognized_pattern": lowering_contract.get("recognized_pattern")})
if lowering_contract.get("kaxi_pattern") != "source_vec_add_f32":
    failures.append({"kind": "kaxi_pattern_mismatch", "kaxi_pattern": lowering_contract.get("kaxi_pattern")})
if not lowering_contract.get("source_lowered_to_kaxi"):
    failures.append({"kind": "source_lowered_to_kaxi_not_true"})
if "profile_directive_present" not in negative_stderr.read_text(encoding="utf-8"):
    failures.append({"kind": "negative_profile_directive_rejection_missing"})
for opcode in ("get_tid", "load_global", "add", "store_global", "ret"):
    if opcode not in assembly_text:
        failures.append({"kind": "missing_lowered_opcode", "opcode": opcode})

payload = {
    "schema": "sounio.kretikos.kaxi-source-lowering-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "pass" if not failures else "fail",
    "out_dir": str(out_dir),
    "source": {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "profile_hint": "none",
    },
    "lowering": {
        "path": lowering_path.name,
        "sha256": hashlib.sha256(lowering_path.read_bytes()).hexdigest(),
        "recognized_pattern": lowering_contract.get("recognized_pattern"),
        "kaxi_pattern": lowering_contract.get("kaxi_pattern"),
        "fallback_path": lowering_contract.get("fallback_path"),
        "source_lowered_to_kaxi": lowering_contract.get("source_lowered_to_kaxi"),
        "semantic_ops": lowering_contract.get("semantic_ops", []),
    },
    "kaxi": {
        "assembly": asm_path.name,
        "assembly_sha256": hashlib.sha256(asm_path.read_bytes()).hexdigest(),
        "witness": witness_path.name,
        "witness_sha256": hashlib.sha256(witness_path.read_bytes()).hexdigest(),
        "opcodes": witness.get("assembly", {}).get("opcodes", []),
        "seq_dense_zero_based": witness.get("assembly", {}).get("seq_dense_zero_based"),
        "store_global_count": witness.get("epistemic_lanes", {}).get("store_global_count"),
    },
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
        "gate_proves_checked_source_to_kaxi_lowering_for_one_shape",
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
