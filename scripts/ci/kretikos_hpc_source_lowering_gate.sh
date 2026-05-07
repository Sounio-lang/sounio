#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="${1:-examples/kretikos/lower_epistemic_dual_output_f32.sio}"
OUT_DIR="${SOUNIO_KRETIKOS_HPC_SOURCE_LOWERING_GATE_DIR:-$(mktemp -d /tmp/kretikos-hpc-source-lowering.XXXXXX)}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-420}"

mkdir -p "$OUT_DIR"

DOCTOR_LOG="$OUT_DIR/kretikos_hpc_doctor.out"
SOURCE_LOG="$OUT_DIR/kretikos_hpc_source_lowering.out"
GATE_JSON="$OUT_DIR/kretikos_hpc_source_lowering_gate.v1.json"

echo "kretikos_hpc_source_lowering_gate: doctor"
./bin/kretikos hpc doctor >"$DOCTOR_LOG" 2>&1
cat "$DOCTOR_LOG"

echo "kretikos_hpc_source_lowering_gate: source=$SOURCE"
WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT_SECONDS" \
KRETIKOS_HPC_PUBLISH_RESULTS=1 \
KRETIKOS_HPC_FETCH_RESULTS=1 \
  ./bin/kretikos hpc source "$SOURCE" >"$SOURCE_LOG" 2>&1
cat "$SOURCE_LOG"

artifact_json="$(awk -F= '/^ArtifactJSON=/ { print $2; exit }' "$SOURCE_LOG")"
if [[ -z "$artifact_json" ]]; then
  echo "kretikos_hpc_source_lowering_gate: FAIL missing ArtifactJSON in $SOURCE_LOG" >&2
  exit 1
fi
if [[ ! -f "$artifact_json" ]]; then
  echo "kretikos_hpc_source_lowering_gate: FAIL artifact json not found: $artifact_json" >&2
  exit 1
fi

python3 - "$GATE_JSON" "$SOURCE" "$DOCTOR_LOG" "$SOURCE_LOG" "$artifact_json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

gate_json, source, doctor_log, source_log, artifact_json = sys.argv[1:]
artifact_path = Path(artifact_json)
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
runtime = artifact.get("runtime_validation") or {}
cert = artifact.get("kaxi_certificate") or {}
cert_payload = cert.get("certificate") or {}
cert_runtime = (cert_payload.get("runtime") or {}).get("runtime_validation") or {}
lowering = cert_payload.get("lowering") or {}
kaxi = cert_payload.get("kaxi") or {}
kaxi_lanes = kaxi.get("epistemic_lanes") or {}

failures = []
if artifact.get("schema") != "sounio.kretikos.hpc-source-result.v1":
    failures.append({"kind": "artifact_schema_mismatch", "schema": artifact.get("schema")})
if runtime.get("status") != "pass":
    failures.append({"kind": "runtime_not_pass", "status": runtime.get("status"), "reason": runtime.get("reason")})
if runtime.get("rung") != "epistemic_dual_output_f32":
    failures.append({"kind": "runtime_rung_mismatch", "rung": runtime.get("rung")})
if runtime.get("kernel") != "sounio_bare_epistemic_dual_output_f32_sm80":
    failures.append({"kind": "runtime_kernel_mismatch", "kernel": runtime.get("kernel")})
if cert.get("status") != "pass":
    failures.append({"kind": "certificate_status_not_pass", "status": cert.get("status"), "reason": cert.get("reason")})
if cert.get("reason") != "runtime_backed_source_lowering_certificate_pass":
    failures.append({"kind": "certificate_reason_mismatch", "reason": cert.get("reason")})
if cert_payload.get("schema") != "sounio.kretikos.kaxi-lowering-certificate.v1":
    failures.append({"kind": "certificate_schema_mismatch", "schema": cert_payload.get("schema")})
if cert_payload.get("status") != "pass":
    failures.append({"kind": "certificate_payload_not_pass", "status": cert_payload.get("status"), "failures": cert_payload.get("failures", [])})
if lowering.get("profile_hint") != "none":
    failures.append({"kind": "profile_hint_not_none", "profile_hint": lowering.get("profile_hint")})
if lowering.get("fallback_path") != "none":
    failures.append({"kind": "fallback_path_not_none", "fallback_path": lowering.get("fallback_path")})
if lowering.get("recognized_pattern") != "indexed_f32_epistemic_dual_output":
    failures.append({"kind": "recognized_pattern_mismatch", "recognized_pattern": lowering.get("recognized_pattern")})
if lowering.get("kaxi_pattern") != "source_epistemic_dual_output_f32":
    failures.append({"kind": "kaxi_pattern_mismatch", "kaxi_pattern": lowering.get("kaxi_pattern")})
if cert_runtime.get("status") != "pass":
    failures.append({"kind": "certificate_runtime_not_pass", "status": cert_runtime.get("status"), "reason": cert_runtime.get("reason")})
if cert_runtime.get("rung") != "epistemic_dual_output_f32":
    failures.append({"kind": "certificate_runtime_rung_mismatch", "rung": cert_runtime.get("rung")})
if kaxi.get("pattern") != "source_epistemic_dual_output_f32":
    failures.append({"kind": "certificate_kaxi_pattern_mismatch", "pattern": kaxi.get("pattern")})
if kaxi_lanes.get("store_global_count", 0) < 2:
    failures.append({"kind": "dual_store_global_count_missing", "store_global_count": kaxi_lanes.get("store_global_count")})

payload = {
    "schema": "sounio.kretikos.hpc-source-lowering-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "pass" if not failures else "fail",
    "source": source,
    "doctor_log": Path(doctor_log).name,
    "source_log": Path(source_log).name,
    "artifact_json": str(artifact_path),
    "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    "job_id": artifact.get("job_id"),
    "host": artifact.get("host"),
    "runtime": {
        "status": runtime.get("status"),
        "reason": runtime.get("reason"),
        "device_name": runtime.get("device_name"),
        "driver_version": runtime.get("driver_version"),
        "cc_major": runtime.get("cc_major"),
        "cc_minor": runtime.get("cc_minor"),
        "rung": runtime.get("rung"),
        "kernel": runtime.get("kernel"),
    },
    "certificate": {
        "status": cert.get("status"),
        "reason": cert.get("reason"),
        "schema": cert_payload.get("schema"),
        "profile": (cert_payload.get("profile") or {}).get("name"),
        "recognized_pattern": lowering.get("recognized_pattern"),
        "kaxi_pattern": lowering.get("kaxi_pattern"),
        "profile_hint": lowering.get("profile_hint"),
        "fallback_path": lowering.get("fallback_path"),
        "store_global_count": kaxi_lanes.get("store_global_count"),
        "runtime_status": cert_runtime.get("status"),
        "runtime_reason": cert_runtime.get("reason"),
        "runtime_rung": cert_runtime.get("rung"),
        "runtime_kernel": cert_runtime.get("kernel"),
    },
    "failures": failures,
    "boundaries": [
        "slurm_worker_is_cuda_runtime_authority",
        "source_has_no_profile_directive_and_must_lower_through_kaxi_lower_source",
        "certificate_must_be_source_lowering_not_profile_level",
        "runtime_acceptance_requires_cuda_driver_load_launch_and_copyback",
        "does_not_claim_first_class_knowledge_generic_lowering",
        "does_not_claim_arbitrary_sounio_gpu_lowering",
    ],
}

gate_path = Path(gate_json)
gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if failures:
    raise SystemExit(json.dumps(failures))
PY

echo "kretikos_hpc_source_lowering_gate: PASS out=$OUT_DIR artifact=$artifact_json"
