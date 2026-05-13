#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${OMEGA_KAXI_CERT_ATTEST_MODE:-auto}"
RUN_HPC="${OMEGA_KAXI_CERT_RUN_HPC:-0}"
HPC_GATE_JSON="${OMEGA_KAXI_CERT_HPC_GATE_JSON:-}"
OUT_JSON="${OMEGA_KAXI_CERT_ATTEST_OUT:-$ROOT_DIR/artifacts/omega/kaxi_certificate_attestation.v1.json}"
LOG_PATH="${OMEGA_KAXI_CERT_ATTEST_LOG:-$ROOT_DIR/artifacts/omega/kaxi_certificate_attestation.log}"
WORK_DIR="${OMEGA_KAXI_CERT_ATTEST_WORK_DIR:-$(mktemp -d /tmp/omega-kaxi-cert-attest.XXXXXX)}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-600}"
REQUIRED_PROFILES="${OMEGA_KAXI_CERT_REQUIRED_PROFILES:-vec_add_f32,vec_sub_f32,vec_mul_f32,vec_div_f32,fma_f32,epistemic_elementwise_f32,epistemic_dual_output_f32}"

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_KAXI_CERT_ATTEST_MODE must be auto, required, or off (got '$MODE')" >&2
    exit 2
    ;;
esac

case "$RUN_HPC" in
  0|1) ;;
  *)
    echo "error: OMEGA_KAXI_CERT_RUN_HPC must be 0 or 1 (got '$RUN_HPC')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")" "$WORK_DIR"
: >"$LOG_PATH"

emit_not_run() {
  local reason="$1"
  python3 - "$OUT_JSON" "$MODE" "$reason" "$LOG_PATH" "$REQUIRED_PROFILES" <<'PY'
import datetime
import json
import sys
from pathlib import Path

out, mode, reason, log_path, required_profiles = sys.argv[1:]
payload = {
    "schema": "sounio.omega.kaxi-certificate-attestation.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": mode,
    "status_summary": "not_run",
    "reason": reason,
    "required_profiles": [x for x in required_profiles.split(",") if x],
    "hpc_gate": None,
    "certificates": [],
    "merkle": {
        "algorithm": "sha256-merkle-v1",
        "leaf_count": 0,
        "leaves": [],
        "root": "",
    },
    "blockers": [reason],
    "log_path": str(Path(log_path)),
    "boundaries": [
        "attestation_requires_a_kretikos_hpc_slurm_runtime_gate_artifact",
        "attestation_does_not_claim_arbitrary_sounio_gpu_lowering",
    ],
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

if [[ "$MODE" == "off" ]]; then
  emit_not_run "gate_disabled"
  echo "omega_kaxi_certificate_attest_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

if [[ "$RUN_HPC" == "1" ]]; then
  HPC_OUT_DIR="$WORK_DIR/hpc"
  mkdir -p "$HPC_OUT_DIR"
  echo "omega_kaxi_certificate_attest_gate: running Slurm runtime gate" | tee -a "$LOG_PATH"
  WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT_SECONDS" \
  SOUNIO_KRETIKOS_HPC_GATE_DIR="$HPC_OUT_DIR" \
    "$ROOT_DIR/scripts/ci/kretikos_hpc_slurm_runtime_gate.sh" >>"$LOG_PATH" 2>&1
  HPC_GATE_JSON="$HPC_OUT_DIR/kretikos_hpc_slurm_runtime_gate.v1.json"
fi

if [[ -z "$HPC_GATE_JSON" || ! -f "$HPC_GATE_JSON" ]]; then
  emit_not_run "hpc_gate_json_missing"
  echo "omega_kaxi_certificate_attest_gate: status=not_run reason=hpc_gate_json_missing report=$OUT_JSON"
  [[ "$MODE" == "required" ]] && exit 1
  exit 0
fi

python3 - "$ROOT_DIR" "$HPC_GATE_JSON" "$OUT_JSON" "$LOG_PATH" "$MODE" "$REQUIRED_PROFILES" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
hpc_gate_path = Path(sys.argv[2]).resolve()
out_path = Path(sys.argv[3])
log_path = Path(sys.argv[4])
mode = sys.argv[5]
required_profiles = [x.strip() for x in sys.argv[6].split(",") if x.strip()]

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_record(path: Path):
    if not path.is_file():
        return None
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }

def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)

def leaf_hash(kind: str, profile: str, name: str, digest: str) -> str:
    return sha256_bytes(f"{kind}\0{profile}\0{name}\0{digest}".encode("utf-8"))

def merkle_root(leaves):
    hashes = [leaf["leaf_hash"] for leaf in leaves]
    if not hashes:
        return ""
    level = hashes
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(sha256_bytes((left + right).encode("utf-8")))
        level = nxt
    return level[0]

blockers = []
certificate_rows = []
leaves = []

try:
    hpc_gate = json.loads(hpc_gate_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"failed to read HPC gate JSON: {exc}")

hpc_gate_record = file_record(hpc_gate_path)
if hpc_gate_record:
    leaves.append({
        "kind": "hpc_gate",
        "profile": "_gate",
        "name": "kretikos_hpc_slurm_runtime_gate.v1.json",
        "sha256": hpc_gate_record["sha256"],
        "leaf_hash": leaf_hash("hpc_gate", "_gate", "kretikos_hpc_slurm_runtime_gate.v1.json", hpc_gate_record["sha256"]),
    })

if hpc_gate.get("schema") != "sounio.kretikos.hpc-slurm-runtime-gate.v1":
    blockers.append({"kind": "hpc_gate_schema_mismatch", "schema": hpc_gate.get("schema")})
if hpc_gate.get("status") != "pass":
    blockers.append({"kind": "hpc_gate_not_pass", "status": hpc_gate.get("status")})
if hpc_gate.get("failed") not in (0, "0"):
    blockers.append({"kind": "hpc_gate_has_failed_cases", "failed": hpc_gate.get("failed")})

cases = hpc_gate.get("cases") if isinstance(hpc_gate.get("cases"), list) else []
case_by_profile = {}
for case in cases:
    artifact_json = case.get("artifact_json") if isinstance(case, dict) else ""
    if not artifact_json:
        continue
    artifact_path = Path(artifact_json)
    if not artifact_path.is_file():
        continue
    try:
        source_result = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    profile = (source_result.get("source_profile") or {}).get("profile") or case.get("label")
    case_by_profile[profile] = (case, artifact_path, source_result)

for profile in required_profiles:
    entry = case_by_profile.get(profile)
    if entry is None:
        blockers.append({"kind": "required_profile_missing", "profile": profile})
        continue

    case, artifact_path, source_result = entry
    source_result_record = file_record(artifact_path)
    if source_result_record:
        leaves.append({
            "kind": "hpc_source_result",
            "profile": profile,
            "name": artifact_path.name,
            "sha256": source_result_record["sha256"],
            "leaf_hash": leaf_hash("hpc_source_result", profile, artifact_path.name, source_result_record["sha256"]),
        })

    runtime = source_result.get("runtime_validation") or {}
    cert_info = source_result.get("kaxi_certificate") or {}
    cert_payload = cert_info.get("certificate") or {}
    cert_runtime = cert_info.get("runtime_validation") or {}
    published_path = cert_info.get("published_path") or ""
    fetched_certificate_path = artifact_path.parent / published_path if published_path else None
    fetched_certificate = None
    fetched_record = None

    if fetched_certificate_path and fetched_certificate_path.is_file():
        fetched_record = file_record(fetched_certificate_path)
        fetched_certificate = json.loads(fetched_certificate_path.read_text(encoding="utf-8"))
        leaves.append({
            "kind": "kaxi_certificate",
            "profile": profile,
            "name": fetched_certificate_path.name,
            "sha256": fetched_record["sha256"],
            "leaf_hash": leaf_hash("kaxi_certificate", profile, fetched_certificate_path.name, fetched_record["sha256"]),
        })
    else:
        blockers.append({
            "kind": "certificate_file_missing",
            "profile": profile,
            "path": str(fetched_certificate_path) if fetched_certificate_path else "",
        })

    cert = fetched_certificate or cert_payload
    source = cert.get("source") or {}
    kaxi = cert.get("kaxi") or {}
    artifact_records = cert.get("artifact_records") or {}
    cert_profile = (cert.get("profile") or {}).get("name")
    cert_runtime = (cert.get("runtime") or {}).get("runtime_validation") or cert_runtime

    checks = []
    checks.append(("hpc_runtime_pass", runtime.get("status") == "pass"))
    checks.append(("certificate_status_pass", cert.get("status") == "pass"))
    checks.append(("certificate_runtime_pass", cert_runtime.get("status") == "pass"))
    checks.append(("profile_matches", cert_profile == profile))
    checks.append(("runtime_rung_matches", cert_runtime.get("rung") == profile))
    checks.append(("source_sha_present", bool(source.get("sha256"))))
    checks.append(("kaxi_witness_sha_present", bool(kaxi.get("witness_sha256"))))
    checks.append(("kaxi_assembly_sha_present", bool((kaxi.get("assembly") or {}).get("sha256"))))
    checks.append(("runtime_bundle_sha_present", bool((artifact_records.get("runtime_bundle") or {}).get("sha256"))))

    failed_checks = [name for name, ok in checks if not ok]
    if failed_checks:
        blockers.append({
            "kind": "certificate_invariant_failed",
            "profile": profile,
            "failed_checks": failed_checks,
        })

    semantic_digests = [
        ("source", "source.sha256", source.get("sha256")),
        ("kaxi_witness", "kaxi.witness_sha256", kaxi.get("witness_sha256")),
        ("kaxi_assembly", "kaxi.assembly.sha256", (kaxi.get("assembly") or {}).get("sha256")),
        ("runtime_bundle", "artifact_records.runtime_bundle.sha256", (artifact_records.get("runtime_bundle") or {}).get("sha256")),
        ("runtime_source_profile", "artifact_records.runtime_source_profile.sha256", (artifact_records.get("runtime_source_profile") or {}).get("sha256")),
        ("source_witness", "artifact_records.source_witness.sha256", (artifact_records.get("source_witness") or {}).get("sha256")),
    ]
    for kind, name, digest in semantic_digests:
        if digest:
            leaves.append({
                "kind": kind,
                "profile": profile,
                "name": name,
                "sha256": digest,
                "leaf_hash": leaf_hash(kind, profile, name, digest),
            })

    certificate_rows.append({
        "profile": profile,
        "label": case.get("label"),
        "job_id": case.get("job_id"),
        "source": {
            "path": source.get("path"),
            "bytes": source.get("bytes"),
            "sha256": source.get("sha256"),
        },
        "hpc_source_result": source_result_record,
        "certificate_file": fetched_record,
        "certificate_status": cert.get("status"),
        "certificate_schema": cert.get("schema"),
        "kaxi": {
            "pattern": kaxi.get("pattern"),
            "semantic_profile": kaxi.get("semantic_profile"),
            "witness_sha256": kaxi.get("witness_sha256"),
            "assembly_sha256": (kaxi.get("assembly") or {}).get("sha256"),
            "assembly_instruction_count": (kaxi.get("assembly") or {}).get("instruction_count"),
            "assembly_opcodes": (kaxi.get("assembly") or {}).get("opcodes"),
            "epistemic_lanes": kaxi.get("epistemic_lanes"),
        },
        "runtime": {
            "status": cert_runtime.get("status"),
            "reason": cert_runtime.get("reason"),
            "rung": cert_runtime.get("rung"),
            "kernel": cert_runtime.get("kernel"),
            "stage": cert_runtime.get("stage"),
            "cuda_result": cert_runtime.get("cuda_result"),
            "driver_version": cert_runtime.get("driver_version"),
            "device_count": cert_runtime.get("device_count"),
            "cc_major": cert_runtime.get("cc_major"),
            "cc_minor": cert_runtime.get("cc_minor"),
            "device_name": cert_runtime.get("device_name"),
        },
        "artifact_record_hashes": {
            name: value.get("sha256")
            for name, value in artifact_records.items()
            if isinstance(value, dict) and value.get("sha256")
        },
        "failed_checks": failed_checks,
    })

leaves.sort(key=lambda row: (row["profile"], row["kind"], row["name"], row["sha256"]))
root_hash = merkle_root(leaves)
status = "pass" if not blockers else "fail"
reason = "kaxi_certificate_attestation_pass" if status == "pass" else blockers[0]["kind"]

payload = {
    "schema": "sounio.omega.kaxi-certificate-attestation.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": mode,
    "status_summary": status,
    "reason": reason,
    "required_profiles": required_profiles,
    "hpc_gate": {
        "path": rel_or_abs(hpc_gate_path),
        "schema": hpc_gate.get("schema"),
        "status": hpc_gate.get("status"),
        "total": hpc_gate.get("total"),
        "failed": hpc_gate.get("failed"),
        "jobs": hpc_gate.get("jobs"),
        "devices": hpc_gate.get("devices"),
        "drivers": hpc_gate.get("drivers"),
        "compute_capabilities": hpc_gate.get("compute_capabilities"),
        "sha256": hpc_gate_record.get("sha256") if hpc_gate_record else None,
    },
    "certificates": certificate_rows,
    "merkle": {
        "algorithm": "sha256-merkle-v1",
        "leaf_count": len(leaves),
        "leaves": leaves,
        "root": root_hash,
    },
    "blockers": blockers,
    "log_path": rel_or_abs(log_path),
    "boundaries": [
        "omega_attestation_consumes_fetched_slurm_worker_evidence",
        "supported_kaxi_profiles_require_certificate_status_pass",
        "certificate_runtime_must_pass_on_the_matching_profile_rung",
        "merkle_root_covers_source_kaxi_witness_assembly_runtime_bundle_and_certificate_hashes",
        "slurm_worker_is_cuda_runtime_authority",
        "attestation_does_not_claim_arbitrary_sounio_gpu_lowering",
        "attestation_does_not_claim_general_gpu_backend_completeness",
    ],
}

out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if status != "pass":
    raise SystemExit(json.dumps(blockers, separators=(",", ":")))
PY

echo "omega_kaxi_certificate_attest_gate: status=pass report=$OUT_JSON"
