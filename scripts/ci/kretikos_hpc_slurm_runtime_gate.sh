#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${1:-examples/kretikos/manifest.tsv}"
OUT_DIR="${SOUNIO_KRETIKOS_HPC_GATE_DIR:-$(mktemp -d /tmp/kretikos-hpc-slurm.XXXXXX)}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-420}"

mkdir -p "$OUT_DIR"

DOCTOR_LOG="$OUT_DIR/kretikos_hpc_doctor.out"
MANIFEST_LOG="$OUT_DIR/kretikos_hpc_manifest.out"
SUMMARY_COPY="$OUT_DIR/kretikos_hpc_manifest_summary.tsv"
GATE_JSON="$OUT_DIR/kretikos_hpc_slurm_runtime_gate.v1.json"

echo "kretikos_hpc_slurm_runtime_gate: doctor"
./bin/kretikos hpc doctor >"$DOCTOR_LOG" 2>&1
cat "$DOCTOR_LOG"

echo "kretikos_hpc_slurm_runtime_gate: submit manifest=$MANIFEST"
set +e
WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT_SECONDS" \
KRETIKOS_HPC_PUBLISH_RESULTS=1 \
KRETIKOS_HPC_FETCH_RESULTS=1 \
  ./bin/kretikos hpc manifest "$MANIFEST" >"$MANIFEST_LOG" 2>&1
manifest_rc=$?
set -e
cat "$MANIFEST_LOG"

if [[ "$manifest_rc" -ne 0 ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL manifest_rc=$manifest_rc log=$MANIFEST_LOG" >&2
  exit "$manifest_rc"
fi

result_line="$(awk '/^kretikos_manifest_result / { line=$0 } END { print line }' "$MANIFEST_LOG")"
if [[ -z "$result_line" ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL missing kretikos_manifest_result" >&2
  exit 1
fi

total="$(sed -n 's/.*total=\([0-9][0-9]*\).*/\1/p' <<<"$result_line")"
failed="$(sed -n 's/.*failed=\([0-9][0-9]*\).*/\1/p' <<<"$result_line")"
summary_path="$(sed -n 's/.*summary=\([^ ]*\).*/\1/p' <<<"$result_line")"

if [[ -z "$total" || -z "$failed" || -z "$summary_path" ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL malformed result line: $result_line" >&2
  exit 1
fi

if [[ "$failed" -ne 0 ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL total=$total failed=$failed" >&2
  exit 1
fi

if [[ -f "$summary_path" ]]; then
  cp "$summary_path" "$SUMMARY_COPY"
else
  printf 'label\tsource\tstatus\tjob_id\tcomment\tartifact_dir\tartifact_json\n' >"$SUMMARY_COPY"
fi

python3 - "$GATE_JSON" "$MANIFEST" "$DOCTOR_LOG" "$MANIFEST_LOG" "$SUMMARY_COPY" "$total" "$failed" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, manifest, doctor_log, manifest_log, summary_tsv, total, failed = sys.argv[1:]

cases = []
summary_path = Path(summary_tsv)
if summary_path.exists():
    with summary_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            cases.append(row)

comments = [case.get("comment", "") for case in cases]
devices = sorted({part.split("=", 1)[1] for comment in comments for part in comment.split() if part.startswith("device=")})
drivers = sorted({part.split("=", 1)[1] for comment in comments for part in comment.split() if part.startswith("driver=")})
ccs = sorted({part.split("=", 1)[1] for comment in comments for part in comment.split() if part.startswith("cc=")})
jobs = [case.get("job_id", "") for case in cases if case.get("job_id")]
artifact_dirs = [case.get("artifact_dir", "") for case in cases if case.get("artifact_dir")]
artifact_jsons = [case.get("artifact_json", "") for case in cases if case.get("artifact_json")]
artifact_runtimes = []
kaxi_certificates = []
missing_artifacts = []
missing_certificates = []
bad_certificates = []
kaxi_supported_profiles = {
    "vec_add_f32",
    "vec_sub_f32",
    "vec_mul_f32",
    "vec_div_f32",
    "fma_f32",
    "epistemic_elementwise_f32",
    "epistemic_dual_output_f32",
}

for case in cases:
    artifact_json = case.get("artifact_json", "")
    if not artifact_json:
        missing_artifacts.append({"label": case.get("label"), "reason": "artifact_json_missing"})
        continue
    artifact_path = Path(artifact_json)
    if not artifact_path.is_file():
        missing_artifacts.append({"label": case.get("label"), "path": artifact_json, "reason": "artifact_json_not_found"})
        continue
    obj = json.load(artifact_path.open(encoding="utf-8"))
    runtime = obj.get("runtime_validation") or {}
    artifact_runtimes.append({
        "label": case.get("label"),
        "artifact_json": artifact_json,
        "status": runtime.get("status"),
        "reason": runtime.get("reason"),
        "rung": runtime.get("rung"),
        "kernel": runtime.get("kernel"),
        "device": runtime.get("device_name"),
    })
    profile = (obj.get("source_profile") or {}).get("profile") or case.get("label")
    cert = obj.get("kaxi_certificate") or {}
    cert_runtime = cert.get("runtime_validation") or {}
    cert_payload = cert.get("certificate") or {}
    published_path = cert.get("published_path")
    fetched_certificate = None
    if published_path:
        fetched_path = artifact_path.parent / published_path
        if fetched_path.is_file():
            fetched_certificate = json.load(fetched_path.open(encoding="utf-8"))
        elif profile in kaxi_supported_profiles:
            missing_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "path": str(fetched_path),
                "reason": "published_certificate_not_found",
            })
    if fetched_certificate is not None:
        fetched_runtime = fetched_certificate.get("runtime", {}).get("runtime_validation") or {}
        if fetched_certificate.get("status") != cert_payload.get("status"):
            bad_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "reason": "embedded_and_fetched_certificate_status_mismatch",
                "embedded_status": cert_payload.get("status"),
                "fetched_status": fetched_certificate.get("status"),
            })
        if fetched_runtime.get("status") != cert_runtime.get("status"):
            bad_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "reason": "embedded_and_fetched_runtime_status_mismatch",
                "embedded_runtime_status": cert_runtime.get("status"),
                "fetched_runtime_status": fetched_runtime.get("status"),
            })
    kaxi_certificates.append({
        "label": case.get("label"),
        "profile": profile,
        "status": cert.get("status"),
        "reason": cert.get("reason"),
        "published_path": published_path,
        "certificate_status": cert_payload.get("status"),
        "certificate_schema": cert_payload.get("schema"),
        "kaxi_pattern": cert.get("kaxi_pattern"),
        "semantic_profile": cert.get("semantic_profile"),
        "runtime_status": cert_runtime.get("status"),
        "runtime_reason": cert_runtime.get("reason"),
        "runtime_rung": cert_runtime.get("rung"),
        "runtime_kernel": cert_runtime.get("kernel"),
        "fetched": fetched_certificate is not None,
    })
    if profile in kaxi_supported_profiles:
        if cert.get("status") != "pass":
            bad_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "reason": "kaxi_certificate_status_not_pass",
                "status": cert.get("status"),
                "detail": cert.get("reason"),
            })
        if cert_payload.get("status") != "pass":
            bad_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "reason": "kaxi_certificate_payload_not_pass",
                "status": cert_payload.get("status"),
            })
        if cert_runtime.get("status") != "pass":
            bad_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "reason": "kaxi_certificate_runtime_not_pass",
                "status": cert_runtime.get("status"),
                "runtime_reason": cert_runtime.get("reason"),
            })
        if cert_runtime.get("rung") != profile:
            bad_certificates.append({
                "label": case.get("label"),
                "profile": profile,
                "reason": "kaxi_certificate_runtime_rung_mismatch",
                "runtime_rung": cert_runtime.get("rung"),
            })

if int(failed) == 0 and missing_artifacts:
    raise SystemExit(f"missing fetched artifacts: {missing_artifacts}")

bad_artifacts = [row for row in artifact_runtimes if row.get("status") != "pass"]
if int(failed) == 0 and bad_artifacts:
    raise SystemExit(f"artifact runtime did not pass: {bad_artifacts}")
if int(failed) == 0 and missing_certificates:
    raise SystemExit(f"missing K-AXI certificates: {missing_certificates}")
if int(failed) == 0 and bad_certificates:
    raise SystemExit(f"K-AXI certificate did not pass: {bad_certificates}")

payload = {
    "schema": "sounio.kretikos.hpc-slurm-runtime-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "manifest": manifest,
    "status": "pass" if int(failed) == 0 else "fail",
    "total": int(total),
    "failed": int(failed),
    "jobs": jobs,
    "devices": devices,
    "drivers": drivers,
    "compute_capabilities": ccs,
    "artifact_dirs": artifact_dirs,
    "artifact_jsons": artifact_jsons,
    "artifact_runtimes": artifact_runtimes,
    "kaxi_certificates": kaxi_certificates,
    "doctor_log": str(Path(doctor_log).name),
    "manifest_log": str(Path(manifest_log).name),
    "summary_tsv": str(Path(summary_tsv).name),
    "cases": cases,
    "boundaries": [
        "local_workspace_is_build_and_inspection_host",
        "slurm_worker_is_cuda_runtime_authority",
        "runtime_acceptance_requires_cuda_driver_load_launch_and_copyback",
        "supported_kaxi_profiles_must_publish_a_runtime_backed_kaxi_certificate",
        "worker_side_ptxas_or_nvdisasm_missing_is_not_a_runtime_failure",
        "each_worker_result_must_publish_and_fetch_a_json_artifact",
        "does_not_claim_arbitrary_sounio_gpu_lowering",
    ],
}

with open(out, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

echo "kretikos_hpc_slurm_runtime_gate: PASS out=$OUT_DIR total=$total failed=$failed"
