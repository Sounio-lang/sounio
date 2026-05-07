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
  printf 'label\tsource\tstatus\tjob_id\tcomment\n' >"$SUMMARY_COPY"
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
    "doctor_log": str(Path(doctor_log).name),
    "manifest_log": str(Path(manifest_log).name),
    "summary_tsv": str(Path(summary_tsv).name),
    "cases": cases,
    "boundaries": [
        "local_workspace_is_build_and_inspection_host",
        "slurm_worker_is_cuda_runtime_authority",
        "runtime_acceptance_requires_cuda_driver_load_launch_and_copyback",
        "worker_side_ptxas_or_nvdisasm_missing_is_not_a_runtime_failure",
        "does_not_claim_arbitrary_sounio_gpu_lowering",
    ],
}

with open(out, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

echo "kretikos_hpc_slurm_runtime_gate: PASS out=$OUT_DIR total=$total failed=$failed"
