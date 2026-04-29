#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_GPU_RUNTIME_PARITY_DIR:-$(mktemp -d /tmp/sounio-native-v2-gpu-runtime-parity.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_epistemic_gpu_runtime_parity.v1.json"
ACCEL_DIR="$OUT_DIR/accel-spine"
MANIFEST="${SOUNIO_NATIVE_V2_GPU_RUNTIME_PARITY_MANIFEST:-tests/gpu/epistemic_runtime/manifest.tsv}"
RUNTIME_SOUC="${SOUNIO_CUDA_RUNTIME_SOUC_BIN:-$ROOT_DIR/bin/souc}"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$ACCEL_DIR"

append_header() {
  cat >"$RESULTS_TSV" <<'EOF'
case_id	program	status	detail	kind
EOF
}

append_row() {
  local case_id="$1"
  local program="$2"
  local status="$3"
  local detail="$4"
  local kind="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "$program" "$status" "$detail" "$kind" >>"$RESULTS_TSV"
}

echo "[native-v2-gpu-runtime] out=$OUT_DIR"

SOUNIO_NATIVE_V2_ACCEL_SPINE_DIR="$ACCEL_DIR" \
SOUNIO_NATIVE_V2_ACCEL_RUN_CUDA=off \
  bash scripts/ci/native_v2_epistemic_accel_spine_gate.sh >"$LOG_DIR/accel_spine.log" 2>&1

append_header
pass_count=0
not_run_count=0
fail_count=0

while IFS=$'\t' read -r case_id program_path expected_stdout kind; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi
  if [[ ! -f "$program_path" ]]; then
    append_row "$case_id" "$program_path" "fail" "missing_program" "$kind"
    fail_count=$((fail_count + 1))
    continue
  fi

  check_log="$LOG_DIR/$case_id.check.log"
  runtime_log="$LOG_DIR/$case_id.runtime.log"

  if ! "$RUNTIME_SOUC" check "$program_path" >"$check_log" 2>&1; then
    append_row "$case_id" "$program_path" "fail" "runtime_fixture_check_failed" "$kind"
    fail_count=$((fail_count + 1))
    continue
  fi

  set +e
  SOUC_BIN="$RUNTIME_SOUC" \
  SOUNIO_CUDA_SMOKE_EXPECT="$expected_stdout" \
    bash scripts/gpu/run_native_cuda_smoke.sh "$program_path" >"$runtime_log" 2>&1
  runtime_rc=$?
  set -e

  if grep -q '^SKIP:' "$runtime_log"; then
    detail="$(grep '^SKIP:' "$runtime_log" | head -n 1)"
    append_row "$case_id" "$program_path" "not_run" "$detail" "$kind"
    not_run_count=$((not_run_count + 1))
  elif [[ "$runtime_rc" -eq 0 ]]; then
    append_row "$case_id" "$program_path" "pass" "cuda_runtime_stdout_ok" "$kind"
    pass_count=$((pass_count + 1))
  else
    detail="cuda_runtime_parity_failed"
    if grep -q '^FAIL:' "$runtime_log"; then
      detail="$(grep '^FAIL:' "$runtime_log" | head -n 1)"
    fi
    append_row "$case_id" "$program_path" "fail" "$detail" "$kind"
    fail_count=$((fail_count + 1))
  fi
done <"$MANIFEST"

python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$ACCEL_DIR/artifacts/epistemic_accel_spine.v1.json" "$MANIFEST" "$RUNTIME_SOUC" "$pass_count" "$not_run_count" "$fail_count" "$OUT_DIR" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
accel_summary_path = pathlib.Path(sys.argv[3])
manifest_path = pathlib.Path(sys.argv[4])
runtime_souc = sys.argv[5]
pass_count = int(sys.argv[6])
not_run_count = int(sys.argv[7])
fail_count = int(sys.argv[8])
out_dir = pathlib.Path(sys.argv[9])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
with open(results_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

status = "pass"
if fail_count:
    status = "fail"
elif not_run_count:
    status = "partial"

payload = {
    "schema": "sounio.native_v2_epistemic_gpu_runtime_parity.v1",
    "status": status,
    "target": "x86_64-linux+cuda-runtime",
    "fallback_path": "none",
    "host_callback": "none",
    "runtime_souc_path": runtime_souc,
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else "",
    "accel_spine_summary": str(accel_summary_path),
    "accel_spine_summary_sha256": sha256(accel_summary_path) if accel_summary_path.exists() else "",
    "pass_count": pass_count,
    "not_run_count": not_run_count,
    "fail_count": fail_count,
    "results_tsv": str(results_path),
    "artifact_dir": str(out_dir),
    "cases": rows,
    "remaining_boundaries": [
        "CUDA driver runtime is required for runtime parity rows" if not_run_count else "",
        "this gate does not prove tensor-core performance, ROCm, Metal, WebGPU, or DDC",
        "public GPU PTX f64 op selection still uses the legalization bridge until the pinned artifact is source-rebuilt",
    ],
}
payload["remaining_boundaries"] = [x for x in payload["remaining_boundaries"] if x]
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if fail_count:
    raise SystemExit(1)
PY

summary_status="$(python3 - "$SUMMARY_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"

echo "[native-v2-gpu-runtime] status=$summary_status summary=$SUMMARY_JSON"
if [[ "$summary_status" == "fail" ]]; then
  exit 1
fi
