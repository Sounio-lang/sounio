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

# ── summary JSON assembly (replaces python3 heredoc) ─────────────────────
__gate_sha256() { sha256sum "$1" 2>/dev/null | cut -c1-64 || echo ""; }

if [[ "$fail_count" -gt 0 ]]; then
    STATUS_VAL="fail"
elif [[ "$not_run_count" -gt 0 ]]; then
    STATUS_VAL="partial"
else
    STATUS_VAL="pass"
fi

# Convert TSV (case_id, program, status, detail, kind) to JSON array
CASES_JSON="$(
  {
    awk -F'\t' 'NR>1 && !/^[[:space:]]*#/ && NF {
      printf "{\"case_id\":\"%s\",\"detail\":\"%s\",\"kind\":\"%s\",\"program\":\"%s\",\"status\":\"%s\"}\n", $1, $4, $5, $2, $3
    }' "$RESULTS_TSV"
  } | ./bin/kretikos json-emit-array
)"

# Conditional CUDA boundary
if [[ "$not_run_count" -gt 0 ]]; then
    CUDA_BOUND="CUDA driver runtime is required for runtime parity rows|"
else
    CUDA_BOUND=""
fi
MANIFEST_SHA="$(__gate_sha256 "$MANIFEST")"
ACCEL_SHA="$(__gate_sha256 "$ACCEL_DIR/artifacts/epistemic_accel_spine.v1.json")"

./bin/kretikos json-emit \
    --string "accel_spine_summary=$ACCEL_DIR/artifacts/epistemic_accel_spine.v1.json" \
    --string "accel_spine_summary_sha256=$ACCEL_SHA" \
    --string "artifact_dir=$OUT_DIR" \
    --raw-json "cases=$CASES_JSON" \
    --string "fallback_path=none" \
    --int "fail_count=$fail_count" \
    --string "host_callback=none" \
    --string "manifest=$MANIFEST" \
    --string "manifest_sha256=$MANIFEST_SHA" \
    --int "not_run_count=$not_run_count" \
    --int "pass_count=$pass_count" \
    --array-strings "remaining_boundaries=${CUDA_BOUND}this gate does not prove tensor-core performance, ROCm, Metal, WebGPU, or DDC|public GPU PTX f64 op selection still uses the legalization bridge until the pinned artifact is source-rebuilt" \
    --string "results_tsv=$RESULTS_TSV" \
    --string "runtime_souc_path=$RUNTIME_SOUC" \
    --string "schema=sounio.native_v2_epistemic_gpu_runtime_parity.v1" \
    --string "status=$STATUS_VAL" \
    --string "target=x86_64-linux+cuda-runtime" \
    > "$SUMMARY_JSON" || exit 1

summary_status="$(./bin/kretikos kaxi-validate-evidence "$SUMMARY_JSON" --print-or-empty status)"

echo "[native-v2-gpu-runtime] status=$summary_status summary=$SUMMARY_JSON"
if [[ "$summary_status" == "fail" ]]; then
  exit 1
fi
