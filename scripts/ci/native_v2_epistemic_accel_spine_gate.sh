#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

OUT_DIR="${SOUNIO_NATIVE_V2_ACCEL_SPINE_DIR:-$(mktemp -d /tmp/sounio-native-v2-accel-spine.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PTX_DIR="$ARTIFACT_DIR/ptx"
CPU_DIR="$OUT_DIR/cpu-oracle"
F64_DIR="$OUT_DIR/f64-ladder"
GPU_PUBLIC_JSON="$ARTIFACT_DIR/gpu_public_contract.v1.json"
GPU_PUBLIC_LOG="$LOG_DIR/gpu_public_contract.log"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
SUMMARY_JSON="$ARTIFACT_DIR/epistemic_accel_spine.v1.json"
CPU_MANIFEST="tests/native-v2/accel_spine/manifest.tsv"
GPU_MANIFEST="tests/gpu/epistemic_accel/manifest.tsv"
GPU_FIXTURE="scripts/fixtures/gpu_minimal.sio"
HYPERCOMPLEX_KERNEL="self-hosted/gpu/kernels/hypercomplex.sio"
OSSM_KERNEL="self-hosted/gpu/kernels/ossm_forward.sio"
HMMA_SIGNS_KERNEL="self-hosted/gpu/kernels/hmma_signs.sio"
PTX_F64_LEGALIZER="scripts/gpu/ptx_f64_legalize.py"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$PTX_DIR" "$CPU_DIR" "$F64_DIR"

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

append_header() {
  cat >"$RESULTS_TSV" <<'EOF'
case_id	lane	program	status	detail	ptx_sha256
EOF
}

append_row() {
  local case_id="$1"
  local lane="$2"
  local program="$3"
  local status="$4"
  local detail="$5"
  local ptx_sha="${6:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$case_id" "$lane" "$program" "$status" "$detail" "$ptx_sha" >>"$RESULTS_TSV"
}

analyze_ptx() {
  local ptx_path="$1"
  python3 - "$ptx_path" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
has_f64_param = ".param .f64" in text
has_f64_reg = ".reg .f64" in text
has_f64_arith = bool(re.search(r"\b(add|sub|mul)\.f64\b|\bdiv(?:\.rn)?\.f64\b", text))
bad_mixed = []
for line in text.splitlines():
    s = line.strip()
    if re.search(r"\b(add|sub|mul|div)\.f32\b", s) and ("f64_" in s or "r64_" in s):
        bad_mixed.append(s)

status = "pass"
detail = "f64_ptx_structural_ok"
if not has_f64_param:
    status = "fail"
    detail = "missing_param_f64"
elif not has_f64_reg:
    status = "fail"
    detail = "missing_reg_f64"
elif bad_mixed:
    status = "blocked"
    detail = "f64_operands_lowered_through_f32_ops"
elif not has_f64_arith:
    status = "blocked"
    detail = "missing_f64_arithmetic_opcode"

print(status + "\t" + detail)
if bad_mixed:
    print("\n".join(bad_mixed[:8]))
PY
}

source_probe_pass() {
  local path="$1"
  shift
  [[ -f "$path" ]] || return 1
  local pattern
  for pattern in "$@"; do
    grep -Fq -- "$pattern" "$path" || return 1
  done
}

append_source_probe() {
  local case_id="$1"
  local program_path="$2"
  local detail="$3"
  shift 3
  if source_probe_pass "$program_path" "$@"; then
    append_row "$case_id" "native_algebra" "$program_path" "pass" "$detail" ""
    pass_count=$((pass_count + 1))
  else
    append_row "$case_id" "native_algebra" "$program_path" "fail" "missing_native_algebra_surface" ""
    fail_count=$((fail_count + 1))
  fi
}

append_native_algebra_probes() {
  append_source_probe \
    "native_octonion_ptx_emitter" \
    "$HYPERCOMPLEX_KERNEL" \
    "octonion_fano_mul_and_epistemic_shadow" \
    "fn oct_mul_sign" \
    "fn oct_mul_basis" \
    "fn hc_emit_oct_mul" \
    "fn hc_emit_oct_eps_mul" \
    "fn hc_emit_oct_mul_kernel_header"

  append_source_probe \
    "native_sedenion_ptx_emitter" \
    "$HYPERCOMPLEX_KERNEL" \
    "sedenion_cayley_dickson_mul_zero_divisor_and_shadow" \
    "struct SixteenVec" \
    "fn hc_emit_sed_mul" \
    "fn hc_emit_sed_eps_mul" \
    "fn hc_emit_sed_is_zero_divisor_check" \
    "fn hc_emit_sed_mul_kernel_header"

  append_source_probe \
    "native_octonion_hmma_signs" \
    "$HMMA_SIGNS_KERNEL" \
    "tensor_core_left_mul_matrix_surface_convention_x" \
    "fn hmma_oct_cd_sigma" \
    "fn hmma_left_mul_sign" \
    "fn hmma_emit_build_left_mul_matrix" \
    "fn hmma_emit_unpack_result" \
    "fn hmma_emit_oct_mul_wmma_kernel" \
    ".visible .entry sounio_oct_mul_wmma" \
    "fn hmma_emit_oct_mul_full_kernel" \
    ".visible .entry sounio_oct_mul_full" \
    "fn hmma_emit_sed_mul_full_kernel" \
    "fn hmma_sed_left_mul_sign" \
    ".visible .entry sounio_sed_mul_full" \
    "fn hmma_right_mul_sign" \
    "fn hmma_emit_oct_associator_kernel"
    "fn hmma_right_mul_sign" \
    "fn hmma_emit_oct_associator_kernel" \
    ".visible .entry sounio_oct_associator" \
    "fn hmma_sed_right_mul_sign" \
    "fn hmma_emit_sed_associator_kernel" \
    ".visible .entry sounio_sed_associator" \
    "fn hmma_emit_ossm_oct_step_kernel" \
    ".visible .entry sounio_ossm_oct_step" \
    "fn hmma_emit_ossm_oct_cell_kernel" \
    ".visible .entry sounio_ossm_oct_cell" \
    "fn hmma_emit_sed_ssm_step_kernel" \
    ".visible .entry sounio_sed_ssm_step"

  append_source_probe \
    "native_ossm_ptx_emitter" \
    "$OSSM_KERNEL" \
    "ossm_forward_backward_associator_f64_f32_surface" \
    "fn ossm_emit_oct_mul_f64" \
    "fn ossm_emit_forward_kernel_f64" \
    "fn ossm_emit_forward_eps_f64" \
    "sounio_ossm_associator_f64" \
    "fn ossm_emit_backward_f64"
}

echo "[native-v2-accel] out=$OUT_DIR"

SOUNIO_NATIVE_V2_F64_LADDER_DIR="$F64_DIR" \
  bash scripts/ci/native_v2_f64_ladder_gate.sh >"$LOG_DIR/native_v2_f64_ladder_gate.log" 2>&1
SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$CPU_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$CPU_MANIFEST" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh >"$LOG_DIR/cpu_oracle_gate.log" 2>&1

OMEGA_GPU_PUBLIC_CONTRACT_OUT="$GPU_PUBLIC_JSON" \
OMEGA_GPU_PUBLIC_CONTRACT_LOG="$GPU_PUBLIC_LOG" \
  bash scripts/omega/omega_gpu_public_contract_gate.sh >"$LOG_DIR/gpu_public_contract_gate.stdout" 2>&1

GPU_SOUC=""
if ! GPU_SOUC="$(sounio_resolve_gpu_souc "$GPU_FIXTURE" "${SOUNIO_GPU_SOUC_BIN:-}" 2>/dev/null)"; then
  reason="$(sounio_gpu_probe_reason)"
  if [[ -z "$reason" ]]; then reason="gpu_backend_unavailable"; fi
  echo "[native-v2-accel] SKIP: $reason"
  python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$reason" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": "sounio.native_v2_epistemic_accel_spine.v1",
    "status": "not_run",
    "reason": sys.argv[3],
    "fallback_path": "none",
    "host_callback": "none",
}, indent=2) + "\n", encoding="utf-8")
PY
  exit 0
fi

append_header
blocked_count=0
fail_count=0
pass_count=0

append_native_algebra_probes

while IFS=$'\t' read -r case_id program_path kind; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi
  if [[ ! -f "$program_path" ]]; then
    append_row "$case_id" "gpu" "$program_path" "fail" "missing_program" ""
    fail_count=$((fail_count + 1))
    continue
  fi

  check_log="$LOG_DIR/$case_id.check.log"
  build_log="$LOG_DIR/$case_id.build.log"
  raw_ptx_path="$PTX_DIR/$case_id.raw.ptx"
  ptx_path="$PTX_DIR/$case_id.ptx"
  legalizer_report="$LOG_DIR/$case_id.f64_legalize.json"

  if ! "$GPU_SOUC" check "$program_path" >"$check_log" 2>&1; then
    append_row "$case_id" "gpu" "$program_path" "fail" "gpu_check_failed" ""
    fail_count=$((fail_count + 1))
    continue
  fi
  if ! "$GPU_SOUC" build "$program_path" --backend gpu -o "$raw_ptx_path" >"$build_log" 2>&1; then
    append_row "$case_id" "gpu" "$program_path" "fail" "gpu_ptx_emit_failed" ""
    fail_count=$((fail_count + 1))
    continue
  fi
  if [[ ! -s "$raw_ptx_path" ]]; then
    append_row "$case_id" "gpu" "$program_path" "fail" "missing_ptx" ""
    fail_count=$((fail_count + 1))
    continue
  fi
  if ! python3 "$PTX_F64_LEGALIZER" "$raw_ptx_path" "$ptx_path" --report "$legalizer_report" >>"$build_log" 2>&1; then
    append_row "$case_id" "gpu" "$program_path" "fail" "gpu_ptx_f64_legalize_failed" ""
    fail_count=$((fail_count + 1))
    continue
  fi
  if [[ ! -s "$ptx_path" ]]; then
    append_row "$case_id" "gpu" "$program_path" "fail" "missing_legalized_ptx" ""
    fail_count=$((fail_count + 1))
    continue
  fi

  analysis="$(analyze_ptx "$ptx_path")"
  status="$(printf '%s\n' "$analysis" | head -n 1 | cut -f1)"
  detail="$(printf '%s\n' "$analysis" | head -n 1 | cut -f2)"
  printf '%s\n' "$analysis" >"$LOG_DIR/$case_id.ptx_analysis.log"
  ptx_sha="$(portable_sha256 "$ptx_path")"
  append_row "$case_id" "gpu" "$program_path" "$status" "$detail" "$ptx_sha"
  if [[ "$status" == "pass" ]]; then
    pass_count=$((pass_count + 1))
  elif [[ "$status" == "blocked" ]]; then
    blocked_count=$((blocked_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
done <"$GPU_MANIFEST"

runtime_status="not_run"
runtime_reason="cuda_runtime_not_attempted"
runtime_log="$LOG_DIR/native_cuda_smoke.log"
runtime_souc="${SOUNIO_CUDA_RUNTIME_SOUC_BIN:-$ROOT_DIR/bin/souc}"
if [[ "${SOUNIO_NATIVE_V2_ACCEL_RUN_CUDA:-auto}" != "off" ]]; then
  set +e
  SOUC_BIN="$runtime_souc" bash scripts/gpu/run_native_cuda_smoke.sh tests/run-pass/gpu_vec_add.sio >"$runtime_log" 2>&1
  runtime_rc=$?
  set -e
  if grep -q '^SKIP:' "$runtime_log"; then
    runtime_status="not_run"
    runtime_reason="$(grep '^SKIP:' "$runtime_log" | head -n 1)"
  elif [[ "$runtime_rc" -eq 0 ]]; then
    runtime_status="pass"
    runtime_reason="cuda_smoke_ok"
  else
    runtime_status="fail"
    runtime_reason="cuda_smoke_failed"
    fail_count=$((fail_count + 1))
  fi
fi

cpu_summary="$CPU_DIR/artifacts/summary.json"
f64_summary="$F64_DIR/artifacts/summary.json"
python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$cpu_summary" "$f64_summary" "$GPU_PUBLIC_JSON" "$GPU_SOUC" "$runtime_souc" "$runtime_status" "$runtime_reason" "$pass_count" "$blocked_count" "$fail_count" "$OUT_DIR" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
cpu_summary_path = pathlib.Path(sys.argv[3])
f64_summary_path = pathlib.Path(sys.argv[4])
gpu_public_path = pathlib.Path(sys.argv[5])
gpu_souc = sys.argv[6]
runtime_souc = sys.argv[7]
runtime_status = sys.argv[8]
runtime_reason = sys.argv[9]
pass_count = int(sys.argv[10])
blocked_count = int(sys.argv[11])
fail_count = int(sys.argv[12])
out_dir = pathlib.Path(sys.argv[13])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

rows = []
with open(results_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

status = "pass"
if fail_count:
    status = "fail"
elif blocked_count:
    status = "blocked"
elif runtime_status == "not_run":
    status = "partial"

payload = {
    "schema": "sounio.native_v2_epistemic_accel_spine.v1",
    "status": status,
    "target": "x86_64-linux+cuda-ptx",
    "fallback_path": "none",
    "host_callback": "none",
    "gpu_souc_path": gpu_souc,
    "cpu_oracle_summary": str(cpu_summary_path),
    "cpu_oracle_summary_sha256": sha256(cpu_summary_path) if cpu_summary_path.exists() else "",
    "f64_ladder_summary": str(f64_summary_path),
    "gpu_public_contract_json": str(gpu_public_path),
    "gpu_public_contract_sha256": sha256(gpu_public_path) if gpu_public_path.exists() else "",
    "ptx_f64_legalizer": "scripts/gpu/ptx_f64_legalize.py",
    "runtime_souc_path": runtime_souc,
    "runtime_status": runtime_status,
    "runtime_reason": runtime_reason,
    "pass_count": pass_count,
    "blocked_count": blocked_count,
    "fail_count": fail_count,
    "results_tsv": str(results_path),
    "artifact_dir": str(out_dir),
    "cases": rows,
    "remaining_boundaries": [
        "pinned GPU artifact still needs a source rebuild; this gate legalizes public PTX f64 dataflow after emission" if not blocked_count else "pinned GPU artifact still lowers some f64 arithmetic through f32 PTX ops",
        "native CUDA runtime proof is environment-dependent unless CUDA driver is visible" if runtime_status == "not_run" else "",
        "this gate does not promote tensor-core performance, ROCm runtime, Metal, WebGPU, or DDC",
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

echo "[native-v2-accel] status=$summary_status summary=$SUMMARY_JSON"
if [[ "$summary_status" == "fail" ]]; then
  exit 1
fi
