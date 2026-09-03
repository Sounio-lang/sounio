#!/usr/bin/env bash
# l4_optimized_no_rust_bench.sh
# L4-first benchmark harness with mandatory value-only + strict + fast variants.
# Emits v2 evidence JSON with per-dimension medians and gate status.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/artifacts/omega/l4_runs}"
TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
REPORT_PATH="${REPORT_PATH:-$REPORT_DIR/l4_optimized_no_rust_report.${TIMESTAMP_UTC}.v2.json}"
LATEST_PATH="${LATEST_PATH:-$REPORT_DIR/l4_optimized_no_rust_report.latest.v2.json}"

GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"

GEMM_DIMS="${GEMM_DIMS:-2048,4096,8192}"
GEMM_ITERS="${GEMM_ITERS:-10}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
REPEATS="${REPEATS:-3}"

# Enforce protocol floor from the plan.
if [ "${GEMM_ITERS}" -lt 10 ]; then
  GEMM_ITERS=10
fi
if [ "${WARMUP_RUNS}" -lt 2 ]; then
  WARMUP_RUNS=2
fi
if [ "${REPEATS}" -lt 3 ]; then
  REPEATS=3
fi

TARGET_DIM="${TARGET_DIM:-4096}"
TARGET_STRICT_OVERHEAD_MAX="${TARGET_STRICT_OVERHEAD_MAX:-3.0}"
TARGET_FAST_SPEEDUP_MIN="${TARGET_FAST_SPEEDUP_MIN:-1.05}"
TARGET_FAST_DRIFT_MAX="${TARGET_FAST_DRIFT_MAX:-0.02}"

VALUE_ONLY_PTX_FILE="${VALUE_ONLY_PTX_FILE:-/tmp/epistemic_gemm_sm7_4096.ptx}"
SHADOW_STRICT_PTX_FILE="${SHADOW_STRICT_PTX_FILE:-/tmp/epistemic_tensor_core_shadow_strict_sm89.ptx}"
SHADOW_FAST_PTX_FILE="${SHADOW_FAST_PTX_FILE:-/tmp/epistemic_tensor_core_shadow_fast_sm89.ptx}"

VALUE_ONLY_MATH_MODE="${VALUE_ONLY_MATH_MODE:-fp32}"
SHADOW_STRICT_MATH_MODE="${SHADOW_STRICT_MATH_MODE:-bf16_accum_f32}"
SHADOW_FAST_MATH_MODE="${SHADOW_FAST_MATH_MODE:-bf16_accum_f32}"

VALUE_ONLY_PIPELINE_MODE="${VALUE_ONLY_PIPELINE_MODE:-cp_async_2stage}"
SHADOW_STRICT_PIPELINE_MODE="${SHADOW_STRICT_PIPELINE_MODE:-cp_async_3stage}"
SHADOW_FAST_PIPELINE_MODE="${SHADOW_FAST_PIPELINE_MODE:-cp_async_3stage}"

VALUE_ONLY_TILE_CONFIG="${VALUE_ONLY_TILE_CONFIG:-128x128x32}"
SHADOW_STRICT_TILE_CONFIG="${SHADOW_STRICT_TILE_CONFIG:-128x128x32}"
SHADOW_FAST_TILE_CONFIG="${SHADOW_FAST_TILE_CONFIG:-128x128x32}"

VALUE_ONLY_STAGES="${VALUE_ONLY_STAGES:-2}"
SHADOW_STRICT_STAGES="${SHADOW_STRICT_STAGES:-3}"
SHADOW_FAST_STAGES="${SHADOW_FAST_STAGES:-3}"

VALUE_ONLY_WARPS="${VALUE_ONLY_WARPS:-4}"
SHADOW_STRICT_WARPS="${SHADOW_STRICT_WARPS:-8}"
SHADOW_FAST_WARPS="${SHADOW_FAST_WARPS:-8}"

VALUE_ONLY_REGS_PER_THREAD="${VALUE_ONLY_REGS_PER_THREAD:-0}"
SHADOW_STRICT_REGS_PER_THREAD="${SHADOW_STRICT_REGS_PER_THREAD:-0}"
SHADOW_FAST_REGS_PER_THREAD="${SHADOW_FAST_REGS_PER_THREAD:-0}"

VALUE_ONLY_SMEM_BYTES="${VALUE_ONLY_SMEM_BYTES:-0}"
SHADOW_STRICT_SMEM_BYTES="${SHADOW_STRICT_SMEM_BYTES:-0}"
SHADOW_FAST_SMEM_BYTES="${SHADOW_FAST_SMEM_BYTES:-0}"

SHADOW_STRICT_PROVENANCE_FOLD_INTERVAL="${SHADOW_STRICT_PROVENANCE_FOLD_INTERVAL:-1}"
SHADOW_FAST_PROVENANCE_FOLD_INTERVAL="${SHADOW_FAST_PROVENANCE_FOLD_INTERVAL:-16}"
VALUE_ONLY_PROVENANCE_FOLD_INTERVAL="${VALUE_ONLY_PROVENANCE_FOLD_INTERVAL:-0}"

SHADOW_STRICT_VALIDITY_PACK_MODE="${SHADOW_STRICT_VALIDITY_PACK_MODE:-full}"
SHADOW_FAST_VALIDITY_PACK_MODE="${SHADOW_FAST_VALIDITY_PACK_MODE:-packed}"
VALUE_ONLY_VALIDITY_PACK_MODE="${VALUE_ONLY_VALIDITY_PACK_MODE:-none}"

# Native shadow generator tuning controls (forwarded to generate_l4_shadow_variants.py).
SHADOW_STRICT_SQRT_INTERVAL="${SHADOW_STRICT_SQRT_INTERVAL:-2}"
SHADOW_STRICT_PROV_INTERVAL="${SHADOW_STRICT_PROV_INTERVAL:-1}"
SHADOW_STRICT_FEEDBACK_INTERVAL="${SHADOW_STRICT_FEEDBACK_INTERVAL:-32}"
SHADOW_FAST_PROV_INTERVAL="${SHADOW_FAST_PROV_INTERVAL:-64}"
SHADOW_FAST_FEEDBACK_INTERVAL="${SHADOW_FAST_FEEDBACK_INTERVAL:-0}"
SHADOW_STRICT_SLOT_CAP="${SHADOW_STRICT_SLOT_CAP:-16}"
SHADOW_FAST_SLOT_CAP="${SHADOW_FAST_SLOT_CAP:-8}"

ENABLE_NCU="${ENABLE_NCU:-1}"
NCU_DIM="${NCU_DIM:-$TARGET_DIM}"
NCU_ITERS="${NCU_ITERS:-5}"
NCU_REPORT_PATH="${NCU_REPORT_PATH:-$REPORT_DIR/l4_ncu_metrics.${TIMESTAMP_UTC}.v1.json}"

# Auto-generation fallback for strict/fast PTX variants.
AUTO_GENERATE_SHADOW_VARIANTS="${AUTO_GENERATE_SHADOW_VARIANTS:-1}"
ALLOW_LEGACY_SHADOW_AUGMENT_FALLBACK="${ALLOW_LEGACY_SHADOW_AUGMENT_FALLBACK:-1}"
REQUIRE_NATIVE_SHADOW_VARIANTS="${REQUIRE_NATIVE_SHADOW_VARIANTS:-0}"
SHADOW_VARIANT_GENERATOR_PREFERENCE="${SHADOW_VARIANT_GENERATOR_PREFERENCE:-native}"
FORCE_REGENERATE_SHADOW_VARIANTS="${FORCE_REGENERATE_SHADOW_VARIANTS:-0}"

mkdir -p "$REPORT_DIR"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=no
)

run_remote() {
  local cmd="$1"
  ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" "$cmd"
}

copy_script_remote() {
  local src="$1"
  local dst="$2"
  scp "${SSH_OPTS[@]}" "$src" "${GPU_USER}@${GPU_HOST}:${dst}" >/dev/null
}

sync_remote_gpu_scripts() {
  run_remote "mkdir -p ${REMOTE_DIR}/scripts/gpu" >/dev/null
  copy_script_remote "$ROOT_DIR/scripts/gpu/cuda_gemm_dispatch.py" "${REMOTE_DIR}/scripts/gpu/cuda_gemm_dispatch.py"
  copy_script_remote "$ROOT_DIR/scripts/gpu/cuda_cublas_baseline.py" "${REMOTE_DIR}/scripts/gpu/cuda_cublas_baseline.py"
}

SHADOW_STRICT_VARIANT_SOURCE="prebuilt"
SHADOW_FAST_VARIANT_SOURCE="prebuilt"

variant_source_for_remote_file() {
  local path="$1"
  if ! run_remote "test -f ${path}" >/dev/null 2>&1; then
    echo "missing"
    return
  fi
  if run_remote "head -n 1 ${path} | grep -q 'native_shadow_variant'" >/dev/null 2>&1; then
    echo "native_shadow_emitter"
    return
  fi
  if run_remote "head -n 1 ${path} | grep -q 'legacy_shadow_augmentation'" >/dev/null 2>&1; then
    echo "legacy_shadow_augment"
    return
  fi
  echo "prebuilt"
}

ensure_shadow_variants() {
  if [ "$AUTO_GENERATE_SHADOW_VARIANTS" != "1" ]; then
    return
  fi

  local need_strict=0
  local need_fast=0
  SHADOW_STRICT_VARIANT_SOURCE="$(variant_source_for_remote_file "$SHADOW_STRICT_PTX_FILE")"
  SHADOW_FAST_VARIANT_SOURCE="$(variant_source_for_remote_file "$SHADOW_FAST_PTX_FILE")"
  if [ "$SHADOW_STRICT_VARIANT_SOURCE" = "missing" ]; then
    need_strict=1
  fi
  if [ "$SHADOW_FAST_VARIANT_SOURCE" = "missing" ]; then
    need_fast=1
  fi
  if [ "$SHADOW_VARIANT_GENERATOR_PREFERENCE" = "native" ]; then
    if [ "$SHADOW_STRICT_VARIANT_SOURCE" = "legacy_shadow_augment" ]; then
      need_strict=1
    fi
    if [ "$SHADOW_FAST_VARIANT_SOURCE" = "legacy_shadow_augment" ]; then
      need_fast=1
    fi
  fi
  if [ "$FORCE_REGENERATE_SHADOW_VARIANTS" = "1" ]; then
    need_strict=1
    need_fast=1
  fi

  if [ "$need_strict" -eq 0 ] && [ "$need_fast" -eq 0 ]; then
    return
  fi

  echo "[shadow] strict/fast PTX regeneration requested (preference=${SHADOW_VARIANT_GENERATOR_PREFERENCE})"
  if ! run_remote "test -f ${VALUE_ONLY_PTX_FILE}" >/dev/null 2>&1; then
    echo "[shadow] baseline PTX missing: ${VALUE_ONLY_PTX_FILE}"
    return
  fi

  run_remote "mkdir -p ${REMOTE_DIR}/scripts/gpu" >/dev/null
  # Keep helpers in sync with local repo so remote runs don't depend on stale copies.
  copy_script_remote "$ROOT_DIR/scripts/gpu/generate_l4_shadow_variants.py" "${REMOTE_DIR}/scripts/gpu/generate_l4_shadow_variants.py"
  copy_script_remote "$ROOT_DIR/scripts/gpu/ptx_shadow_augment.py" "${REMOTE_DIR}/scripts/gpu/ptx_shadow_augment.py"

  if [ "$need_strict" -eq 1 ]; then
    if run_remote "cd ${REMOTE_DIR} && python3 scripts/gpu/generate_l4_shadow_variants.py --mode strict --strict-sqrt-interval ${SHADOW_STRICT_SQRT_INTERVAL} --strict-prov-interval ${SHADOW_STRICT_PROV_INTERVAL} --strict-feedback-interval ${SHADOW_STRICT_FEEDBACK_INTERVAL} --fast-prov-interval ${SHADOW_FAST_PROV_INTERVAL} --fast-feedback-interval ${SHADOW_FAST_FEEDBACK_INTERVAL} --strict-slot-cap ${SHADOW_STRICT_SLOT_CAP} --fast-slot-cap ${SHADOW_FAST_SLOT_CAP} ${VALUE_ONLY_PTX_FILE} ${SHADOW_STRICT_PTX_FILE}" >/dev/null 2>&1; then
      SHADOW_STRICT_VARIANT_SOURCE="native_shadow_emitter"
    elif [ "$ALLOW_LEGACY_SHADOW_AUGMENT_FALLBACK" = "1" ] && run_remote "cd ${REMOTE_DIR} && python3 scripts/gpu/ptx_shadow_augment.py --mode strict ${VALUE_ONLY_PTX_FILE} ${SHADOW_STRICT_PTX_FILE}" >/dev/null 2>&1; then
      SHADOW_STRICT_VARIANT_SOURCE="legacy_shadow_augment"
    else
      SHADOW_STRICT_VARIANT_SOURCE="missing"
    fi
  fi
  if [ "$need_fast" -eq 1 ]; then
    if run_remote "cd ${REMOTE_DIR} && python3 scripts/gpu/generate_l4_shadow_variants.py --mode fast --strict-sqrt-interval ${SHADOW_STRICT_SQRT_INTERVAL} --strict-prov-interval ${SHADOW_STRICT_PROV_INTERVAL} --strict-feedback-interval ${SHADOW_STRICT_FEEDBACK_INTERVAL} --fast-prov-interval ${SHADOW_FAST_PROV_INTERVAL} --fast-feedback-interval ${SHADOW_FAST_FEEDBACK_INTERVAL} --strict-slot-cap ${SHADOW_STRICT_SLOT_CAP} --fast-slot-cap ${SHADOW_FAST_SLOT_CAP} ${VALUE_ONLY_PTX_FILE} ${SHADOW_FAST_PTX_FILE}" >/dev/null 2>&1; then
      SHADOW_FAST_VARIANT_SOURCE="native_shadow_emitter"
    elif [ "$ALLOW_LEGACY_SHADOW_AUGMENT_FALLBACK" = "1" ] && run_remote "cd ${REMOTE_DIR} && python3 scripts/gpu/ptx_shadow_augment.py --mode fast ${VALUE_ONLY_PTX_FILE} ${SHADOW_FAST_PTX_FILE}" >/dev/null 2>&1; then
      SHADOW_FAST_VARIANT_SOURCE="legacy_shadow_augment"
    else
      SHADOW_FAST_VARIANT_SOURCE="missing"
    fi
  fi
}

extract_first_gflops() {
  local log_path="$1"
  grep -E "GFLOPS:" "$log_path" | head -1 | awk '{print $2}' || true
}

extract_result0() {
  local log_path="$1"
  grep -E "result\[(0|0,0)\]:" "$log_path" | head -1 | awk '{print $2}' || true
}

quote_json() {
  # Tiny helper for inline JSON-safe string via python json encoder.
  local raw="$1"
  python3 - "$raw" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

run_cublas_once() {
  local dim="$1"
  local log_path="$2"
  if run_remote "cd ${REMOTE_DIR} && GEMM_DIMS=${dim} GEMM_ITERS=${GEMM_ITERS} python3 scripts/gpu/cuda_cublas_baseline.py" >"$log_path" 2>&1; then
    local gflops
    gflops="$(extract_first_gflops "$log_path")"
    local r0
    r0="$(extract_result0 "$log_path")"
    if [ -n "$gflops" ]; then
      echo "PASS,$gflops,$r0"
      return
    fi
  fi
  echo "FAIL,,"
}

run_variant_once() {
  local dim="$1"
  local ptx_file="$2"
  local log_path="$3"
  if run_remote "cd ${REMOTE_DIR} && GEMM_M=${dim} GEMM_N=${dim} GEMM_K=${dim} GEMM_ITERS=${GEMM_ITERS} GEMM_PTX_FILE=${ptx_file} python3 scripts/gpu/cuda_gemm_dispatch.py" >"$log_path" 2>&1; then
    local gflops
    gflops="$(extract_first_gflops "$log_path")"
    local r0
    r0="$(extract_result0 "$log_path")"
    if [ -n "$gflops" ]; then
      echo "PASS,$gflops,$r0"
      return
    fi
  fi
  echo "FAIL,,"
}

trim() {
  local x="$1"
  x="${x#${x%%[![:space:]]*}}"
  x="${x%${x##*[![:space:]]}}"
  printf '%s' "$x"
}

IFS=',' read -r -a DIMS_RAW <<<"$GEMM_DIMS"
DIMS=()
for d in "${DIMS_RAW[@]}"; do
  td="$(trim "$d")"
  if [ -n "$td" ]; then
    DIMS+=("$td")
  fi
done
if [ "${#DIMS[@]}" -eq 0 ]; then
  DIMS=(2048 4096 8192)
fi

REMOTE_GPU_NAME="$(run_remote "nvidia-smi --query-gpu=name --format=csv,noheader | head -1" 2>/dev/null || true)"
if [ -z "$REMOTE_GPU_NAME" ]; then
  REMOTE_GPU_NAME="UNKNOWN"
fi

# CUDA toolchain provenance (ptxas/nvcc/driver) of the host that produced these
# GFLOPS. ptxas codegen + cuBLAS baselines change across toolkit versions, so a
# measured number is only comparable when the toolchain is recorded alongside it.
OMEGA_GPU_TOOLCHAIN_JSON="$(SSH_OPTS="${SSH_OPTS[*]}" "$ROOT_DIR/scripts/omega/omega_capture_toolchain.sh" "${GPU_USER}@${GPU_HOST}" 2>/dev/null || true)"
export OMEGA_GPU_TOOLCHAIN_JSON

SAMPLES_CSV="$(mktemp "${TMPDIR:-/tmp}/l4_bench_samples.${TIMESTAMP_UTC}.XXXXXX.csv")"
cleanup() {
  rm -f "$SAMPLES_CSV"
}
trap cleanup EXIT
printf 'variant,dim,repeat,status,gflops,result0,log\n' > "$SAMPLES_CSV"

echo "======================================================================="
echo "  L4 Optimized No-Rust Benchmark (v2)"
echo "  Host       : ${GPU_USER}@${GPU_HOST}"
echo "  Remote dir : ${REMOTE_DIR}"
echo "  Dims       : ${DIMS[*]}"
echo "  Protocol   : warmup=${WARMUP_RUNS} iters=${GEMM_ITERS} repeats=${REPEATS}"
echo "  Variants   : value_only_baseline, shadow_strict, shadow_fast"
echo "======================================================================="

declare -A VARIANT_PTX
declare -A VARIANT_MATH
declare -A VARIANT_PIPELINE
declare -A VARIANT_TILE
declare -A VARIANT_STAGES
declare -A VARIANT_WARPS
declare -A VARIANT_REGS
declare -A VARIANT_SMEM
declare -A VARIANT_EPI
declare -A VARIANT_PROV_FOLD
declare -A VARIANT_VALIDITY

VARIANT_PTX[value_only_baseline]="$VALUE_ONLY_PTX_FILE"
VARIANT_PTX[shadow_strict]="$SHADOW_STRICT_PTX_FILE"
VARIANT_PTX[shadow_fast]="$SHADOW_FAST_PTX_FILE"

VARIANT_MATH[value_only_baseline]="$VALUE_ONLY_MATH_MODE"
VARIANT_MATH[shadow_strict]="$SHADOW_STRICT_MATH_MODE"
VARIANT_MATH[shadow_fast]="$SHADOW_FAST_MATH_MODE"

VARIANT_PIPELINE[value_only_baseline]="$VALUE_ONLY_PIPELINE_MODE"
VARIANT_PIPELINE[shadow_strict]="$SHADOW_STRICT_PIPELINE_MODE"
VARIANT_PIPELINE[shadow_fast]="$SHADOW_FAST_PIPELINE_MODE"

VARIANT_TILE[value_only_baseline]="$VALUE_ONLY_TILE_CONFIG"
VARIANT_TILE[shadow_strict]="$SHADOW_STRICT_TILE_CONFIG"
VARIANT_TILE[shadow_fast]="$SHADOW_FAST_TILE_CONFIG"

VARIANT_STAGES[value_only_baseline]="$VALUE_ONLY_STAGES"
VARIANT_STAGES[shadow_strict]="$SHADOW_STRICT_STAGES"
VARIANT_STAGES[shadow_fast]="$SHADOW_FAST_STAGES"

VARIANT_WARPS[value_only_baseline]="$VALUE_ONLY_WARPS"
VARIANT_WARPS[shadow_strict]="$SHADOW_STRICT_WARPS"
VARIANT_WARPS[shadow_fast]="$SHADOW_FAST_WARPS"

VARIANT_REGS[value_only_baseline]="$VALUE_ONLY_REGS_PER_THREAD"
VARIANT_REGS[shadow_strict]="$SHADOW_STRICT_REGS_PER_THREAD"
VARIANT_REGS[shadow_fast]="$SHADOW_FAST_REGS_PER_THREAD"

VARIANT_SMEM[value_only_baseline]="$VALUE_ONLY_SMEM_BYTES"
VARIANT_SMEM[shadow_strict]="$SHADOW_STRICT_SMEM_BYTES"
VARIANT_SMEM[shadow_fast]="$SHADOW_FAST_SMEM_BYTES"

VARIANT_EPI[value_only_baseline]="disabled"
VARIANT_EPI[shadow_strict]="strict"
VARIANT_EPI[shadow_fast]="fast"

VARIANT_PROV_FOLD[value_only_baseline]="$VALUE_ONLY_PROVENANCE_FOLD_INTERVAL"
VARIANT_PROV_FOLD[shadow_strict]="$SHADOW_STRICT_PROVENANCE_FOLD_INTERVAL"
VARIANT_PROV_FOLD[shadow_fast]="$SHADOW_FAST_PROVENANCE_FOLD_INTERVAL"

VARIANT_VALIDITY[value_only_baseline]="$VALUE_ONLY_VALIDITY_PACK_MODE"
VARIANT_VALIDITY[shadow_strict]="$SHADOW_STRICT_VALIDITY_PACK_MODE"
VARIANT_VALIDITY[shadow_fast]="$SHADOW_FAST_VALIDITY_PACK_MODE"

ensure_shadow_variants
sync_remote_gpu_scripts

for dim in "${DIMS[@]}"; do
  echo ""
  echo "[dim ${dim}] cuBLAS baseline"

  warm=1
  while [ "$warm" -le "$WARMUP_RUNS" ]; do
    run_remote "cd ${REMOTE_DIR} && GEMM_DIMS=${dim} GEMM_ITERS=${GEMM_ITERS} python3 scripts/gpu/cuda_cublas_baseline.py" >/dev/null 2>&1 || true
    warm=$((warm + 1))
  done

  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    log_path="$REPORT_DIR/l4_optimized_no_rust.${TIMESTAMP_UTC}.cublas_baseline.d${dim}.r${rep}.log"
    result="$(run_cublas_once "$dim" "$log_path")"
    IFS=',' read -r status gflops r0 <<<"$result"
    printf 'cublas_baseline,%s,%s,%s,%s,%s,%s\n' "$dim" "$rep" "$status" "$gflops" "$r0" "$log_path" >> "$SAMPLES_CSV"
    echo "  rep=${rep} status=${status} gflops=${gflops:-NA}"
    rep=$((rep + 1))
  done

  for variant in value_only_baseline shadow_strict shadow_fast; do
    ptx_file="${VARIANT_PTX[$variant]}"
    echo ""
    echo "[dim ${dim}] ${variant}"

    if ! run_remote "test -f ${ptx_file}" >/dev/null 2>&1; then
      log_path="$REPORT_DIR/l4_optimized_no_rust.${TIMESTAMP_UTC}.${variant}.d${dim}.missing.log"
      printf 'missing PTX: %s\n' "$ptx_file" > "$log_path"
      printf '%s,%s,0,NOT_FOUND,,,%s\n' "$variant" "$dim" "$log_path" >> "$SAMPLES_CSV"
      echo "  status=NOT_FOUND ptx=${ptx_file}"
      continue
    fi

    warm=1
    while [ "$warm" -le "$WARMUP_RUNS" ]; do
      run_remote "cd ${REMOTE_DIR} && GEMM_M=${dim} GEMM_N=${dim} GEMM_K=${dim} GEMM_ITERS=${GEMM_ITERS} GEMM_PTX_FILE=${ptx_file} python3 scripts/gpu/cuda_gemm_dispatch.py" >/dev/null 2>&1 || true
      warm=$((warm + 1))
    done

    rep=1
    while [ "$rep" -le "$REPEATS" ]; do
      log_path="$REPORT_DIR/l4_optimized_no_rust.${TIMESTAMP_UTC}.${variant}.d${dim}.r${rep}.log"
      result="$(run_variant_once "$dim" "$ptx_file" "$log_path")"
      IFS=',' read -r status gflops r0 <<<"$result"
      printf '%s,%s,%s,%s,%s,%s,%s\n' "$variant" "$dim" "$rep" "$status" "$gflops" "$r0" "$log_path" >> "$SAMPLES_CSV"
      echo "  rep=${rep} status=${status} gflops=${gflops:-NA}"
      rep=$((rep + 1))
    done
  done

done

ncu_status="NOT RUN"
if [ "$ENABLE_NCU" = "1" ]; then
  set +e
  GPU_HOST="$GPU_HOST" \
  GPU_USER="$GPU_USER" \
  REMOTE_DIR="$REMOTE_DIR" \
  NCU_DIM="$NCU_DIM" \
  NCU_ITERS="$NCU_ITERS" \
  VALUE_ONLY_PTX_FILE="$VALUE_ONLY_PTX_FILE" \
  SHADOW_STRICT_PTX_FILE="$SHADOW_STRICT_PTX_FILE" \
  SHADOW_FAST_PTX_FILE="$SHADOW_FAST_PTX_FILE" \
  REPORT_PATH="$NCU_REPORT_PATH" \
  bash "$ROOT_DIR/scripts/gpu/gpu_ncu_l4_collect.sh"
  rc_ncu=$?
  set -e
  if [ "$rc_ncu" -eq 0 ] && [ -f "$NCU_REPORT_PATH" ]; then
    ncu_status="PASS"
  else
    ncu_status="FAIL"
  fi
fi

export REPEATS TARGET_DIM TARGET_STRICT_OVERHEAD_MAX TARGET_FAST_SPEEDUP_MIN TARGET_FAST_DRIFT_MAX
export GEMM_DIMS WARMUP_RUNS GEMM_ITERS
export GPU_HOST REMOTE_DIR REMOTE_GPU_NAME
export VALUE_ONLY_PTX_FILE SHADOW_STRICT_PTX_FILE SHADOW_FAST_PTX_FILE
export VALUE_ONLY_MATH_MODE SHADOW_STRICT_MATH_MODE SHADOW_FAST_MATH_MODE
export VALUE_ONLY_PIPELINE_MODE SHADOW_STRICT_PIPELINE_MODE SHADOW_FAST_PIPELINE_MODE
export VALUE_ONLY_TILE_CONFIG SHADOW_STRICT_TILE_CONFIG SHADOW_FAST_TILE_CONFIG
export VALUE_ONLY_STAGES SHADOW_STRICT_STAGES SHADOW_FAST_STAGES
export VALUE_ONLY_WARPS SHADOW_STRICT_WARPS SHADOW_FAST_WARPS
export VALUE_ONLY_REGS_PER_THREAD SHADOW_STRICT_REGS_PER_THREAD SHADOW_FAST_REGS_PER_THREAD
export VALUE_ONLY_SMEM_BYTES SHADOW_STRICT_SMEM_BYTES SHADOW_FAST_SMEM_BYTES
export SHADOW_STRICT_VARIANT_SOURCE SHADOW_FAST_VARIANT_SOURCE
export VALUE_ONLY_PROVENANCE_FOLD_INTERVAL SHADOW_STRICT_PROVENANCE_FOLD_INTERVAL SHADOW_FAST_PROVENANCE_FOLD_INTERVAL
export VALUE_ONLY_VALIDITY_PACK_MODE SHADOW_STRICT_VALIDITY_PACK_MODE SHADOW_FAST_VALIDITY_PACK_MODE
export SHADOW_STRICT_SQRT_INTERVAL SHADOW_STRICT_PROV_INTERVAL SHADOW_STRICT_FEEDBACK_INTERVAL
export SHADOW_FAST_PROV_INTERVAL SHADOW_FAST_FEEDBACK_INTERVAL
export SHADOW_STRICT_SLOT_CAP SHADOW_FAST_SLOT_CAP
export NCU_REPORT_PATH
export NCU_COLLECTION_STATUS="$ncu_status"
export REQUIRE_NATIVE_SHADOW_VARIANTS

python3 - "$SAMPLES_CSV" "$REPORT_PATH" <<'PY'
import csv
import json
import os
import math
import statistics
import sys
from datetime import datetime, timezone

samples_path = sys.argv[1]
report_path = sys.argv[2]

repeats = int(os.environ["REPEATS"])
target_dim = int(os.environ["TARGET_DIM"])
target_strict_overhead_max = float(os.environ["TARGET_STRICT_OVERHEAD_MAX"])
target_fast_speedup_min = float(os.environ["TARGET_FAST_SPEEDUP_MIN"])
target_fast_drift_max = float(os.environ["TARGET_FAST_DRIFT_MAX"])

dims = [int(x.strip()) for x in os.environ["GEMM_DIMS"].split(",") if x.strip()]
if not dims:
    dims = [2048, 4096, 8192]

variants = ["cublas_baseline", "value_only_baseline", "shadow_strict", "shadow_fast"]

variant_meta = {
    "value_only_baseline": {
        "kernel_variant": "value_only_baseline",
        "math_mode": os.environ["VALUE_ONLY_MATH_MODE"],
        "epistemic_mode": "disabled",
        "pipeline_mode": os.environ["VALUE_ONLY_PIPELINE_MODE"],
        "tile_config": os.environ["VALUE_ONLY_TILE_CONFIG"],
        "stages": int(os.environ["VALUE_ONLY_STAGES"]),
        "warps": int(os.environ["VALUE_ONLY_WARPS"]),
        "regs_per_thread": int(os.environ["VALUE_ONLY_REGS_PER_THREAD"]),
        "smem_bytes": int(os.environ["VALUE_ONLY_SMEM_BYTES"]),
        "provenance_fold_interval": int(os.environ["VALUE_ONLY_PROVENANCE_FOLD_INTERVAL"]),
        "validity_pack_mode": os.environ["VALUE_ONLY_VALIDITY_PACK_MODE"],
    },
    "shadow_strict": {
        "kernel_variant": "shadow_strict",
        "variant_generation": os.environ.get("SHADOW_STRICT_VARIANT_SOURCE", "prebuilt"),
        "math_mode": os.environ["SHADOW_STRICT_MATH_MODE"],
        "epistemic_mode": "strict",
        "pipeline_mode": os.environ["SHADOW_STRICT_PIPELINE_MODE"],
        "tile_config": os.environ["SHADOW_STRICT_TILE_CONFIG"],
        "stages": int(os.environ["SHADOW_STRICT_STAGES"]),
        "warps": int(os.environ["SHADOW_STRICT_WARPS"]),
        "regs_per_thread": int(os.environ["SHADOW_STRICT_REGS_PER_THREAD"]),
        "smem_bytes": int(os.environ["SHADOW_STRICT_SMEM_BYTES"]),
        "provenance_fold_interval": int(os.environ["SHADOW_STRICT_PROVENANCE_FOLD_INTERVAL"]),
        "validity_pack_mode": os.environ["SHADOW_STRICT_VALIDITY_PACK_MODE"],
        "shadow_tuning": {
            "sqrt_interval": int(os.environ["SHADOW_STRICT_SQRT_INTERVAL"]),
            "prov_interval": int(os.environ["SHADOW_STRICT_PROV_INTERVAL"]),
            "feedback_interval": int(os.environ["SHADOW_STRICT_FEEDBACK_INTERVAL"]),
            "slot_cap": int(os.environ["SHADOW_STRICT_SLOT_CAP"]),
        },
    },
    "shadow_fast": {
        "kernel_variant": "shadow_fast",
        "variant_generation": os.environ.get("SHADOW_FAST_VARIANT_SOURCE", "prebuilt"),
        "math_mode": os.environ["SHADOW_FAST_MATH_MODE"],
        "epistemic_mode": "fast",
        "pipeline_mode": os.environ["SHADOW_FAST_PIPELINE_MODE"],
        "tile_config": os.environ["SHADOW_FAST_TILE_CONFIG"],
        "stages": int(os.environ["SHADOW_FAST_STAGES"]),
        "warps": int(os.environ["SHADOW_FAST_WARPS"]),
        "regs_per_thread": int(os.environ["SHADOW_FAST_REGS_PER_THREAD"]),
        "smem_bytes": int(os.environ["SHADOW_FAST_SMEM_BYTES"]),
        "provenance_fold_interval": int(os.environ["SHADOW_FAST_PROVENANCE_FOLD_INTERVAL"]),
        "validity_pack_mode": os.environ["SHADOW_FAST_VALIDITY_PACK_MODE"],
        "shadow_tuning": {
            "prov_interval": int(os.environ["SHADOW_FAST_PROV_INTERVAL"]),
            "feedback_interval": int(os.environ["SHADOW_FAST_FEEDBACK_INTERVAL"]),
            "slot_cap": int(os.environ["SHADOW_FAST_SLOT_CAP"]),
        },
    },
}

rows = []
with open(samples_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

by_key = {}
for row in rows:
    dim = int(row["dim"])
    key = (dim, row["variant"])
    by_key.setdefault(key, []).append(row)


def _to_float(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except ValueError:
        return None


def summarize_variant(dim: int, variant: str):
    recs = by_key.get((dim, variant), [])
    statuses = [r["status"] for r in recs]
    pass_recs = [r for r in recs if r["status"] == "PASS" and _to_float(r["gflops"]) is not None]
    gflops_vals = [_to_float(r["gflops"]) for r in pass_recs]
    gflops_vals = [x for x in gflops_vals if x is not None]
    result_vals = [_to_float(r["result0"]) for r in pass_recs]
    result_vals = [x for x in result_vals if x is not None]

    status = "NOT RUN"
    if any(s == "NOT_FOUND" for s in statuses):
        status = "NOT_FOUND"
    elif len(pass_recs) >= repeats:
        status = "PASS"
    elif any(s == "FAIL" for s in statuses):
        status = "FAIL"
    elif recs:
        status = recs[0]["status"]

    return {
        "status": status,
        "samples": len(pass_recs),
        "expected_samples": repeats,
        "gflops_median": statistics.median(gflops_vals) if gflops_vals else None,
        "gflops_samples": gflops_vals,
        "result0_median": statistics.median(result_vals) if result_vals else None,
        "runs": [
            {
                "repeat": int(r["repeat"]),
                "status": r["status"],
                "gflops": _to_float(r["gflops"]),
                "result0": _to_float(r["result0"]),
                "log": r["log"],
            }
            for r in recs
        ],
    }


def ratio_or_none(num, den):
    if num is None or den is None:
        return None
    if num <= 0 or den <= 0:
        return None
    return num / den


def drift_against_strict(dim: int):
    strict_rows = [r for r in by_key.get((dim, "shadow_strict"), []) if r["status"] == "PASS"]
    fast_rows = [r for r in by_key.get((dim, "shadow_fast"), []) if r["status"] == "PASS"]
    strict_by_rep = {
        int(r["repeat"]): _to_float(r["result0"]) for r in strict_rows if _to_float(r["result0"]) is not None
    }
    fast_by_rep = {
        int(r["repeat"]): _to_float(r["result0"]) for r in fast_rows if _to_float(r["result0"]) is not None
    }
    shared = sorted(set(strict_by_rep.keys()) & set(fast_by_rep.keys()))
    drifts = []
    for rep in shared:
        s = strict_by_rep[rep]
        f = fast_by_rep[rep]
        denom = max(abs(s), 1e-12)
        drifts.append(abs(f - s) / denom)
    return {
        "paired_repeats": shared,
        "numerical_drift_vs_strict": drifts,
        "numerical_drift_vs_strict_median": statistics.median(drifts) if drifts else None,
    }


per_dim = {}
for dim in dims:
    cublas = summarize_variant(dim, "cublas_baseline")
    value = summarize_variant(dim, "value_only_baseline")
    strict = summarize_variant(dim, "shadow_strict")
    fast = summarize_variant(dim, "shadow_fast")

    cublas_med = cublas["gflops_median"]
    value_med = value["gflops_median"]
    strict_med = strict["gflops_median"]
    fast_med = fast["gflops_median"]

    ratios = {
        "overhead_x_value_only_vs_cublas": ratio_or_none(cublas_med, value_med),
        "overhead_x_shadow_strict_vs_cublas": ratio_or_none(cublas_med, strict_med),
        "overhead_x_shadow_fast_vs_cublas": ratio_or_none(cublas_med, fast_med),
        "speedup_x_shadow_fast_vs_shadow_strict": ratio_or_none(fast_med, strict_med),
    }

    drift = drift_against_strict(dim)

    per_dim[str(dim)] = {
        "cublas_baseline": cublas,
        "value_only_baseline": value,
        "shadow_strict": strict,
        "shadow_fast": fast,
        "ratios": ratios,
        "drift": drift,
    }

if target_dim not in dims:
    target_dim = 4096 if 4096 in dims else max(dims)

target = per_dim[str(target_dim)]
strict_overhead = target["ratios"]["overhead_x_shadow_strict_vs_cublas"]
fast_speedup = target["ratios"]["speedup_x_shadow_fast_vs_shadow_strict"]
fast_drift = target["drift"]["numerical_drift_vs_strict_median"]

gates = {
    "strict_overhead": {
        "dimension": target_dim,
        "target_max": target_strict_overhead_max,
        "actual": strict_overhead,
        "pass": strict_overhead is not None and strict_overhead <= target_strict_overhead_max,
    },
    "fast_speedup_vs_strict": {
        "dimension": target_dim,
        "target_min": target_fast_speedup_min,
        "actual": fast_speedup,
        "pass": fast_speedup is not None and fast_speedup >= target_fast_speedup_min,
    },
    "fast_drift_vs_strict": {
        "dimension": target_dim,
        "target_max": target_fast_drift_max,
        "actual": fast_drift,
        "pass": fast_drift is not None and fast_drift <= target_fast_drift_max,
    },
}

required_statuses = [
    target["cublas_baseline"]["status"] == "PASS",
    target["value_only_baseline"]["status"] == "PASS",
    target["shadow_strict"]["status"] == "PASS",
    target["shadow_fast"]["status"] == "PASS",
]

overall_pass = all(g["pass"] for g in gates.values()) and all(required_statuses)
overall_status = "PASS" if overall_pass else "BLOCKED"

blocked_reasons = []
if not all(required_statuses):
    blocked_reasons.append("missing_required_variant_at_target_dim")
if os.environ.get("REQUIRE_NATIVE_SHADOW_VARIANTS", "0") == "1":
    if variant_meta["shadow_strict"]["variant_generation"] != "native_shadow_emitter":
        blocked_reasons.append("strict_variant_not_native_shadow_emitter")
    if variant_meta["shadow_fast"]["variant_generation"] != "native_shadow_emitter":
        blocked_reasons.append("fast_variant_not_native_shadow_emitter")
for gate_name, gate in gates.items():
    if not gate["pass"]:
        blocked_reasons.append(f"gate_failed:{gate_name}")

ncu_status = os.environ.get("NCU_COLLECTION_STATUS", "NOT RUN")
ncu_report_path = os.environ.get("NCU_REPORT_PATH", "")
ncu_payload = {
    "status": ncu_status,
    "report_path": ncu_report_path,
}
if ncu_report_path and os.path.exists(ncu_report_path):
    try:
        with open(ncu_report_path) as nf:
            ncu_payload["details"] = json.load(nf)
    except Exception:
        ncu_payload["details"] = {"status": "FAIL", "reason": "invalid_ncu_json"}

try:
    _tc = os.environ.get("OMEGA_GPU_TOOLCHAIN_JSON", "")
    toolchain_obj = json.loads(_tc) if _tc else {"capture_status": "not_captured"}
    if not isinstance(toolchain_obj, dict):
        toolchain_obj = {"capture_status": "not_captured"}
except Exception:
    toolchain_obj = {"capture_status": "not_captured"}

report = {
    "schema": "sounio.benchmark.l4-optimized-no-rust.v2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "overall_status": overall_status,
    "blocked_reasons": blocked_reasons,
    "host": os.environ["GPU_HOST"],
    "gpu": os.environ["REMOTE_GPU_NAME"],
    "remote_dir": os.environ["REMOTE_DIR"],
    "toolchain": toolchain_obj,
    "protocol": {
        "warmup_runs": int(os.environ["WARMUP_RUNS"]),
        "iters": int(os.environ["GEMM_ITERS"]),
        "repeats": repeats,
        "dimensions": dims,
    },
    "variants": variant_meta,
    "ptx_files": {
        "value_only_baseline": os.environ["VALUE_ONLY_PTX_FILE"],
        "shadow_strict": os.environ["SHADOW_STRICT_PTX_FILE"],
        "shadow_fast": os.environ["SHADOW_FAST_PTX_FILE"],
    },
    "results": {
        "by_dimension": per_dim,
    },
    "targets": {
        "target_dimension": target_dim,
        "strict_overhead_max": target_strict_overhead_max,
        "fast_speedup_min": target_fast_speedup_min,
        "fast_drift_max": target_fast_drift_max,
    },
    "gates": gates,
    "ncu_metrics": ncu_payload,
}

with open(report_path, "w") as out:
    json.dump(report, out, indent=2)
    out.write("\n")
PY

cp "$REPORT_PATH" "$LATEST_PATH"

overall_status="$(python3 - "$REPORT_PATH" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("overall_status", "BLOCKED"))
PY
)"

echo ""
echo "Report: $REPORT_PATH"
echo "Latest: $LATEST_PATH"
echo "NCU:    $ncu_status (${NCU_REPORT_PATH})"
echo "Status: $overall_status"

if [ "$overall_status" = "PASS" ]; then
  exit 0
fi
exit 4
