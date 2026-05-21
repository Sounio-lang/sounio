#!/usr/bin/env bash
# CUDA timing gate for Kretikos cross-backend PTX-side artifacts.
#
# This gate is intentionally narrow: it reports CUDA event timings for the
# generated vec_add and epistemic dual-output CUBINs. It does not claim
# cross-backend runtime equivalence or octonion benchmark performance.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_CROSS_BACKEND_CUDA_BENCH_DIR:-$ROOT_DIR/artifacts/omega/kretikos_cross_backend_cuda_benchmark}"
REPORT_JSON="${SOUNIO_KRETIKOS_CROSS_BACKEND_CUDA_BENCH_JSON:-$OUT_DIR/kretikos_cross_backend_cuda_benchmark.v1.json}"
SEMANTIC_DIR="${SOUNIO_KRETIKOS_CROSS_BACKEND_DIR:-$ROOT_DIR/artifacts/omega/kretikos_cross_backend_semantic}"
BENCH_SRC="$ROOT_DIR/scripts/gpu/kretikos_cross_backend_cuda_bench.c"
BENCH_BIN="$OUT_DIR/kretikos_cross_backend_cuda_bench"
VEC_CUBIN="$SEMANTIC_DIR/vec_add.cubin"
EPI_CUBIN="$SEMANTIC_DIR/epistemic_dual_output_f32.cubin"
VEC_STDOUT="$OUT_DIR/vec_add.benchmark.txt"
EPI_STDOUT="$OUT_DIR/epistemic_dual_output_f32.benchmark.txt"

mkdir -p "$OUT_DIR"

rel_path() {
  local p="$1"
  if [[ "$p" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${p#$ROOT_DIR/}"
  else
    printf '%s' "$p"
  fi
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

size_file() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

kv() {
  local key="$1"
  local file="$2"
  grep -m1 -o "${key}=[^ ]*" "$file" | cut -d= -f2-
}

bash scripts/ci/kretikos_cross_backend_semantic_gate.sh >/dev/null

cc -O2 "$BENCH_SRC" -ldl -lm -o "$BENCH_BIN"

"$BENCH_BIN" "$VEC_CUBIN" vec_add vec_add_f32 > "$VEC_STDOUT"
"$BENCH_BIN" "$EPI_CUBIN" epistemic_dual_output_f32 epistemic_dual_output_f32 > "$EPI_STDOUT"

vec_status="$(kv status "$VEC_STDOUT")"
epi_status="$(kv status "$EPI_STDOUT")"
device_name="$(kv device_name "$VEC_STDOUT")"
driver_version="$(kv driver_version "$VEC_STDOUT")"
compute_capability="$(kv cc "$VEC_STDOUT")"
epi_device_name="$(kv device_name "$EPI_STDOUT")"
epi_driver_version="$(kv driver_version "$EPI_STDOUT")"
epi_compute_capability="$(kv cc "$EPI_STDOUT")"

if [[ "$vec_status" != "pass" || "$epi_status" != "pass" ]]; then
  echo "kretikos_cross_backend_cuda_benchmark_gate: FAIL benchmark status" >&2
  exit 1
fi

if [[ -z "$device_name" || "$device_name" != "$epi_device_name" ||
      "$driver_version" != "$epi_driver_version" ||
      "$compute_capability" != "$epi_compute_capability" ]]; then
  echo "kretikos_cross_backend_cuda_benchmark_gate: FAIL device identity missing or inconsistent" >&2
  exit 1
fi

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

./bin/kretikos json-emit \
  --raw-json "artifacts={\"epistemic_dual_output_f32\":{\"cubin\":{\"bytes\":$(size_file "$EPI_CUBIN"),\"path\":\"$(rel_path "$EPI_CUBIN")\",\"sha256\":\"$(sha_file "$EPI_CUBIN")\"},\"stdout\":{\"bytes\":$(size_file "$EPI_STDOUT"),\"path\":\"$(rel_path "$EPI_STDOUT")\",\"sha256\":\"$(sha_file "$EPI_STDOUT")\"}},\"vec_add_f32\":{\"cubin\":{\"bytes\":$(size_file "$VEC_CUBIN"),\"path\":\"$(rel_path "$VEC_CUBIN")\",\"sha256\":\"$(sha_file "$VEC_CUBIN")\"},\"stdout\":{\"bytes\":$(size_file "$VEC_STDOUT"),\"path\":\"$(rel_path "$VEC_STDOUT")\",\"sha256\":\"$(sha_file "$VEC_STDOUT")\"}},\"runner\":{\"binary\":{\"bytes\":$(size_file "$BENCH_BIN"),\"path\":\"$(rel_path "$BENCH_BIN")\",\"sha256\":\"$(sha_file "$BENCH_BIN")\"},\"source\":{\"bytes\":$(size_file "$BENCH_SRC"),\"path\":\"$(rel_path "$BENCH_SRC")\",\"sha256\":\"$(sha_file "$BENCH_SRC")\"}}}" \
  --array-strings "boundaries=cuda_event_timing_for_ptx_side_only|does_not_claim_metal_runtime|does_not_claim_spirv_runtime|does_not_claim_octonion_workload_performance|scalar_ops_are_kernel_model_counts" \
  --bool "checks.epistemic_dual_output_f32_benchmark_pass=true" \
  --bool "checks.vec_add_f32_benchmark_pass=true" \
  --string "generated_at_utc=$generated_at_utc" \
  --int "protocol.iterations=$(kv iterations "$VEC_STDOUT")" \
  --int "protocol.warmup=$(kv warmup "$VEC_STDOUT")" \
  --string "runtime.compute_capability=$compute_capability" \
  --string "runtime.device_name=$device_name" \
  --string "runtime.driver_version=$driver_version" \
  --string "schema=sounio.kretikos.cross-backend-cuda-benchmark.v1" \
  --array-strings "semantics=vec_add_f32|epistemic_dual_output_f32" \
  --string "status=pass" \
  --int "results.epistemic_dual_output_f32.n=$(kv n "$EPI_STDOUT")" \
  --int "results.epistemic_dual_output_f32.scalar_ops_per_elem=$(kv scalar_ops_per_elem "$EPI_STDOUT")" \
  --string "results.epistemic_dual_output_f32.median_ms=$(kv median_ms "$EPI_STDOUT")" \
  --string "results.epistemic_dual_output_f32.iqr_ms=$(kv iqr_ms "$EPI_STDOUT")" \
  --string "results.epistemic_dual_output_f32.gscalarops_per_s=$(kv gscalarops_per_s "$EPI_STDOUT")" \
  --string "results.epistemic_dual_output_f32.value_max_abs_err=$(kv value_max_abs_err "$EPI_STDOUT")" \
  --string "results.epistemic_dual_output_f32.eps_max_abs_err=$(kv eps_max_abs_err "$EPI_STDOUT")" \
  --int "results.epistemic_dual_output_f32.valid_mismatch=$(kv valid_mismatch "$EPI_STDOUT")" \
  --int "results.epistemic_dual_output_f32.prov_mismatch=$(kv prov_mismatch "$EPI_STDOUT")" \
  --int "results.vec_add_f32.n=$(kv n "$VEC_STDOUT")" \
  --int "results.vec_add_f32.scalar_ops_per_elem=$(kv scalar_ops_per_elem "$VEC_STDOUT")" \
  --string "results.vec_add_f32.median_ms=$(kv median_ms "$VEC_STDOUT")" \
  --string "results.vec_add_f32.iqr_ms=$(kv iqr_ms "$VEC_STDOUT")" \
  --string "results.vec_add_f32.gscalarops_per_s=$(kv gscalarops_per_s "$VEC_STDOUT")" \
  --string "results.vec_add_f32.max_abs_err=$(kv max_abs_err "$VEC_STDOUT")" \
  > "$REPORT_JSON"

./bin/kretikos kaxi-validate-evidence "$REPORT_JSON" \
  --expect "status=pass" \
  --expect "checks.vec_add_f32_benchmark_pass=true" \
  --expect "checks.epistemic_dual_output_f32_benchmark_pass=true" \
  --expect "runtime.device_name=$device_name" \
  --expect "runtime.compute_capability=$compute_capability" >/dev/null

echo "kretikos_cross_backend_cuda_benchmark_gate: PASS report=$(rel_path "$REPORT_JSON")"
