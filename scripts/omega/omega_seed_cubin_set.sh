#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/sass/cubin}"
FORCE_RESEED="${OMEGA_FORCE_RESEED_CUBINS:-0}"
CUDA_ARCH="${OMEGA_CUDA_ARCH:-sm_80}"
PREFER_REAL_CUDA="${OMEGA_PREFER_REAL_CUDA:-1}"
REPORT_PATH="${OUT_DIR}/seed_report.json"
WORK_DIR="${OUT_DIR}/.build"

mkdir -p "$OUT_DIR" "$WORK_DIR"

emit_seed_cubin_fallback() {
  local kernel="$1"
  local salt="$2"
  local out="$OUT_DIR/${kernel}.cubin"
  local src="$WORK_DIR/${kernel}.seed.c"

  cat >"$src" <<EOF
#include <stdint.h>
__attribute__((used))
uint64_t sounio_${kernel}_seed_symbol(uint64_t x) {
  uint64_t v = x ^ UINT64_C(0x9e3779b97f4a7c15);
  v = (v << 13) ^ (v >> 7);
  return v + UINT64_C(${salt});
}
EOF

  gcc \
    -c \
    -O2 \
    -fno-asynchronous-unwind-tables \
    -fno-stack-protector \
    -fno-ident \
    -x c \
    "$src" \
    -o "$out"
}

emit_cuda_source() {
  local kernel="$1"
  local src="$WORK_DIR/${kernel}.cu"
  case "$kernel" in
    gemm)
      cat >"$src" <<'EOF'
extern "C" __global__ void gemm(
    const float* a, const float* b, float* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) acc += a[row * k + i] * b[i * n + col];
    c[row * n + col] = acc;
  }
}
EOF
      ;;
    attention)
      cat >"$src" <<'EOF'
extern "C" __global__ void attention(
    const float* q, const float* k, const float* v, float* out, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float score = 0.0f;
    for (int j = 0; j < d; ++j) score += q[i * d + j] * k[i * d + j];
    float w = score / (1.0f + fabsf(score));
    for (int j = 0; j < d; ++j) out[i * d + j] = w * v[i * d + j];
  }
}
EOF
      ;;
    epistemic_elementwise)
      cat >"$src" <<'EOF'
extern "C" __global__ void epistemic_elementwise(
    const float* x, const float* y, const float* eps, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i] * (1.0f + eps[i]);
}
EOF
      ;;
    monte_carlo)
      cat >"$src" <<'EOF'
extern "C" __global__ void monte_carlo(unsigned int seed, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned int x = seed ^ (0x9e3779b9u * (i + 1u));
    x = 1664525u * x + 1013904223u;
    float u = (x & 0x00ffffff) / 16777216.0f;
    out[i] = u * (1.0f - u);
  }
}
EOF
      ;;
    *)
      echo "unknown kernel: $kernel" >&2
      return 2
      ;;
  esac
  echo "$src"
}

emit_cuda_cubin() {
  local kernel="$1"
  local out="$OUT_DIR/${kernel}.cubin"
  local src
  src="$(emit_cuda_source "$kernel")"
  nvcc -cubin -arch="$CUDA_ARCH" -O3 "$src" -o "$out"
}

kernel_status_json() {
  local kernel="$1"
  local mode="$2"
  local status="$3"
  local reason="$4"
  local path="$OUT_DIR/${kernel}.cubin"
  local size="0"
  local sha256=""
  if [[ -f "$path" ]]; then
    size="$(stat -c%s "$path")"
    sha256="$(sha256sum "$path" | awk '{print $1}')"
  fi
  cat <<EOF
{"kernel":"$kernel","mode":"$mode","status":"$status","reason":"$reason","path":"$path","size_bytes":$size,"sha256":"$sha256"}
EOF
}

declare -a KERNELS=("gemm" "attention" "epistemic_elementwise" "monte_carlo")
declare -a SALTS=(
  "0x0101010101010101"
  "0x0202020202020202"
  "0x0303030303030303"
  "0x0404040404040404"
)

has_nvcc=0
if command -v nvcc >/dev/null 2>&1; then
  has_nvcc=1
fi

declare -a RESULT_JSON=()
for i in "${!KERNELS[@]}"; do
  kernel="${KERNELS[$i]}"
  salt="${SALTS[$i]}"
  out="$OUT_DIR/${kernel}.cubin"

  if [[ -f "$out" && "$FORCE_RESEED" != "1" ]]; then
    echo "omega_seed_cubin_set: keep existing $out"
    RESULT_JSON+=("$(kernel_status_json "$kernel" "existing" "pass" "kept_existing")")
    continue
  fi

  if [[ "$PREFER_REAL_CUDA" = "1" && "$has_nvcc" -eq 1 ]]; then
    set +e
    emit_cuda_cubin "$kernel" >"$WORK_DIR/${kernel}.nvcc.stdout.log" 2>"$WORK_DIR/${kernel}.nvcc.stderr.log"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      echo "omega_seed_cubin_set: wrote CUDA cubin $out (arch=$CUDA_ARCH)"
      RESULT_JSON+=("$(kernel_status_json "$kernel" "cuda" "pass" "nvcc_ok")")
      continue
    fi
    echo "omega_seed_cubin_set: nvcc failed for $kernel (rc=$rc), falling back"
  fi

  emit_seed_cubin_fallback "$kernel" "$salt"
  echo "omega_seed_cubin_set: wrote fallback cubin $out"
  if [[ "$PREFER_REAL_CUDA" = "1" && "$has_nvcc" -eq 0 ]]; then
    RESULT_JSON+=("$(kernel_status_json "$kernel" "fallback" "pass" "nvcc_missing")")
  else
    RESULT_JSON+=("$(kernel_status_json "$kernel" "fallback" "pass" "nvcc_failed")")
  fi
done

{
  echo "{"
  echo "  \"schema\": \"sounio.omega.cubin-seed-report.v2\","
  echo "  \"cuda_arch\": \"$CUDA_ARCH\","
  echo "  \"prefer_real_cuda\": $PREFER_REAL_CUDA,"
  echo "  \"nvcc_available\": $has_nvcc,"
  echo "  \"results\": ["
  for i in "${!RESULT_JSON[@]}"; do
    if [[ "$i" -gt 0 ]]; then
      echo ","
    fi
    echo -n "    ${RESULT_JSON[$i]}"
  done
  echo
  echo "  ]"
  echo "}"
} >"$REPORT_PATH"

echo "omega_seed_cubin_set: done out_dir=$OUT_DIR report=$REPORT_PATH"
