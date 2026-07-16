#!/usr/bin/env bash
# Capture a narrow, retained GB10 hardware receipt for oct_wmma_validate.cu.
# This checks the documented CUDA witness only; it is not a general GPU backend gate.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_REL="docs/gpu/oct_wmma_validate.cu"
SOURCE_PATH="$ROOT_DIR/$SOURCE_REL"
NVCC_BIN="${SOUNIO_GB10_NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
DEFAULT_RECEIPT_DIR="$ROOT_DIR/artifacts/gpu/gb10-wmma-receipt"
RECEIPT_DIR="${SOUNIO_GB10_WMMA_RECEIPT_DIR:-$DEFAULT_RECEIPT_DIR}"

usage() {
    cat <<'EOF'
Usage: scripts/ci/gb10_wmma_receipt_gate.sh [--output DIR]

Compiles and runs docs/gpu/oct_wmma_validate.cu on exactly one NVIDIA GB10
device with compute capability 12.1, then writes the command, environment,
source hash, compiler output, runtime output, and receipt summary to DIR.

Environment:
  SOUNIO_GB10_WMMA_RECEIPT_DIR  Default output directory.
  SOUNIO_GB10_NVCC              CUDA compiler path (default: /usr/local/cuda-13.0/bin/nvcc).
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--output" ]]; then
    [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
    RECEIPT_DIR="$2"
elif [[ "$#" -ne 0 ]]; then
    usage >&2
    exit 2
fi

if [[ "$RECEIPT_DIR" != /* ]]; then
    RECEIPT_DIR="$ROOT_DIR/$RECEIPT_DIR"
fi

if [[ -e "$RECEIPT_DIR" ]]; then
    printf 'refusing to overwrite existing receipt directory: %s\n' "$RECEIPT_DIR" >&2
    exit 2
fi

mkdir -p "$(dirname "$RECEIPT_DIR")"
mkdir "$RECEIPT_DIR"

on_exit() {
    local rc=$?
    if [[ -d "$RECEIPT_DIR" ]]; then
        if [[ "$rc" -eq 0 ]]; then
            printf 'status=pass\nexit_code=0\n' > "$RECEIPT_DIR/status.txt" || true
        else
            printf 'status=fail\nexit_code=%d\n' "$rc" > "$RECEIPT_DIR/status.txt" || true
        fi
    fi
}
trap on_exit EXIT

fail() {
    printf 'FAIL: %s\n' "$*" | tee -a "$RECEIPT_DIR/gate.stderr" >&2
    exit 1
}

cd "$ROOT_DIR"

[[ -f "$SOURCE_PATH" ]] || fail "missing witness source: $SOURCE_REL"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required"
[[ -x "$NVCC_BIN" ]] || fail "CUDA compiler is not executable: $NVCC_BIN"

GPU_INFO="$(nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader)"
GPU_COUNT="$(printf '%s\n' "$GPU_INFO" | sed '/^$/d' | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -ne 1 ]]; then
    fail "expected exactly one visible GPU, found $GPU_COUNT"
fi
if ! printf '%s\n' "$GPU_INFO" | grep -Eq '^[0-9]+, NVIDIA GB10, [^,]+, 12\.1$'; then
    fail "expected one NVIDIA GB10 with compute capability 12.1; inventory: $GPU_INFO"
fi

{
    printf 'hostname=%s\n' "$(hostname)"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf '\nGPU inventory (index, name, driver, compute capability):\n%s\n' "$GPU_INFO"
    printf '\nCUDA compiler path:\n%s\n' "$NVCC_BIN"
    printf '\nCUDA compiler version:\n'
    "$NVCC_BIN" --version
} > "$RECEIPT_DIR/environment.txt"

sha256sum "$SOURCE_REL" > "$RECEIPT_DIR/source.sha256"

BINARY_PATH="$RECEIPT_DIR/oct_wmma_validate"
COMPILE_COMMAND=("$NVCC_BIN" -std=c++17 -O2 -arch=sm_121 "$SOURCE_REL" -o "$BINARY_PATH")
{
    printf 'working_directory=%q\n' "$ROOT_DIR"
    printf 'command='
    printf '%q ' "${COMPILE_COMMAND[@]}"
    printf '\n'
} > "$RECEIPT_DIR/compile-command.txt"

"${COMPILE_COMMAND[@]}" > "$RECEIPT_DIR/compile.stdout" 2> "$RECEIPT_DIR/compile.stderr"
printf 'compile_exit_code=0\n' > "$RECEIPT_DIR/compile.status"
sha256sum "$BINARY_PATH" > "$RECEIPT_DIR/binary.sha256"

"$BINARY_PATH" > "$RECEIPT_DIR/run.stdout" 2> "$RECEIPT_DIR/run.stderr"
printf 'run_exit_code=0\n' > "$RECEIPT_DIR/run.status"

require_output() {
    local expected="$1"
    grep -Fqx "$expected" "$RECEIPT_DIR/run.stdout" || fail "runtime output did not contain expected line: $expected"
}

require_output 'e1*e2 on tensor core: comp3=1.00 comp4=0.00  (X: comp3=+1,comp4=0)'
require_output 'batch: 0/128 comps mismatch, maxerr=0.000 (f16 tile precision)'
require_output 'PASS: WMMA octonion multiply is Convention X on GB10'

cat > "$RECEIPT_DIR/receipt.md" <<EOF
# GB10 WMMA CI Receipt

Status: pass.

This artifact records one execution of \`$SOURCE_REL\` from commit
\`$(git rev-parse HEAD)\` on the runner inventory in \`environment.txt\`.
The source hash, compile command, compiler output, and runtime output are
retained beside this summary.

## Scope boundary

This is a narrow CUDA WMMA witness for the documented Convention X discriminator
and the 16-vector batch in \`oct_wmma_validate.cu\`. It does not claim general
GPU backend correctness, code-generation coverage, or validation of unrelated
GPU hardware.
EOF

printf 'PASS: retained GB10 WMMA receipt written to %s\n' "$RECEIPT_DIR"
