#!/usr/bin/env bash
# Validate the COMPILER-GENERATED epistemic WMMA matmul kernel
# (self-hosted/gpu/kernel_ir.sio: gpu_build_epistemic_wmma_matmul_16x16_ir)
# end-to-end on the DGX Spark: emit PTX -> ptxas native sm_121 CUBIN (no JIT)
# -> CUDA Driver API launch -> compare against the CPU GUM/RSS oracle.
#
# DISTINCT from:
#   - scripts/dev/dgx_spark_public_gpu_gate.sh (narrow public `kernel fn`
#     surface, souc --backend gpu, loop+scalar f32 only).
#   - the PR #487 Blackwell receipt (docs/research/solver-gpu-native-path-2026-06-27.md),
#     which loaded the HAND-WRITTEN self-hosted/gpu/epistemic_mma_reference.ptx,
#     not this compiler-generated kernel.
#   - the "13 L4-validated profiles" K-AXI/nvidia_bare.sio SM80 hand-assembled
#     SASS path (unrelated code path, unrelated target arch).
#
# This gate keeps the local workspace as the compiler/source authority and
# uses the DGX Spark only as the CUDA toolchain/runtime authority. It does not
# manage passwords — use an SSH key or an existing ControlMaster socket.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# Role selection: canonical (Spark #1, must stay green) vs experimental
# (Spark #2, allowed to fail). See docs/ops/dgx_spark_gpu_dev.md.
DGX_SPARK_ROLE="${DGX_SPARK_ROLE:-canonical}"
if [ "$DGX_SPARK_ROLE" = "experimental" ]; then
  DGX_SPARK_HOST="${DGX_SPARK_HOST:-192.168.3.24}"
else
  DGX_SPARK_HOST="${DGX_SPARK_HOST:-192.168.3.43}"
fi
DGX_SPARK_USER="${DGX_SPARK_USER:-demetrios}"
DGX_SPARK_TARGET="${DGX_SPARK_TARGET:-${DGX_SPARK_USER}@${DGX_SPARK_HOST}}"
DGX_SPARK_REMOTE_DIR="${DGX_SPARK_REMOTE_DIR:-/tmp/sounio-dgx-spark-epistemic-wmma}"
DGX_SPARK_PTXAS="${DGX_SPARK_PTXAS:-/usr/local/cuda-13.0/bin/ptxas}"
DGX_SPARK_NVCC="${DGX_SPARK_NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
DGX_SPARK_ARCH="${DGX_SPARK_ARCH:-sm_121}"
DGX_SPARK_JSON="${DGX_SPARK_JSON:-$ROOT_DIR/artifacts/gpu/dgx_spark_epistemic_wmma_matmul_gate.v1.json}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=accept-new
)
if [ -n "${DGX_SPARK_SSH_CONTROL_PATH:-}" ]; then
  SSH_OPTS+=(-o "ControlPath=${DGX_SPARK_SSH_CONTROL_PATH}")
fi

case "$DGX_SPARK_REMOTE_DIR" in
  *[!a-zA-Z0-9_./-]*|'')
    echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL invalid DGX_SPARK_REMOTE_DIR=$DGX_SPARK_REMOTE_DIR" >&2
    exit 2
    ;;
esac

if [ ! -x "$SOUC" ]; then
  echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL souc not executable: $SOUC" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOCAL_REPORT="$TMP_DIR/report.json"
mkdir -p "$(dirname "$DGX_SPARK_JSON")"

write_json() {
  local status="$1" reason="$2" runtime_output="${3:-}"
  python3 - "$LOCAL_REPORT" "$status" "$reason" "$DGX_SPARK_TARGET" "$DGX_SPARK_ARCH" \
    "$DGX_SPARK_ROLE" "$runtime_output" <<'PY'
import json, pathlib, sys
out, status, reason, target, arch, role, runtime_output = sys.argv[1:]
payload = {
    "schema": "sounio.dgx-spark-epistemic-wmma-matmul-gate.v1",
    "status": status,
    "reason": reason,
    "target": target,
    "arch": arch,
    "role": role,
    "kernel": "epi_wmma_mm16 (self-hosted/gpu/kernel_ir.sio, compiler-generated)",
    "runtime_output": runtime_output,
    "boundaries": [
        "local_workspace_is_compiler_authority",
        "dgx_spark_is_cuda_toolchain_and_runtime_authority",
        "validates_compiler_generated_epistemic_wmma_kernel_only",
        "does_not_store_or_manage_ssh_passwords",
        "distinct_from_hand_written_epistemic_mma_reference_ptx_receipt",
        "distinct_from_13_l4_validated_kaxi_sm80_profiles",
        "provenance_output_is_printed_not_asserted_pointer_dependent",
    ],
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  cp "$LOCAL_REPORT" "$DGX_SPARK_JSON"
}

run_ssh() { ssh "${SSH_OPTS[@]}" "$DGX_SPARK_TARGET" "$@"; }
run_scp_to_remote() { scp "${SSH_OPTS[@]}" "$1" "$DGX_SPARK_TARGET:$2"; }

echo "=== DGX Spark Epistemic WMMA Matmul Gate (role=$DGX_SPARK_ROLE) ==="
echo "target: $DGX_SPARK_TARGET   arch: $DGX_SPARK_ARCH"
echo ""

echo "=== Local: build PTX emitter driver and emit PTX ==="
DRIVER_ELF="$TMP_DIR/kretikos_emit_epistemic_wmma.elf"
PTX_OUT="$TMP_DIR/epi_wmma_mm16.ptx"

if ! timeout 120 "$SOUC" build self-hosted/gpu/kretikos_emit_epistemic_wmma.sio -o "$DRIVER_ELF"; then
  write_json "fail" "local_driver_build_failed"
  echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL local souc build of the PTX-emitter driver failed" >&2
  echo "hint: this is a known pre-existing Madaros native-codegen limitation on large merged" >&2
  echo "self-hosted GPU modules (unrelated to the kernel math) — see memory m2-effect-firewall / sounio-gpu-lane-state" >&2
  exit 1
fi

"$DRIVER_ELF" epi_wmma_mm16 > "$PTX_OUT"
if [ ! -s "$PTX_OUT" ]; then
  write_json "fail" "local_ptx_empty"
  echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL emitted PTX is empty" >&2
  exit 1
fi

echo ""
echo "=== Remote CUDA toolchain probe ==="
run_ssh "test -x '$DGX_SPARK_PTXAS' && test -x '$DGX_SPARK_NVCC'" || {
  write_json "fail" "ssh_or_cuda_toolchain_unreachable"
  echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL cannot reach Spark CUDA toolchain over BatchMode SSH" >&2
  exit 1
}

run_ssh "rm -rf '$DGX_SPARK_REMOTE_DIR' && mkdir -p '$DGX_SPARK_REMOTE_DIR'"
run_scp_to_remote "$PTX_OUT" "$DGX_SPARK_REMOTE_DIR/epi_wmma_mm16.ptx"
run_scp_to_remote "benchmarks/solver/gpu/run_epistemic_wmma_mm16_native_sm121.c" \
  "$DGX_SPARK_REMOTE_DIR/run_epi_wmma.c"

echo ""
echo "=== Remote: ptxas native CUBIN (no JIT) + Driver API launch ==="
RUNTIME_OUTPUT="$(
  run_ssh "set -euo pipefail
cd '$DGX_SPARK_REMOTE_DIR'
'$DGX_SPARK_PTXAS' -arch='$DGX_SPARK_ARCH' epi_wmma_mm16.ptx -o epi_wmma_mm16.cubin
test -s epi_wmma_mm16.cubin
wc -c epi_wmma_mm16.cubin
'$DGX_SPARK_NVCC' -std=c++17 run_epi_wmma.c -lcuda -o run_epi_wmma
./run_epi_wmma epi_wmma_mm16.cubin"
)" || {
  write_json "fail" "remote_ptxas_or_runtime_failed"
  echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL remote ptxas/build/launch failed" >&2
  exit 1
}
printf '%s\n' "$RUNTIME_OUTPUT"

if ! printf '%s' "$RUNTIME_OUTPUT" | grep -q "RESULT: dataPath=PASS epistemicShadow(RSS-quadrature)=PASS"; then
  write_json "fail" "runtime_assertion_failed" "$RUNTIME_OUTPUT"
  echo "dgx_spark_epistemic_wmma_matmul_gate: FAIL numeric assertions did not pass" >&2
  exit 1
fi

write_json "pass" "dgx_spark_epistemic_wmma_matmul_validated" "$RUNTIME_OUTPUT"
echo ""
echo "dgx_spark_epistemic_wmma_matmul_gate: PASS report=$DGX_SPARK_JSON"
