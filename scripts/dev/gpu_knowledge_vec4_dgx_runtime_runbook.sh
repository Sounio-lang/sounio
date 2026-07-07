#!/usr/bin/env bash
# Runbook wrapper for the GPU Knowledge Vec4 DGX handoff.
#
# This script is intentionally thin: it composes the package, package verifier,
# DGX runtime gate, and runtime receipt verifier without adding another evidence
# surface. Use `all-local` for no-GPU validation and `run-dgx` only when DGX SSH
# and CUDA runtime are intentionally available.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${SOUNIO_GPU_KNOWLEDGE_VEC4_PACKAGE_MANIFEST:-artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json}"
RECEIPT="${SOUNIO_GPU_KNOWLEDGE_VEC4_RUNTIME_RECEIPT:-artifacts/gpu/dgx_spark_public_gpu_gate.v1.json}"

usage() {
  cat <<'USAGE'
usage: scripts/dev/gpu_knowledge_vec4_dgx_runtime_runbook.sh <mode>

modes:
  package-only    Prepare the local marker-only DGX package without SSH/GPU.
  verify-package  Verify package manifest hashes and launch-contract boundary.
  run-dgx         Run the opt-in DGX marker route; requires SSH/CUDA runtime.
  verify-runtime Verify DGX runtime receipt requires runtime_pass and PASS output.
  all-local       Run package-only + verify-package + verify not-run receipt.
USAGE
}

mode="${1:-}"
case "$mode" in
  package-only)
    DGX_SPARK_PACKAGE_ONLY=1 \
    DGX_SPARK_PUBLIC_KERNELS=0 \
    DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER=1 \
    DGX_SPARK_RUNTIME=0 \
      bash scripts/dev/dgx_spark_public_gpu_gate.sh
    ;;
  verify-package)
    scripts/dev/gpu_knowledge_vec4_package_verify.py "$MANIFEST"
    ;;
  run-dgx)
    DGX_SPARK_PUBLIC_KERNELS="${DGX_SPARK_PUBLIC_KERNELS:-0}" \
    DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER=1 \
      bash scripts/dev/dgx_spark_public_gpu_gate.sh
    ;;
  verify-runtime)
    scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode runtime-pass "$RECEIPT"
    ;;
  all-local)
    "$0" package-only
    "$0" verify-package
    scripts/dev/gpu_knowledge_vec4_runtime_receipt_verify.py --mode not-run "$RECEIPT"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
