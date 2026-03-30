#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-gpu-capability-gate}"
ABI_LAYOUT_WORK_DIR="$WORK_DIR/abi_layout"
SURFACE_WORK_DIR="$WORK_DIR/surface_lowering"
COMPILE_PROOF_WORK_DIR="$WORK_DIR/compile_proof"
SIM_RUNTIME_WORK_DIR="$WORK_DIR/sim_runtime"
HARDWARE_WORK_DIR="$WORK_DIR/hardware_runtime"

echo "GPU_CAPABILITY_GATE_START"
echo "work_dir=$WORK_DIR"

WORK_DIR="$ABI_LAYOUT_WORK_DIR" bash "$ROOT_DIR/scripts/gpu/gpu_abi_layout_gate.sh"
WORK_DIR="$SURFACE_WORK_DIR" bash "$ROOT_DIR/scripts/gpu/gpu_surface_lowering_gate.sh"
WORK_DIR="$COMPILE_PROOF_WORK_DIR" bash "$ROOT_DIR/scripts/gpu/gpu_compile_proof_gate.sh"
WORK_DIR="$SIM_RUNTIME_WORK_DIR" bash "$ROOT_DIR/scripts/gpu/gpu_sim_runtime_gate.sh"
WORK_DIR="$HARDWARE_WORK_DIR" bash "$ROOT_DIR/scripts/gpu/gpu_hardware_runtime_gate.sh"

echo "GPU_CAPABILITY_GATE_SUMMARY status=pass"
