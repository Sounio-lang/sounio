#!/bin/bash
cd /workspace/.wt/claude-1
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_PARITY_MADAROS=/tmp/qwen_madaros_bigstack
export SOUNIO_PARITY_LEAN=/tmp/qwen_lean_bigstack
export SOUNIO_PARITY_JOBS=4
export SOUNIO_PARITY_TIMEOUT=120
bash scripts/ci/engine_parity_gate.sh > /tmp/qwen_parity_run4.log 2>&1
echo "gate_rc=$?" >> /tmp/qwen_parity_run4.log
