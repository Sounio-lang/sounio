#!/usr/bin/env bash
# Sounio repo CI hook: SWOW unified ORC parity (hyperbolic-semantic-networks).
set -euo pipefail
HSN_ROOT="${HSN_ROOT:-/workspace/hyperbolic-semantic-networks}"
exec bash "$HSN_ROOT/experiments/03_semantic_networks/run_swow_unified_orc_gate.sh"
