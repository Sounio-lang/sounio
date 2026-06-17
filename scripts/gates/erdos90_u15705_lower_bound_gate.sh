#!/usr/bin/env bash
# scripts/gates/erdos90_u15705_lower_bound_gate.sh
#
# Certify the new lower bound u(15705) ≥ 176768.
#
# The computational witness was produced by stdlib/research/erdos90_optimize.sio
# (compact disk x²+y² ≤ 5000, unit distance² = 1105). This gate verifies that
# the Lean theorem Sounio.Erdos90Planar.erdos90_compact_disk_u15705 builds
# successfully; native_decide re-checks the integer count inside Lean core.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ELAN_HOME="${ELAN_HOME:-$ROOT/formal/lean4/.elan}"
export PATH="$ELAN_HOME/bin:$PATH"

command -v lake >/dev/null 2>&1 || { echo "error: lake not in PATH" >&2; exit 2; }

echo "[erdos90-u15705] build Lean certificate SounioErdos90PlanarLowerBound"
cd "$ROOT/formal/lean4"
lake build SounioErdos90PlanarLowerBound

echo "[erdos90-u15705] PASS — u(15705) ≥ 176768 certified by Lean native_decide"
