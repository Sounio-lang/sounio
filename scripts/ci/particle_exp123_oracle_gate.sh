#!/usr/bin/env bash
# N5 — independent oracle audit of EXP123 receipts
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
OUT=/tmp/particle_exp123_oracle_out.txt

echo "== particle exp123 oracle (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>/tmp/particle_exp123_oracle_err.txt
grep -q 'PARTICLE_EXP123_OK' "$OUT"
python3 scripts/research/particle_exp123_oracle.py "$OUT"
echo "PARTICLE_EXP123_ORACLE_GATE_OK"
