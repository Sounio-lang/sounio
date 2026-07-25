#!/usr/bin/env bash
# Gate for examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
#
# Three executable verticals on the particle_physics stdlib:
#   EXP1 — Z metrology (GUM Γ(Z→ee) + budget + confidence gate)
#   EXP2 — NonUnitary at Z pole (deficit scan + peak xsec with NonUnitary effect)
#   EXP3 — EW tension (M_W pull, S/T/U, Δρ)
#
# Prefer lean_single for this package (Madaros science-boundary may block
# examples → protected scientific modules). Override with SOUNIO_SOUC_ENGINE.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
OUT=/tmp/particle_exp123_out.txt

echo "== particle exp123 (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>/tmp/particle_exp123_err.txt || {
  echo "run failed; stderr:" >&2
  tail -40 /tmp/particle_exp123_err.txt >&2
  exit 1
}

grep -q 'PARTICLE_EXP123_OK' "$OUT"
grep -q 'PARTICLE_EXP123_PASS 42' "$OUT"
grep -q 'EXP1_DOMINANT_SOURCE' "$OUT"
grep -q 'EXP2_DEFICIT_POLE 1.000000' "$OUT"
grep -q 'EXP3_MW_PULL' "$OUT"
grep -q 'EXP1_DONE' "$OUT"
grep -q 'EXP2_DONE' "$OUT"
grep -q 'EXP3_DONE' "$OUT"

# no unexpected FAIL lines
if grep -q '^FAIL ' "$OUT"; then
  echo "unexpected FAIL lines in output" >&2
  grep '^FAIL ' "$OUT" >&2
  exit 1
fi

echo "PARTICLE_EXP123_GATE_OK"
