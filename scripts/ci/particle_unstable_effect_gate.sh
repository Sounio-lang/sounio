#!/usr/bin/env bash
# Gate for examples/particle_physics/exp4_unstable_spectrum.sio
#
# N1: NonUnitary deficit axioms for Z and W (unstable spectrum, not Z-only).
# Prefer lean_single for full run; Madaros check must also pass.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp4_unstable_spectrum.sio
OUT=/tmp/particle_exp4_out.txt
ERR=/tmp/particle_exp4_err.txt

echo "== particle exp4 unstable spectrum (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>"$ERR" || {
  echo "run failed; stderr:" >&2
  tail -40 "$ERR" >&2
  exit 1
}

grep -q 'PARTICLE_EXP4_OK' "$OUT"
grep -q 'PARTICLE_EXP4_PASS 21' "$OUT"
grep -q 'EXP4_SPECIES Z' "$OUT"
grep -q 'EXP4_SPECIES W' "$OUT"
grep -q 'EXP4_DEFICIT_POLE 1.000000' "$OUT"
grep -q 'EXP4_SPECIES_JSON' "$OUT"
grep -q 'sounio.broken_structure.v1' "$OUT"
grep -q 'EXP4_THRESH_Z' "$OUT"
grep -q 'EXP4_THRESH_W' "$OUT"

if grep -q '^FAIL ' "$OUT"; then
  echo "unexpected FAIL lines in output" >&2
  grep '^FAIL ' "$OUT" >&2
  exit 1
fi

# Madaros typecheck + no E-SRB-002
echo "== particle exp4 Madaros check =="
BOUND_ERR=/tmp/particle_exp4_boundary_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc check "$SRC" >"$BOUND_ERR" 2>&1
MAD_RC=$?
set -e
if grep -q 'E-SRB-002' "$BOUND_ERR"; then
  echo "E-SRB-002 still present" >&2
  grep 'E-SRB-002' "$BOUND_ERR" >&2
  exit 1
fi
if [[ "$MAD_RC" -ne 0 ]] || grep -q 'error\[' "$BOUND_ERR"; then
  if ! grep -q 'check: OK' "$BOUND_ERR" && ! grep -q 'verdict=0' "$BOUND_ERR"; then
    echo "Madaros typecheck failed:" >&2
    tail -40 "$BOUND_ERR" >&2
    exit 1
  fi
fi


# Madaros full run (SEGV closed via split helpers + local peak; 2026-07-26)
echo "== particle exp4 Madaros full run =="
FULL_OUT=/tmp/particle_exp4_madaros_full_out.txt
FULL_ERR=/tmp/particle_exp4_madaros_full_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$FULL_OUT" 2>"$FULL_ERR"
FULL_RC=$?
set -e
if ! grep -q 'PARTICLE_EXP4_OK' "$FULL_OUT"; then
  echo "Madaros full EXP4 failed (rc=$FULL_RC):" >&2
  tail -40 "$FULL_ERR" >&2
  tail -20 "$FULL_OUT" >&2
  exit 1
fi
if grep -q '^FAIL ' "$FULL_OUT"; then
  echo "unexpected FAIL in Madaros EXP4" >&2
  grep '^FAIL ' "$FULL_OUT" >&2
  exit 1
fi
echo "PARTICLE_EXP4_MADAROS_RUN_OK"

echo "PARTICLE_EXP4_GATE_OK"
