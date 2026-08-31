#!/usr/bin/env bash
# Gate for examples/particle_physics/exp13_amplitude_honesty.sio
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=examples/particle_physics/exp13_amplitude_honesty.sio

run_eng() {
  local eng="$1" out="$2" err="$3"
  echo "== particle exp13 engine=$eng =="
  set +e
  SOUNIO_SOUC_ENGINE="$eng" ./bin/souc run "$SRC" >"$out" 2>"$err"
  local rc=$?
  set -e
  if ! grep -q 'PARTICLE_EXP13_OK' "$out"; then
    echo "engine=$eng failed rc=$rc" >&2
    tail -40 "$err" >&2
    tail -30 "$out" >&2
    exit 1
  fi
  grep -q 'PARTICLE_EXP13_PASS 4' "$out"
  if grep -q '^FAIL ' "$out"; then
    echo "FAIL under $eng" >&2
    grep '^FAIL ' "$out" >&2
    exit 1
  fi
}

run_eng lean_single /tmp/particle_exp13_lean_out.txt /tmp/particle_exp13_lean_err.txt
run_eng madaros /tmp/particle_exp13_mad_out.txt /tmp/particle_exp13_mad_err.txt

echo "PARTICLE_EXP13_GATE_OK"
