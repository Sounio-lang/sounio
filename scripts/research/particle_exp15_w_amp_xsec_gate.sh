#!/usr/bin/env bash
# Gate for examples/particle_physics/exp15_w_amp_to_xsec.sio
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=examples/particle_physics/exp15_w_amp_to_xsec.sio
RECEIPT_DIR=examples/particle_physics/results
RECEIPT="$RECEIPT_DIR/exp15_w_amp_to_xsec.json"

run_eng() {
  local eng="$1" out="$2" err="$3"
  echo "== particle exp15 engine=$eng =="
  set +e
  SOUNIO_SOUC_ENGINE="$eng" ./bin/souc run "$SRC" >"$out" 2>"$err"
  local rc=$?
  set -e
  if ! grep -q 'PARTICLE_EXP15_OK' "$out"; then
    echo "engine=$eng failed rc=$rc" >&2
    tail -40 "$err" >&2
    tail -40 "$out" >&2
    exit 1
  fi
  grep -q 'PARTICLE_EXP15_PASS 5' "$out"
  if grep -q '^FAIL ' "$out"; then
    echo "FAIL under $eng" >&2
    grep '^FAIL ' "$out" >&2
    exit 1
  fi
}

run_eng lean_single /tmp/particle_exp15_lean_out.txt /tmp/particle_exp15_lean_err.txt
run_eng madaros /tmp/particle_exp15_mad_out.txt /tmp/particle_exp15_mad_err.txt

mkdir -p "$RECEIPT_DIR"
grep 'EXP15_LEDGER_JSON' /tmp/particle_exp15_lean_out.txt | sed 's/^EXP15_LEDGER_JSON //' >"$RECEIPT"
test -s "$RECEIPT"

echo "PARTICLE_EXP15_GATE_OK"
echo "receipt=$RECEIPT"
