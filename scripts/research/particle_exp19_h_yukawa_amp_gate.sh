#!/usr/bin/env bash
# Gate for examples/particle_physics/exp19_h_yukawa_amp_to_xsec.sio
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=examples/particle_physics/exp19_h_yukawa_amp_to_xsec.sio
RECEIPT_DIR=examples/particle_physics/results
RECEIPT="$RECEIPT_DIR/exp19_h_yukawa_amp_to_xsec.json"

run_eng() {
  local eng="$1" out="$2" err="$3"
  local attempts=1
  if [ "$eng" = madaros ]; then
    attempts=3
  fi
  local try=1
  while [ "$try" -le "$attempts" ]; do
    echo "== particle exp19 engine=$eng try=$try =="
    set +e
    SOUNIO_SOUC_ENGINE="$eng" ./bin/souc run "$SRC" >"$out" 2>"$err"
    local rc=$?
    set -e
    if grep -q 'PARTICLE_EXP19_OK' "$out"; then
      break
    fi
    if [ "$try" -eq "$attempts" ]; then
      echo "engine=$eng failed rc=$rc" >&2
      tail -40 "$err" >&2
      tail -40 "$out" >&2
      exit 1
    fi
    echo "engine=$eng soft-fail rc=$rc; retrying Madaros native flake" >&2
    try=$((try + 1))
    sleep 1
  done
  grep -q 'PARTICLE_EXP19_PASS 5' "$out"
  if grep -q '^FAIL ' "$out"; then
    echo "FAIL under $eng" >&2
    grep '^FAIL ' "$out" >&2
    exit 1
  fi
}

run_eng lean_single /tmp/particle_exp19_lean_out.txt /tmp/particle_exp19_lean_err.txt
run_eng madaros /tmp/particle_exp19_mad_out.txt /tmp/particle_exp19_mad_err.txt

mkdir -p "$RECEIPT_DIR"
grep 'EXP19_LEDGER_JSON' /tmp/particle_exp19_lean_out.txt | sed 's/^EXP19_LEDGER_JSON //' >"$RECEIPT"
test -s "$RECEIPT"

echo "PARTICLE_EXP19_GATE_OK"
echo "receipt=$RECEIPT"
