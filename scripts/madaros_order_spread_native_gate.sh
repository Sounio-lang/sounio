#!/usr/bin/env bash
# Madaros native gate — CPC order_spread4 under default Madaros (not lean_single).
#
# Acceptance:
#   - Default ./bin/souc (Madaros) compile+run of the CPC generic 4-tuple witness
#     prints scaled micro-units 2044225 or 2044226 (exact spread ≈ 2.044226).
#   - TRUST sentinel ORDER_SPREAD_TRUST_OK from tests/epistemic_trust/order_spread_trust.sio
#
# Does NOT claim lean_single is Madaros. Rebuild of Madaros is not required for
# this stdlib-only leaf; measure with the current default engine.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Refuse to silently run lean_single — this gate is Madaros-only.
engine_line="$($SOUC --version 2>&1 | head -1 || true)"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single ($engine_line)"
  exit 1
fi
if ! echo "$engine_line" | grep -qi Madaros; then
  echo "WARN: version string does not mention Madaros: $engine_line"
fi
echo "engine: $engine_line"

fail=0

echo "== witness_import_order_spread (scaled micro-units) =="
if $SOUC compile tests/epistemic_trust/witness_import_order_spread.sio -o "$OUT/w.elf" >"$OUT/w.compile" 2>&1; then
  scaled="$("$OUT/w.elf" | tr -d '[:space:]')"
  echo "scaled=$scaled"
  if [ "$scaled" = "2044225" ] || [ "$scaled" = "2044226" ]; then
    echo "PASS: exact spread ~2.044226 (scaled $scaled)"
  else
    echo "FAIL: unexpected scaled value $scaled (want 2044225|2044226)"
    fail=1
  fi
else
  echo "FAIL: native compile"
  tail -20 "$OUT/w.compile"
  fail=1
fi

echo "== order_spread_trust sentinel =="
if $SOUC compile tests/epistemic_trust/order_spread_trust.sio -o "$OUT/t.elf" >"$OUT/t.compile" 2>&1; then
  if "$OUT/t.elf" | grep -q 'ORDER_SPREAD_TRUST_OK'; then
    echo "PASS: ORDER_SPREAD_TRUST_OK"
  else
    echo "FAIL: sentinel missing"
    "$OUT/t.elf" || true
    fail=1
  fi
else
  echo "FAIL: trust compile"
  tail -20 "$OUT/t.compile"
  fail=1
fi

# Optional: souc run path (wrapper)
echo "== souc run trust driver =="
if $SOUC run tests/epistemic_trust/order_spread_trust.sio >"$OUT/run.out" 2>"$OUT/run.err"; then
  if grep -q 'ORDER_SPREAD_TRUST_OK' "$OUT/run.out"; then
    echo "PASS: souc run"
  else
    echo "FAIL: souc run missing sentinel"
    cat "$OUT/run.out"
    fail=1
  fi
else
  echo "FAIL: souc run"
  tail -20 "$OUT/run.err"
  fail=1
fi

echo
if [ $fail -eq 0 ]; then
  echo "MADAROS_ORDER_SPREAD_NATIVE_GATE_OK"
  exit 0
fi
echo "MADAROS_ORDER_SPREAD_NATIVE_GATE_FAIL"
exit 1
