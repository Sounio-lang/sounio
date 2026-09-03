#!/usr/bin/env bash
# Madaros native gate — product_nonassoc structural variance under default Madaros.
#
# Acceptance:
#   - Default ./bin/souc (Madaros) compile+run of the knowledge-free witness prints
#     scaled micro-units fano=250000 and nonfano=4250000 (variances 0.25 / 4.25).
#   - TRUST sentinel PRODUCT_NONASSOC_TRUST_OK from free-function Epistemic path.
#   - Historic L0 run-pass (propagate_nonassoc_variance.sio) prints ALL PASS.
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

echo "== witness_import_product_nonassoc (scaled micro-units) =="
if $SOUC compile tests/epistemic_trust/witness_import_product_nonassoc.sio -o "$OUT/w.elf" >"$OUT/w.compile" 2>&1; then
  mapfile -t lines < <("$OUT/w.elf" | tr -d '[:space:]' | sed 's/\n/\n/g' ; "$OUT/w.elf")
  # Prefer raw two-line capture
  out="$("$OUT/w.elf" 2>/dev/null || true)"
  fano_scaled="$(printf '%s\n' "$out" | sed -n '1p' | tr -d '[:space:]')"
  nonfano_scaled="$(printf '%s\n' "$out" | sed -n '2p' | tr -d '[:space:]')"
  echo "fano_scaled=$fano_scaled nonfano_scaled=$nonfano_scaled"
  if [ "$fano_scaled" = "250000" ] && [ "$nonfano_scaled" = "4250000" ]; then
    echo "PASS: fano=0.25 nonfano=4.25 (scaled $fano_scaled / $nonfano_scaled)"
  else
    echo "FAIL: unexpected scaled values (want 250000 and 4250000)"
    printf '%s\n' "$out"
    fail=1
  fi
else
  echo "FAIL: native compile"
  tail -30 "$OUT/w.compile"
  fail=1
fi

echo "== product_nonassoc_trust sentinel =="
if $SOUC compile tests/epistemic_trust/product_nonassoc_trust.sio -o "$OUT/t.elf" >"$OUT/t.compile" 2>&1; then
  if "$OUT/t.elf" | grep -q 'PRODUCT_NONASSOC_TRUST_OK'; then
    echo "PASS: PRODUCT_NONASSOC_TRUST_OK"
  else
    echo "FAIL: sentinel missing"
    "$OUT/t.elf" || true
    fail=1
  fi
else
  echo "FAIL: trust compile"
  tail -30 "$OUT/t.compile"
  fail=1
fi

echo "== souc run trust driver =="
if $SOUC run tests/epistemic_trust/product_nonassoc_trust.sio >"$OUT/run.out" 2>"$OUT/run.err"; then
  if grep -q 'PRODUCT_NONASSOC_TRUST_OK' "$OUT/run.out"; then
    echo "PASS: souc run trust"
  else
    echo "FAIL: souc run missing sentinel"
    cat "$OUT/run.out"
    fail=1
  fi
else
  echo "FAIL: souc run trust"
  tail -30 "$OUT/run.err"
  fail=1
fi

echo "== souc run L0 propagate_nonassoc_variance =="
if $SOUC run tests/run-pass/propagate_nonassoc_variance.sio >"$OUT/l0.out" 2>"$OUT/l0.err"; then
  if grep -q 'ALL PASS' "$OUT/l0.out"; then
    echo "PASS: L0 ALL PASS"
  else
    echo "FAIL: L0 missing ALL PASS"
    cat "$OUT/l0.out"
    fail=1
  fi
else
  echo "FAIL: L0 souc run"
  tail -30 "$OUT/l0.err"
  fail=1
fi

echo
if [ $fail -eq 0 ]; then
  echo "MADAROS_PRODUCT_NONASSOC_NATIVE_GATE_OK"
  exit 0
fi
echo "MADAROS_PRODUCT_NONASSOC_NATIVE_GATE_FAIL"
exit 1
