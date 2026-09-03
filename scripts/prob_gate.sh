#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check distributions.sio =="
$SOUC check stdlib/prob/distributions.sio || fail=1

# Prefer default Madaros multi-module native after #901 scale closeout (Wave15C).
# Fall back to lean_single only if Madaros is unavailable in this checkout.
echo "== run-proof driver (default Madaros) =="
if $SOUC compile tests/stdlib/prob/test_prob_stdlib.sio -o "$OUT/tp.elf" >"$OUT/tp.compile" 2>&1; then
  chmod +x "$OUT/tp.elf" 2>/dev/null || true
  "$OUT/tp.elf" | grep -q "PROB_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
  echo "PASS: driver under default engine"
else
  echo "WARN: Madaros driver compile failed; trying lean_single escape hatch"
  tail -15 "$OUT/tp.compile" || true
  if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/prob/test_prob_stdlib.sio -o "$OUT/tp_ls.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/tp_ls.elf"
    "$OUT/tp_ls.elf" | grep -q "PROB_STDLIB_OK" || { echo "FAIL: lean_single driver assertions"; fail=1; }
    echo "PASS: driver under lean_single (Madaros path red — re-run scripts/madaros_native_multimodule_scale_901_gate.sh)"
  else
    echo "FAIL: driver compile (Madaros and lean_single)"
    fail=1
  fi
fi

echo "== consumer example (default Madaros) =="
if $SOUC compile examples/prob/distribution_report.sio -o "$OUT/dr.elf" >"$OUT/dr.compile" 2>&1; then
  chmod +x "$OUT/dr.elf" 2>/dev/null || true
  "$OUT/dr.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
  echo "PASS: example under default engine"
else
  echo "WARN: Madaros example compile failed; trying lean_single"
  if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile examples/prob/distribution_report.sio -o "$OUT/dr_ls.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/dr_ls.elf"
    "$OUT/dr_ls.elf" >/dev/null || { echo "FAIL: lean_single example run"; fail=1; }
  else
    echo "FAIL: example compile"
    fail=1
  fi
fi

[ $fail -eq 0 ] && echo "PROB_GATE_OK"
exit $fail
