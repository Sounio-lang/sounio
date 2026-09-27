#!/usr/bin/env bash
# Named residual gate: epistemic::uncertain_eq under default Madaros native import.
# Promotes the former trust-map "native-import-blocked" row after the IO-driver
# misdiagnosis was corrected (2026-08-04, Attention P0=C).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== uncertain_eq trust witness =="
if ! $SOUC compile tests/epistemic_trust/uncertain_eq_trust.sio -o "$OUT/a.elf" >/dev/null 2>"$OUT/e"; then
  echo "FAIL: compile"
  tail -20 "$OUT/e" || true
  exit 1
fi
out="$("$OUT/a.elf")"
echo "$out"
echo "$out" | grep -q 'UNCERTAIN_EQ_TRUST_OK' || {
  echo "FAIL: missing UNCERTAIN_EQ_TRUST_OK"
  exit 1
}

echo "== uncertain_eq_bernoulli run-pass (IO-fixed) =="
if ! $SOUC compile tests/run-pass/uncertain_eq_bernoulli.sio -o "$OUT/b.elf" >/dev/null 2>"$OUT/be"; then
  echo "FAIL: bernoulli compile"
  tail -20 "$OUT/be" || true
  exit 1
fi
bout="$("$OUT/b.elf")"
echo "$bout" | grep -q 'ALL PASS' || {
  echo "FAIL: bernoulli assertions"
  echo "$bout"
  exit 1
}

echo "MADAROS_UNCERTAIN_EQ_TRUST_GATE_OK"
