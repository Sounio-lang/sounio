#!/usr/bin/env bash
# Cross-toolchain gate: the ZD-geometry graphs have INTEGRAL spectra (Frente B, vector 4/1), certified
# by spectral moments trace(A^k). souc vs Python oracle; /usr/bin/diff.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/s.elf" >/dev/null 2>&1; chmod +x "$WORK/s.elf"; "$WORK/s.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_spectra.sio | grep -E '^(FIBM|K7M|VERTS|SPECTRA)' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_spectra_oracle.py | grep -E '^(FIBM|K7M|VERTS|SPECTRA)' | sort > "$WORK/py.txt"
if diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null && grep -q '^SPECTRA OK' "$WORK/souc.txt"; then
  echo "CROSS-VERIFIED: integral spectra — fiber {+-4,+-2^2,0^6}, 2*K_7 {12,-2^6}; spectral gaps 2 and 14."
  echo "spectra gate: PASS"
else echo "spectra gate: FAIL"; diff "$WORK/souc.txt" "$WORK/py.txt"; exit 1; fi
