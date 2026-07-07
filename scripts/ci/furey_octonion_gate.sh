#!/usr/bin/env bash
# Cross-toolchain gate: Sounio reproduces Furey's octonion -> one SM generation (Frente B, vector 4/3 A).
# Fermionic ladder algebra {A_i,A_j}=0, {A_i,A_j^dag}=4 delta_ij I over Z[i] + one-generation charge
# multiplicities {1,3,3,1}. souc vs Python oracle; /usr/bin/diff on the value lines.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/s.elf" >/dev/null 2>&1; chmod +x "$WORK/s.elf"; "$WORK/s.elf" 2>/dev/null; fi }
run_souc tests/run-pass/furey_octonion_generation.sio | grep -E '^(LADDER_OK|CHARGE3_|FUREY)' | sort > "$WORK/souc.txt"
python3 scripts/research/furey_octonion_oracle.py | grep -E '^(LADDER_OK|CHARGE3_|FUREY)' | sort > "$WORK/py.txt"
if diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null && grep -q '^FUREY OK' "$WORK/souc.txt"; then
  echo "CROSS-VERIFIED: Furey octonion -> one SM generation — fermionic ladder {A_i,A_j}=0,"
  echo "  {A_i,A_j^dag}=4 delta_ij I over Z[i]; charge multiplicities Q*3 -> {0:1,1:3,2:3,3:1} (SU(3) colour)."
  echo "furey octonion gate: PASS"
else echo "furey octonion gate: FAIL"; diff "$WORK/souc.txt" "$WORK/py.txt"; exit 1; fi
