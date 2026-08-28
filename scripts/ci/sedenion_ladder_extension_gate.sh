#!/usr/bin/env bash
# Cross-toolchain gate: the SEDENION EXTENSION of the Furey ladder (Frente B, vector 4/3 Part B) —
# the octonion SM generation persists (B1) and the doubling adds exactly ONE more fermionic mode
# (greedy rank 3 -> 4), NOT a clean second generation. souc vs Python oracle; /usr/bin/diff on the
# value lines.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/s.elf" >/dev/null 2>&1; chmod +x "$WORK/s.elf"; "$WORK/s.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_ladder_extension.sio | grep -E '^(B1_OK|OCT_RANK|SED_RANK|SEDEXT)' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_ladder_extension_oracle.py | grep -E '^(B1_OK|OCT_RANK|SED_RANK|SEDEXT)' | sort > "$WORK/py.txt"
if diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null && grep -q '^SEDEXT OK' "$WORK/souc.txt"; then
  echo "CROSS-VERIFIED: sedenion ladder extension — octonion SM generation persists (B1_OK=1),"
  echo "  greedy max fermionic rank 3 (octonion) -> 4 (sedenion): the doubling adds exactly ONE mode,"
  echo "  NOT a clean second generation. Particle-physics interpretation OPEN."
  echo "sedenion ladder extension gate: PASS"
else echo "sedenion ladder extension gate: FAIL"; diff "$WORK/souc.txt" "$WORK/py.txt"; exit 1; fi
