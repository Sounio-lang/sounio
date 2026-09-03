#!/usr/bin/env bash
# Cross-toolchain gate: Furey Cl(6) charge + G2 automorphism does-not-preserve-charge (vector 4/3).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/furey_charge_g2.sio | grep -E '^(WITT_OK|COMM_NONZERO|CHARGE_OK|FUREYCHARGE)' | sort > "$WORK/souc.txt"
python3 scripts/research/furey_charge_g2_oracle.py | grep -E '^(WITT_OK|COMM_NONZERO|CHARGE_OK|FUREYCHARGE)' | sort > "$WORK/py.txt"
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt"; echo "furey charge gate: FAIL"; exit 1; }
grep -q '^FUREYCHARGE OK' "$WORK/souc.txt" || { echo "verdict != FUREYCHARGE OK"; echo "furey charge gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: Furey Cl(6) Witt ladder relations exact; the G2 automorphism phi does NOT preserve"
echo "  the charge operator ([P,D]!=0) => phi is not a family symmetry at the state level. E1 stands."
echo "furey charge gate: PASS"
