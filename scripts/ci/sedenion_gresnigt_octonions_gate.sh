#!/usr/bin/env bash
# Cross-toolchain gate: Gresnigt octonion triple + G2 (color-side) automorphism (Frente B vector 4/3).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_gresnigt_octonions.sio | grep -E '^(AUT_OK|ORD3|FIX_|G2_|OCTS_|CYCLE_|GRESNIGT)' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_gresnigt_octonions_oracle.py | grep -E '^(AUT_OK|ORD3|FIX_|G2_|OCTS_|CYCLE_|GRESNIGT)' | sort > "$WORK/py.txt"
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt"; echo "gresnigt gate: FAIL"; exit 1; }
grep -q '^GRESNIGT OK' "$WORK/souc.txt" || { echo "verdict != GRESNIGT OK"; echo "gresnigt gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: Gresnigt's 3 octonions (ZD-free, ambient-equiv, shared quaternion {4,8,12})"
echo "  cyclically permuted by an explicit G2 (color-side) monomial automorphism. NOT the family S3."
echo "gresnigt gate: PASS"
