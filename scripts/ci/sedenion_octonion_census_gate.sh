#!/usr/bin/env bash
# Cross-toolchain gate: octonion-subalgebra census (Erratum E1 corroboration, Frente B vector 4/3).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_octonion_census.sio | grep -E '^(NSUB|ZDFREE|QUASI|PURE|QUAT|OCTCENSUS)' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_octonion_census_oracle.py | grep -E '^(NSUB|ZDFREE|QUASI|PURE|QUAT|OCTCENSUS)' | sort > "$WORK/py.txt"
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt"; echo "octonion census gate: FAIL"; exit 1; }
grep -q '^OCTCENSUS OK' "$WORK/souc.txt" || { echo "verdict != OCTCENSUS OK"; echo "octonion census gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: 15 basis-aligned octonions, exactly 1 Clifford-pure (={1..7}); quaternion-triple"
echo "  L-non-anti counts {0,6,12} ⟹ family-S3 octonion copies are non-monomial (Erratum E1)."
echo "octonion census gate: PASS"
