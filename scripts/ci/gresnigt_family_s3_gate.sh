#!/usr/bin/env bash
# Cross-toolchain gate: Gresnigt family S3 generator (non-monomial) + frame-relative commutators.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/gresnigt_family_s3.sio | grep -E '^(PSI_|FAMILYS3)' | sort > "$WORK/souc.txt"
python3 scripts/research/gresnigt_family_s3_oracle.py > "$WORK/py_all.txt"
grep -E '^(PSI_|FAMILYS3)' "$WORK/py_all.txt" | sort > "$WORK/py.txt"
fail=0
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt"; fail=1; }
grep -q '^FAMILYS3 OK' "$WORK/souc.txt" || { echo "verdict != FAMILYS3 OK"; fail=1; }
# frame-relative mechanism (oracle-verified, matched in Lean): [phi,N]=0, [psi,N]!=0
[ "$(grep '^COMM_PHI_ZERO ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] || { echo "[phi,N] != 0 unexpectedly"; fail=1; }
[ "$(grep '^COMM_PSI_NONZERO ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] || { echo "[psi,N] == 0 unexpectedly"; fail=1; }
[ "$fail" -eq 0 ] || { echo "family S3 gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: Gresnigt's family S3 generator psi is a sedenion automorphism (order 3, maps A->B)"
echo "  and is NON-MONOMIAL (sqrt3) => family symmetry outside the ZD monomial-168. No bridge. E1 vindicated."
echo "  Mechanism: [phi(color-Weyl),N]=0, [psi(family),N]!=0 (frame-relative)."
echo "family S3 gate: PASS"
