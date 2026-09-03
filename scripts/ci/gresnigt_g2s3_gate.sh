#!/usr/bin/env bash
# Cross-toolchain gate: Aut(S)=G2xS3 executed (Frente B vector 4/3 capstone^2).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/gresnigt_g2s3.sio | grep -E '^(EPS_|S3_|COMM_|PSI_|ONLY_|G2S3)' | sort > "$WORK/souc.txt"
python3 scripts/research/gresnigt_g2s3_oracle.py | grep -E '^(EPS_|S3_|COMM_|PSI_|ONLY_|G2S3)' | sort > "$WORK/py.txt"
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt"; echo "g2s3 gate: FAIL"; exit 1; }
grep -q '^G2S3 OK' "$WORK/souc.txt" || { echo "verdict != G2S3 OK"; echo "g2s3 gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: Aut(S)=G2xS3 executed -- family S3=<psi,eps> (braid relation), color-Weyl phi"
echo "  commutes with both family generators (direct product), psi non-monomial and sole octonion<->new mixer."
echo "g2s3 gate: PASS"
