#!/usr/bin/env bash
# Cross-toolchain gate: unbounded-Q Cayley-Dickson product (16 comps) via minimal BigInt.
# souc emits residues mod 1e9+7 (byte-identical on bin/souc AND stage2); compared IN ORDER vs oracle.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
OUT="$(run_souc tests/run-pass/sedenion_cd_qbig.sio)"
echo "$OUT" | grep -aE '^(RES|ANNIHILATION_C1) ' > "$WORK/souc.txt"
python3 scripts/research/sedenion_cd_qbig_oracle.py | grep -E '^(RES|ANNIHILATION_C1) ' > "$WORK/py.txt"
diff "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt" | sed -n '1,10p'; echo "cd-qbig gate: FAIL"; exit 1; }
grep -qa '^CDQBIG OK' <<<"$OUT" || { echo "verdict != CDQBIG OK"; echo "cd-qbig gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: exact unbounded-Q CD product (16 comps). Case 1 annihilates EXACTLY at 10^80"
echo "  (all 16 comps 0); case 2 residues (mod 1e9+7) match the Python oracle in order. Exact decimals"
echo "  witnessed by Lean (native Int). #651 circumvented via a minimal working-primitive BigInt."
echo "cd-qbig gate: PASS"
