#!/usr/bin/env bash
# Cross-toolchain gate: the e_top-seam bridge across the Cayley-Dickson tower (dim 16, 32, 64).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
OUT="$(run_souc tests/run-pass/cd_tower_seam.sio)"
echo "$OUT" | grep -aE '^(LSQ|FWD|SEAM|ZDEQ|NA|OS)[0-9]' | sort > "$WORK/souc.txt"
python3 scripts/research/cd_tower_seam_oracle.py > "$WORK/py_all.txt"
# souc omits ZDEQ64 (dim-64 ZD scan left to the oracle); compare the lines souc DID emit.
grep -aE '^(LSQ|FWD|SEAM|ZDEQ|NA|OS)[0-9]' "$WORK/py_all.txt" | grep -v '^ZDEQ64 ' | sort > "$WORK/py.txt"
fail=0
diff "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt" | sed -n '1,10p'; fail=1; }
grep -qa '^TOWER OK' <<<"$OUT" || { echo "verdict != TOWER OK"; fail=1; }
# the converse (off-seam <=> ZD) at dim 64, carried by the oracle's full scan:
[ "$(grep '^ZDEQ64 ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] || { echo "dim-64 ZD coincidence failed"; fail=1; }
[ "$fail" -eq 0 ] || { echo "cd-tower gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: non-anticommuting = zero-divisor = off-seam coincide at dim 16, 32, 64;"
echo "  cocycle lemma L_i^2=-I holds at each; dim-64 ZD coincidence via the oracle's full scan."
echo "cd-tower gate: PASS"
