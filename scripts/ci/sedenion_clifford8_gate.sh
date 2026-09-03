#!/usr/bin/env bash
# Cross-toolchain gate: the sedenion left-mult algebra is Cℓ(8) (Frente B, vector 4/3). souc vs the
# Python oracle (exact port of the operator's numpy script). The souc leg certifies the Clifford
# presentation + fingerprint + charges; the oracle additionally confirms the full dim=256 (=Cℓ(8)=M16(C)).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_clifford8.sio | grep -E '^(CL8_|NONANTI|GENS|RANK|Q1_|CL8 )' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_clifford8_oracle.py > "$WORK/py_all.txt"
grep -E '^(CL8_|NONANTI|GENS|RANK|Q1_|CL8 )' "$WORK/py_all.txt" | sort > "$WORK/py.txt"
fail=0
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH souc vs oracle:"; diff "$WORK/souc.txt" "$WORK/py.txt"; fail=1; }
grep -q '^CL8 OK' "$WORK/souc.txt" || { echo "souc verdict != CL8 OK"; fail=1; }
[ "$(grep '^DIM256 ' "$WORK/py_all.txt" | awk '{print $2}')" = "256" ] || { echo "oracle DIM256 != 256"; fail=1; }
[ "$fail" -eq 0 ] || { echo "clifford8 gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: sedenion left-mult algebra = Cℓ(8) (8 anticommuting √(−I) gens, dim 256 = M16(C));"
echo "  ladder rank 4; 42 non-anticommuting pairs all lower-upper; Gresnigt Q_1 = SM electric charges."
echo "clifford8 gate: PASS"
