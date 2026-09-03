#!/usr/bin/env bash
# Cross-toolchain gate: the substrate DYNAMICS of the ZD-geometry graphs (Frente B, vector 4/2) —
# spanning-tree complexity tau (Matrix-Tree, exact Bareiss integer det) + random-walk return counts.
# souc vs Python oracle; /usr/bin/diff on the value lines.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/s.elf" >/dev/null 2>&1; chmod +x "$WORK/s.elf"; "$WORK/s.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_dynamics.sio | grep -E '^(SPANTREE|FIB_|K7_|VERTS|DYNAMICS)' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_dynamics_oracle.py | grep -E '^(SPANTREE|FIB_|K7_|VERTS|DYNAMICS)' | sort > "$WORK/py.txt"
if diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null && grep -q '^DYNAMICS OK' "$WORK/souc.txt"; then
  echo "CROSS-VERIFIED: substrate dynamics — spanning trees tau(fiber)=393216, tau(2*K_7)=1075648"
  echo "  (Matrix-Tree via exact Bareiss integer det = prod(nonzero Laplacian eigenvalues)/n); walk counts agree."
  echo "dynamics gate: PASS"
else echo "dynamics gate: FAIL"; diff "$WORK/souc.txt" "$WORK/py.txt"; exit 1; fi
