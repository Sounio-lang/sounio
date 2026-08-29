#!/usr/bin/env bash
# Dual-codebase machine-check of the sedenion Φ-map injectivity structure
# (correction to Theorem 2 of "A Dual Pathway to 168"):
#   1. Python oracle (exact integer arithmetic, reuses the compiler-faithful cd_sigma):
#      168 admissible / Eq.12 = unique ZD sign / I1-I4 / |Im Φ̄|=126 / 42 collisions /
#      GL(3,2) transitive ⇒ equivariant gauge obstructed / cocycle ω=∏_Q ε_g / F_21 /
#      Fano 21×4 split 2/2. ALL CHECKS PASS (exit 0).
#   2. Sounio (self-hosted compiler): T9/T10 in test_fractal_sedenion_e2e.sio mirror the
#      core algebraic facts and must agree (10/10 passed, exit 0).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"

echo "[phi-inj] (1/2) Python oracle ..."
if ! python3 scripts/research/generate_sedenion_phi_map_injectivity.py >/tmp/phi_inj_py.log 2>&1; then
  echo "[phi-inj] FAIL: Python oracle reported a failed check"; tail -30 /tmp/phi_inj_py.log; exit 1
fi
grep -q "ALL CHECKS PASS" /tmp/phi_inj_py.log || { echo "[phi-inj] FAIL: no ALL CHECKS PASS"; tail -30 /tmp/phi_inj_py.log; exit 1; }
echo "[phi-inj]   Python: ALL CHECKS PASS"

echo "[phi-inj] (2/2) Sounio T9/T10 ..."
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64|Linux/amd64) ;;
  *) echo "[phi-inj] SKIP Sounio leg: x86-64 Linux only (Python leg passed)"; exit 0;;
esac
SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
OUT="$("$SOUC" run tests/stdlib/nn/test_fractal_sedenion_e2e.sio 2>/tmp/phi_inj_sio.log)" || {
  echo "[phi-inj] FAIL: Sounio test exited nonzero"; echo "$OUT"; tail -20 /tmp/phi_inj_sio.log; exit 1; }
# run-pass exit 0 above already guarantees fail==0 (main returns 0 iff all pass);
# these greps confirm the new tests actually ran.
grep -q "T9 OK" <<<"$OUT"  || { echo "[phi-inj] FAIL: T9 missing"; echo "$OUT"; exit 1; }
grep -q "T10 OK" <<<"$OUT" || { echo "[phi-inj] FAIL: T10 missing"; echo "$OUT"; exit 1; }
grep -q "/ 10 passed" <<<"$OUT" || { echo "[phi-inj] FAIL: passed-line missing"; echo "$OUT"; exit 1; }
echo "[phi-inj]   Sounio: 10 / 10 passed (T9, T10 present; exit 0)"

echo "[phi-inj] PASS: Φ̄ non-injective (126, 42 collisions) machine-checked in both codebases"
