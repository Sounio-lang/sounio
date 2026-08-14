#!/usr/bin/env bash
# Dual-codebase machine-check of the sedenion Φ-map injectivity structure
# (correction to Theorem 2 of "A Dual Pathway to 168"):
#   1. Python: corroboration (ADR-008 soft unless SOUNIO_FOREIGN_ORACLE_HARD=1)
#   2. Sounio claim: T9/T10 in test_fractal_sedenion_e2e.sio (hard)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"
# shellcheck source=scripts/ci/lib_sounio_claim_oracle.sh
source "$ROOT_DIR/scripts/ci/lib_sounio_claim_oracle.sh"

fail=0
echo "[phi-inj] (1/2) Python corroboration ..."
if ! python3 scripts/research/generate_sedenion_phi_map_injectivity.py >/tmp/phi_inj_py.log 2>&1; then
  sounio_foreign_mismatch "Python oracle reported a failed check" || fail=1
  tail -30 /tmp/phi_inj_py.log || true
elif ! grep -q "ALL CHECKS PASS" /tmp/phi_inj_py.log; then
  sounio_foreign_mismatch "Python missing ALL CHECKS PASS" || fail=1
  tail -30 /tmp/phi_inj_py.log || true
else
  echo "[phi-inj]   Python: ALL CHECKS PASS"
fi

echo "[phi-inj] (2/2) Sounio claim T9/T10 ..."
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[phi-inj] SKIP Sounio leg: x86-64 Linux only"
    [ "$fail" -eq 0 ] || exit 1
    exit 0
    ;;
esac
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
# Historical path is lean_single (Madaros hits private-fn visibility on this graph).
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
set +e
OUT="$("$SOUC" run tests/stdlib/nn/test_fractal_sedenion_e2e.sio 2>/tmp/phi_inj_sio.log)"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "CLAIM FAIL: Sounio test exited nonzero"
  echo "$OUT"
  tail -20 /tmp/phi_inj_sio.log || true
  fail=1
fi
echo "$OUT" | grep -q "T9 OK"  || { echo "CLAIM FAIL: T9 missing"; fail=1; }
echo "$OUT" | grep -q "T10 OK" || { echo "CLAIM FAIL: T10 missing"; fail=1; }
echo "$OUT" | grep -q "/ 10 passed" || { echo "CLAIM FAIL: passed-line missing"; fail=1; }
if [ "$fail" -eq 0 ]; then
  echo "[phi-inj]   Sounio: 10 / 10 passed (T9, T10 present; exit 0)"
  echo "[phi-inj] PASS"
  exit 0
fi
echo "[phi-inj] FAIL"
exit 1
