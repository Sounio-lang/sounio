#!/usr/bin/env bash
# CI gate: multi-dose / τ-series FO science companion.
#
# Driver: examples/epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio
# Family: fo_pk_struct_method_driver + fo_pk_struct_rho_tau_driver
#
# Freeze:
#   Css(τ=8/12/24) = 10 / 6.666… / 3.333…
#   Var ∝ 1/τ²: 1.790625 / 0.795833 / 0.198958
#   scales 8/12 → 2.25, 24/12 → 0.25
#   kel=0.1, Var(kel)=0.000052 (η cancels)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-multidose] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[fo-pk-multidose] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"

fail() {
  echo "FO_PK_STRUCT_MULTIDOSE_GATE_FAIL: $*" >&2
  exit 1
}

[[ -f "$DRIVER" ]] || fail "missing driver $DRIVER"
[[ -x "$SOUC" ]] || fail "missing souc at $SOUC"
[[ -x "$MADAROS_BIN" ]] || fail "missing Madaros ELF at $MADAROS_BIN"

OUT_DIR="${SOUNIO_FO_PK_MULTIDOSE_DIR:-$(mktemp -d /tmp/sounio-fo-pk-multidose.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

echo "[fo-pk-multidose] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e
"$SOUC" run "$DRIVER" >"$LOG" 2>&1
rc=$?
set -e

if grep -q 'FO_PK_STRUCT_MULTIDOSE_DRIVER_FAIL' "$LOG" 2>/dev/null; then
  fail "driver printed FAIL (log=$LOG)"
fi
if ! grep -q 'FO_PK_STRUCT_MULTIDOSE_DRIVER_PASS' "$LOG" 2>/dev/null; then
  fail "no PASS token (rc=$rc log=$LOG)"
fi

grep -q 'css8=10.000000' "$LOG" || fail "missing/wrong css8"
grep -q 'css12=6.666666' "$LOG" || fail "missing/wrong css12"
grep -q 'css24=3.333333' "$LOG" || fail "missing/wrong css24"
# Var ∝ 1/τ² freeze (implies scales 2.25 and 0.25)
grep -q 'v_css8=1.790625' "$LOG" || fail "missing/wrong v_css8"
grep -q 'v_css12=0.795833' "$LOG" || fail "missing/wrong v_css12"
grep -q 'v_css24=0.198958' "$LOG" || fail "missing/wrong v_css24"
grep -q 'm2_css12=6.724' "$LOG" || fail "missing/wrong m2_css12"
# ΣH under multi-site FO load prints ~7.20 (parent solo Css path prints 7.292592)
grep -qE 'h_css12=7\.[12]' "$LOG" || fail "missing/wrong h_css12"
grep -q 'kel_point=0.100000' "$LOG" || fail "missing/wrong kel"
grep -q 'v_kel=0.000052' "$LOG" || fail "missing/wrong v_kel"
# Call-result / free parity at τ=12
grep -q 'v_css12_call=0.795833' "$LOG" || fail "missing/wrong v_css12_call"
grep -q 'v_css12_free=0.795833' "$LOG" || fail "missing/wrong v_css12_free"

echo "[fo-pk-multidose] log=$LOG"
echo "FO_PK_STRUCT_MULTIDOSE_GATE_OK"
exit 0
