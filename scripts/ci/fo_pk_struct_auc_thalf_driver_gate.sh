#!/usr/bin/env bash
# CI gate: FO PK R5 — oral AUC + half-life method FO science driver.
#
# Spec: examples/epistemic_fo_second_order/fo_pk_struct_auc_thalf_driver.sio
# Freezes: AUC Var=114.6 E2=80.688; t½ Var=0.249835; kel Var=5.2e-5
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-auc-thalf] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[fo-pk-auc-thalf] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_auc_thalf_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"

fail() {
  echo "FO_PK_STRUCT_AUC_THALF_GATE_FAIL: $*" >&2
  exit 1
}

[[ -f "$DRIVER" ]] || fail "missing driver $DRIVER"
[[ -x "$SOUC" ]] || fail "missing souc at $SOUC"
[[ -x "$MADAROS_BIN" ]] || fail "missing Madaros ELF at $MADAROS_BIN (set MADAROS_RAW_BIN or rebuild)"

OUT_DIR="${SOUNIO_FO_PK_AUC_THALF_DIR:-$(mktemp -d /tmp/sounio-fo-pk-auc-thalf.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

echo "[fo-pk-auc-thalf] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e
"$SOUC" run "$DRIVER" >"$LOG" 2>&1
rc=$?
set -e

if grep -q 'FO_PK_STRUCT_AUC_THALF_DRIVER_FAIL' "$LOG" 2>/dev/null; then
  fail "driver printed FAIL (log=$LOG)"
fi
if ! grep -q 'FO_PK_STRUCT_AUC_THALF_DRIVER_PASS' "$LOG" 2>/dev/null; then
  fail "no PASS token (rc=$rc log=$LOG)"
fi

grep -q 'auc_point=80.000000' "$LOG" || fail "missing/wrong auc_point"
grep -q 'v_auc=114.600000' "$LOG" || fail "missing/wrong v_auc"
grep -q 'e2_auc=80.688000' "$LOG" || fail "missing/wrong e2_auc"
grep -q 'v_auc_call=114.600000' "$LOG" || fail "missing/wrong v_auc_call"
grep -q 'v_auc_free=114.600000' "$LOG" || fail "missing/wrong v_auc_free"
grep -q 'v_auc_site=114.600000' "$LOG" || fail "missing/wrong v_auc_site"
grep -q 'kel_point=0.100000' "$LOG" || fail "missing/wrong kel_point"
grep -q 'v_kel=0.000052' "$LOG" || fail "missing/wrong v_kel"
grep -q 'thalf_point=6.931471' "$LOG" || fail "missing/wrong thalf_point"
grep -q 'v_thalf=0.249835' "$LOG" || fail "missing/wrong v_thalf"
grep -q 'v_thalf_call=0.249835' "$LOG" || fail "missing/wrong v_thalf_call"
grep -q 'v_thalf_free=0.249835' "$LOG" || fail "missing/wrong v_thalf_free"
grep -q 'v_thalf_peel=0.249835' "$LOG" || fail "missing/wrong v_thalf_peel"

echo "[fo-pk-auc-thalf] log=$LOG"
echo "FO_PK_STRUCT_AUC_THALF_GATE_OK"
exit 0
