#!/usr/bin/env bash
# CI gate: FO PK R5 import ↔ method parity (AUC + half-life).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-import-auc-thalf] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[fo-pk-import-auc-thalf] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_import_auc_thalf_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"

fail() {
  echo "FO_PK_IMPORT_AUC_THALF_GATE_FAIL: $*" >&2
  exit 1
}

[[ -f "$DRIVER" ]] || fail "missing driver $DRIVER"
[[ -x "$SOUC" ]] || fail "missing souc at $SOUC"
[[ -x "$MADAROS_BIN" ]] || fail "missing Madaros ELF at $MADAROS_BIN"

OUT_DIR="${SOUNIO_FO_PK_IMPORT_AUC_DIR:-$(mktemp -d /tmp/sounio-fo-pk-import-auc.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

echo "[fo-pk-import-auc-thalf] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e
"$SOUC" run "$DRIVER" >"$LOG" 2>&1
rc=$?
set -e

if ! grep -q 'FO_PK_IMPORT_AUC_THALF_DRIVER_PASS' "$LOG" 2>/dev/null; then
  fail "no PASS token (rc=$rc log=$LOG)"
fi

for tok in \
  'v_auc_imp=114.600000' 'v_auc_meth=114.600000' 'v_auc_call=114.600000' 'v_auc_site=114.600000' \
  'e2_auc_imp=80.688000' 'e2_auc_meth=80.688000' \
  'v_kel_imp=0.000052' 'v_kel_meth=0.000052' \
  'v_thalf_imp=0.249835' 'v_thalf_meth=0.249835' 'v_thalf_call=0.249835' 'v_thalf_peel=0.249835' \
  'v_cl_imp=0.340000'
do
  grep -q "$tok" "$LOG" || fail "missing/wrong $tok"
done

echo "[fo-pk-import-auc-thalf] log=$LOG"
echo "FO_PK_IMPORT_AUC_THALF_GATE_OK"
exit 0
