#!/usr/bin/env bash
# CI gate: FO PK R12 — AUC_τ SS method FO (= Css·τ).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-auct] SKIP: Linux-only gate" >&2; exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in x86_64|amd64) ;; *)
  echo "[fo-pk-auct] SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac
DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_auct_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"
fail() { echo "FO_PK_STRUCT_AUCT_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$DRIVER" && -x "$SOUC" && -x "$MADAROS_BIN" ]] || fail "missing driver/souc/madaros"
OUT_DIR="${SOUNIO_FO_PK_AUCT_DIR:-$(mktemp -d /tmp/sounio-fo-pk-auct.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"
echo "[fo-pk-auct] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e; "$SOUC" run "$DRIVER" >"$LOG" 2>&1; rc=$?; set -e
grep -q 'FO_PK_STRUCT_AUCT_DRIVER_PASS' "$LOG" || fail "no PASS (rc=$rc log=$LOG)"
for tok in \
  'auct_point=80.000000' \
  'v_auct=114.600000' 'e2_auct=80.688000' \
  'v_css_tau=114.600000' 'e2_css_tau=80.688000' \
  'css_point=6.666666' 'v_css=0.795833'
do
  grep -q "$tok" "$LOG" || fail "missing/wrong $tok"
done
echo "[fo-pk-auct] log=$LOG"
echo "FO_PK_STRUCT_AUCT_GATE_OK"
exit 0
