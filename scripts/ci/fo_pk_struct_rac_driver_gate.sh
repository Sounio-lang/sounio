#!/usr/bin/env bash
# CI gate: FO PK R6 — accumulation ratio + residual fraction method FO.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-rac] SKIP: Linux-only gate" >&2; exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in x86_64|amd64) ;; *)
  echo "[fo-pk-rac] SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac
DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_rac_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"
fail() { echo "FO_PK_STRUCT_RAC_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$DRIVER" && -x "$SOUC" && -x "$MADAROS_BIN" ]] || fail "missing driver/souc/madaros"
OUT_DIR="${SOUNIO_FO_PK_RAC_DIR:-$(mktemp -d /tmp/sounio-fo-pk-rac.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"
echo "[fo-pk-rac] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e; "$SOUC" run "$DRIVER" >"$LOG" 2>&1; rc=$?; set -e
grep -q 'FO_PK_STRUCT_RAC_DRIVER_PASS' "$LOG" || fail "no PASS (rc=$rc log=$LOG)"
for tok in \
  'frac_point=0.301195' \
  'v_frac=0.000679' 'v_frac_peel=0.000679' \
  'rac_point=1.431014' 'v_rac=0.002848' 'e2_rac=1.434130' \
  'v_rac_free=0.002848' 'v_rac_peel=0.002848'
do
  grep -q "$tok" "$LOG" || fail "missing/wrong $tok"
done
echo "[fo-pk-rac] log=$LOG"
echo "FO_PK_STRUCT_RAC_GATE_OK"
exit 0
