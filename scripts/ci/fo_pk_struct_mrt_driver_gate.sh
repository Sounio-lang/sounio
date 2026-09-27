#!/usr/bin/env bash
# CI gate: FO PK R10 — MRT + t90 method FO.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-mrt] SKIP: Linux-only gate" >&2; exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in x86_64|amd64) ;; *)
  echo "[fo-pk-mrt] SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac
DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_mrt_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"
fail() { echo "FO_PK_STRUCT_MRT_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$DRIVER" && -x "$SOUC" && -x "$MADAROS_BIN" ]] || fail "missing driver/souc/madaros"
OUT_DIR="${SOUNIO_FO_PK_MRT_DIR:-$(mktemp -d /tmp/sounio-fo-pk-mrt.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"
echo "[fo-pk-mrt] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e; "$SOUC" run "$DRIVER" >"$LOG" 2>&1; rc=$?; set -e
grep -q 'FO_PK_STRUCT_MRT_DRIVER_PASS' "$LOG" || fail "no PASS (rc=$rc log=$LOG)"
for tok in \
  'mrt_point=10.000000' \
  'v_mrt=0.519999' 'e2_mrt=10.035999' 'v_mrt_peel=0.520000' \
  't90_point=23.025850' 'v_t90=2.756987' 'e2_t90=23.108743' 'v_t90_peel=2.756987'
do
  grep -q "$tok" "$LOG" || fail "missing/wrong $tok"
done
echo "[fo-pk-mrt] log=$LOG"
echo "FO_PK_STRUCT_MRT_GATE_OK"
exit 0
