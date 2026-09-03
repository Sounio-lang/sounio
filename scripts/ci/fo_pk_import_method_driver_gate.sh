#!/usr/bin/env bash
# CI gate: import epistemic::fo helpers ↔ Pk method FO parity.
#
# Driver: examples/epistemic_fo_second_order/fo_pk_import_method_driver.sio
# Requires multi-mod FO prepass + method FO stack (FO trust ≥42).
#
# Freeze: Var Css/CL/rate and E₂ agree across import, method, call-result, site.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-import-method] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[fo-pk-import-method] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_import_method_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"

fail() {
  echo "FO_PK_IMPORT_METHOD_GATE_FAIL: $*" >&2
  exit 1
}

[[ -f "$DRIVER" ]] || fail "missing driver $DRIVER"
[[ -x "$SOUC" ]] || fail "missing souc at $SOUC"
[[ -x "$MADAROS_BIN" ]] || fail "missing Madaros ELF at $MADAROS_BIN"

OUT_DIR="${SOUNIO_FO_PK_IMPORT_METHOD_DIR:-$(mktemp -d /tmp/sounio-fo-pk-import-method.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

echo "[fo-pk-import-method] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e
"$SOUC" run "$DRIVER" >"$LOG" 2>&1
rc=$?
set -e

if grep -q 'FO_PK_IMPORT_METHOD_DRIVER_FAIL' "$LOG" 2>/dev/null; then
  fail "driver printed FAIL (log=$LOG)"
fi
if ! grep -q 'FO_PK_IMPORT_METHOD_DRIVER_PASS' "$LOG" 2>/dev/null; then
  fail "no PASS token (rc=$rc log=$LOG)"
fi

# Css FO parity across all surfaces
for key in v_css_imp v_css_meth v_css_call v_css_site v_css_imp_site; do
  grep -q "${key}=0.795833" "$LOG" || fail "missing/wrong $key"
done
# E₂ parity
for key in m2_css_imp m2_css_meth m2_css_call; do
  grep -q "${key}=6.724" "$LOG" || fail "missing/wrong $key"
done
# CL FO parity
for key in v_cl_imp v_cl_meth v_cl_call; do
  grep -q "${key}=0.340000" "$LOG" || fail "missing/wrong $key"
done
# Rate FO
grep -q 'v_rate_imp=4.784722' "$LOG" || fail "missing/wrong v_rate_imp"
grep -q 'v_rate_meth=4.784722' "$LOG" || fail "missing/wrong v_rate_meth"
# Point agreement
grep -q 'css_imp=6.666666' "$LOG" || fail "missing/wrong css_imp"
grep -q 'css_meth=6.666666' "$LOG" || fail "missing/wrong css_meth"

echo "[fo-pk-import-method] log=$LOG"
echo "FO_PK_IMPORT_METHOD_GATE_OK"
exit 0
