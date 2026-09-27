#!/usr/bin/env bash
# CI gate: dissertation-shaped PK method FO science driver.
#
# Spec surface: examples/epistemic_fo_second_order/fo_pk_struct_method_driver.sio
# Audit:       docs/audit/MADAROS_FO_GUM_STACK_2026-07-27.md (science drivers)
#
# Requires Madaros with FO stack ≥42/42 (method call-recv, free-fn field,
# correlate, struct methods). Token: FO_PK_STRUCT_METHOD_DRIVER_PASS
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-struct-method] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[fo-pk-struct-method] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_method_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"

fail() {
  echo "FO_PK_STRUCT_METHOD_GATE_FAIL: $*" >&2
  exit 1
}

[[ -f "$DRIVER" ]] || fail "missing driver $DRIVER"
[[ -x "$SOUC" ]] || fail "missing souc at $SOUC"
[[ -x "$MADAROS_BIN" ]] || fail "missing Madaros ELF at $MADAROS_BIN (set MADAROS_RAW_BIN or rebuild)"

OUT_DIR="${SOUNIO_FO_PK_STRUCT_DIR:-$(mktemp -d /tmp/sounio-fo-pk-struct.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

echo "[fo-pk-struct-method] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e
"$SOUC" run "$DRIVER" >"$LOG" 2>&1
rc=$?
set -e

if grep -q 'FO_PK_STRUCT_METHOD_DRIVER_FAIL' "$LOG" 2>/dev/null; then
  fail "driver printed FAIL (log=$LOG)"
fi
if ! grep -q 'FO_PK_STRUCT_METHOD_DRIVER_PASS' "$LOG" 2>/dev/null; then
  fail "no PASS token (rc=$rc log=$LOG)"
fi
# Spot-check science table numbers (re-derivable)
grep -q 'v_css_lit=0.795833' "$LOG" || fail "missing/wrong v_css_lit"
grep -q 'v_exp_shared=2825' "$LOG" || fail "missing/wrong v_exp_shared"
grep -q 'v_exp_indep=1575' "$LOG" || fail "missing/wrong v_exp_indep"
grep -q 'v_exp_rho1=2825' "$LOG" || fail "missing/wrong v_exp_rho1"
grep -q 'm2_css_lit=6.724' "$LOG" || fail "missing/wrong m2_css_lit"

echo "[fo-pk-struct-method] log=$LOG"
echo "FO_PK_STRUCT_METHOD_GATE_OK"
exit 0
