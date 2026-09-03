#!/usr/bin/env bash
# CI gate: Pk-method FO ρ-sweep + τ-uncertainty science companion.
#
# Driver: examples/epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio
# Parent: scripts/ci/fo_pk_struct_method_driver_gate.sh
#
# Analytic freeze:
#   exposure Var: ρ=0 → 1575; ρ=0.5 → 2200; ρ=1 → 2825 (= shared peel)
#   Css with σ_τ=0.5 → Var≈0.872993 (> fixed-τ 0.795833)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[fo-pk-rho-tau] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[fo-pk-rho-tau] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

DRIVER="${ROOT_DIR}/examples/epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio"
MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$MADAROS_BIN"
export PATH="$ROOT_DIR/bin:$PATH"

fail() {
  echo "FO_PK_STRUCT_RHO_TAU_GATE_FAIL: $*" >&2
  exit 1
}

[[ -f "$DRIVER" ]] || fail "missing driver $DRIVER"
[[ -x "$SOUC" ]] || fail "missing souc at $SOUC"
[[ -x "$MADAROS_BIN" ]] || fail "missing Madaros ELF at $MADAROS_BIN"

OUT_DIR="${SOUNIO_FO_PK_RHO_TAU_DIR:-$(mktemp -d /tmp/sounio-fo-pk-rho-tau.XXXXXX)}"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

echo "[fo-pk-rho-tau] madaros=$(basename "$MADAROS_BIN") driver=$(basename "$DRIVER")"
set +e
"$SOUC" run "$DRIVER" >"$LOG" 2>&1
rc=$?
set -e

if grep -q 'FO_PK_STRUCT_RHO_TAU_DRIVER_FAIL' "$LOG" 2>/dev/null; then
  fail "driver printed FAIL (log=$LOG)"
fi
if ! grep -q 'FO_PK_STRUCT_RHO_TAU_DRIVER_PASS' "$LOG" 2>/dev/null; then
  fail "no PASS token (rc=$rc log=$LOG)"
fi

grep -q 'v_exp_rho0=1575' "$LOG" || fail "missing/wrong v_exp_rho0"
grep -q 'v_exp_rho05=2200' "$LOG" || fail "missing/wrong v_exp_rho05"
grep -q 'v_exp_rho1=2825' "$LOG" || fail "missing/wrong v_exp_rho1"
grep -q 'v_exp_shared=2825' "$LOG" || fail "missing/wrong v_exp_shared"
grep -q 'v_css_tau=0.872993' "$LOG" || fail "missing/wrong v_css_tau"
grep -q 'v_css_fixed_tau=0.795833' "$LOG" || fail "missing/wrong v_css_fixed_tau"
# Monotone ρ-sweep: indep < mid < full
# (numeric freeze already implies 1575 < 2200 < 2825)

echo "[fo-pk-rho-tau] log=$LOG"
echo "FO_PK_STRUCT_RHO_TAU_GATE_OK"
exit 0
