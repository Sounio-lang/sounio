#!/usr/bin/env bash
# KL-11 (partial): pow FO / Hessian transfer on Madaros.
#
# Witness: tests/run-pass/kl11_pow_fo_hessian.sio
#   hessian_of(pow(x, 3.0), 0, 0) at x=0.5 must be 3.0 (analytic 6x), not the
#   opaque-call zero that fo_apply_call_transfer used to emit.
#
# Sabotage: SOUNIO_SABOTAGE_HESSIAN_CHAIN=1 must kill the sentinel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
unset SOUNIO_SABOTAGE_HESSIAN_CHAIN || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_kl11_pow_fo_gate =="
echo "compiler under test: $SOUC ${SOUNIO_MADAROS_BIN:+(SOUNIO_MADAROS_BIN=$SOUNIO_MADAROS_BIN)}"

fails=0
SRC=tests/run-pass/kl11_pow_fo_hessian.sio
SENT=KL11_POW_FO_HESSIAN_PASS

run_one() {
  local label="$1" expect="$2"; shift 2
  local elf="$OUT/$label.elf" log="$OUT/$label.log"
  if ! env "$@" "$SOUC" compile "$SRC" -o "$elf" >"$OUT/$label.compile.log" 2>&1; then
    echo "FAIL  $label -- compile"
    tail -40 "$OUT/$label.compile.log" || true
    fails=$((fails + 1)); return
  fi
  chmod +x "$elf"
  set +e
  "$elf" >"$log" 2>&1
  local rc=$?
  set -e
  local got=absent
  grep -q "$SENT" "$log" && got=present
  case "$expect" in
    pass)
      if [[ "$rc" -eq 0 && "$got" == present ]]; then
        echo "ok    $label -- $SENT"
        sed 's/^/        /' "$log"
      else
        echo "FAIL  $label -- rc=$rc sentinel=$got"
        sed 's/^/        /' "$log" || true
        fails=$((fails + 1))
      fi ;;
    fail)
      if [[ "$got" == absent ]]; then
        echo "ok    $label -- sabotage kills it (rc=$rc)"
      else
        echo "FAIL  $label -- sabotage armed and $SENT still printed"
        sed 's/^/        /' "$log" || true
        fails=$((fails + 1))
      fi ;;
  esac
}

run_one pow_hess pass
run_one pow_hess_sab fail SOUNIO_SABOTAGE_HESSIAN_CHAIN=1

if [[ "$fails" -ne 0 ]]; then
  echo "MADAROS_KL11_POW_FO_GATE=FAIL ($fails)"
  exit 1
fi
echo "MADAROS_KL11_POW_FO_GATE=PASS"
