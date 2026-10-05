#!/usr/bin/env bash
# KL-16f: pin lean_single loop accumulation of FO SSHADOW (+ H00) on mut f64.
#
# Engine: SOUNIO_SOUC_ENGINE=lean_single (committed seed via bin/souc).
# Witness: tests/run-pass/kl16f_loop_accum.sio
#   s = 0; while i < 3 { s = s + x } with x = measure(1).value
#   sensitivity_of(s, 0) == 3.0
#   hessian_of(s, 0, 0) == 0.0
#
# No Madaros sabotage — this is a seed pin. Residual still OPEN:
#   HSHADOW multi-pair interproc, non-H00 pairs through if/else, a64 atan2/pow AD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[kl16f-loop-accum] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="$(mktemp -d /tmp/sounio-kl16f-loop.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== lean_single_kl16f_loop_accum_gate =="
echo "engine: SOUNIO_SOUC_ENGINE=$SOUNIO_SOUC_ENGINE"
echo "souc:   $SOUC"

SRC=tests/run-pass/kl16f_loop_accum.sio
LOG="$OUT/kl16f.log"
ERR="$OUT/kl16f.err"
set +e
"$SOUC" run "$SRC" >"$LOG" 2>"$ERR"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAIL  compile/run rc=$rc"
  sed 's/^/        /' "$ERR" "$LOG" || true
  echo "LEAN_SINGLE_KL16F_LOOP_ACCUM_GATE_FAIL"
  exit 1
fi

expect=$'3.000000\n0.000000'
if ! diff -u <(printf '%s\n' "$expect") "$LOG" >"$OUT/kl16f.diff"; then
  echo "FAIL  stdout mismatch"
  sed 's/^/        /' "$OUT/kl16f.diff" || true
  echo "LEAN_SINGLE_KL16F_LOOP_ACCUM_GATE_FAIL"
  exit 1
fi

echo "ok    kl16f_loop_accum"
echo "LEAN_SINGLE_KL16F_LOOP_ACCUM_GATE_OK"
