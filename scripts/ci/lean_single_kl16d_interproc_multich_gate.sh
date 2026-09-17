#!/usr/bin/env bash
# KL-16d: pin lean_single inter-procedural FO channels 0–7 across user fns.
#
# Engine: SOUNIO_SOUC_ENGINE=lean_single (committed seed via bin/souc).
# Witness: tests/run-pass/kl16d_fo_multich_across_user_fn.sio
#   sensitivity_of(add(x, y), 0) == 1.0  (x = measure #0 .value)
#   sensitivity_of(add(x, y), 1) == 1.0  (y = measure #1 .value)
#
# No Madaros sabotage — this is a seed pin. Residual still OPEN:
#   loop accumulation, if/else merge, HSHADOW multi-pair interproc,
#   a64 atan2/pow AD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[kl16d-interproc-multich] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="$(mktemp -d /tmp/sounio-kl16d-interproc.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== lean_single_kl16d_interproc_multich_gate =="
echo "engine: SOUNIO_SOUC_ENGINE=$SOUNIO_SOUC_ENGINE"
echo "souc:   $SOUC"

SRC=tests/run-pass/kl16d_fo_multich_across_user_fn.sio
LOG="$OUT/kl16d.log"
ERR="$OUT/kl16d.err"
set +e
"$SOUC" run "$SRC" >"$LOG" 2>"$ERR"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAIL  compile/run rc=$rc"
  sed 's/^/        /' "$ERR" "$LOG" || true
  echo "LEAN_SINGLE_KL16D_INTERPROC_MULTICH_GATE_FAIL"
  exit 1
fi

expect=$'1.000000\n1.000000\nKL16D_FO_MULTICH_ACROSS_USER_FN_PASS'
if ! diff -u <(printf '%s\n' "$expect") "$LOG" >"$OUT/kl16d.diff"; then
  echo "FAIL  stdout mismatch"
  sed 's/^/        /' "$OUT/kl16d.diff" || true
  echo "LEAN_SINGLE_KL16D_INTERPROC_MULTICH_GATE_FAIL"
  exit 1
fi

echo "ok    kl16d_fo_multich_across_user_fn"
echo "LEAN_SINGLE_KL16D_INTERPROC_MULTICH_GATE_OK"
