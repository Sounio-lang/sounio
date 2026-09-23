#!/usr/bin/env bash
# KL-16c: pin lean_single inter-procedural FO ch0 + H[0,0] across user fns.
#
# Engine: SOUNIO_SOUC_ENGINE=lean_single (committed seed via bin/souc).
# Witness: tests/run-pass/kl16c_fo_across_user_fn.sio
#   sensitivity_of(id(x), 0) == 1.0
#   sensitivity_of(sq(x), 0) == 4.0 at x=2
#   hessian_of(sq(x), 0, 0)  == 2.0
#
# No Madaros sabotage — this is a seed pin. Residual still OPEN:
#   multi-channel interproc, loop accumulation, if/else merge, a64 atan2/pow AD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[kl16c-interproc] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="$(mktemp -d /tmp/sounio-kl16c-interproc.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== lean_single_kl16c_interproc_shadow_gate =="
echo "engine: SOUNIO_SOUC_ENGINE=$SOUNIO_SOUC_ENGINE"
echo "souc:   $SOUC"

SRC=tests/run-pass/kl16c_fo_across_user_fn.sio
LOG="$OUT/kl16c.log"
ERR="$OUT/kl16c.err"
set +e
"$SOUC" run "$SRC" >"$LOG" 2>"$ERR"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAIL  compile/run rc=$rc"
  sed 's/^/        /' "$ERR" "$LOG" || true
  echo "LEAN_SINGLE_KL16C_INTERPROC_SHADOW_GATE_FAIL"
  exit 1
fi

expect=$'1.000000\n4.000000\n2.000000\nKL16C_FO_ACROSS_USER_FN_PASS'
if ! diff -u <(printf '%s\n' "$expect") "$LOG" >"$OUT/kl16c.diff"; then
  echo "FAIL  stdout mismatch"
  sed 's/^/        /' "$OUT/kl16c.diff" || true
  echo "LEAN_SINGLE_KL16C_INTERPROC_SHADOW_GATE_FAIL"
  exit 1
fi

echo "ok    kl16c_fo_across_user_fn"
echo "LEAN_SINGLE_KL16C_INTERPROC_SHADOW_GATE_OK"
