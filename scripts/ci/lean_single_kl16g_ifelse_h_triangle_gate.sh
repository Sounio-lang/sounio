#!/usr/bin/env bash
# KL-16g: pin lean_single if/else merge of the full Hessian upper triangle.
#
# Engine: SOUNIO_SOUC_ENGINE=lean_single (committed seed via bin/souc).
# Witness: tests/run-pass/kl16g_ifelse_h_triangle.sio
#   if true  { x*y } else { 0.0 } → hessian_of(_, 0, 1) == 1.0
#   if false { 0.0 } else { x*y } → hessian_of(_, 0, 1) == 1.0
#   if true  { x*x } else { x*x } → hessian_of(_, 0, 0) == 2.0
#
# No Madaros sabotage — this is a seed pin. Residual still OPEN:
#   HSHADOW multi-pair interproc, a64 atan2/pow AD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[kl16g-ifelse-h-triangle] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="$(mktemp -d /tmp/sounio-kl16g-ifelse.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== lean_single_kl16g_ifelse_h_triangle_gate =="
echo "engine: SOUNIO_SOUC_ENGINE=$SOUNIO_SOUC_ENGINE"
echo "souc:   $SOUC"

SRC=tests/run-pass/kl16g_ifelse_h_triangle.sio
LOG="$OUT/kl16g.log"
ERR="$OUT/kl16g.err"
set +e
"$SOUC" run "$SRC" >"$LOG" 2>"$ERR"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAIL  compile/run rc=$rc"
  sed 's/^/        /' "$ERR" "$LOG" || true
  echo "LEAN_SINGLE_KL16G_IFELSE_H_TRIANGLE_GATE_FAIL"
  exit 1
fi

expect=$'1.000000\n1.000000\n2.000000\nKL16G_IFELSE_H_TRIANGLE_PASS'
if ! diff -u <(printf '%s\n' "$expect") "$LOG" >"$OUT/kl16g.diff"; then
  echo "FAIL  stdout mismatch"
  sed 's/^/        /' "$OUT/kl16g.diff" || true
  echo "LEAN_SINGLE_KL16G_IFELSE_H_TRIANGLE_GATE_FAIL"
  exit 1
fi

echo "ok    kl16g_ifelse_h_triangle"
echo "LEAN_SINGLE_KL16G_IFELSE_H_TRIANGLE_GATE_OK"
