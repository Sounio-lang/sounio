#!/usr/bin/env bash
# KL-16e: pin lean_single if/else FO SSHADOW (+ H00 plumbing) merge.
#
# Engine: SOUNIO_SOUC_ENGINE=lean_single (committed seed via bin/souc).
# Witness: tests/run-pass/kl16e_ifelse_shadow_merge.sio
#   if true  { x*2 } else { x*3 } → sensitivity_of(_,0) == 2.0
#   if false { x*2 } else { x*3 } → sensitivity_of(_,0) == 3.0
#   if true  { x } else { y } → s0==1, s1==0
#   if false { x } else { y } → s0==0, s1==1
#
# No Madaros sabotage — this is a seed pin. Residual still OPEN:
#   loop accumulation, HSHADOW multi-pair interproc, a64 atan2/pow AD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[kl16e-ifelse-shadow-merge] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="$(mktemp -d /tmp/sounio-kl16e-ifelse.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== lean_single_kl16e_ifelse_shadow_merge_gate =="
echo "engine: SOUNIO_SOUC_ENGINE=$SOUNIO_SOUC_ENGINE"
echo "souc:   $SOUC"

SRC=tests/run-pass/kl16e_ifelse_shadow_merge.sio
LOG="$OUT/kl16e.log"
ERR="$OUT/kl16e.err"
set +e
"$SOUC" run "$SRC" >"$LOG" 2>"$ERR"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAIL  compile/run rc=$rc"
  sed 's/^/        /' "$ERR" "$LOG" || true
  echo "LEAN_SINGLE_KL16E_IFELSE_SHADOW_MERGE_GATE_FAIL"
  exit 1
fi

expect=$'2.000000\n3.000000\n1.000000\n0.000000\n0.000000\n1.000000\nKL16E_IFELSE_SHADOW_MERGE_PASS'
if ! diff -u <(printf '%s\n' "$expect") "$LOG" >"$OUT/kl16e.diff"; then
  echo "FAIL  stdout mismatch"
  sed 's/^/        /' "$OUT/kl16e.diff" || true
  echo "LEAN_SINGLE_KL16E_IFELSE_SHADOW_MERGE_GATE_FAIL"
  exit 1
fi

echo "ok    kl16e_ifelse_shadow_merge"
echo "LEAN_SINGLE_KL16E_IFELSE_SHADOW_MERGE_GATE_OK"
