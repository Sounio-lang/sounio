#!/usr/bin/env bash
# Madaros gate — bare float intrinsic results must participate in f64 arithmetic.
#
# Residual (Wave14e tip measure, stock prebuilt post-#1392):
#   cos(0.0) bits / print_f64 look correct (integer ABI on IEEE bits)
#   cos(0.0) * 1000.0 → ~4.6e21 (cvtsi2sd on bit pattern of 1.0)
# Root: empty float-builtin stubs (ir_module_ensure_builtin_call_targets) left
# returns_float=0 → call sites omit IR_FLOAT_REG_MARKER_FLAG → native core
# marks the result INT → float binop path converts via cvtsi2sd.
#
# Fix: advertise returns_float=1 on empty float-intrinsic stubs and stamp the
# call-site float marker.
#
# Claim boundary:
#   GREEN: cos/sin/sqrt/exp results in subsequent f64 mul/add (scaled i64)
#   NON-CLAIM: exp(0) series soft spot; bare crossmod f64 Ident; D1 cast shapes
#              beyond this witness; language Knowledge<T> generic import
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

engine_line="$($SOUC --version 2>&1 | head -1 || true)"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single ($engine_line)"
  exit 1
fi
echo "== madaros_bare_float_arith_gate =="
echo "engine: $engine_line"

SRC=tests/compiler/bare_float_arith/main.sio
if ! $SOUC compile "$SRC" -o "$OUT/bare_float_arith.elf" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT/compile.log"
  exit 1
fi
set +e
"$OUT/bare_float_arith.elf" >"$OUT/run.out" 2>"$OUT/run.err"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "FAIL: rc=$rc"
  cat "$OUT/run.out" || true
  cat "$OUT/run.err" || true
  exit 1
fi
if ! grep -q "BARE_FLOAT_ARITH_OK" "$OUT/run.out"; then
  echo "FAIL: missing BARE_FLOAT_ARITH_OK"
  cat "$OUT/run.out" || true
  exit 1
fi
# Line order: 1000, 2000, 1000, ~2718
mapfile -t lines < <(grep -E '^[0-9-]+$' "$OUT/run.out" || true)
if [ "${#lines[@]}" -lt 4 ]; then
  echo "FAIL: expected >=4 integer lines"
  cat "$OUT/run.out"
  exit 1
fi
if [ "${lines[0]}" != "1000" ]; then echo "FAIL: cos*1000 want 1000 got ${lines[0]}"; exit 1; fi
if [ "${lines[1]}" != "2000" ]; then echo "FAIL: sqrt*1000 want 2000 got ${lines[1]}"; exit 1; fi
if [ "${lines[2]}" != "1000" ]; then echo "FAIL: sin+1 want 1000 got ${lines[2]}"; exit 1; fi
em="${lines[3]}"
if [ "$em" -lt 2717 ] || [ "$em" -gt 2719 ]; then
  echo "FAIL: exp(1)*1000 want 2717..2719 got $em"
  exit 1
fi

echo "PASS: cos*1000=1000 sqrt*1000=2000 sin+1=1000 exp(1)*1000=$em"
echo "MADAROS_BARE_FLOAT_ARITH_GATE_OK"
