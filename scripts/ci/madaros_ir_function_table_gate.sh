#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

RAW_BIN="${MADAROS_RAW_BIN:-$WORK_DIR/madaros-current-source}"
REPS="${MADAROS_STACK_REPS:-10}"

if ! grep -Eq 'pub functions: \*mut IrFunctionTable' "$ROOT_DIR/self-hosted/ir/ir.sio"; then
  echo "FAIL: IrModule.functions is not backed by IrFunctionTable" >&2
  exit 1
fi
if grep -Eq '^[[:space:]]*(let|var)[[:space:]].*\[IrFunction;[[:space:]]*(8192|16384)\]' \
  "$ROOT_DIR/self-hosted/ir/normalize.sio" \
  "$ROOT_DIR/self-hosted/ir/serialize.sio"; then
  echo "FAIL: stack-sized IrFunction table remains in normalize/serialize" >&2
  exit 1
fi
for witness in \
  'ir_module_fn_set(&! module, 511' \
  'ir_module_fn_set(&! module, 512' \
  'ir_module_fn_set(&! module, IR_MAX_FUNCS - 1' \
  'ir_module_fn_set(&! module, IR_MAX_FUNCS,'; do
  if ! grep -Fq "$witness" "$ROOT_DIR/self-hosted/test_ir.sio"; then
    echo "FAIL: missing function-table witness: $witness" >&2
    exit 1
  fi
done

if [[ -z "${MADAROS_RAW_BIN:-}" ]]; then
  echo "gate: building current-source Madaros at $RAW_BIN"
  (
    ulimit -s 524288
    unset SOUC_BIN SOUNIO_SOUC_BIN
    bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW_BIN"
  )
fi

if [[ ! -x "$RAW_BIN" ]]; then
  echo "FAIL: current-source raw Madaros is not executable: $RAW_BIN" >&2
  exit 1
fi

expected='Hello, Sounio'
pass=0
for rep in $(seq 1 "$REPS"); do
  out_elf="$WORK_DIR/hello-$rep.elf"
  compile_log="$WORK_DIR/hello-$rep.compile.log"
  (
    ulimit -s 8192
    unset SOUC_BIN SOUNIO_SOUC_BIN
    "$RAW_BIN" compile "$ROOT_DIR/examples/hello.sio" -o "$out_elf"
  ) >"$compile_log" 2>&1 || {
    rc=$?
    echo "FAIL: raw current-source compile rep=$rep rc=$rc stack_kib=8192" >&2
    tail -40 "$compile_log" >&2
    exit "$rc"
  }
  chmod +x "$out_elf"
  actual="$($out_elf)"
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: hello rep=$rep expected='$expected' actual='$actual'" >&2
    exit 1
  fi
  pass=$((pass + 1))
done

echo "PASS: raw current-source Madaros hello stack_kib=8192 reps=$pass/$REPS"
