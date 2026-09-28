#!/usr/bin/env bash
# scripts/madaros_const_i64_array_image_gate.sh
#
# Const [i64; N] literal image gate. A let-bound all-i64 literal is stamped
# into the flat rodata; loads read the image. Stores to it are a compile
# error; var bindings keep the store path; escape into a callee keeps the
# heap fill.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== madaros_const_i64_array_image_gate =="

# 1. let literal compiles, runs, and the image is in the ELF
if ! "$SOUC" compile tests/run-pass/const_i64_array_image.sio -o "$OUT/img.elf" >"$OUT/img.log" 2>&1; then
  echo "FAIL: compile const_i64_array_image.sio"; tail -15 "$OUT/img.log" || true; fail=1
else
  chmod +x "$OUT/img.elf"
  if ! "$OUT/img.elf"; then echo "FAIL: run const_i64_array_image.sio"; fail=1
  elif ! python3 scripts/check_const_i64_array_image.py "$OUT/img.elf" | grep -q PASS; then
    echo "FAIL: image absent from ELF"; fail=1
  else echo "PASS: image present and program runs"
  fi
fi

# 2. var binding keeps the store path (123)
if ! "$SOUC" compile tests/run-pass/const_i64_array_var_store.sio -o "$OUT/var.elf" >"$OUT/var.log" 2>&1; then
  echo "FAIL: compile var store"; tail -15 "$OUT/var.log" || true; fail=1
else
  chmod +x "$OUT/var.elf"
  if ! "$OUT/var.elf"; then echo "FAIL: run var store"; fail=1; else echo "PASS: var store"; fi
fi

# 3. escape into a callee keeps the heap fill (26)
if ! "$SOUC" compile tests/run-pass/const_i64_array_escape.sio -o "$OUT/esc.elf" >"$OUT/esc.log" 2>&1; then
  echo "FAIL: compile escape"; tail -15 "$OUT/esc.log" || true; fail=1
else
  chmod +x "$OUT/esc.elf"
  if ! "$OUT/esc.elf"; then echo "FAIL: run escape"; fail=1; else echo "PASS: escape"; fi
fi

# 4. store through an immutable stamped binding is a compile error
cat > "$OUT/store_let.sio" << "SIO"
fn main() -> i64 {
    let xs: [i64; 4] = [3, 5, 7, 11]
    xs[0] = 100
    xs[0]
}
SIO
if "$SOUC" compile "$OUT/store_let.sio" -o "$OUT/store_let.elf" >"$OUT/store_let.log" 2>&1; then
  echo "FAIL: store to let-bound const literal compiled"; fail=1
else
  echo "PASS: store to let-bound const literal rejected"
fi

if [ "$fail" -ne 0 ]; then exit 1; fi
echo "CONST_I64_ARRAY_IMAGE_GATE_PASS"
