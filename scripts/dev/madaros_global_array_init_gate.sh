#!/usr/bin/env bash
# Gate: Madaros native-v2 global array init — scalar, i64/f64 array-repeat, element-list.
# Exit 0 only when compile+run yields correct fills under default Madaros.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
TMP="${TMPDIR:-/tmp}/madaros_global_array_init_gate_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

fail() {
  echo "GLOBAL_ARRAY_INIT_GATE_FAIL $*"
  exit 1
}

run_case() {
  local name="$1" src="$2" expect="$3"
  local f="$TMP/${name}.sio"
  local elf="$TMP/${name}.elf"
  printf '%s\n' "$src" > "$f"
  echo "[gate] $name"
  "$SOUC" compile "$f" -o "$elf"
  chmod +x "$elf"
  local out rc
  set +e
  out="$("$elf" 2>&1)"
  rc=$?
  set -e
  echo "[gate]   stdout: $out"
  echo "[gate]   rc: $rc"
  [[ "$rc" -eq 0 ]] || fail "$name rc=$rc out=$out"
  [[ "$out" == *"$expect"* ]] || fail "$name expected '$expect' got: $out"
}

# 1) i64 array-repeat + f64 print coexistence (#1305)
run_case "i64_repeat" '
var CT: [i64; 2] = [42; 2]
fn main() -> i64 with IO {
  let p = 5.7
  print(p)
  print(" (")
  print_int(CT[0])
  print(")\n")
  if CT[0] != 42 { return 1 }
  if CT[1] != 42 { return 2 }
  0
}
' '5.700000 (42)'

# 2) f64 array-repeat residual
run_case "f64_repeat" '
var F: [f64; 2] = [1.5; 2]
fn main() -> i64 with IO {
  print(F[0])
  print(" ")
  print(F[1])
  print("\n")
  if F[0] != 1.5 { return 1 }
  if F[1] != 1.5 { return 2 }
  0
}
' '1.500000 1.500000'

# 3) element-list i64 + f64
run_case "elem_list" '
var A: [i64; 3] = [10, 20, 30]
var B: [f64; 2] = [1.5, 2.5]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("\n")
  print(B[0]); print(" "); print(B[1]); print("\n")
  if A[0] != 10 { return 1 }
  if A[1] != 20 { return 2 }
  if A[2] != 30 { return 3 }
  if B[0] != 1.5 { return 4 }
  if B[1] != 2.5 { return 5 }
  0
}
' '10 20 30'

# 4) scalar control (explicit print_int — always worked)
run_case "scalar" '
var G: i64 = 42
fn main() -> i64 with IO {
  print_int(G)
  print("\n")
  if G != 42 { return 1 }
  0
}
' '42'

# 5) scalar f64 print/println via auto-dispatch (#1325 residual)
run_case "scalar_f64_print" '
var G: f64 = 1.5
fn main() -> i64 with IO {
  print(G)
  print("
")
  println(G)
  if G != 1.5 { return 1 }
  0
}
' '1.500000'

# 6) scalar i64 print/println via auto-dispatch (same class as f64 char* SEGV)
run_case "scalar_i64_print" '
var G: i64 = 42
fn main() -> i64 with IO {
  print(G)
  print("
")
  println(G)
  if G != 42 { return 1 }
  0
}
' '42'

# 7) Wave8: packed i8 signed load (movsx) + u8 zero-extend (#1325 residual)
run_case "i8_signed" '
var NEG: [i8; 4] = [-1, -128, 127, 0]
var UPOS: [u8; 2] = [255, 128]
fn main() -> i64 with IO {
  print_int(NEG[0] as i64); print(" "); print_int(NEG[1] as i64); print(" "); print_int(NEG[2] as i64); print("
")
  print_int(UPOS[0] as i64); print(" "); print_int(UPOS[1] as i64); print("
")
  if NEG[0] as i64 != 0 - 1 { return 1 }
  if NEG[1] as i64 != 0 - 128 { return 2 }
  if NEG[2] as i64 != 127 { return 3 }
  if UPOS[0] as i64 != 255 { return 4 }
  if UPOS[1] as i64 != 128 { return 5 }
  0
}
' '-1 -128 127'

# 8) Wave8: const-fold element-list (`0 - n`, bool, binary arith)
run_case "constfold_list" '
var A: [i64; 3] = [0 - 1, 0 - 2, 3]
var B: [bool; 2] = [true, false]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  if A[0] != 0 - 1 { return 1 }
  if A[1] != 0 - 2 { return 2 }
  if A[2] != 3 { return 3 }
  if !B[0] { return 4 }
  if B[1] { return 5 }
  0
}
' '-1 -2 3'


# 9) Wave9: cast element-list (`1 as i64`) — was fully dropped → zeros
run_case "cast_list" '
var A: [i64; 2] = [1 as i64, 2 as i64]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print("
")
  if A[0] != 1 { return 1 }
  if A[1] != 2 { return 2 }
  0
}
' '1 2'

# 10) Wave9: ident of earlier scalar global in element-list
run_case "ident_list" '
var X: i64 = 7
var A: [i64; 3] = [X, 20, 30]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  if A[0] != 7 { return 1 }
  if A[1] != 20 { return 2 }
  if A[2] != 30 { return 3 }
  0
}
' '7 20 30'

# 11) Wave10: pure zero-arg call in element-list folds to real values
# (wave9 residual was fail-closed BSS zero; must never left-shift to `20 30 0`).
run_case "call_list_pure" '
fn ten() -> i64 { 10 }
var A: [i64; 3] = [ten(), 20, 30]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  if A[0] != 10 { return 1 }
  if A[1] != 20 { return 2 }
  if A[2] != 30 { return 3 }
  0
}
' '10 20 30'

# 12) Wave12: multi-stmt pure return chain folds (was Wave10 residual BSS zero).
# Must never left-shift remaining const words to `20 30 0`.
run_case "call_list_multistmt" '
fn multi() -> i64 {
  let x = 10
  x
}
var A: [i64; 3] = [multi(), 20, 30]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  if A[0] != 10 { return 1 }
  if A[1] != 20 { return 2 }
  if A[2] != 30 { return 3 }
  0
}
' '10 20 30'

# 12b) Wave12: multi-stmt pure chain with dependent lets
run_case "call_list_multistmt_chain" '
fn multi() -> i64 {
  let x = 10
  let y = x + 5
  y
}
var A: [i64; 3] = [multi(), 20, 30]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  if A[0] != 15 { return 1 }
  if A[1] != 20 { return 2 }
  if A[2] != 30 { return 3 }
  0
}
' '15 20 30'

# 12c) Residual: effectful callee stays fail-closed zeros (no left-shift).
run_case "nonconst_failclosed" '
fn impure() -> i64 with IO {
  10
}
var A: [i64; 3] = [impure(), 20, 30]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  // Fail-closed: effects prevent fold → BSS zero (not shifted 20 30 0).
  if A[0] != 0 { return 1 }
  if A[1] != 0 { return 2 }
  if A[2] != 0 { return 3 }
  0
}
' '0 0 0'

# 12d) Wave13e: pure paramful SINGLE-STMT calls with const args fold.
# Shapes: binary-of-Idents (kind1) and bare Ident identity (kind2).
# Must never left-shift remaining const words to `1 2 0`.
run_case "call_list_args" '
fn add2(a: i64, b: i64) -> i64 { a + b }
fn mul2(a: i64, b: i64) -> i64 { a * b }
fn id1(x: i64) -> i64 { x }
var A: [i64; 3] = [add2(10, 20), 1, 2]
var B: [i64; 3] = [mul2(3, 7), id1(9), 5]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  print_int(B[0]); print(" "); print_int(B[1]); print(" "); print_int(B[2]); print("
")
  if A[0] != 30 { return 1 }
  if A[1] != 1 { return 2 }
  if A[2] != 2 { return 3 }
  if B[0] != 21 { return 4 }
  if B[1] != 9 { return 5 }
  if B[2] != 5 { return 6 }
  0
}
' '30 1 2
21 9 5'

# 12e) Wave15a: multi-stmt pure paramful single-return folds (was Wave13e residual).
# Shape: `{ let x = a + 1; x }` with const args → real BSS words, no left-shift.
run_case "call_list_args_multistmt" '
fn f(a: i64) -> i64 {
  let x = a + 1
  x
}
var C: [i64; 2] = [f(9), 2]
fn main() -> i64 with IO {
  print_int(C[0]); print(" "); print_int(C[1]); print("
")
  if C[0] != 10 { return 1 }
  if C[1] != 2 { return 2 }
  0
}
' '10 2'

# 12f) Wave15a poison regression: multi-stmt KIND 3 registration must NOT
# poison same-module KIND 1/2 (Wave13e body-pointer table did: add2+f → 0 0 0).
run_case "call_list_args_kind3_no_poison" '
fn add2(a: i64, b: i64) -> i64 { a + b }
fn f(a: i64) -> i64 {
  let x = a + 1
  x
}
var A: [i64; 3] = [add2(10, 20), 1, 2]
var C: [i64; 2] = [f(9), 2]
fn main() -> i64 with IO {
  print_int(A[0]); print(" "); print_int(A[1]); print(" "); print_int(A[2]); print("
")
  print_int(C[0]); print(" "); print_int(C[1]); print("
")
  if A[0] != 30 { return 1 }
  if A[1] != 1 { return 2 }
  if A[2] != 2 { return 3 }
  if C[0] != 10 { return 4 }
  if C[1] != 2 { return 5 }
  0
}
' '30 1 2
10 2'

# 12g) Wave15a: multi-stmt pure paramful dependent-let chain (param/local + lit).
# Uses Wave12-proven expression shapes (`x + 5`). Pre-existing residual (Wave12):
# Ident+Ident binary pure fold evaluates both sides as the RHS Ident
# (`a+b` with a=9,b=5 → 10; also zero-arg `let a=9; let b=5; a+b`).
# Do not claim Ident+Ident body folds here.
run_case "call_list_args_multistmt_chain" '
fn g(a: i64) -> i64 {
  let x = a + 1
  let y = x + 5
  y
}
var D: [i64; 3] = [g(9), 1, 2]
fn main() -> i64 with IO {
  print_int(D[0]); print(" "); print_int(D[1]); print(" "); print_int(D[2]); print("
")
  if D[0] != 15 { return 1 }
  if D[1] != 1 { return 2 }
  if D[2] != 2 { return 3 }
  0
}
' '15 1 2'

# 13) Wave9: packed i8 BSS size is physical (adjacent arrays independent + dense)
run_case "i8_adjacent" '
var A: [i8; 4] = [1, 2, 3, 4]
var B: [i8; 4] = [10, 20, 30, 40]
fn main() -> i64 with IO {
  print_int(A[0] as i64); print(" "); print_int(A[3] as i64); print(" ");
  print_int(B[0] as i64); print(" "); print_int(B[3] as i64); print("
")
  if A[0] as i64 != 1 { return 1 }
  if A[3] as i64 != 4 { return 2 }
  if B[0] as i64 != 10 { return 3 }
  if B[3] as i64 != 40 { return 4 }
  0
}
' '1 4 10 40'


echo "GLOBAL_ARRAY_INIT_GATE_OK"
