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


echo "GLOBAL_ARRAY_INIT_GATE_OK"
