#!/usr/bin/env bash
# Gate: Madaros length-1 BSS global array IndexGet (Wave9 residual).
# Exit 0 only when compile+run of [T;1] globals yields correct values (no SEGV).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
TMP="${TMPDIR:-/tmp}/madaros_len1_global_array_gate_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

fail() {
  echo "LEN1_GLOBAL_ARRAY_GATE_FAIL $*"
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

# 1) The measured residual: [i64;1] read-only IndexGet
run_case "i64_len1_ro" '
var A: [i64; 1] = [7; 1]
fn main() -> i64 with IO {
  print_int(A[0])
  print("\n")
  if A[0] != 7 { return 1 }
  0
}
' '7'

# 2) Write via helper then read (original #891-class note)
run_case "i64_len1_write" '
var A: [i64; 1] = [7; 1]
fn set() with Mut { A[0] = 9 }
fn main() -> i64 with IO, Mut {
  set()
  print_int(A[0])
  print("\n")
  if A[0] != 9 { return 1 }
  0
}
' '9'

# 3) f64 length-1 (same gsize<=8 trap)
run_case "f64_len1" '
var F: [f64; 1] = [1.5; 1]
fn set() with Mut { F[0] = 2.5 }
fn main() -> i64 with IO, Mut {
  set()
  print(F[0])
  print("\n")
  if F[0] != 2.5 { return 1 }
  0
}
' '2.500000'

# 4) element-list form [V] not only [V;1]
run_case "i64_len1_elemlist" '
var A: [i64; 1] = [42]
fn main() -> i64 with IO {
  print_int(A[0])
  print("\n")
  if A[0] != 42 { return 1 }
  0
}
' '42'

# 5) N=2 control (must stay green)
run_case "i64_len2_control" '
var A: [i64; 2] = [7; 2]
fn set() with Mut { A[0] = 9 }
fn main() -> i64 with IO, Mut {
  set()
  print_int(A[0])
  print(" ")
  print_int(A[1])
  print("\n")
  if A[0] != 9 { return 1 }
  if A[1] != 7 { return 2 }
  0
}
' '9 7'

# 6) Scalar control (must still load the *value*, not SEGV on print_int)
run_case "scalar_i64_control" '
var G: i64 = 42
fn main() -> i64 with IO {
  print_int(G)
  print("\n")
  if G != 42 { return 1 }
  0
}
' '42'

echo "LEN1_GLOBAL_ARRAY_GATE_OK"
