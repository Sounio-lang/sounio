#!/usr/bin/env bash
# Gate: Madaros native-v2 global array-repeat init `[V; N]`.
# Exit 0 only when compile+run yields non-zero fill under default Madaros.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
TMP="${TMPDIR:-/tmp}/madaros_global_array_init_gate_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/repro.sio" << 'SIO'
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
SIO

echo "[gate] compile $TMP/repro.sio under default Madaros"
"$SOUC" compile "$TMP/repro.sio" -o "$TMP/repro.elf"
chmod +x "$TMP/repro.elf"
out="$("$TMP/repro.elf")"; rc=$?
echo "[gate] stdout: $out"
echo "[gate] rc: $rc"
if [[ "$rc" -ne 0 ]]; then
  echo "GLOBAL_ARRAY_INIT_GATE_FAIL rc=$rc"
  exit 1
fi
if [[ "$out" != *"5.700000 (42)"* ]]; then
  echo "GLOBAL_ARRAY_INIT_GATE_FAIL expected '5.700000 (42)' got: $out"
  exit 1
fi
echo "GLOBAL_ARRAY_INIT_GATE_OK"
