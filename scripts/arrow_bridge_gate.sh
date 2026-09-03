#!/usr/bin/env bash
# Campaign C4 -- SIO1 Arrow-bridge round-trip gate.
#
# Proves real Sounio -> Python interop end to end:
#   1. compile + RUN the Sounio driver, which builds a known 3x5 DataFrame and
#      writes it to a SIO1 binary file; the driver also prints each cell's exact
#      IEEE-754 i64 bit pattern (via f64_to_bits).
#   2. run the Python bridge, which reads the SAME file back and prints each
#      cell's bits (struct.unpack '<q') plus the reconstructed float.
#   3. assert BIT-EXACT equality of every cell between the two sides.
#
# Exit 0 only if the round trip is lossless for every cell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export SOUNIO_STDLIB_PATH="$PWD/stdlib"
ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-./bin/souc}"

DRIVER_SRC="tests/interop/arrow_bridge_driver.sio"
BRIDGE_PY="scripts/research/arrow_bridge_bridge.py"
SIO1_FILE="/tmp/arrow_bridge.sio1"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
DRIVER_BIN="$WORK/arrow_driver"
DRIVER_OUT="$WORK/driver_stdout.txt"
PY_OUT="$WORK/py_cells.txt"

echo "== [1/4] compile Sounio driver ($ENGINE) =="
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$DRIVER_SRC" -o "$DRIVER_BIN" >/dev/null
chmod +x "$DRIVER_BIN"

echo "== [2/4] run driver (writes $SIO1_FILE) =="
rm -f "$SIO1_FILE"
"$DRIVER_BIN" > "$DRIVER_OUT"
cat "$DRIVER_OUT"
test -f "$SIO1_FILE" || { echo "FAIL: driver did not write $SIO1_FILE"; exit 1; }

echo "== [3/4] Python bridge dump =="
python3 "$BRIDGE_PY" "$SIO1_FILE"
python3 "$BRIDGE_PY" "$SIO1_FILE" --cells > "$PY_OUT"

echo "== [4/4] bit-exact cross-check (Sounio CELL vs Python PYCELL) =="
python3 - "$DRIVER_OUT" "$PY_OUT" <<'PYEOF'
import sys

driver_path, py_path = sys.argv[1], sys.argv[2]

# Parse the Sounio driver stdout. Integer printing inserts newlines, so we
# tokenize the whole stream on whitespace and walk it: after a "CELL" token
# come col, row, bits.
toks = open(driver_path).read().split()
driver = {}
i = 0
while i < len(toks):
    if toks[i] == "CELL":
        col = int(toks[i + 1]); row = int(toks[i + 2]); bits = int(toks[i + 3])
        driver[(col, row)] = bits
        i += 4
    else:
        i += 1

# Parse the Python PYCELL lines: PYCELL <col> <row> <bits> <float>
py = {}
for line in open(py_path):
    p = line.split()
    if p and p[0] == "PYCELL":
        col = int(p[1]); row = int(p[2]); bits = int(p[3])
        py[(col, row)] = bits

if not driver:
    print("FAIL: no CELL records parsed from driver output"); sys.exit(1)

if driver.keys() != py.keys():
    print("FAIL: cell-set mismatch")
    print("  driver-only:", sorted(driver.keys() - py.keys()))
    print("  python-only:", sorted(py.keys() - driver.keys()))
    sys.exit(1)

mism = [(k, driver[k], py[k]) for k in driver if driver[k] != py[k]]
if mism:
    print("FAIL: %d cell(s) differ in exact bits:" % len(mism))
    for k, d, p in mism:
        print("  cell %s: driver=%d python=%d" % (k, d, p))
    sys.exit(1)

print("OK: %d cells round-tripped BIT-EXACT (Sounio f64_to_bits == Python struct '<q')"
      % len(driver))
PYEOF

echo
echo "GATE PASS: SIO1 Sounio->Python round trip is lossless."
