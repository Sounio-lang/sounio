#!/bin/bash
# scripts/dev/run_kaxi_ptx.sh — K-AXI → PTX → ptxas → cuobjdump round-trip gate
#
# Compiles the test_kaxi_ptx.sio harness, captures PTX for each kernel,
# assembles each via ptxas, and disassembles each cubin via cuobjdump.
# PASS condition: every kernel assembles to a valid sm_50 cubin whose
# SASS contains an EXIT or RET (proves the entry function is well-formed).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-$ROOT_DIR/bin/souc-linux-x86_64}"
TEST_SRC="self-hosted/gpu/test_kaxi_ptx.sio"
TEST_BIN="/tmp/test_kaxi_ptx"
OUTDIR="${SOUNIO_KAXI_PTX_GATE_DIR:-/tmp/kaxi_ptx_gate}"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "=== Compiling Sounio K-AXI → PTX harness ==="
$SOUC "$TEST_SRC" "$TEST_BIN" >"$OUTDIR/compile.log" 2>&1
chmod +x "$TEST_BIN"

echo "=== Emitting PTX for each K-AXI kernel ==="
$TEST_BIN > "$OUTDIR/all.ptx.txt"

if ! command -v ptxas >/dev/null 2>&1; then
  echo "FAIL: ptxas not on PATH"
  exit 1
fi

echo "=== Assembling and disassembling each kernel ==="
python3 <<PYEOF
import re, subprocess, sys, os

outdir = "$OUTDIR"
text = open(f"{outdir}/all.ptx.txt").read()
blocks = re.findall(r'---KERNEL (\S+)\n(.*?)\n---KERNEL_END', text, re.DOTALL)
if not blocks:
    print("FAIL: no kernels emitted")
    sys.exit(1)

passed = failed = 0
for name, ptx in blocks:
    p = f"{outdir}/{name}.ptx"
    c = f"{outdir}/{name}.cubin"
    open(p, "w").write(ptx + "\n")

    r = subprocess.run(["ptxas", "-arch=sm_50", "-o", c, p],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL {name}: ptxas — {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}")
        failed += 1
        continue

    d = subprocess.run(["cuobjdump", "--dump-sass", c],
                       capture_output=True, text=True)
    if "EXIT" not in d.stdout and "RET" not in d.stdout:
        print(f"FAIL {name}: cubin lacks EXIT/RET in SASS")
        failed += 1
        continue

    print(f"PASS {name}")
    passed += 1

print(f"")
print(f"{passed}/{passed+failed} kernels round-trip K-AXI → PTX → cubin → SASS")
if failed > 0:
    sys.exit(1)
print("PASS kaxi_ptx_gate")
PYEOF
