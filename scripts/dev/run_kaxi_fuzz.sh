#!/bin/bash
# scripts/dev/run_kaxi_fuzz.sh
# Randomized differential fuzzer: warp simulator vs C-transpiled native binary

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-$ROOT_DIR/bin/souc-linux-x86_64}"
TEST_SRC="self-hosted/gpu/test_kaxi_fuzz.sio"
TEST_BIN="/tmp/test_kaxi_fuzz"
OUTDIR="/tmp/kaxi_fuzz_c"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "=== Compiling Sounio fuzz harness ==="
$SOUC "$TEST_SRC" "$TEST_BIN"
chmod +x "$TEST_BIN"

echo "=== Running Sounio fuzz harness ==="
$TEST_BIN > "$OUTDIR/sounio_out.txt" 2>&1

echo "=== Extracting and compiling C kernels ==="

python3 << 'PYEOF'
import os, re, subprocess, sys

outdir = "/tmp/kaxi_fuzz_c"
with open(f"{outdir}/sounio_out.txt") as f:
    text = f.read()

# Verify Sounio harness passed
if "PASS fuzz_all" not in text:
    print("Sounio harness did not report PASS fuzz_all")
    print(text[-2000:])
    sys.exit(1)

# Extract kernels
kernel_blocks = re.findall(
    r'---KERNEL\s*\n(.*?)\n---\s*\nSIM_MEM:\s*(.*?)\nWARP_MEM:\s*(.*?)\n---C_CODE---\s*\n(.*?)\n---C_CODE_END---', text, re.DOTALL)

if not kernel_blocks:
    print("No kernel blocks found in output")
    sys.exit(1)

all_ok = True
for idx, (name, sim_mem, warp_mem, c_code) in enumerate(kernel_blocks):
    name = name.strip()
    sim_vals = [int(x) for x in sim_mem.strip().split()]
    warp_vals = [int(x) for x in warp_mem.strip().split()]

    c_file = f"{outdir}/kernel_{idx}_{name}.c"
    bin_file = f"{outdir}/kernel_{idx}_{name}"
    with open(c_file, "w") as f:
        f.write(c_code)

    # Compile
    result = subprocess.run(
        ["gcc", "-O2", "-I", os.getcwd(), "-o", bin_file, c_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"FAIL {name}: gcc compilation failed")
        print(result.stderr)
        all_ok = False
        continue

    # Run
    result = subprocess.run([bin_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL {name}: native binary crashed")
        print(result.stderr)
        all_ok = False
        continue

    # Parse C_MEM
    c_mem_match = re.search(r'C_MEM:\s*(.*)', result.stdout)
    if not c_mem_match:
        print(f"FAIL {name}: no C_MEM in native output")
        print(result.stdout)
        all_ok = False
        continue

    c_vals = [int(x) for x in c_mem_match.group(1).strip().split()]

    # Primary gate: warp vs native must match
    if warp_vals != c_vals:
        print(f"FAIL {name}: WARP_MEM != C_MEM")
        print(f"  WARP: {warp_vals}")
        print(f"  C:    {c_vals}")
        all_ok = False
        continue

    # Informational: note if scalar diverges from warp (expected for warp-op kernels)
    if sim_vals != warp_vals:
        print(f"PASS {name}: warp/native agree (scalar diverges - expected if warp ops used)")
    else:
        print(f"PASS {name}: scalar/warp/native all agree")

if all_ok:
    print(f"PASS fuzz_validation ({len(kernel_blocks)} kernels)")
    sys.exit(0)
else:
    print("FAIL fuzz_validation")
    sys.exit(1)
PYEOF
