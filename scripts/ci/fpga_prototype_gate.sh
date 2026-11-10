#!/usr/bin/env bash
# Prototype gate for the FPGA backend MVP (ADR-022).
#
# Pipeline under test:
#   examples/fpga/san_scan_kernel.sio
#     -> tools/fpga/sio_fpga_emit.py        (prototype emitter, tooling)
#     -> generated HLS C++ compiled with g++ (native csim-equivalent;
#        no Xilinx toolchain on this node, so no vitis csim/cosim here)
#     -> outputs compared against the Futhark reference binary.
#
# The claim clock is the Futhark reference (ADR-021, pinned by
# scripts/ci/san_fpga_futhark_reference_gate.sh). Python below is
# orchestration/measurement tooling only (ADR-009).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

cd "${REPO_ROOT}"
FUTHARK_BIN="${FUTHARK:-futhark}"
command -v "${FUTHARK_BIN}" >/dev/null 2>&1 || { echo "FPGA_PROTOTYPE_FAIL: futhark not found"; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "FPGA_PROTOTYPE_FAIL: g++ not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "FPGA_PROTOTYPE_FAIL: python3 not found"; exit 1; }

# 1. Emit HLS C++ + testbench from the Sounio kernel.
python3 tools/fpga/sio_fpga_emit.py examples/fpga/san_scan_kernel.sio \
    "${WORK}/krnl_san_scan_gen.cpp" --tb "${WORK}/tb_san_scan_gen.cpp"

# 2. Build the generated kernel natively and the Futhark reference.
g++ -O2 -std=c++17 -Wno-unknown-pragmas -o "${WORK}/tb" \
    "${WORK}/krnl_san_scan_gen.cpp" "${WORK}/tb_san_scan_gen.cpp"
"${FUTHARK_BIN}" c --entry-point=san_scan -o "${WORK}/fut_ref" \
    scripts/research/san_scan_reference.fut >/dev/null

# 3. One deterministic random cohort for both implementations.
python3 - "${WORK}" << 'PY'
import random, sys
work = sys.argv[1]
random.seed(220307)
n, k, q = 200, 7, 18022
lut = [100 * (i + 1) for i in range(8)]
confs = [[random.randint(0, 32767) for _ in range(k)] for _ in range(n)]
with open(f"{work}/fut_in.txt", "w") as f:
    f.write("[" + ", ".join("[" + ", ".join(str(c) for c in r) + "]" for r in confs) + "]")
    f.write(f" {q} [" + ", ".join(str(v) for v in lut) + "]\n")
with open(f"{work}/tb_in.txt", "w") as f:
    f.write(f"{n} {q}\n")
    f.write(" ".join(str(v) for v in lut) + "\n")
    for r in confs:
        f.write(" ".join(str(c) for c in r) + "\n")
PY

# 4. Run both and compare bit-exactly.
"${WORK}/fut_ref" -e san_scan < "${WORK}/fut_in.txt" > "${WORK}/fut_out.txt"
"${WORK}/tb" < "${WORK}/tb_in.txt" > "${WORK}/tb_out.txt"

python3 - "${WORK}" << 'PY'
import re, sys
work = sys.argv[1]
fut = open(f"{work}/fut_out.txt").read().replace("i64", "").replace("[", "").replace("]", "")
fut_vals = [int(x) for x in re.split(r"[,\s]+", fut.strip()) if x]
tb_lines = open(f"{work}/tb_out.txt").read().splitlines()
tb_hist = [int(x) for x in tb_lines[0].split()]
tb_macs = int(tb_lines[1])
expected = fut_vals  # hist[8], catastrophes, macs
got = tb_hist + [tb_hist[7], tb_macs]
if got != expected:
    print(f"FPGA_PROTOTYPE_FAIL: generated kernel != Futhark reference\n  expected {expected}\n  got      {got}")
    sys.exit(1)
print(f"FPGA_PROTOTYPE_PASS hist={tb_hist} macs={tb_macs} cohort=200 samples (seed 220307)")
PY
