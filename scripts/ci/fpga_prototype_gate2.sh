#!/usr/bin/env bash
# Prototype gate #2 (ADR-022 promotion criterion): a differently
# shaped kernel (1D dot product) must pass through the SAME emitter
# translation rules as the SAN scan. Only the emitter's kernel-name
# CLI argument differs; no translation rule is kernel-specific.
#
# Claim clock: scripts/research/dot_product_reference.fut (Futhark,
# pinned by the same toolchain as ADR-021). Python is orchestration
# tooling only (ADR-009).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

cd "${REPO_ROOT}"
FUTHARK_BIN="${FUTHARK:-futhark}"
command -v "${FUTHARK_BIN}" >/dev/null 2>&1 || { echo "FPGA_PROTOTYPE2_FAIL: futhark not found"; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "FPGA_PROTOTYPE2_FAIL: g++ not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "FPGA_PROTOTYPE2_FAIL: python3 not found"; exit 1; }

python3 tools/fpga/sio_fpga_emit.py examples/fpga/dot_product_kernel.sio \
    "${WORK}/krnl_dot_gen.cpp" --kernel dot_product

cat > "${WORK}/tb_dot.cpp" << 'EOF'
#include <cstdint>
#include <cstdio>
extern "C" int64_t dot_product(int64_t a[1024], int64_t b[1024], int64_t n);
int main() {
    static int64_t a[1024], b[1024];
    long long n;
    if (scanf("%lld", &n) != 1) return 2;
    for (long long i = 0; i < n; i++) scanf("%lld", (long long*)&a[i]);
    for (long long i = 0; i < n; i++) scanf("%lld", (long long*)&b[i]);
    printf("%lld\n", (long long)dot_product(a, b, n));
    return 0;
}
EOF

g++ -O2 -std=c++17 -Wno-unknown-pragmas -o "${WORK}/tb" \
    "${WORK}/krnl_dot_gen.cpp" "${WORK}/tb_dot.cpp"
"${FUTHARK_BIN}" c --entry-point=dot_product -o "${WORK}/fut_ref" \
    scripts/research/dot_product_reference.fut >/dev/null

python3 - "${WORK}" << 'PY'
import random, sys
work = sys.argv[1]
random.seed(220308)
n = 1024
a = [random.randint(-1000, 1000) for _ in range(n)]
b = [random.randint(-1000, 1000) for _ in range(n)]
with open(f"{work}/fut_in.txt", "w") as f:
    f.write("[" + ", ".join(str(v) for v in a) + "] [" + ", ".join(str(v) for v in b) + "]\n")
with open(f"{work}/tb_in.txt", "w") as f:
    f.write(f"{n}\n")
    f.write(" ".join(str(v) for v in a) + "\n")
    f.write(" ".join(str(v) for v in b) + "\n")
PY

"${WORK}/fut_ref" -e dot_product < "${WORK}/fut_in.txt" > "${WORK}/fut_out.txt"
"${WORK}/tb" < "${WORK}/tb_in.txt" > "${WORK}/tb_out.txt"

python3 - "${WORK}" << 'PY'
import sys
work = sys.argv[1]
expected = int(open(f"{work}/fut_out.txt").read().strip().replace("i64", ""))
got = int(open(f"{work}/tb_out.txt").read().strip())
if got != expected:
    print(f"FPGA_PROTOTYPE2_FAIL: generated kernel != Futhark reference: expected {expected}, got {got}")
    sys.exit(1)
print(f"FPGA_PROTOTYPE2_PASS dot={got} cohort=1024 samples (seed 220308)")
PY
