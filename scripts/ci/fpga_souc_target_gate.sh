#!/usr/bin/env bash
# CI gate: souc --target fpga-u250 (SIO-307, ADR-022 subset v0 promotion).
#
# The self-hosted emitter (self-hosted/compiler/codegen/hardware/fpga_emit.sio)
# must translate both example kernels to Vitis HLS C++, reject out-of-subset
# input loudly (SIO_FPGA_EMIT_FAIL), and the generated HLS must reproduce
# hand-computable closed-form results under host C++ simulation.
# Semantic oracle of record remains the Futhark reference
# (scripts/ci/san_fpga_futhark_reference_gate.sh, ADR-021); this gate checks
# the compiler path, not the semantics contract.
#
# Requires a Madaros built from this tree (make build-madaros). Uses
# artifacts/self-hosted/madaros when present via MADAROS_RAW_BIN.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x artifacts/self-hosted/madaros ]]; then
    export MADAROS_RAW_BIN="${REPO_ROOT}/artifacts/self-hosted/madaros"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail() { echo "FPGA_SOUC_TARGET_FAIL: $*" >&2; exit 1; }

# 1. Both kernels translate through the compiler target.
./bin/souc compile examples/fpga/san_scan_kernel.sio -t fpga-u250 -o "${TMP}/san_scan.cpp" \
    > "${TMP}/san.log" 2>&1 || fail "san_scan compile exited nonzero: $(cat "${TMP}/san.log")"
grep -q "SIO_FPGA_EMIT_OK" "${TMP}/san.log" || fail "san_scan missing SIO_FPGA_EMIT_OK marker"
grep -q 'extern "C" int64_t san_scan' "${TMP}/san_scan.cpp" || fail "san_scan HLS signature missing"
grep -q "m_axi port=confs" "${TMP}/san_scan.cpp" || fail "san_scan m_axi pragma missing"

./bin/souc compile examples/fpga/dot_product_kernel.sio -t fpga-u250 -o "${TMP}/dot_product.cpp" \
    > "${TMP}/dot.log" 2>&1 || fail "dot_product compile exited nonzero: $(cat "${TMP}/dot.log")"
grep -q "SIO_FPGA_EMIT_OK" "${TMP}/dot.log" || fail "dot_product missing SIO_FPGA_EMIT_OK marker"
grep -q 'extern "C" int64_t dot_product' "${TMP}/dot_product.cpp" || fail "dot_product HLS signature missing"

# 2. Out-of-subset input is rejected loudly (while is not in subset v0).
cat > "${TMP}/bad_kernel.sio" <<'EOF'
fn bad_kernel(xs: [i64; 16], n: i64) -> i64 {
    var i = 0
    var acc: i64 = 0
    while i < n {
        acc = acc + xs[i as usize]
        i = i + 1
    }
    acc
}

fn main() -> i64 with IO {
    0
}
EOF
if ./bin/souc compile "${TMP}/bad_kernel.sio" -t fpga-u250 -o "${TMP}/bad.cpp" > "${TMP}/bad.log" 2>&1; then
    fail "while-kernel unexpectedly accepted by fpga-u250 target"
fi
# Fail-closed contract: nonzero exit (asserted above) AND no HLS artifact.
# Madaros panic text is not surfaced through the compiler driver, so this
# gate asserts rc + absence, the same convention as the other negative
# gates; the emitter still panics with SIO_FPGA_EMIT_FAIL at every
# subset violation.
[[ -s "${TMP}/bad.cpp" ]] && fail "while-kernel left an HLS artifact behind"
grep -q "SIO_FPGA_EMIT_FAIL" "${TMP}/bad.log" && echo "note: SIO_FPGA_EMIT_FAIL marker surfaced" || true

# 3. Host C++ simulation of the emitted HLS against closed-form expectations.
# SAN: q=50 -> exits at k=0 (100), k=2 (60), never (zeros) => hist[0]=hist[2]=hist[7]=1,
# macs = lut[0]+lut[2]+lut[7] = 10+30+80 = 120. DOT: [1,2,3,4].[5,6,7,8] = 70.
cat > "${TMP}/tb.cpp" <<'EOF'
#include <cstdint>
#include <cstddef>
#include <cstdio>
extern "C" int64_t san_scan(int64_t confs[256][7], int64_t n_samples, int64_t q_delta, int64_t lut[8], int64_t hist[8]);
extern "C" int64_t dot_product(int64_t a[1024], int64_t b[1024], int64_t n);
int main() {
    static int64_t confs[256][7] = {};
    confs[0][0] = 100;
    confs[1][2] = 60;
    int64_t lut[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int64_t hist[8] = {};
    int64_t macs = san_scan(confs, 3, 50, lut, hist);
    if (macs != 120 || hist[0] != 1 || hist[2] != 1 || hist[7] != 1 ||
        hist[1] != 0 || hist[3] != 0 || hist[4] != 0 || hist[5] != 0 || hist[6] != 0) {
        std::printf("SAN_MISMATCH macs=%lld hist=%lld %lld %lld %lld %lld %lld %lld %lld\n",
                    (long long)macs, (long long)hist[0], (long long)hist[1], (long long)hist[2],
                    (long long)hist[3], (long long)hist[4], (long long)hist[5], (long long)hist[6],
                    (long long)hist[7]);
        return 1;
    }
    int64_t a[1024] = {1, 2, 3, 4};
    int64_t b[1024] = {5, 6, 7, 8};
    int64_t dot = dot_product(a, b, 4);
    if (dot != 70) {
        std::printf("DOT_MISMATCH dot=%lld\n", (long long)dot);
        return 1;
    }
    std::printf("HLS_SIM_OK macs=120 dot=70\n");
    return 0;
}
EOF
g++ -std=c++17 -O1 -o "${TMP}/tb" "${TMP}/tb.cpp" "${TMP}/san_scan.cpp" "${TMP}/dot_product.cpp" \
    || fail "host g++ failed to build emitted HLS"
"${TMP}/tb" > "${TMP}/tb.log" 2>&1 || fail "HLS simulation mismatch: $(cat "${TMP}/tb.log")"
grep -q "HLS_SIM_OK" "${TMP}/tb.log" || fail "HLS simulation marker missing"

echo "FPGA_SOUC_TARGET_PASS san=ok dot=ok negative=ok sim=ok"
