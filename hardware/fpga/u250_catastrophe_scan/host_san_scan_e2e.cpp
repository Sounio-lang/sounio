// host_san_scan_e2e.cpp — XRT host for the SAN catastrophe-scan kernel,
// end-to-end edition.
//
// Reads a packed cohort from stdin, runs krnl_san_scan on the AMD Alveo U250,
// and prints the result plus per-phase timing. This host deliberately does NOT
// load cohort files or verify golden values; it is the raw PCIe side of the
// Python-orchestrated end-to-end loop (scripts/research/san_fpga_endtoend.py).
//
// Binary stdin protocol (all little-endian):
//   uint32 n_samples      // number of samples (>= 1)
//   uint32 n_conf         // number of confidence fields per sample (<= 7)
//   uint32 q_delta        // integer threshold in Q0.15
//   uint32 reserved       // padding, must be 0
//   uint64 lut[8]         // stage-cost LUT (n_conf+1 entries used)
//   uint64 beats[n_words * 8]  // packed 512-bit beats, n_words = ceil(n_samples/4)
//
// Output (stdout, one line):
//   E2E_RESULT n=<n> catastrophes=<c> flops_macs=<f> dma_h2d_ms=<a> kernel_ms=<b> dma_d2h_ms=<c> total_ms=<d> Msamples/s=<e>
//
// Build (on the DL380, XRT sourced):
//   g++ -O2 -std=c++17 host_san_scan_e2e.cpp -o host_san_scan_e2e \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -lpthread
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#define MAX_POINTS 8
#define LANES 4

struct e2e_header {
    uint32_t n_samples;
    uint32_t n_conf;
    uint32_t q_delta;
    uint32_t reserved;
    uint64_t lut[MAX_POINTS];
};

static void read_all(void *dst, size_t n) {
    char *p = reinterpret_cast<char *>(dst);
    size_t got = 0;
    while (got < n) {
        size_t r = fread(p + got, 1, n - got, stdin);
        if (r == 0) {
            if (feof(stdin)) {
                fprintf(stderr, "host_san_scan_e2e: unexpected EOF after %zu of %zu bytes\n",
                        got, n);
            } else {
                fprintf(stderr, "host_san_scan_e2e: stdin read error\n");
            }
            exit(2);
        }
        got += r;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <xclbin>\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1];

    e2e_header hdr;
    read_all(&hdr, sizeof(hdr));
    if (hdr.n_conf == 0 || hdr.n_conf > MAX_POINTS - 1) {
        fprintf(stderr, "host_san_scan_e2e: n_conf=%u out of range [1,%d]\n",
                hdr.n_conf, MAX_POINTS - 1);
        return 2;
    }
    const int n_points = (int)hdr.n_conf + 1;
    const size_t n_words = ((size_t)hdr.n_samples + LANES - 1) / LANES;
    const size_t beats_bytes = n_words * sizeof(uint64_t) * 8;

    std::vector<uint64_t> beats(n_words * 8, 0);
    read_all(beats.data(), beats_bytes);

    // ---- XRT run ----------------------------------------------------------
    auto t_setup_0 = std::chrono::steady_clock::now();
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);
    auto krnl = xrt::kernel(device, uuid, "krnl_san_scan");

    auto bo_samples = xrt::bo(device, beats_bytes, krnl.group_id(0));
    auto bo_lut = xrt::bo(device, sizeof(hdr.lut), krnl.group_id(1));
    auto bo_hist = xrt::bo(device, MAX_POINTS * 4, krnl.group_id(5));
    auto bo_cat = xrt::bo(device, 4, krnl.group_id(6));
    auto bo_flops = xrt::bo(device, 8, krnl.group_id(7));

    auto *smap = bo_samples.map<uint64_t *>();
    memcpy(smap, beats.data(), beats_bytes);
    auto *lmap = bo_lut.map<uint64_t *>();
    memcpy(lmap, hdr.lut, sizeof(hdr.lut));
    auto t_setup_1 = std::chrono::steady_clock::now();

    auto t_h2d_0 = std::chrono::steady_clock::now();
    bo_samples.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_lut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t_h2d_1 = std::chrono::steady_clock::now();

    auto t_k0 = std::chrono::steady_clock::now();
    auto run = krnl(bo_samples, bo_lut, hdr.q_delta, (uint32_t)n_points,
                    hdr.n_samples, bo_hist, bo_cat, bo_flops);
    run.wait();
    auto t_k1 = std::chrono::steady_clock::now();

    auto t_d2h_0 = std::chrono::steady_clock::now();
    bo_hist.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_cat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_flops.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto t_d2h_1 = std::chrono::steady_clock::now();

    auto *hmap = bo_hist.map<uint32_t *>();
    uint32_t cat_out = *bo_cat.map<uint32_t *>();
    uint64_t flops_out = *bo_flops.map<uint64_t *>();

    double setup_ms = std::chrono::duration<double, std::milli>(t_setup_1 - t_setup_0).count();
    double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_1 - t_h2d_0).count();
    double kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();
    double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_1 - t_d2h_0).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_d2h_1 - t_setup_0).count();
    double msamples = total_ms > 0 ? hdr.n_samples / total_ms / 1e3 : 0;

    printf("E2E_RESULT n=%u n_conf=%u q_delta=%u catastrophes=%u flops_macs=%llu "
           "setup_ms=%.3f dma_h2d_ms=%.3f kernel_ms=%.3f dma_d2h_ms=%.3f total_ms=%.3f "
           "Msamples/s=%.2f\n",
           hdr.n_samples, hdr.n_conf, hdr.q_delta, cat_out,
           (unsigned long long)flops_out, setup_ms, h2d_ms, kernel_ms, d2h_ms,
           total_ms, msamples);

    // Print histogram as additional machine-readable lines.
    for (int b = 0; b < n_points; b++) {
        printf("E2E_HIST %d %u\n", b, hmap[b]);
    }
    printf("HOST_SAN_SCAN_E2E_PASS\n");
    return 0;
}
