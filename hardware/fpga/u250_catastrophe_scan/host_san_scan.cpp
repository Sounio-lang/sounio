// host_san_scan.cpp — XRT host for the SAN catastrophe-scan kernel on the
// AMD Alveo U250 (DL380 deployment).
//
// Loads a quantized cohort exported by san_dl380_t3_export.py
// (artifacts/san_dl380_t3/*.u16 + expected.txt), packs samples into
// 512-bit beats (128-bit records, 7 x 15-bit Q0.15 fields — the packing
// the kernel expects), runs krnl_san_scan, and verifies exit histogram,
// catastrophe count, and metered MACs BIT-EXACTLY against the control
// VM's gated outputs (spec theorem T3, clause I6). Any mismatch is a loud
// failure, never a silent fallback.
//
// Usage:
//   host_san_scan <krnl.xclbin> <artifacts_dir> <dataset>
//   dataset: val_resnet | val_vit | stress_1p2M
//
// Build (on the DL380, XRT sourced):
//   g++ -O2 -std=c++17 host_san_scan.cpp -o host_san_scan \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -lpthread
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#define MAX_POINTS 8
#define LANES 4

static std::map<std::string, std::string> load_expected(const std::string &path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::map<std::string, std::string> kv;
    std::string k, v;
    while (f >> k) { std::getline(f, v); kv[k] = v.substr(1); }
    return kv;
}

static std::vector<unsigned long long> parse_ull(const std::string &s) {
    std::istringstream is(s);
    std::vector<unsigned long long> out;
    unsigned long long v;
    while (is >> v) out.push_back(v);
    return out;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <xclbin> <artifacts_dir> <dataset>\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1], dir = argv[2], ds = argv[3];
    auto kv = load_expected(dir + "/expected.txt");

    auto shape = parse_ull(kv[ds + "_shape"]);
    const std::string family = kv[ds + "_family"];
    const int n_samples = (int)shape[0], n_conf = (int)shape[1];
    const int n_points = n_conf + 1;
    auto lut = parse_ull(kv["lut_" + family]);
    const unsigned q_delta = std::stoul(kv["q_delta_" + family]);
    auto exp_hist = parse_ull(kv[ds + "_hist"]);
    const unsigned long long exp_cat = std::stoull(kv[ds + "_catastrophes"]);
    const unsigned long long exp_flops = std::stoull(kv[ds + "_flops_macs"]);

    // load quantized cohort (uint16 LE, row-major [n_samples, n_conf])
    std::ifstream uf(dir + "/" + kv[ds + "_file"], std::ios::binary);
    if (!uf) { fprintf(stderr, "cannot open cohort file %s\n",
                       (dir + "/" + kv[ds + "_file"]).c_str()); return 2; }
    std::vector<uint16_t> q((size_t)n_samples * n_conf);
    uf.read(reinterpret_cast<char *>(q.data()), q.size() * 2);
    if (!uf) { fprintf(stderr, "short read on cohort\n"); return 2; }

    // pack into 512-bit beats: sample s at lane s%LANES, field k at bits
    // [128*lane + 15k + 14 : 128*lane + 15k]  (mirrors the kernel exactly)
    const size_t n_words = ((size_t)n_samples + LANES - 1) / LANES;
    std::vector<uint64_t> beats(n_words * 8, 0);
    auto set15 = [&](size_t bit_lo, uint16_t v) {
        size_t wi = bit_lo / 64, off = bit_lo % 64;
        beats[wi] &= ~(0x7FFFULL << off);
        beats[wi] |= ((uint64_t)v & 0x7FFF) << off;
        if (off > 49) {
            int rem = (int)off - 49;
            beats[wi + 1] &= ~(0x7FFFULL >> (15 - rem));
            beats[wi + 1] |= ((uint64_t)v & 0x7FFF) >> (15 - rem);
        }
    };
    for (int s = 0; s < n_samples; s++)
        for (int k = 0; k < n_conf; k++)
            set15((size_t)(s % LANES) * 128 + (size_t)k * 15 + (size_t)(s / LANES) * 512,
                  q[(size_t)s * n_conf + k]);

    uint64_t lut_buf[MAX_POINTS] = {0};
    for (int i = 0; i < n_points && i < MAX_POINTS; i++) lut_buf[i] = lut[i];

    // ---- XRT run ----------------------------------------------------------
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);
    auto krnl = xrt::kernel(device, uuid, "krnl_san_scan");

    auto bo_samples = xrt::bo(device, beats.size() * 8, krnl.group_id(0));
    auto bo_lut = xrt::bo(device, sizeof(lut_buf), krnl.group_id(1));
    auto bo_hist = xrt::bo(device, MAX_POINTS * 4, krnl.group_id(5));
    auto bo_cat = xrt::bo(device, 4, krnl.group_id(6));
    auto bo_flops = xrt::bo(device, 8, krnl.group_id(7));

    auto *smap = bo_samples.map<uint64_t *>();
    memcpy(smap, beats.data(), beats.size() * 8);
    auto *lmap = bo_lut.map<uint64_t *>();
    memcpy(lmap, lut_buf, sizeof(lut_buf));
    bo_samples.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_lut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    printf("host_san_scan: dataset=%s n=%d points=%d q_delta=%u family=%s\n",
           argv[3], n_samples, n_points, q_delta, family.c_str());
    auto t0 = std::chrono::steady_clock::now();
    auto run = krnl(bo_samples, bo_lut, (uint32_t)q_delta, (uint32_t)n_points,
                    (uint32_t)n_samples, bo_hist, bo_cat, bo_flops);
    run.wait();
    auto t1 = std::chrono::steady_clock::now();
    bo_hist.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_cat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_flops.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    double wall_s = std::chrono::duration<double>(t1 - t0).count();

    auto *hmap = bo_hist.map<uint32_t *>();
    uint32_t cat_out = *bo_cat.map<uint32_t *>();
    uint64_t flops_out = *bo_flops.map<uint64_t *>();

    // ---- bit-exact verification (spec T3 / clause I6) ---------------------
    int fail = 0;
    for (int b = 0; b < n_points; b++)
        if (hmap[b] != exp_hist[b]) {
            printf("MISMATCH hist[%d]: card=%u expected=%llu\n", b, hmap[b], exp_hist[b]);
            fail = 1;
        }
    if (cat_out != exp_cat) {
        printf("MISMATCH catastrophes: card=%u expected=%llu\n", cat_out, exp_cat);
        fail = 1;
    }
    if (flops_out != exp_flops) {
        printf("MISMATCH flops: card=%llu expected=%llu\n",
               (unsigned long long)flops_out, exp_flops);
        fail = 1;
    }
    double msamples = wall_s > 0 ? n_samples / wall_s / 1e6 : 0;
    printf("CARD_RESULT n=%d catastrophes=%u flops_macs=%llu wall=%.3fms "
           "(%.1f Msamples/s kernel-only)\n",
           n_samples, cat_out, (unsigned long long)flops_out, wall_s * 1e3, msamples);
    printf("HOST_SAN_SCAN_%s (%s)\n", fail ? "FAIL" : "PASS", argv[3]);
    return fail;
}
