// host_san_scan_bench.cpp — XRT throughput/power benchmark for krnl_san_scan.
//
// Loads the cohort once, then enqueues the kernel repeatedly for a requested
// duration without reloading the xclbin. The kernel-only wall time is measured
// per-iteration; the aggregate samples/s is reported at the end. Designed to
// be paired with an external power sampler (xrt-smi examine -r electrical)
// so that the U250 board power can be read under steady load.
//
// Usage:
//   host_san_scan_bench <xclbin> <artifacts_dir> <dataset> <seconds>
//
// Build (on the DL380, XRT sourced):
//   g++ -O2 -std=c++17 host_san_scan_bench.cpp -o host_san_scan_bench \
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
    if (argc != 5) {
        fprintf(stderr, "usage: %s <xclbin> <artifacts_dir> <dataset> <seconds>\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1], dir = argv[2], ds = argv[3];
    const double bench_seconds = std::atof(argv[4]);
    if (bench_seconds <= 0) {
        fprintf(stderr, "seconds must be positive\n");
        return 2;
    }

    auto kv = load_expected(dir + "/expected.txt");
    auto shape = parse_ull(kv[ds + "_shape"]);
    const std::string family = kv[ds + "_family"];
    const int n_samples = (int)shape[0], n_conf = (int)shape[1];
    const int n_points = n_conf + 1;
    auto lut = parse_ull(kv["lut_" + family]);
    const unsigned q_delta = std::stoul(kv["q_delta_" + family]);

    std::ifstream uf(dir + "/" + kv[ds + "_file"], std::ios::binary);
    if (!uf) { fprintf(stderr, "cannot open cohort file\n"); return 2; }
    std::vector<uint16_t> q((size_t)n_samples * n_conf);
    uf.read(reinterpret_cast<char *>(q.data()), q.size() * 2);
    if (!uf) { fprintf(stderr, "short read on cohort\n"); return 2; }

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

    printf("host_san_scan_bench: dataset=%s n=%d points=%d q_delta=%u family=%s duration=%.1fs\n",
           argv[3], n_samples, n_points, q_delta, family.c_str(), bench_seconds);

    auto t_start = std::chrono::steady_clock::now();
    uint64_t total_samples = 0;
    uint64_t iterations = 0;
    double total_kernel_s = 0.0;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (elapsed >= bench_seconds) break;

        auto t0 = std::chrono::steady_clock::now();
        auto run = krnl(bo_samples, bo_lut, (uint32_t)q_delta, (uint32_t)n_points,
                        (uint32_t)n_samples, bo_hist, bo_cat, bo_flops);
        run.wait();
        auto t1 = std::chrono::steady_clock::now();

        total_kernel_s += std::chrono::duration<double>(t1 - t0).count();
        total_samples += (uint64_t)n_samples;
        iterations++;
    }

    auto t_end = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();

    printf("BENCH_RESULT dataset=%s iterations=%llu total_samples=%llu wall=%.3fs "
           "kernel_time=%.3fs aggregate=%.1f Msamples/s avg_per_iter=%.3fms\n",
           argv[3], (unsigned long long)iterations, (unsigned long long)total_samples,
           wall_s, total_kernel_s, total_samples / wall_s / 1e6,
           iterations ? total_kernel_s * 1e3 / iterations : 0.0);
    return 0;
}
