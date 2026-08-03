#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static constexpr int WORDS_PER_LEAF = 8;
static constexpr int FRAC_BITS = 40;
static constexpr double ZS = 0x1.653d4a9e20f75p+4;
static constexpr double Q0_AREA = -0x1.221ef15087f44p-10;

static std::vector<int64_t> read_words(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    auto bytes = stream.tellg();
    if (bytes < 0 || bytes % 8 != 0) { std::fprintf(stderr, "bad binary size\n"); std::exit(2); }
    std::vector<int64_t> words(static_cast<size_t>(bytes) / 8);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(words.data()), bytes);
    if (!stream) { std::fprintf(stderr, "short binary read\n"); std::exit(2); }
    return words;
}

static std::vector<std::vector<std::string>> read_tsv(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(stream, line)) {
        std::vector<std::string> row;
        std::stringstream parser(line);
        std::string cell;
        while (std::getline(parser, cell, '\t')) row.push_back(cell);
        rows.push_back(std::move(row));
    }
    return rows;
}

static int column(const std::vector<std::string> &header, const std::string &name) {
    for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return static_cast<int>(i);
    std::fprintf(stderr, "missing TSV column %s\n", name.c_str()); std::exit(2);
}

static double qdouble(int64_t value) { return std::ldexp(static_cast<double>(value), -FRAC_BITS); }

int main(int argc, char **argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s XCLBIN INPUTS EXPECTED DECIMAL_RESULTS REPEATS OUT_TSV\n", argv[0]);
        return 2;
    }
    auto inputs = read_words(argv[2]);
    auto expected = read_words(argv[3]);
    if (inputs.size() % 2 || expected.size() != inputs.size() / 2 * WORDS_PER_LEAF) {
        std::fprintf(stderr, "input/expected cardinality mismatch\n"); return 2;
    }
    const int n = static_cast<int>(inputs.size() / 2);
    const int repeats = std::atoi(argv[5]);
    if (n != 331 || repeats < 1 || repeats > 1000) {
        std::fprintf(stderr, "frozen cardinality or repeats mismatch\n"); return 2;
    }
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(argv[1]);
    auto kernel = xrt::kernel(device, uuid, "target23_batch");
    auto input_bo = xrt::bo(device, inputs.size() * sizeof(int64_t), kernel.group_id(0));
    auto output_bo = xrt::bo(device, expected.size() * sizeof(int64_t), kernel.group_id(1));
    std::memcpy(input_bo.map<void *>(), inputs.data(), inputs.size() * sizeof(int64_t));

    auto h2d_start = std::chrono::steady_clock::now();
    input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto h2d_end = std::chrono::steady_clock::now();
    double kernel_seconds = 0.0;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        auto run = kernel(input_bo, output_bo, n);
        run.wait();
        auto end = std::chrono::steady_clock::now();
        kernel_seconds += std::chrono::duration<double>(end - start).count();
    }
    auto d2h_start = std::chrono::steady_clock::now();
    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto d2h_end = std::chrono::steady_clock::now();
    auto *actual = output_bo.map<int64_t *>();
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (mismatches < 20) std::printf("WORD_MISMATCH index=%zu card=%lld expected=%lld\n",
                i, static_cast<long long>(actual[i]), static_cast<long long>(expected[i]));
            ++mismatches;
        }
    }

    auto decimal = read_tsv(argv[4]);
    if (decimal.size() != 332) { std::fprintf(stderr, "Decimal row count mismatch\n"); return 2; }
    int det_col = column(decimal[0], "FINE_DETERMINANT");
    int h0l_col = column(decimal[0], "C0HORECT2_LOWER");
    int h0u_col = column(decimal[0], "C0HORECT2_UPPER");
    int r0l_col = column(decimal[0], "C0RECT2_LOWER");
    int r0u_col = column(decimal[0], "C0RECT2_UPPER");
    std::ofstream rows(argv[6]);
    rows << "LEAF_INDEX\tSTEPS\tEVENT1_TIME\tEVENT2_TIME\tX2\tY2\tELL2\tDETERMINANT\tDECIMAL_DETERMINANT\tABS_DELTA\tINSIDE_BOTH_CAPD\n";
    double max_delta = 0.0;
    double min_margin = HUGE_VAL;
    int negative = 0;
    int inside = 0;
    int event_pass = 0;
    for (int leaf = 0; leaf < n; ++leaf) {
        const int64_t *word = actual + WORDS_PER_LEAF * leaf;
        double x0 = qdouble(inputs[2 * leaf]);
        double y0 = qdouble(inputs[2 * leaf + 1]);
        double x2 = qdouble(word[4]);
        double y2 = qdouble(word[5]);
        double ell2 = qdouble(word[6]);
        double normal0 = x0 * y0 - ZS;
        double normal2 = x2 * y2 - ZS;
        double determinant = std::exp(ell2) * normal0 / normal2 * Q0_AREA;
        double decimal_det = std::strtod(decimal[leaf + 1][det_col].c_str(), nullptr);
        double delta = std::abs(determinant - decimal_det);
        max_delta = std::max(max_delta, delta);
        double h0l = std::strtod(decimal[leaf + 1][h0l_col].c_str(), nullptr);
        double h0u = std::strtod(decimal[leaf + 1][h0u_col].c_str(), nullptr);
        double r0l = std::strtod(decimal[leaf + 1][r0l_col].c_str(), nullptr);
        double r0u = std::strtod(decimal[leaf + 1][r0u_col].c_str(), nullptr);
        bool in = h0l < determinant && determinant < h0u && r0l < determinant && determinant < r0u;
        if (determinant < 0) ++negative;
        if (in) ++inside;
        if (word[1] == 2 && word[7] == 7) ++event_pass;
        min_margin = std::min(min_margin, std::min({determinant - h0l, h0u - determinant,
                                                    determinant - r0l, r0u - determinant}));
        rows << leaf + 1 << '\t' << word[0] << '\t' << std::setprecision(17)
             << qdouble(word[2]) << '\t' << qdouble(word[3]) << '\t' << x2 << '\t' << y2
             << '\t' << ell2 << '\t' << determinant << '\t' << decimal_det << '\t' << delta
             << '\t' << (in ? "true" : "false") << '\n';
    }
    double h2d = std::chrono::duration<double>(h2d_end - h2d_start).count();
    double d2h = std::chrono::duration<double>(d2h_end - d2h_start).count();
    double mean_kernel = kernel_seconds / repeats;
    std::printf("SCHEMA=sounio.cs6.v7b-target23-u250-hardware.v1\n");
    std::printf("LEAVES=%d\nREPEATS=%d\nBIT_EXACT_WORDS=%zu\nBIT_MISMATCHES=%zu\n", n, repeats, expected.size(), mismatches);
    std::printf("EVENT_ORBITS_PASS=%d\nNEGATIVE_DETERMINANTS=%d\nINSIDE_BOTH_CAPD=%d\n", event_pass, negative, inside);
    std::printf("MAX_ABS_DELTA_VS_DECIMAL=%.17g\nMIN_CAPD_MARGIN=%.17g\n", max_delta, min_margin);
    std::printf("H2D_BYTES=%zu\nD2H_BYTES=%zu\nH2D_SECONDS=%.9f\nMEAN_KERNEL_SECONDS=%.9f\nD2H_SECONDS=%.9f\n",
                inputs.size() * sizeof(int64_t), expected.size() * sizeof(int64_t), h2d, mean_kernel, d2h);
    std::printf("ORBITS_PER_SECOND=%.6f\n", n / mean_kernel);
    bool pass = mismatches == 0 && event_pass == n && negative == n && inside == n;
    std::printf("TARGET23_U250_HARDWARE_PASS=%s\n", pass ? "true" : "false");
    std::printf("RIGOROUS_INTERVAL_CERTIFICATE=false\nLEAF_WIDE_CERTIFICATE=false\nGLOBAL_HPG_CERTIFICATE=false\n");
    return pass ? 0 : 1;
}
