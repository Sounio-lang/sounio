#include <ap_int.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using q_t = ap_int<224>;
using uq_t = ap_uint<224>;
using limb_t = ap_uint<32>;

extern "C" void target23_chained_taylor41(const limb_t *input, limb_t *output, int partition);

static std::vector<q_t> read_words(const char *path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    auto bytes = stream.tellg();
    if (bytes < 0 || bytes % 28 != 0) { std::fprintf(stderr, "invalid F192 binary size\n"); std::exit(2); }
    std::vector<uint8_t> raw(static_cast<size_t>(bytes));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(raw.data()), bytes);
    std::vector<q_t> result(raw.size() / 28);
    for (size_t word = 0; word < result.size(); ++word) {
        uq_t bits = 0;
        for (int byte = 0; byte < 28; ++byte) bits.range(8 * byte + 7, 8 * byte) = raw[28 * word + byte];
        result[word] = q_t(bits);
    }
    return result;
}

static std::vector<limb_t> limbs(const std::vector<q_t> &words) {
    std::vector<limb_t> result(words.size() * 7);
    for (size_t word = 0; word < words.size(); ++word) {
        uq_t bits = uq_t(words[word]);
        for (int limb = 0; limb < 7; ++limb) result[7 * word + limb] = bits.range(32 * limb + 31, 32 * limb);
    }
    return result;
}

static q_t from_limbs(const std::vector<limb_t> &memory, size_t word) {
    uq_t bits = 0;
    for (int limb = 0; limb < 7; ++limb) bits.range(32 * limb + 31, 32 * limb) = memory[7 * word + limb];
    return q_t(bits);
}

int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s hardware_inputs.bin expected.bin\n", argv[0]); return 2; }
    auto input_words = read_words(argv[1]);
    auto expected = read_words(argv[2]);
    if (input_words.size() != 26 || expected.size() != 16860) { std::fprintf(stderr, "frozen cardinality mismatch\n"); return 2; }
    auto input = limbs(input_words);
    std::vector<limb_t> output(8430 * 7);
    size_t mismatches = 0;
    for (int partition = 0; partition < 2; ++partition) {
        std::fill(output.begin(), output.end(), limb_t(0));
        target23_chained_taylor41(input.data(), output.data(), partition);
        for (size_t local = 0; local < 8430; ++local) {
            q_t actual = from_limbs(output, local);
            q_t wanted = expected[partition * 8430 + local];
            if (actual != wanted) {
                if (mismatches < 20) std::printf("CSIM_MISMATCH partition=%d word=%zu\n", partition, local);
                ++mismatches;
            }
        }
    }
    std::printf("CSIM_PARTITIONS=2\nCSIM_STEPS=1686\nCSIM_WORDS=16860\nCSIM_MISMATCHES=%zu\n", mismatches);
    std::printf("TARGET23_CHAINED_TAYLOR41_CSIM_PASS=%s\n", mismatches == 0 ? "true" : "false");
    return mismatches == 0 ? 0 : 1;
}
