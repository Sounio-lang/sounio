#include <ap_int.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

using q_t = ap_int<64>;

extern "C" void target23_batch(const q_t *initial_xy, q_t *output, int n_leaves);

static std::vector<int64_t> read_words(const char *path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    auto bytes = stream.tellg();
    if (bytes < 0 || bytes % 8 != 0) {
        std::fprintf(stderr, "invalid binary size\n");
        std::exit(2);
    }
    std::vector<int64_t> words(static_cast<size_t>(bytes) / 8);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(words.data()), bytes);
    if (!stream) {
        std::fprintf(stderr, "short read\n");
        std::exit(2);
    }
    return words;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s inputs.bin expected.bin\n", argv[0]);
        return 2;
    }
    auto raw_inputs = read_words(argv[1]);
    auto expected = read_words(argv[2]);
    if (raw_inputs.size() != 662 || expected.size() != 2648) {
        std::fprintf(stderr, "frozen cardinality mismatch\n");
        return 2;
    }
    std::vector<q_t> inputs(raw_inputs.size());
    std::vector<q_t> outputs(expected.size());
    for (size_t index = 0; index < inputs.size(); ++index) inputs[index] = raw_inputs[index];
    target23_batch(inputs.data(), outputs.data(), 331);
    size_t mismatches = 0;
    for (size_t index = 0; index < outputs.size(); ++index) {
        int64_t actual = outputs[index].to_int64();
        if (actual != expected[index]) {
            if (mismatches < 20) std::printf("CSIM_MISMATCH index=%zu actual=%lld expected=%lld\n",
                index, static_cast<long long>(actual), static_cast<long long>(expected[index]));
            ++mismatches;
        }
    }
    std::printf("CSIM_WORDS=%zu\nCSIM_MISMATCHES=%zu\n", outputs.size(), mismatches);
    std::printf("TARGET23_U250_CSIM_PASS=%s\n", mismatches == 0 ? "true" : "false");
    return mismatches == 0 ? 0 : 1;
}
