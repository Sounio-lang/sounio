#include <ap_int.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using q_t = ap_int<128>;
using uq_t = ap_uint<128>;

extern "C" void target23_picard_step(const q_t *input, q_t *output, int n_cases);

static std::vector<q_t> read_words(const char *path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    auto bytes = stream.tellg();
    if (bytes < 0 || bytes % 16 != 0) {
        std::fprintf(stderr, "invalid binary size\n");
        std::exit(2);
    }
    std::vector<uint8_t> raw(static_cast<size_t>(bytes));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(raw.data()), bytes);
    if (!stream) {
        std::fprintf(stderr, "short read\n");
        std::exit(2);
    }
    std::vector<q_t> words(raw.size() / 16);
    for (size_t index = 0; index < words.size(); ++index) {
        uq_t assembled = 0;
        for (int byte = 0; byte < 16; ++byte) {
            assembled.range(8 * byte + 7, 8 * byte) = raw[16 * index + byte];
        }
        words[index] = q_t(assembled);
    }
    return words;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s inputs.bin expected.bin\n", argv[0]);
        return 2;
    }
    auto inputs = read_words(argv[1]);
    auto expected = read_words(argv[2]);
    if (inputs.size() != 72 || expected.size() != 88) {
        std::fprintf(stderr, "frozen cardinality mismatch\n");
        return 2;
    }
    std::vector<q_t> output(expected.size());
    target23_picard_step(inputs.data(), output.data(), 4);
    size_t mismatches = 0;
    for (size_t index = 0; index < output.size(); ++index) {
        if (output[index] != expected[index]) {
            if (mismatches < 20) {
                std::printf("CSIM_MISMATCH index=%zu actual=%s expected=%s\n",
                    index, output[index].to_string(10).c_str(),
                    expected[index].to_string(10).c_str());
            }
            ++mismatches;
        }
    }
    std::printf("CSIM_CASES=4\nCSIM_WORDS=%zu\nCSIM_MISMATCHES=%zu\n", output.size(), mismatches);
    std::printf("TARGET23_PICARD_CSIM_PASS=%s\n", mismatches == 0 ? "true" : "false");
    return mismatches == 0 ? 0 : 1;
}
