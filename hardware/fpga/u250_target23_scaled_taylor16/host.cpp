#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<uint8_t> read_bytes(const char *path, size_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || static_cast<size_t>(stream.tellg()) != expected) {
        std::fprintf(stderr, "invalid input size: %s\n", path);
        std::exit(2);
    }
    std::vector<uint8_t> bytes(expected);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(bytes.data()), expected);
    if (!stream) {
        std::fprintf(stderr, "short read: %s\n", path);
        std::exit(2);
    }
    return bytes;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s kernel.xclbin inputs.bin expected.bin\n", argv[0]);
        return 2;
    }
    constexpr size_t input_bytes = 54 * 16;
    constexpr size_t output_bytes = 459 * 16;
    auto inputs = read_bytes(argv[2], input_bytes);
    auto expected = read_bytes(argv[3], output_bytes);

    try {
        auto device = xrt::device(0);
        auto xclbin = xrt::xclbin(std::string(argv[1]));
        auto uuid = device.register_xclbin(xclbin);
        auto context = xrt::hw_context(device, uuid);
        auto kernel = xrt::kernel(context, "target23_scaled_taylor16");
        auto input_bo = xrt::bo(device, input_bytes, kernel.group_id(0));
        auto output_bo = xrt::bo(device, output_bytes, kernel.group_id(1));
        auto input_map = input_bo.map<uint8_t *>();
        auto output_map = output_bo.map<uint8_t *>();
        std::copy(inputs.begin(), inputs.end(), input_map);
        std::fill(output_map, output_map + output_bytes, 0);
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_bytes, 0);
        auto run = kernel(input_bo, output_bo, 3);
        run.wait();
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, output_bytes, 0);

        size_t mismatches = 0;
        for (size_t word = 0; word < 459; ++word) {
            const uint8_t *actual = output_map + 16 * word;
            const uint8_t *wanted = expected.data() + 16 * word;
            if (!std::equal(actual, actual + 16, wanted)) {
                if (mismatches < 20) {
                    std::printf("FPGA_MISMATCH_WORD=%zu\n", word);
                }
                ++mismatches;
            }
        }
        std::printf("FPGA_DEVICE_INDEX=0\nFPGA_CASES=3\nFPGA_WORDS=459\n");
        std::printf("FPGA_MISMATCHES=%zu\n", mismatches);
        std::printf("TARGET23_SCALED_TAYLOR16_PHYSICAL_PASS=%s\n",
                    mismatches == 0 ? "true" : "false");
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "XRT_ERROR=%s\n", error.what());
        return 1;
    }
}
