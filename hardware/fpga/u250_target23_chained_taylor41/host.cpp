#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
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
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "usage: %s kernel.xclbin hardware_inputs.bin expected.bin [device [partition]]\n", argv[0]);
        return 2;
    }
    constexpr size_t input_bytes = 26 * 28;
    constexpr size_t partition_words = 8430;
    constexpr size_t partition_bytes = partition_words * 28;
    constexpr size_t expected_bytes = 2 * partition_bytes;
    const unsigned device_index = argc >= 5 ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 0;
    const int selected_partition = argc == 6 ? static_cast<int>(std::strtol(argv[5], nullptr, 10)) : -1;
    if (selected_partition < -1 || selected_partition > 1) {
        std::fprintf(stderr, "partition must be 0 or 1\n");
        return 2;
    }
    auto inputs = read_bytes(argv[2], input_bytes);
    auto expected = read_bytes(argv[3], expected_bytes);

    try {
        auto device = xrt::device(device_index);
        auto xclbin = xrt::xclbin(std::string(argv[1]));
        auto uuid = device.register_xclbin(xclbin);
        auto context = xrt::hw_context(device, uuid);
        auto kernel = xrt::kernel(context, "target23_chained_taylor41");
        auto input_bo = xrt::bo(device, input_bytes, kernel.group_id(0));
        auto output_bo = xrt::bo(device, partition_bytes, kernel.group_id(1));
        auto input_map = input_bo.map<uint8_t *>();
        auto output_map = output_bo.map<uint8_t *>();
        std::copy(inputs.begin(), inputs.end(), input_map);
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_bytes, 0);

        size_t mismatches = 0;
        auto started = std::chrono::steady_clock::now();
        const int first_partition = selected_partition < 0 ? 0 : selected_partition;
        const int last_partition = selected_partition < 0 ? 2 : selected_partition + 1;
        for (int partition = first_partition; partition < last_partition; ++partition) {
            std::fill(output_map, output_map + partition_bytes, 0);
            output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, partition_bytes, 0);
            auto run = kernel(input_bo, output_bo, partition);
            run.wait();
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, partition_bytes, 0);
            const uint8_t *wanted = expected.data() + partition * partition_bytes;
            for (size_t word = 0; word < partition_words; ++word) {
                const uint8_t *actual_word = output_map + 28 * word;
                const uint8_t *wanted_word = wanted + 28 * word;
                if (!std::equal(actual_word, actual_word + 28, wanted_word)) {
                    if (mismatches < 20) std::printf("FPGA_MISMATCH partition=%d word=%zu\n", partition, word);
                    ++mismatches;
                }
            }
            std::printf("FPGA_PARTITION_%d_WORDS=%zu\n", partition, partition_words);
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        const int partitions_run = last_partition - first_partition;
        std::printf("FPGA_DEVICE_INDEX=%u\nFPGA_SELECTED_PARTITION=%d\n", device_index, selected_partition);
        std::printf("FPGA_PARTITIONS=%d\nFPGA_STEPS=%d\nFPGA_WORDS=%zu\n",
                    partitions_run, partitions_run * 843, partitions_run * partition_words);
        std::printf("FPGA_ELAPSED_MS=%lld\nFPGA_MISMATCHES=%zu\n", static_cast<long long>(elapsed), mismatches);
        std::printf("TARGET23_CHAINED_TAYLOR41_PHYSICAL_PASS=%s\n", mismatches == 0 ? "true" : "false");
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "XRT_ERROR=%s\n", error.what());
        return 1;
    }
}
