// ADR-009 companion probe: dumps the raw GPU-measured output matrix of
// the shfl.sync.bfly.b32 XOR-butterfly kernel for independent
// comparison against tools/pireus/dgx_ptx_shfl_material_parity.fut.
//
// This does NOT modify or replace dgx_ptx_shfl_material_parity.cu (its
// frozen source hash is unaffected). It runs the identical kernel
// definition and identical deterministic input generator, and prints
// the raw 16x16 u64 output matrix as plain decimal text, one row per
// line, space-separated -- for a Futhark `check` call, not for the
// SHA256 digest scheme used by the frozen probe.

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

constexpr std::size_t kDimension = 16;
constexpr unsigned kMemberMask = 0x0000ffffU;

constexpr std::uint64_t payload_for_source(std::size_t source) {
    return UINT64_C(0xfedcba9876543210) ^
        (UINT64_C(0x1111111111111111) * source);
}

using Payloads = std::array<std::uint64_t, kDimension>;
using Outputs = std::array<std::uint64_t, kDimension * kDimension>;

__global__ void shfl_bfly_material_kernel(const std::uint64_t* input,
                                          std::uint64_t* output) {
    const unsigned lane = threadIdx.x;
    if (lane >= kDimension) {
        return;
    }
    const std::uint64_t value = input[lane];
    const std::uint32_t low = static_cast<std::uint32_t>(value);
    const std::uint32_t high = static_cast<std::uint32_t>(value >> 32U);
#pragma unroll
    for (unsigned displacement = 0; displacement < kDimension; ++displacement) {
        const std::uint32_t selected_low =
            __shfl_xor_sync(kMemberMask, low, displacement, kDimension);
        const std::uint32_t selected_high =
            __shfl_xor_sync(kMemberMask, high, displacement, kDimension);
        output[displacement * kDimension + lane] =
            static_cast<std::uint64_t>(selected_low) |
            (static_cast<std::uint64_t>(selected_high) << 32U);
    }
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << "PIREUS_DGX_CUDA_ERROR operation=" << operation
                  << " code=" << static_cast<int>(status)
                  << " detail=" << cudaGetErrorString(status) << '\n';
        std::exit(2);
    }
}

}  // namespace

int main() {
    Payloads input{};
    Outputs output{};
    for (std::size_t source = 0; source < input.size(); ++source) {
        input[source] = payload_for_source(source);
    }

    std::uint64_t* device_input = nullptr;
    std::uint64_t* device_output = nullptr;
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(input)),
                 "cudaMalloc-input");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(output)),
                 "cudaMalloc-output");
    require_cuda(cudaMemcpy(device_input, input.data(), sizeof(input),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy-input");
    shfl_bfly_material_kernel<<<1, kDimension>>>(device_input, device_output);
    require_cuda(cudaGetLastError(), "kernel-launch");
    require_cuda(cudaDeviceSynchronize(), "kernel-synchronize");
    require_cuda(cudaMemcpy(output.data(), device_output, sizeof(output),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy-output");
    require_cuda(cudaFree(device_output), "cudaFree-output");
    require_cuda(cudaFree(device_input), "cudaFree-input");

    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::cerr << "gpu_name=" << properties.name
              << " compute_capability=" << properties.major << '.'
              << properties.minor << '\n';

    for (std::size_t displacement = 0; displacement < kDimension; ++displacement) {
        for (std::size_t lane = 0; lane < kDimension; ++lane) {
            std::cout << output[displacement * kDimension + lane];
            if (lane + 1 < kDimension) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
    return EXIT_SUCCESS;
}
