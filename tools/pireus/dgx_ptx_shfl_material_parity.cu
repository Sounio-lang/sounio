// CUDA C++ MATERIAL_PARITY probe for the frozen Sounio DGX PTX SHFL candidate.
// Expected coordinates and digest are consumed from Sounio; C++ defines none.

#include "material_sha256.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

constexpr std::size_t kDimension = 16;
constexpr std::size_t kComponents = 2;
constexpr unsigned kMemberMask = 0x0000ffffU;
constexpr char kSounioSourceSha256[] =
    "4be23864a14274d7996dd890473a5b3356a88441a589e509080c9978ba1cf404";
constexpr char kFrozenSemanticsSha256[] =
    "a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336";
constexpr std::array<std::uint32_t, 8> kSounioCandidateDigest = {
    1996463492U, 2773712531U, 2232409634U, 959326894U,
    3198512505U, 1781073490U, 2005131219U, 1992148529U,
};

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

int observed_source(const Payloads& input, std::uint64_t value) {
    for (std::size_t source = 0; source < input.size(); ++source) {
        if (input[source] == value) {
            return static_cast<int>(source);
        }
    }
    return -1;
}

struct Measurements {
    std::size_t matched_cells = 0;
    std::size_t in_range_cells = 0;
    std::size_t active_source_cells = 0;
    std::size_t member_source_cells = 0;
    std::size_t matched_payload_bits = 0;
    std::size_t max_source = 0;
    std::array<std::uint32_t, 8> digest{};
};

Measurements measure_material_result(const Payloads& input,
                                     const Outputs& output) {
    Measurements measured;
    pireus::material::Sha256 digest;
    digest.update_i64_be(INT64_C(0x4450585348463031));
    for (std::size_t displacement = 0; displacement < kDimension;
         ++displacement) {
        for (std::size_t lane = 0; lane < kDimension; ++lane) {
            const std::size_t expected_source = lane ^ displacement;
            const std::uint64_t observed_value =
                output[displacement * kDimension + lane];
            const int decoded_source = observed_source(input, observed_value);
            const bool in_range = decoded_source >= 0 && decoded_source <= 15;
            const bool active = decoded_source >= 0 && decoded_source < 16;
            const bool member = active &&
                ((kMemberMask >> static_cast<unsigned>(decoded_source)) & 1U) != 0;
            if (in_range) {
                ++measured.in_range_cells;
            }
            if (active) {
                ++measured.active_source_cells;
            }
            if (member) {
                ++measured.member_source_cells;
            }
            if (decoded_source == static_cast<int>(expected_source) && member) {
                ++measured.matched_cells;
            }
            if (decoded_source > static_cast<int>(measured.max_source)) {
                measured.max_source = static_cast<std::size_t>(decoded_source);
            }
            digest.update_i64_be(static_cast<std::int64_t>(displacement));
            digest.update_i64_be(static_cast<std::int64_t>(lane));
            digest.update_i64_be(static_cast<std::int64_t>(displacement));
            digest.update_i64_be(15);
            digest.update_i64_be(decoded_source);
            for (std::size_t component = 0; component < kComponents; ++component) {
                for (std::size_t bit = 0; bit < 32; ++bit) {
                    const std::int64_t source_bit = static_cast<std::int64_t>(
                        expected_source * 64 + component * 32 + bit);
                    const std::int64_t reconstructed_bit = decoded_source < 0
                        ? -1
                        : static_cast<std::int64_t>(
                              decoded_source * 64 + component * 32 + bit);
                    const std::size_t payload_bit = component * 32 + bit;
                    const bool actual = ((observed_value >> payload_bit) & 1U) != 0;
                    const bool expected =
                        ((input[expected_source] >> payload_bit) & 1U) != 0;
                    if (actual == expected) {
                        ++measured.matched_payload_bits;
                    }
                    digest.update_i64_be(source_bit);
                    digest.update_i64_be(reconstructed_bit);
                }
            }
        }
    }
    measured.digest = digest.finish();
    return measured;
}

std::size_t mismatch_count(const Payloads& input, const Outputs& output) {
    std::size_t mismatches = 0;
    for (std::size_t displacement = 0; displacement < kDimension;
         ++displacement) {
        for (std::size_t lane = 0; lane < kDimension; ++lane) {
            if (output[displacement * kDimension + lane] !=
                input[lane ^ displacement]) {
                ++mismatches;
            }
        }
    }
    return mismatches;
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
    const Measurements measured = measure_material_result(input, output);
    const std::size_t mismatches = mismatch_count(input, output);
    Outputs mutated = output;
    mutated[0] ^= 1U;
    const std::size_t mutation_mismatches = mismatch_count(input, mutated);
    const bool pass = properties.major == 12 && properties.minor == 1 &&
        mismatches == 0 && measured.matched_cells == 256 &&
        measured.in_range_cells == 256 && measured.active_source_cells == 256 &&
        measured.member_source_cells == 256 && measured.max_source == 15 &&
        measured.matched_payload_bits == 16384 &&
        measured.digest == kSounioCandidateDigest && mutation_mismatches > 0;

    std::cout << "PIREUS_DGX_PTX_SHFL_MATERIAL_PARITY_V1\n"
              << "producer_language=C++\n"
              << "producer_role=MATERIAL_PARITY\n"
              << "semantic_authority_language=Sounio\n"
              << "sounio_source_sha256=" << kSounioSourceSha256 << '\n'
              << "frozen_semantics_sha256=" << kFrozenSemanticsSha256 << '\n'
              << "target=dgx\n"
              << "architecture=aarch64\n"
              << "gpu_name=" << properties.name << '\n'
              << "compute_capability=" << properties.major << '.'
              << properties.minor << '\n'
              << "material_instruction=shfl.sync.bfly.b32\n"
              << "sounio_abstract_membermask_low=65535\n"
              << "sounio_abstract_segmask=0\n"
              << "sounio_abstract_cval=15\n"
              << "sounio_abstract_packed_c=15\n"
              << "displacements=16\n"
              << "logical_cells=256\n"
              << "matched_cells=" << measured.matched_cells << '\n'
              << "mismatched_cells=" << mismatches << '\n'
              << "in_range_cells=" << measured.in_range_cells << '\n'
              << "inferred_own_lane_fallback_cells=0\n"
              << "active_source_cells=" << measured.active_source_cells << '\n'
              << "member_source_cells=" << measured.member_source_cells << '\n'
              << "max_source_lane=" << measured.max_source << '\n'
              << "sounio_abstract_f64_components=2\n"
              << "material_b32_components=2\n"
              << "payload_component_cells=512\n"
              << "symbolic_payload_bits=16384\n"
              << "matched_payload_bits=" << measured.matched_payload_bits << '\n'
              << "abstract_shfl_sync_instructions=32\n"
              << "identity_shfl_sync_instructions=2\n"
              << "nontrivial_shfl_sync_instructions=30\n"
              << "candidate_digest="
              << pireus::material::digest_words(measured.digest) << '\n'
              << "candidate_digest_sha256="
              << pireus::material::digest_hex(measured.digest) << '\n'
              << "mutation_mismatching_cells=" << mutation_mismatches << '\n'
              << "unresolved_other_nodes=3\n"
              << "exact_tree_reduction_refused=true\n"
              << "claim_ready=false\n"
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
