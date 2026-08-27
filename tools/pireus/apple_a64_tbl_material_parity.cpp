// C++ MATERIAL_PARITY probe for the frozen Sounio Apple A64 TBL candidate.
// Expected coordinates and digest are consumed from Sounio; C++ defines none.

#include "material_sha256.hpp"

#include <arm_neon.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

constexpr std::size_t kDimension = 16;
constexpr std::size_t kPayloadBytes = 8;
constexpr std::size_t kOutputElements = 2;
constexpr std::size_t kTableElements = 8;
constexpr std::size_t kVectorsPerDisplacement = 8;
constexpr char kSounioSourceSha256[] =
    "79c2e859ffe81f3add1ebb36608a5995672c10a5c1645ec4500a03fcd9bcd031";
constexpr char kFrozenSemanticsSha256[] =
    "377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6";
constexpr std::array<std::uint32_t, 8> kSounioCandidateDigest = {
    472477255U, 3903797350U, 1348128039U, 308036239U,
    1349218781U, 3920188038U, 1317640714U, 1357523880U,
};

constexpr std::uint64_t payload_for_source(std::size_t source) {
    return UINT64_C(0x0123456789abcdef) ^
        (UINT64_C(0x1111111111111111) * source);
}

using Payloads = std::array<std::uint64_t, kDimension>;
using Outputs = std::array<std::uint64_t, kDimension * kDimension>;

__attribute__((noinline)) void tbl_material_kernel(const Payloads& input,
                                                    Outputs& output) {
    const auto* input_bytes = reinterpret_cast<const std::uint8_t*>(input.data());
    for (std::size_t displacement = 0; displacement < kDimension;
         ++displacement) {
        for (std::size_t vector = 0; vector < kVectorsPerDisplacement;
             ++vector) {
            const std::size_t logical = vector * kOutputElements;
            const std::size_t source0 = logical ^ displacement;
            const std::size_t source1 = (logical + 1) ^ displacement;
            const std::size_t source_group = source0 / kTableElements;
            const std::size_t source0_local = source0 % kTableElements;
            const std::size_t source1_local = source1 % kTableElements;

            uint8x16x4_t table{};
            const std::uint8_t* table_bytes =
                input_bytes + source_group * kTableElements * kPayloadBytes;
            table.val[0] = vld1q_u8(table_bytes);
            table.val[1] = vld1q_u8(table_bytes + 16);
            table.val[2] = vld1q_u8(table_bytes + 32);
            table.val[3] = vld1q_u8(table_bytes + 48);

            alignas(16) std::array<std::uint8_t, 16> controls{};
            for (std::size_t byte = 0; byte < kPayloadBytes; ++byte) {
                controls[byte] = static_cast<std::uint8_t>(
                    source0_local * kPayloadBytes + byte);
                controls[kPayloadBytes + byte] = static_cast<std::uint8_t>(
                    source1_local * kPayloadBytes + byte);
            }
            const uint8x16_t selected =
                vqtbl4q_u8(table, vld1q_u8(controls.data()));
            auto* destination = reinterpret_cast<std::uint8_t*>(
                output.data() + displacement * kDimension + logical);
            vst1q_u8(destination, selected);
        }
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
    std::size_t matched_bytes = 0;
    std::size_t matched_bits = 0;
    std::size_t in_domain_cells = 0;
    std::size_t bijective_displacements = 0;
    std::size_t out_of_range_controls = 0;
    std::size_t max_control = 0;
    bool same_source_group = true;
    std::array<std::uint32_t, 8> digest{};
};

Measurements measure_material_result(const Payloads& input,
                                     const Outputs& output) {
    Measurements result;
    pireus::material::Sha256 digest;
    digest.update_i64_be(INT64_C(0x415054424c583031));
    for (std::size_t displacement = 0; displacement < kDimension;
         ++displacement) {
        std::array<std::size_t, kDimension> visits{};
        for (std::size_t vector = 0; vector < kVectorsPerDisplacement;
             ++vector) {
            int vector_group = -1;
            for (std::size_t element = 0; element < kOutputElements; ++element) {
                const std::size_t logical = vector * kOutputElements + element;
                const std::size_t expected_source = logical ^ displacement;
                const std::uint64_t observed_value =
                    output[displacement * kDimension + logical];
                const int decoded_source = observed_source(input, observed_value);
                const std::size_t source_group =
                    expected_source / kTableElements;
                const std::size_t source_local =
                    expected_source % kTableElements;
                if (vector_group < 0) {
                    vector_group = static_cast<int>(source_group);
                }
                if (vector_group != static_cast<int>(source_group)) {
                    result.same_source_group = false;
                }
                if (decoded_source >= 0 && decoded_source < 16) {
                    ++result.in_domain_cells;
                    ++visits[static_cast<std::size_t>(decoded_source)];
                }
                if (decoded_source == static_cast<int>(expected_source)) {
                    ++result.matched_cells;
                }
                for (std::size_t byte = 0; byte < kPayloadBytes; ++byte) {
                    const std::size_t control = source_local * kPayloadBytes + byte;
                    if (control > result.max_control) {
                        result.max_control = control;
                    }
                    if (control >= kTableElements * kPayloadBytes) {
                        ++result.out_of_range_controls;
                    }
                    const auto* observed_bytes = reinterpret_cast<const std::uint8_t*>(
                        &observed_value);
                    const auto* expected_bytes = reinterpret_cast<const std::uint8_t*>(
                        &input[expected_source]);
                    if (observed_bytes[byte] == expected_bytes[byte]) {
                        ++result.matched_bytes;
                    }
                    for (std::size_t bit = 0; bit < 8; ++bit) {
                        const std::int64_t expected_bit = static_cast<std::int64_t>(
                            expected_source * 64 + byte * 8 + bit);
                        const std::int64_t reconstructed_bit = decoded_source < 0
                            ? -1
                            : static_cast<std::int64_t>(
                                  decoded_source * 64 + byte * 8 + bit);
                        const bool actual = ((observed_value >> (byte * 8 + bit)) & 1U) != 0;
                        const bool expected =
                            ((input[expected_source] >> (byte * 8 + bit)) & 1U) != 0;
                        if (actual == expected) {
                            ++result.matched_bits;
                        }
                        digest.update_i64_be(expected_bit);
                        digest.update_i64_be(reconstructed_bit);
                    }
                    digest.update_i64_be(static_cast<std::int64_t>(displacement));
                    digest.update_i64_be(static_cast<std::int64_t>(vector));
                    digest.update_i64_be(static_cast<std::int64_t>(element));
                    digest.update_i64_be(static_cast<std::int64_t>(byte));
                    digest.update_i64_be(static_cast<std::int64_t>(source_group));
                    digest.update_i64_be(static_cast<std::int64_t>(control));
                    digest.update_i64_be(decoded_source);
                }
            }
        }
        bool bijective = true;
        for (const std::size_t count : visits) {
            if (count != 1) {
                bijective = false;
            }
        }
        if (bijective) {
            ++result.bijective_displacements;
        }
    }
    result.digest = digest.finish();
    return result;
}

std::size_t mismatch_count(const Payloads& input, const Outputs& output) {
    std::size_t mismatches = 0;
    for (std::size_t displacement = 0; displacement < kDimension;
         ++displacement) {
        for (std::size_t logical = 0; logical < kDimension; ++logical) {
            if (output[displacement * kDimension + logical] !=
                input[logical ^ displacement]) {
                ++mismatches;
            }
        }
    }
    return mismatches;
}

}  // namespace

int main() {
    static_assert(std::endian::native == std::endian::little);
    alignas(64) Payloads input{};
    alignas(64) Outputs output{};
    for (std::size_t source = 0; source < input.size(); ++source) {
        input[source] = payload_for_source(source);
    }
    tbl_material_kernel(input, output);
    const Measurements measured = measure_material_result(input, output);
    const std::size_t mismatches = mismatch_count(input, output);
    Outputs mutated = output;
    mutated[0] ^= 1U;
    const std::size_t mutation_mismatches = mismatch_count(input, mutated);
    const bool pass = mismatches == 0 && measured.matched_cells == 256 &&
        measured.in_domain_cells == 256 && measured.bijective_displacements == 16 &&
        measured.matched_bytes == 2048 && measured.matched_bits == 16384 &&
        measured.max_control == 63 && measured.out_of_range_controls == 0 &&
        measured.same_source_group && measured.digest == kSounioCandidateDigest &&
        mutation_mismatches > 0;

    std::cout << "PIREUS_APPLE_A64_TBL_MATERIAL_PARITY_V1\n"
              << "producer_language=C++\n"
              << "producer_role=MATERIAL_PARITY\n"
              << "semantic_authority_language=Sounio\n"
              << "sounio_source_sha256=" << kSounioSourceSha256 << '\n'
              << "frozen_semantics_sha256=" << kFrozenSemanticsSha256 << '\n'
              << "target=apple_silicon\n"
              << "architecture=arm64\n"
              << "material_instruction=TBL\n"
              << "displacements=16\n"
              << "logical_cells=256\n"
              << "matched_cells=" << measured.matched_cells << '\n'
              << "mismatched_cells=" << mismatches << '\n'
              << "in_domain_source_cells=" << measured.in_domain_cells << '\n'
              << "bijective_displacements=" << measured.bijective_displacements << '\n'
              << "byte_control_cells=2048\n"
              << "matched_byte_controls=" << measured.matched_bytes << '\n'
              << "max_control=" << measured.max_control << '\n'
              << "out_of_range_controls=" << measured.out_of_range_controls << '\n'
              << "abstract_tbl_applications=128\n"
              << "symbolic_payload_bits=16384\n"
              << "matched_payload_bits=" << measured.matched_bits << '\n'
              << "same_source_group_per_output=" << std::boolalpha
              << measured.same_source_group << '\n'
              << "candidate_digest="
              << pireus::material::digest_words(measured.digest) << '\n'
              << "candidate_digest_sha256="
              << pireus::material::digest_hex(measured.digest) << '\n'
              << "mutation_mismatching_cells=" << mutation_mismatches << '\n'
              << "unresolved_other_nodes=4\n"
              << "exact_tree_reduction_refused=true\n"
              << "claim_ready=false\n"
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
