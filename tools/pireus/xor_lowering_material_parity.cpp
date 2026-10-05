// MATERIAL_PARITY probe for the frozen Sounio Pireus XOR-lowering semantics.
// This file consumes expected bits and masks from Sounio; it defines neither.

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>

namespace {

constexpr std::size_t kDimension = 16;
constexpr std::size_t kChunkLanes = 8;
constexpr std::size_t kGroupCount = 32;

constexpr char kFrozenSemanticsSha256[] =
    "9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970";

constexpr std::array<std::uint8_t, kGroupCount> kNegativeLaneMasks = {
    254, 255, 104, 150, 194, 60, 164, 90,
    14, 240, 84, 170, 152, 102, 50, 204,
    254, 0, 148, 149, 56, 57, 82, 83,
    224, 225, 138, 139, 38, 39, 76, 77,
};

alignas(64) constexpr std::array<std::array<std::int64_t, kChunkLanes>,
                                 kChunkLanes>
    kXorControls = {{
        {{0, 1, 2, 3, 4, 5, 6, 7}},
        {{1, 0, 3, 2, 5, 4, 7, 6}},
        {{2, 3, 0, 1, 6, 7, 4, 5}},
        {{3, 2, 1, 0, 7, 6, 5, 4}},
        {{4, 5, 6, 7, 0, 1, 2, 3}},
        {{5, 4, 7, 6, 1, 0, 3, 2}},
        {{6, 7, 4, 5, 2, 3, 0, 1}},
        {{7, 6, 5, 4, 3, 2, 1, 0}},
    }};

constexpr std::array<std::uint64_t, kDimension> kFrozenOutputBits = {
    UINT64_C(4613118981945187609),
    static_cast<std::uint64_t>(-INT64_C(4611510553506841004)),
    static_cast<std::uint64_t>(-INT64_C(4614727410383534212)),
    static_cast<std::uint64_t>(-INT64_C(4608966312158910915)),
    UINT64_C(4604258003457569030),
    static_cast<std::uint64_t>(-INT64_C(4604813642372634225)),
    UINT64_C(4612475610569848966),
    UINT64_C(4616423571282154268),
    static_cast<std::uint64_t>(-INT64_C(4600193066131565794)),
    UINT64_C(4615458514219146306),
    static_cast<std::uint64_t>(-INT64_C(4606699890268513426)),
    static_cast<std::uint64_t>(-INT64_C(4603058993167165201)),
    static_cast<std::uint64_t>(-INT64_C(4605968786432901330)),
    static_cast<std::uint64_t>(-INT64_C(4614201015621893512)),
    static_cast<std::uint64_t>(-INT64_C(4608907823852061948)),
    static_cast<std::uint64_t>(-INT64_C(4604608933298662840)),
};

struct Comparison {
    std::size_t matches = 0;
    std::size_t mismatches = 0;
    int first_mismatch = -1;
};

void generate_frozen_inputs(std::array<double, kDimension>& a,
                            std::array<double, kDimension>& b) {
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(kDimension); ++i) {
        const std::int64_t ai = ((i * 17 + 5) % 29) - 14;
        const std::int64_t bi = ((i * i * 11 + i * 7 + 3) % 31) - 15;
        a[static_cast<std::size_t>(i)] = static_cast<double>(ai) / 7.0;
        b[static_cast<std::size_t>(i)] = static_cast<double>(bi) / 11.0;
    }
}

__attribute__((noinline)) double strict_ascending_sum(const double* terms) {
    double sum = 0.0;
    for (std::size_t i = 0; i < kDimension; ++i) {
        sum = sum + terms[i];
    }
    return sum;
}

void frozen_scalar_reference(
    const std::array<double, kDimension>& a,
    const std::array<double, kDimension>& b,
    const std::array<std::uint8_t, kGroupCount>& masks,
    std::array<double, kDimension>& output) {
    for (std::size_t d = 0; d < kDimension; ++d) {
        double sum = 0.0;
        for (std::size_t i = 0; i < kDimension; ++i) {
            const std::size_t group = d * 2 + i / kChunkLanes;
            const std::size_t lane = i % kChunkLanes;
            const bool negative = ((masks[group] >> lane) & 1U) != 0;
            const double coefficient = negative ? -1.0 : 1.0;
            const std::size_t partner = i ^ d;
            sum = sum + coefficient * a[i] * b[partner];
        }
        output[d] = sum;
    }
}

__attribute__((target("avx512f,avx512dq"), noinline)) void avx512_material_kernel(
    const double* a,
    const double* b,
    const std::uint8_t* masks,
    const std::int64_t* controls,
    double* terms,
    double* output) {
    const __m512d a_low = _mm512_loadu_pd(a);
    const __m512d a_high = _mm512_loadu_pd(a + kChunkLanes);
    const __m512d b_low = _mm512_loadu_pd(b);
    const __m512d b_high = _mm512_loadu_pd(b + kChunkLanes);
    const __m512i sign_bit =
        _mm512_set1_epi64(std::numeric_limits<std::int64_t>::min());

    for (std::size_t d = 0; d < kDimension; ++d) {
        const __m512i control = _mm512_loadu_si512(
            controls + (d & (kChunkLanes - 1)) * kChunkLanes);
        const bool crosses_half = (d & kChunkLanes) != 0;
        const __m512d partner_low = _mm512_permutexvar_pd(
            control, crosses_half ? b_high : b_low);
        const __m512d partner_high = _mm512_permutexvar_pd(
            control, crosses_half ? b_low : b_high);

        const __mmask8 low_mask = static_cast<__mmask8>(masks[d * 2]);
        const __mmask8 high_mask = static_cast<__mmask8>(masks[d * 2 + 1]);
        const __m512i a_low_bits = _mm512_castpd_si512(a_low);
        const __m512i a_high_bits = _mm512_castpd_si512(a_high);
        const __m512d signed_a_low = _mm512_castsi512_pd(
            _mm512_mask_xor_epi64(a_low_bits, low_mask, a_low_bits, sign_bit));
        const __m512d signed_a_high = _mm512_castsi512_pd(
            _mm512_mask_xor_epi64(a_high_bits, high_mask, a_high_bits, sign_bit));

        const __m512d product_low = _mm512_mul_pd(signed_a_low, partner_low);
        const __m512d product_high = _mm512_mul_pd(signed_a_high, partner_high);
        double* displacement_terms = terms + d * kDimension;
        _mm512_storeu_pd(displacement_terms, product_low);
        _mm512_storeu_pd(displacement_terms + kChunkLanes, product_high);
        output[d] = strict_ascending_sum(displacement_terms);
    }
}

Comparison compare_to_frozen(const std::array<double, kDimension>& values) {
    Comparison comparison;
    for (std::size_t d = 0; d < kDimension; ++d) {
        if (std::bit_cast<std::uint64_t>(values[d]) == kFrozenOutputBits[d]) {
            ++comparison.matches;
        } else {
            if (comparison.first_mismatch < 0) {
                comparison.first_mismatch = static_cast<int>(d);
            }
            ++comparison.mismatches;
        }
    }
    return comparison;
}

std::size_t compare_terms(
    const std::array<double, kDimension>& a,
    const std::array<double, kDimension>& b,
    const std::array<double, kDimension * kDimension>& terms) {
    std::size_t matches = 0;
    for (std::size_t d = 0; d < kDimension; ++d) {
        for (std::size_t i = 0; i < kDimension; ++i) {
            const std::size_t group = d * 2 + i / kChunkLanes;
            const std::size_t lane = i % kChunkLanes;
            const bool negative = ((kNegativeLaneMasks[group] >> lane) & 1U) != 0;
            const double coefficient = negative ? -1.0 : 1.0;
            const double expected = coefficient * a[i] * b[i ^ d];
            if (std::bit_cast<std::uint64_t>(terms[d * kDimension + i]) ==
                std::bit_cast<std::uint64_t>(expected)) {
                ++matches;
            }
        }
    }
    return matches;
}

std::size_t count_negative_cells() {
    std::size_t total = 0;
    for (const std::uint8_t mask : kNegativeLaneMasks) {
        total += static_cast<std::size_t>(std::popcount(mask));
    }
    return total;
}

std::size_t validate_partner_cells() {
    std::size_t valid = 0;
    for (std::size_t d = 0; d < kDimension; ++d) {
        for (std::size_t i = 0; i < kDimension; ++i) {
            const std::size_t lane = i % kChunkLanes;
            const std::size_t partner_lane =
                static_cast<std::size_t>(kXorControls[d & 7U][lane]);
            const std::size_t partner_half = ((i / kChunkLanes) ^ (d >> 3U));
            if (partner_half * kChunkLanes + partner_lane == (i ^ d)) {
                ++valid;
            }
        }
    }
    return valid;
}

void print_lane_bits(const std::array<double, kDimension>& values) {
    for (std::size_t d = 0; d < kDimension; ++d) {
        std::cout << "lane[" << d << "].bits=0x" << std::hex
                  << std::setw(16) << std::setfill('0')
                  << std::bit_cast<std::uint64_t>(values[d]) << std::dec << '\n';
    }
}

}  // namespace

int main() {
#if !defined(__x86_64__)
    std::cerr << "PIREUS_XOR_MATERIAL_PARITY_UNSUPPORTED architecture!=x86_64\n";
    return 20;
#else
    __builtin_cpu_init();
    const bool avx512f = __builtin_cpu_supports("avx512f");
    const bool avx512dq = __builtin_cpu_supports("avx512dq");
    if (!avx512f || !avx512dq) {
        std::cerr << "PIREUS_XOR_MATERIAL_PARITY_UNSUPPORTED avx512f=" << avx512f
                  << " avx512dq=" << avx512dq << '\n';
        return 21;
    }
    const int rounding_mode = std::fegetround();
    const unsigned int mxcsr = _mm_getcsr();
    constexpr unsigned int kDenormalsAreZeroMask = 1U << 6U;
    const bool flush_to_zero = (mxcsr & _MM_FLUSH_ZERO_MASK) != 0;
    const bool denormals_are_zero = (mxcsr & kDenormalsAreZeroMask) != 0;
    if (rounding_mode != FE_TONEAREST || flush_to_zero || denormals_are_zero) {
        std::cerr << "PIREUS_XOR_MATERIAL_PARITY_UNSUPPORTED rounding_mode="
                  << rounding_mode << " mxcsr=0x" << std::hex << mxcsr << std::dec
                  << " flush_to_zero=" << std::boolalpha << flush_to_zero
                  << " denormals_are_zero=" << denormals_are_zero << '\n';
        return 22;
    }

    alignas(64) std::array<double, kDimension> a{};
    alignas(64) std::array<double, kDimension> b{};
    alignas(64) std::array<double, kDimension> scalar_output{};
    alignas(64) std::array<double, kDimension> vector_output{};
    alignas(64) std::array<double, kDimension * kDimension> terms{};
    generate_frozen_inputs(a, b);
    frozen_scalar_reference(a, b, kNegativeLaneMasks, scalar_output);
    avx512_material_kernel(a.data(), b.data(), kNegativeLaneMasks.data(),
                           kXorControls[0].data(), terms.data(),
                           vector_output.data());

    auto mutated_masks = kNegativeLaneMasks;
    mutated_masks[0] ^= 1U;
    std::array<double, kDimension> sign_mutation_output{};
    std::array<double, kDimension * kDimension> sign_mutation_terms{};
    avx512_material_kernel(a.data(), b.data(), mutated_masks.data(),
                           kXorControls[0].data(), sign_mutation_terms.data(),
                           sign_mutation_output.data());

    auto mutated_controls = kXorControls;
    mutated_controls[0][0] = 1;
    std::array<double, kDimension> selector_mutation_output{};
    std::array<double, kDimension * kDimension> selector_mutation_terms{};
    avx512_material_kernel(a.data(), b.data(), kNegativeLaneMasks.data(),
                           mutated_controls[0].data(),
                           selector_mutation_terms.data(),
                           selector_mutation_output.data());

    const Comparison scalar = compare_to_frozen(scalar_output);
    const Comparison vector = compare_to_frozen(vector_output);
    const Comparison sign_mutation = compare_to_frozen(sign_mutation_output);
    const Comparison selector_mutation = compare_to_frozen(selector_mutation_output);
    const std::size_t negative_cells = count_negative_cells();
    const std::size_t partner_cells = validate_partner_cells();
    const std::size_t term_matches = compare_terms(a, b, terms);

    const bool pass = scalar.matches == kDimension && scalar.mismatches == 0 &&
        vector.matches == kDimension && vector.mismatches == 0 &&
        term_matches == kDimension * kDimension && negative_cells == 120 &&
        partner_cells == kDimension * kDimension &&
        sign_mutation.mismatches > 0 && selector_mutation.mismatches > 0;

    std::cout << "PIREUS_XOR_MATERIAL_PARITY_V1\n"
              << "producer_language=C++\n"
              << "producer_role=MATERIAL_PARITY\n"
              << "semantic_authority_language=Sounio\n"
              << "semantic_authority_role=SEMANTIC_AUTHORITY\n"
              << "source_semantics_sha256=" << kFrozenSemanticsSha256 << '\n'
              << "target=darwin_xeon\n"
              << "avx512f=" << std::boolalpha << avx512f << '\n'
              << "avx512dq=" << avx512dq << '\n'
              << "rounding_mode=FE_TONEAREST\n"
              << "mxcsr=0x" << std::hex << mxcsr << std::dec << '\n'
              << "flush_to_zero=" << flush_to_zero << '\n'
              << "denormals_are_zero=" << denormals_are_zero << '\n'
              << "dimension=" << kDimension << '\n'
              << "selector_groups=" << kGroupCount << '\n'
              << "partner_cells=" << partner_cells << '\n'
              << "partner_failures=" << kDimension * kDimension - partner_cells << '\n'
              << "sign_mask_groups=" << kNegativeLaneMasks.size() << '\n'
              << "negative_cells=" << negative_cells << '\n'
              << "positive_cells=" << kDimension * kDimension - negative_cells << '\n'
              << "vector_term_matching_cells=" << term_matches << '\n'
              << "frozen_scalar_matching_lanes=" << scalar.matches << '\n'
              << "vector_matching_lanes=" << vector.matches << '\n'
              << "vector_mismatching_lanes=" << vector.mismatches << '\n'
              << "vector_first_mismatch=" << vector.first_mismatch << '\n'
              << "sign_mutation_mismatching_lanes=" << sign_mutation.mismatches << '\n'
              << "selector_mutation_mismatching_lanes="
              << selector_mutation.mismatches << '\n'
              << "ascending_i=true\n"
              << "reassociated=false\n"
              << "vpermpd_one_source_per_group=true\n"
              << "material_nodes_realized=5\n"
              << "apple_silicon_observed=false\n"
              << "dgx_observed=false\n"
              << "generic_cost_claim=false\n"
              << "claim_ready=false\n";
    print_lane_bits(vector_output);
    std::cout << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
