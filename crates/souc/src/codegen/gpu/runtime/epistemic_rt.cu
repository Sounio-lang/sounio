// Sounio Omega L2 - CUDA epistemic runtime stubs.
//
// This file provides device-side fallbacks used by the PTX epistemic bridge.
// It intentionally stays self-contained so it can be compiled to cubin with:
//   nvcc -cubin -arch=sm_80 epistemic_rt.cu -o epistemic_rt.cubin

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct alignas(32) uint256_t {
    uint64_t lane0;
    uint64_t lane1;
    uint64_t lane2;
    uint64_t lane3;
};

struct KnowledgeLane {
    double value;
    double variance;
    uint256_t provenance;
    double beta_alpha;
    double beta_beta;
};

struct KaxiPayload {
    double value;
    double variance;
    uint256_t provenance;
    uint64_t confidence_bits;
    uint32_t flags;
    uint32_t reserved;
};

static_assert(sizeof(uint256_t) == 32, "uint256_t must be 256 bits");

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int shift) {
    return (x << shift) | (x >> (64 - shift));
}

static constexpr unsigned KAXI_RING_SIZE = 64;
static_assert((KAXI_RING_SIZE & (KAXI_RING_SIZE - 1)) == 0, "KAXI_RING_SIZE must be power of two");

static constexpr uint8_t KAXI_OP_ADD = 0x1;
static constexpr uint8_t KAXI_OP_MUL = 0x2;
static constexpr uint8_t KAXI_OP_DIV = 0x3;
static constexpr uint8_t KAXI_OP_FMA = 0x4;
static constexpr uint8_t KAXI_OP_PROP_VAR = 0x5;
static constexpr uint8_t KAXI_OP_MERGE_PROV = 0x6;

// K-AXI publish ring: this is a no-op-safe bridge point for FPGA/ASIC wiring.
__device__ KaxiPayload g_kaxi_ring[KAXI_RING_SIZE];
__device__ unsigned int g_kaxi_write_idx = 0;
// Mirror of hardware accumulator register (Q32.32).
__device__ unsigned long long g_epistemic_power_log_hw_q32_32 = 0ull;

__device__ __forceinline__ unsigned long long __knowledge_epistemic_power_delta(uint32_t flags) {
    const uint8_t op_kind = static_cast<uint8_t>(flags & 0x0Fu);
    const uint8_t failure_inc = static_cast<uint8_t>((flags >> 8) & 0xFFu);
    const uint8_t success_inc = static_cast<uint8_t>((flags >> 16) & 0xFFu);
    const uint8_t alpha_q8 = static_cast<uint8_t>((flags >> 24) & 0xFFu);

    unsigned long long delta = 1ull << 16; // base log-space increment
    if (op_kind == KAXI_OP_FMA) {
        delta += (1ull << 15);
    } else if (op_kind == KAXI_OP_MERGE_PROV) {
        delta += (1ull << 13);
    }

    delta += (static_cast<unsigned long long>(alpha_q8) << 8);
    delta += (static_cast<unsigned long long>(success_inc) << 12);
    const unsigned long long penalty = static_cast<unsigned long long>(failure_inc) << 10;
    if (delta > penalty) {
        delta -= penalty;
    } else {
        delta = 1ull << 8;
    }
    return delta;
}

__device__ __forceinline__ void __knowledge_kaxi_publish(
    double value,
    double variance,
    const uint256_t& prov,
    double confidence,
    uint32_t flags
) {
    const unsigned int idx = atomicAdd(&g_kaxi_write_idx, 1u) & (KAXI_RING_SIZE - 1);
    g_kaxi_ring[idx].value = value;
    g_kaxi_ring[idx].variance = fmax(0.0, variance);
    g_kaxi_ring[idx].provenance = prov;
    g_kaxi_ring[idx].confidence_bits = static_cast<uint64_t>(__double_as_longlong(confidence));
    g_kaxi_ring[idx].flags = flags;
    g_kaxi_ring[idx].reserved = 0;
    atomicAdd(&g_epistemic_power_log_hw_q32_32, __knowledge_epistemic_power_delta(flags));
}

__device__ __forceinline__ uint8_t __knowledge_kaxi_pack_alpha_q8(double alpha) {
    const double clamped = fmin(1.0, fmax(0.0, alpha));
    return static_cast<uint8_t>(clamped * 255.0 + 0.5);
}

__device__ __forceinline__ uint8_t __knowledge_kaxi_success_inc(double confidence) {
    return confidence >= 0.5 ? 1u : 0u;
}

__device__ __forceinline__ uint8_t __knowledge_kaxi_failure_inc(double confidence) {
    return confidence < 0.5 ? 1u : 0u;
}

__device__ __forceinline__ uint32_t __knowledge_kaxi_pack_flags(
    uint8_t op_kind,
    uint8_t alpha_q8,
    uint8_t success_inc,
    uint8_t failure_inc
) {
    return (static_cast<uint32_t>(alpha_q8) << 24) |
           (static_cast<uint32_t>(success_inc) << 16) |
           (static_cast<uint32_t>(failure_inc) << 8) |
           (static_cast<uint32_t>(op_kind) & 0x0Fu);
}

extern "C" __device__ const KaxiPayload* __knowledge_kaxi_ring_base() {
    return g_kaxi_ring;
}

extern "C" __device__ unsigned int __knowledge_kaxi_ring_head() {
    return g_kaxi_write_idx;
}

extern "C" __device__ uint64_t __knowledge_get_epistemic_power_log_q32_32() {
    return static_cast<uint64_t>(g_epistemic_power_log_hw_q32_32);
}

extern "C" __device__ void __knowledge_set_epistemic_power_log_q32_32(uint64_t value) {
    g_epistemic_power_log_hw_q32_32 = static_cast<unsigned long long>(value);
}

// Fast Merkle-like mix (XOR + rotate + multiplicative avalanche).
__device__ __forceinline__ uint256_t __knowledge_merge_prov_fast(
    const uint256_t& a,
    const uint256_t& b
) {
    uint256_t out;
    const uint64_t k0 = 0x9E3779B185EBCA87ULL;
    const uint64_t k1 = 0xC2B2AE3D27D4EB4FULL;

    out.lane0 = rotl64(a.lane0 ^ b.lane0, 13) * k0;
    out.lane1 = rotl64(a.lane1 ^ b.lane1, 29) * k1;
    out.lane2 = rotl64(a.lane2 ^ b.lane2, 41) * k0;
    out.lane3 = rotl64(a.lane3 ^ b.lane3, 53) * k1;
    return out;
}

__device__ __forceinline__ double __knowledge_beta_confidence(double alpha, double beta) {
    const double denom = alpha + beta;
    if (denom <= 0.0) {
        return 0.0;
    }
    return alpha / denom;
}

__device__ __forceinline__ void __knowledge_beta_update(
    double prior_alpha,
    double prior_beta,
    double success,
    double* out_alpha,
    double* out_beta
) {
    // success is expected in [0, 1].
    const double s = fmin(1.0, fmax(0.0, success));
    *out_alpha = prior_alpha + s;
    *out_beta = prior_beta + (1.0 - s);
}

extern "C" __device__ double __knowledge_add(
    const KnowledgeLane a,
    const KnowledgeLane b,
    double* var_out,
    uint256_t* prov_out,
    double* conf_out
) {
    const double val = a.value + b.value;
    const double var = fmax(0.0, a.variance) + fmax(0.0, b.variance);
    const uint256_t prov = __knowledge_merge_prov_fast(a.provenance, b.provenance);
    const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
    const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
    const double conf = fmin(ca, cb);
    const uint32_t flags = __knowledge_kaxi_pack_flags(
        KAXI_OP_ADD,
        __knowledge_kaxi_pack_alpha_q8(0.03125), // ~8/256
        __knowledge_kaxi_success_inc(conf),
        __knowledge_kaxi_failure_inc(conf)
    );

    if (var_out) *var_out = var;
    if (prov_out) *prov_out = prov;
    if (conf_out) *conf_out = conf;
    __knowledge_kaxi_publish(val, var, prov, conf, flags);

    return val;
}

extern "C" __device__ double __knowledge_mul(
    const KnowledgeLane a,
    const KnowledgeLane b,
    double* var_out,
    uint256_t* prov_out,
    double* conf_out
) {
    const double val = a.value * b.value;
    // GUM delta method: var = b^2 * var(a) + a^2 * var(b)
    const double var = (b.value * b.value * fmax(0.0, a.variance)) +
                       (a.value * a.value * fmax(0.0, b.variance));
    const uint256_t prov = __knowledge_merge_prov_fast(a.provenance, b.provenance);
    const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
    const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
    const double conf = fmin(ca, cb);
    const uint32_t flags = __knowledge_kaxi_pack_flags(
        KAXI_OP_MUL,
        __knowledge_kaxi_pack_alpha_q8(0.03125),
        __knowledge_kaxi_success_inc(conf),
        __knowledge_kaxi_failure_inc(conf)
    );

    if (var_out) *var_out = var;
    if (prov_out) *prov_out = prov;
    if (conf_out) *conf_out = conf;
    __knowledge_kaxi_publish(val, var, prov, conf, flags);

    return val;
}

extern "C" __device__ double __knowledge_div(
    const KnowledgeLane a,
    const KnowledgeLane b,
    double* var_out,
    uint256_t* prov_out,
    double* conf_out
) {
    const double eps = 1e-12;
    const double denom = fabs(b.value) < eps ? copysign(eps, b.value == 0.0 ? 1.0 : b.value) : b.value;
    const double val = a.value / denom;

    // GUM delta method for division:
    // var = (var(a)/b^2) + (a^2 * var(b)/b^4)
    const double b2 = denom * denom;
    const double b4 = b2 * b2;
    const double var = (fmax(0.0, a.variance) / b2) +
                       (a.value * a.value * fmax(0.0, b.variance) / b4);
    const uint256_t prov = __knowledge_merge_prov_fast(a.provenance, b.provenance);
    const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
    const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
    const double conf = fmin(ca, cb);
    const uint32_t flags = __knowledge_kaxi_pack_flags(
        KAXI_OP_DIV,
        __knowledge_kaxi_pack_alpha_q8(0.03125),
        __knowledge_kaxi_success_inc(conf),
        __knowledge_kaxi_failure_inc(conf)
    );

    if (var_out) *var_out = var;
    if (prov_out) *prov_out = prov;
    if (conf_out) *conf_out = conf;
    __knowledge_kaxi_publish(val, var, prov, conf, flags);

    return val;
}

extern "C" __device__ double __knowledge_fma(
    const KnowledgeLane a,
    const KnowledgeLane b,
    const KnowledgeLane c,
    double* var_out,
    uint256_t* prov_out,
    double* conf_out
) {
    const double val = fma(a.value, b.value, c.value);
    const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
    const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
    const double cc = __knowledge_beta_confidence(c.beta_alpha, c.beta_beta);
    const double observed_success = fmin(ca, fmin(cb, cc));
    const uint256_t ab = __knowledge_merge_prov_fast(a.provenance, b.provenance);
    const uint256_t prov = __knowledge_merge_prov_fast(ab, c.provenance);
    const uint32_t flags = __knowledge_kaxi_pack_flags(
        KAXI_OP_FMA,
        __knowledge_kaxi_pack_alpha_q8(0.0625), // ~16/256
        __knowledge_kaxi_success_inc(observed_success),
        __knowledge_kaxi_failure_inc(observed_success)
    );

    // Exact first-order GUM delta for FMA: b^2*var(a) + a^2*var(b) + var(c)
    const double var = (b.value * b.value * fmax(0.0, a.variance)) +
                       (a.value * a.value * fmax(0.0, b.variance)) +
                       fmax(0.0, c.variance);

    if (var_out) *var_out = var;

    if (prov_out) *prov_out = prov;

    if (conf_out) {
        double alpha_next = 1.0;
        double beta_next = 1.0;
        __knowledge_beta_update(1.0, 1.0, observed_success, &alpha_next, &beta_next);
        *conf_out = __knowledge_beta_confidence(alpha_next, beta_next);
    }

    __knowledge_kaxi_publish(val, var, prov, observed_success, flags);

    return val;
}

extern "C" __device__ double __knowledge_prop_var(
    double var_in,
    double degradation_alpha
) {
    // Epistemic degradation model: var_out = var_in * (1 + alpha)
    const double v = fmax(0.0, var_in);
    const double alpha = fmax(0.0, degradation_alpha);
    const double out = v * (1.0 + alpha);
    const uint256_t zero = {0, 0, 0, 0};
    const uint32_t flags = __knowledge_kaxi_pack_flags(
        KAXI_OP_PROP_VAR,
        __knowledge_kaxi_pack_alpha_q8(alpha),
        1u,
        0u
    );
    __knowledge_kaxi_publish(0.0, out, zero, 1.0, flags);
    return out;
}

extern "C" __device__ void __knowledge_merge_prov(
    uint256_t* dst,
    const uint256_t* a,
    const uint256_t* b
) {
    if (!a || !b) return;
    const uint256_t merged = __knowledge_merge_prov_fast(*a, *b);
    if (dst) *dst = merged;
    const uint32_t flags = __knowledge_kaxi_pack_flags(
        KAXI_OP_MERGE_PROV,
        0u,
        1u,
        0u
    );
    __knowledge_kaxi_publish(0.0, 0.0, merged, 1.0, flags);
}
