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

static_assert(sizeof(uint256_t) == 32, "uint256_t must be 256 bits");

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int shift) {
    return (x << shift) | (x >> (64 - shift));
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

    if (var_out) *var_out = var;
    if (prov_out) *prov_out = __knowledge_merge_prov_fast(a.provenance, b.provenance);

    if (conf_out) {
        const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
        const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
        *conf_out = fmin(ca, cb);
    }

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

    if (var_out) *var_out = var;
    if (prov_out) *prov_out = __knowledge_merge_prov_fast(a.provenance, b.provenance);

    if (conf_out) {
        const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
        const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
        *conf_out = fmin(ca, cb);
    }

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

    if (var_out) *var_out = var;
    if (prov_out) *prov_out = __knowledge_merge_prov_fast(a.provenance, b.provenance);

    if (conf_out) {
        const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
        const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
        *conf_out = fmin(ca, cb);
    }

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

    // Exact first-order GUM delta for FMA: b^2*var(a) + a^2*var(b) + var(c)
    const double var = (b.value * b.value * fmax(0.0, a.variance)) +
                       (a.value * a.value * fmax(0.0, b.variance)) +
                       fmax(0.0, c.variance);

    if (var_out) *var_out = var;

    if (prov_out) {
        const uint256_t ab = __knowledge_merge_prov_fast(a.provenance, b.provenance);
        *prov_out = __knowledge_merge_prov_fast(ab, c.provenance);
    }

    if (conf_out) {
        const double ca = __knowledge_beta_confidence(a.beta_alpha, a.beta_beta);
        const double cb = __knowledge_beta_confidence(b.beta_alpha, b.beta_beta);
        const double cc = __knowledge_beta_confidence(c.beta_alpha, c.beta_beta);
        const double observed_success = fmin(ca, fmin(cb, cc));
        double alpha_next = 1.0;
        double beta_next = 1.0;
        __knowledge_beta_update(1.0, 1.0, observed_success, &alpha_next, &beta_next);
        *conf_out = __knowledge_beta_confidence(alpha_next, beta_next);
    }

    return val;
}

extern "C" __device__ double __knowledge_prop_var(
    double var_in,
    double degradation_alpha
) {
    // Epistemic degradation model: var_out = var_in * (1 + alpha)
    const double v = fmax(0.0, var_in);
    const double alpha = fmax(0.0, degradation_alpha);
    return v * (1.0 + alpha);
}

extern "C" __device__ void __knowledge_merge_prov(
    uint256_t* dst,
    const uint256_t* a,
    const uint256_t* b
) {
    if (!dst || !a || !b) return;
    *dst = __knowledge_merge_prov_fast(*a, *b);
}
