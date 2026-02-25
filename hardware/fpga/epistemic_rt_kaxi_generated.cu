// GENERATED_FROM_STDLIB: stdlib/hardware/kaxi.sio
// GENERATED_BY: self-hosted/compiler/codegen/hardware/kaxi_emitter.sio
// CUDA sidecar for K-AXI publish path.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

struct alignas(32) uint256_t {
    uint64_t lane0;
    uint64_t lane1;
    uint64_t lane2;
    uint64_t lane3;
};

struct KaxiPayload {
    double value;
    double variance;
    uint256_t provenance;
    uint64_t confidence_bits;
    uint32_t flags;
    uint32_t reserved;
};

static constexpr unsigned KAXI_RING_SIZE = 64;
static constexpr uint8_t OP_PASS = 0x0;
static constexpr uint8_t OP_ADD = 0x1;
static constexpr uint8_t OP_MUL = 0x2;
static constexpr uint8_t OP_DIV = 0x3;
static constexpr uint8_t OP_FMA = 0x4;
static constexpr uint8_t OP_PROP_VAR = 0x5;
__device__ KaxiPayload g_kaxi_ring[KAXI_RING_SIZE];
__device__ unsigned int g_kaxi_write_idx = 0;
__device__ unsigned long long g_epistemic_power_log_hw_q32_32 = 0ull;

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

__device__ __forceinline__ void __knowledge_kaxi_unpack_flags(
    uint32_t flags,
    uint8_t* op_kind,
    uint8_t* alpha_q8,
    uint8_t* success_inc,
    uint8_t* failure_inc
) {
    if (op_kind != nullptr) {
        *op_kind = static_cast<uint8_t>(flags & 0x0Fu);
    }
    if (alpha_q8 != nullptr) {
        *alpha_q8 = static_cast<uint8_t>((flags >> 24) & 0xFFu);
    }
    if (success_inc != nullptr) {
        *success_inc = static_cast<uint8_t>((flags >> 16) & 0xFFu);
    }
    if (failure_inc != nullptr) {
        *failure_inc = static_cast<uint8_t>((flags >> 8) & 0xFFu);
    }
}

__device__ __forceinline__ int __knowledge_kaxi_is_epistemic(uint32_t flags) {
    const uint8_t op_kind = static_cast<uint8_t>(flags & 0x0Fu);
    return (op_kind >= OP_ADD && op_kind <= OP_PROP_VAR) ? 1 : 0;
}

__device__ __forceinline__ void __knowledge_kaxi_publish(
    double value,
    double variance,
    const uint256_t& prov,
    double confidence,
    uint32_t flags
) {
    const unsigned idx = atomicAdd(&g_kaxi_write_idx, 1u) & (KAXI_RING_SIZE - 1);
    g_kaxi_ring[idx].value = value;
    g_kaxi_ring[idx].variance = fmax(0.0, variance);
    g_kaxi_ring[idx].provenance = prov;
    g_kaxi_ring[idx].confidence_bits = static_cast<uint64_t>(__double_as_longlong(confidence));
    g_kaxi_ring[idx].flags = flags;
    g_kaxi_ring[idx].reserved = 0u;
    atomicAdd(&g_epistemic_power_log_hw_q32_32, 1ull << 16);
}

extern "C" __device__ const KaxiPayload* __knowledge_kaxi_ring_base() {
    return g_kaxi_ring;
}

extern "C" __device__ const KaxiPayload* __knowledge_kaxi_ring_at(uint32_t idx) {
    return &g_kaxi_ring[idx & (KAXI_RING_SIZE - 1)];
}

extern "C" __device__ uint32_t __knowledge_kaxi_ring_size() {
    return static_cast<uint32_t>(KAXI_RING_SIZE);
}

extern "C" __device__ unsigned int __knowledge_kaxi_ring_head() {
    return g_kaxi_write_idx;
}

extern "C" __device__ uint64_t __knowledge_get_epistemic_power_log_q32_32() {
    return static_cast<uint64_t>(g_epistemic_power_log_hw_q32_32);
}
