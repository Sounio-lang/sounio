#include <metal_stdlib>
using namespace metal;

kernel void epistemic_dual_output_f32(
    device const float* a_value [[buffer(0)]],
    device const float* b_value [[buffer(1)]],
    device const float* a_eps [[buffer(2)]],
    device const float* b_eps [[buffer(3)]],
    device const uint* a_valid [[buffer(4)]],
    device const uint* b_valid [[buffer(5)]],
    device float* out_value [[buffer(6)]],
    device float* out_eps [[buffer(7)]],
    device uint* out_valid [[buffer(8)]],
    device uint* out_prov [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
)
{
    out_value[tid] = a_value[tid] + b_value[tid];
    out_eps[tid] = (a_eps[tid] * a_eps[tid]) + (b_eps[tid] * b_eps[tid]);
    out_valid[tid] = a_valid[tid] & b_valid[tid];
    out_prov[tid] = (tid ^ 0x12345678u) ^ out_valid[tid];
}
