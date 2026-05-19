#include <metal_stdlib>
using namespace metal;

kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b_in [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
)
{
    c[tid] = a[tid] + b_in[tid];
}
