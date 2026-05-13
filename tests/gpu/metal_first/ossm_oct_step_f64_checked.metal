#include <metal_stdlib>
using namespace metal;

// sounio.source_semantics=f64
// sounio.metal_f64_policy=f32_checked
// sounio.algebra_law_profile=octonion:alternative,non_associative,epistemic_shadow
//
// This fixture is intentionally not a native f64 claim. It is the first Metal
// witness for O-SSM source f64 semantics lowered through checked f32.
kernel void ossm_oct_step_f64_checked(
    device int* n [[buffer(0)]],
    device float* state [[buffer(1)]],
    device float* out_buf [[buffer(2)]],
    device float* shadow [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n[0]) return;

    int upper = (int)tid + 64;
    float h0 = state[(int)tid];
    float h1 = state[upper];
    float x0 = h0 + 1.0f;
    float x1 = h1 + 0.25f;

    float ah0 = (0.5f * h0) - (0.125f * h1);
    float ah1 = (0.5f * h1) + (0.125f * h0);
    float bx0 = (0.25f * x0) - (0.5f * x1);
    float bx1 = (0.25f * x1) + (0.5f * x0);
    float next0 = ah0 + bx0;
    float next1 = ah1 + bx1;
    float energy = (next0 * next0) + (next1 * next1);

    out_buf[(int)tid] = next0;
    out_buf[upper] = next1;
    shadow[(int)tid] = energy + 0.015625f;
}
