#include <metal_stdlib>
using namespace metal;

// sounio.source_semantics=f64
// sounio.metal_f64_policy=f32_checked
// sounio.algebra_law_profile=sedenion:cayley_dickson,non_associative,zero_divisor_sensitive,epistemic_shadow
//
// This fixture keeps the S-SSM boundary honest: source f64 intent is present,
// but Metal receives a checked f32 Cayley-Dickson witness.
kernel void sedenion_cd_step_f64_checked(
    device int* n [[buffer(0)]],
    device float* state [[buffer(1)]],
    device float* out_buf [[buffer(2)]],
    device float* shadow [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n[0]) return;

    int o1 = (int)tid + 64;
    int o2 = (int)tid + 128;
    int o3 = (int)tid + 192;

    float lo0 = state[(int)tid];
    float lo1 = state[o1];
    float hi0 = state[o2];
    float hi1 = state[o3];
    float x0 = lo0 + 1.0f;
    float x1 = 0.5f;

    float lo_mul0 = (lo0 * x0) - (lo1 * x1);
    float lo_mul1 = (lo0 * x1) + (lo1 * x0);
    float hi_mul0 = (0.25f * hi0) - (0.125f * hi1);
    float hi_mul1 = (0.25f * hi1) + (0.125f * hi0);
    float cd_lo0 = lo_mul0 - hi_mul0;
    float cd_hi0 = hi_mul1 + lo_mul1;
    float norm_shadow = (cd_lo0 * cd_lo0) + (cd_hi0 * cd_hi0);

    out_buf[(int)tid] = cd_lo0;
    out_buf[o1] = cd_hi0;
    shadow[(int)tid] = norm_shadow + 0.015625f;
}
