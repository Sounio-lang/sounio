// Sounio hetero: attention x V gather for one KV head group.
// Entry: attn_av_f32 (float* probs, float* v_cache, float* out, int rows, int upto,
//                     int kvdim, int dh, int kv_head, int ld_probs).
// out[r*dh+d] = sum_t probs[r*ld+t] * V[t*kvdim + kv_head*dh + d].
// One block per (row); threads split dh. GQA fan-out is host-launched per head.
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o attn_av_f32.ptx attn_av_f32.cu
#include <cuda_fp16.h>
#include <stdint.h>
extern "C" __global__ void attn_av_f32(float* probs, float* v_cache, float* out,
                                       int rows, int upto, int kvdim, int dh,
                                       int kv_head, int ld_probs) {
    int r = (int)blockIdx.x;
    int d = (int)threadIdx.x;
    if (r >= rows || d >= dh) return;
    float acc = 0.0f;
    size_t voff = (size_t)kv_head * (size_t)dh + (size_t)d;
    for (int t = 0; t < upto; ++t) {
        float p = probs[(size_t)r * (size_t)ld_probs + t];
        acc += p * v_cache[(size_t)t * (size_t)kvdim + voff];
    }
    out[(size_t)r * (size_t)dh + d] = acc;
}
