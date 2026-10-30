// Sounio hetero: causal row softmax for attention scores.
// Entry: attn_softmax_causal_f32 (float* scores, float* probs, int rows, int upto, int ld).
// Row r holds valid scores in [0, upto); output probs sum to 1 over that range.
// Host fills scores; this kernel only normalizes (max-subtracted exp).
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o attn_softmax_causal_f32.ptx attn_softmax_causal_f32.cu
#include <cuda_fp16.h>
#include <stdint.h>
extern "C" __global__ void attn_softmax_causal_f32(float* scores, float* probs, int rows, int upto, int ld) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float mx = -1.0e30f;
    for (int t = 0; t < upto; ++t) {
        float v = scores[(size_t)r * (size_t)ld + t];
        if (v > mx) mx = v;
    }
    float sum = 0.0f;
    for (int t = 0; t < upto; ++t) {
        float e = expf(scores[(size_t)r * (size_t)ld + t] - mx);
        probs[(size_t)r * (size_t)ld + t] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int t = 0; t < upto; ++t) probs[(size_t)r * (size_t)ld + t] *= inv;
}
