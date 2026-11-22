// Sounio hetero: RoPE (rotate-half, HF convention) for attention Q/K.
// Entry: attn_rope_f32 (float* qk, int rows, int heads, int dh, int pos0, float theta).
// Each thread handles one (row, head, pair). Matches ll_rope in stdlib/llm/llama.sio.
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o attn_rope_f32.ptx attn_rope_f32.cu
#include <cuda_fp16.h>
#include <stdint.h>
extern "C" __global__ void attn_rope_f32(float* qk, int rows, int heads, int dh, int pos0, float theta) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int hh = blockIdx.y;
    int rr = blockIdx.z;
    int half = dh / 2;
    if (rr >= rows || hh >= heads || pair >= half) return;
    int pos = pos0 + rr;
    float freq = powf(theta, (0.0f - 2.0f * (float)pair) / (float)dh);
    float ang = (float)pos * freq;
    float c = cosf(ang);
    float s = sinf(ang);
    size_t base = (size_t)rr * (size_t)(heads * dh) + (size_t)hh * (size_t)dh;
    float a = qk[base + pair];
    float b = qk[base + pair + half];
    qk[base + pair] = a * c - b * s;
    qk[base + pair + half] = b * c + a * s;
}
