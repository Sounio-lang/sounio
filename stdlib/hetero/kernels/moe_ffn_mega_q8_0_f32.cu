// Sounio hetero: fused MoE FFN mega-kernel Q8_0 (gate+up SiLU + down + weighted combine).
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o moe_ffn_mega_q8_0_f32.ptx moe_ffn_mega_q8_0_f32.cu
// Lab host:  nvcc -O3 -arch=sm_89 -DMOE_MEGA_LAB_HOST -o moe_mega_lab moe_ffn_mega_q8_0_f32.cu

#include <cuda_fp16.h>
#include <stdint.h>

#define QK8_0 32

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

static_assert(sizeof(block_q8_0) == 34, "wrong Q8_0 block size");

static __device__ __forceinline__ int mega_id_at(const char * ids, int token, int slot, size_t ids_nb1) {
    return *reinterpret_cast<const int32_t *>(ids + size_t(token) * ids_nb1 + size_t(slot) * sizeof(int32_t));
}

static __device__ __forceinline__ float mega_weight_at(const float * weights, int token, int slot, size_t w_nb1) {
    return *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(weights) + size_t(token) * w_nb1 + size_t(slot) * sizeof(float));
}

static __device__ __forceinline__ uint32_t mega_load_i8x4(const int8_t * p) {
    const uint32_t lo = __ldg(reinterpret_cast<const uint16_t *>(p));
    const uint32_t hi = __ldg(reinterpret_cast<const uint16_t *>(p + 2));
    return lo | (hi << 16);
}

static __device__ float mega_dot_q8_0(const block_q8_0 * row, int nqb, const float * y) {
    float acc = 0.0f;
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q8_0 & b = row[ib];
        const float d = float(b.d);
        const float * yb = y + ib * QK8_0;
#pragma unroll
        for (int j = 0; j < QK8_0; j += 4) {
            const uint32_t raw = mega_load_i8x4(b.qs + j);
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int q = int(int8_t((raw >> (8 * lane)) & 0xFFu));
                acc = fmaf(d * float(q), yb[j + lane], acc);
            }
        }
    }
    return acc;
}

static __device__ __forceinline__ float mega_silu(float g) {
    return g / (1.0f + expf(0.0f - g));
}

extern "C" __global__ void moe_ffn_mega_q8_0_f32(
        const char * __restrict__ gate_w, const char * __restrict__ up_w, const char * __restrict__ down_w,
        const float * __restrict__ x, const int * __restrict__ ids, const float * __restrict__ weights,
        float * __restrict__ dst,
        int hidden, int inter, int experts, int slots, int ntokens,
        size_t expert_stride_g, size_t row_stride_g,
        size_t expert_stride_u, size_t row_stride_u,
        size_t expert_stride_d, size_t row_stride_d,
        size_t x_nb1, size_t ids_nb1, size_t w_nb1, size_t dst_nb1) {
    const int token = int(blockIdx.x);
    if (token >= ntokens) {
        return;
    }

    extern __shared__ float shmem[];
    float * xsh = shmem;
    float * actsh = shmem + hidden;

    const float * xin = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(x) + size_t(token) * x_nb1);
    for (int i = threadIdx.x; i < hidden; i += int(blockDim.x)) {
        xsh[i] = xin[i];
    }
    __syncthreads();

    float * dout = reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + size_t(token) * dst_nb1);
    for (int o = threadIdx.x; o < hidden; o += int(blockDim.x)) {
        dout[o] = 0.0f;
    }
    __syncthreads();

    const int nqb_h = hidden / QK8_0;
    const int nqb_i = inter / QK8_0;
    const char * ids_c = reinterpret_cast<const char *>(ids);

    for (int slot = 0; slot < slots; ++slot) {
        const int id = mega_id_at(ids_c, token, slot, ids_nb1);
        const float wt = mega_weight_at(weights, token, slot, w_nb1);
        if (id < 0 || id >= experts) {
            continue;
        }

        for (int r = threadIdx.x; r < inter; r += int(blockDim.x)) {
            const block_q8_0 * gate_row = reinterpret_cast<const block_q8_0 *>(
                    gate_w + size_t(id) * expert_stride_g + size_t(r) * row_stride_g);
            const block_q8_0 * up_row = reinterpret_cast<const block_q8_0 *>(
                    up_w + size_t(id) * expert_stride_u + size_t(r) * row_stride_u);
            const float g = mega_dot_q8_0(gate_row, nqb_h, xsh);
            const float u = mega_dot_q8_0(up_row, nqb_h, xsh);
            actsh[r] = mega_silu(g) * u;
        }
        __syncthreads();

        for (int o = threadIdx.x; o < hidden; o += int(blockDim.x)) {
            const block_q8_0 * down_row = reinterpret_cast<const block_q8_0 *>(
                    down_w + size_t(id) * expert_stride_d + size_t(o) * row_stride_d);
            const float partial = mega_dot_q8_0(down_row, nqb_i, actsh);
            atomicAdd(dout + o, wt * partial);
        }
        __syncthreads();
    }
}

#ifdef MOE_MEGA_LAB_HOST
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static float half_bits_to_f32(uint16_t h) {
    half hv;
    memcpy(&hv, &h, 2);
    return float(hv);
}

static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    const uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00);
    }
    return (uint16_t)(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

static void dequant_row_q8_0(const block_q8_0 * row, int nqb, float * out) {
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q8_0 & b = row[ib];
        uint16_t dh = 0;
        memcpy(&dh, &b.d, 2);
        const float d = half_bits_to_f32(dh);
        float * y = out + ib * QK8_0;
        for (int i = 0; i < QK8_0; ++i) {
            y[i] = d * float(b.qs[i]);
        }
    }
}

static void pack_q8_0_row(const float * w, int k, block_q8_0 * row) {
    const int nqb = k / QK8_0;
    for (int ib = 0; ib < nqb; ++ib) {
        block_q8_0 & b = row[ib];
        const float * src = w + ib * QK8_0;
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; ++i) {
            amax = fmaxf(amax, fabsf(src[i]));
        }
        const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
        const uint16_t dh = f32_to_f16_bits(d);
        memcpy(&b.d, &dh, 2);
        const float inv = 1.0f / d;
        for (int i = 0; i < QK8_0; ++i) {
            const float q = fminf(127.0f, fmaxf(-127.0f, rintf(src[i] * inv)));
            b.qs[i] = (int8_t)q;
        }
    }
}

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
    }
    return den > 0.0 ? num / den : num;
}

static bool has_nan(const float * a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != a[i]) {
            return true;
        }
    }
    return false;
}

static float host_silu(float g) {
    return g / (1.0f + expf(0.0f - g));
}

static float dot_dequant_row(const block_q8_0 * row, int nqb, const float * x, float * wdeq) {
    dequant_row_q8_0(row, nqb, wdeq);
    float acc = 0.0f;
    const int k = nqb * QK8_0;
    for (int i = 0; i < k; ++i) {
        acc += wdeq[i] * x[i];
    }
    return acc;
}

static void host_oracle_ffn(
        const char * gate_w, const char * up_w, const char * down_w,
        const float * x, const int32_t * ids, const float * weights,
        float * dst, int hidden, int inter, int experts, int slots, int ntokens,
        size_t expert_stride_g, size_t row_stride_g,
        size_t expert_stride_u, size_t row_stride_u,
        size_t expert_stride_d, size_t row_stride_d,
        size_t x_nb1, size_t ids_nb1, size_t w_nb1, size_t dst_nb1) {
    const int nqb_h = hidden / QK8_0;
    const int nqb_i = inter / QK8_0;
    std::vector<float> wdeq_h(hidden);
    std::vector<float> wdeq_i(inter);
    std::vector<float> act(inter);

    for (int t = 0; t < ntokens; ++t) {
        float * dt = reinterpret_cast<float *>(
                reinterpret_cast<char *>(dst) + size_t(t) * dst_nb1);
        for (int o = 0; o < hidden; ++o) {
            dt[o] = 0.0f;
        }
        const float * xt = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(x) + size_t(t) * x_nb1);
        for (int s = 0; s < slots; ++s) {
            const int e = ids[size_t(t) * slots + s];
            const float wt = *reinterpret_cast<const float *>(
                    reinterpret_cast<const char *>(weights) + size_t(t) * w_nb1 + size_t(s) * sizeof(float));
            if (e < 0 || e >= experts) {
                continue;
            }
            for (int r = 0; r < inter; ++r) {
                const block_q8_0 * grow = reinterpret_cast<const block_q8_0 *>(
                        gate_w + size_t(e) * expert_stride_g + size_t(r) * row_stride_g);
                const block_q8_0 * urow = reinterpret_cast<const block_q8_0 *>(
                        up_w + size_t(e) * expert_stride_u + size_t(r) * row_stride_u);
                const float g = dot_dequant_row(grow, nqb_h, xt, wdeq_h.data());
                const float u = dot_dequant_row(urow, nqb_h, xt, wdeq_h.data());
                act[size_t(r)] = host_silu(g) * u;
            }
            for (int o = 0; o < hidden; ++o) {
                const block_q8_0 * drow = reinterpret_cast<const block_q8_0 *>(
                        down_w + size_t(e) * expert_stride_d + size_t(o) * row_stride_d);
                const float partial = dot_dequant_row(drow, nqb_i, act.data(), wdeq_i.data());
                dt[o] += wt * partial;
            }
        }
    }
}

static int run_case(int hidden, int inter, int experts, int slots, int ntokens, float amax) {
    const int nqb_h = hidden / QK8_0;
    const int nqb_i = inter / QK8_0;
    const size_t row_bytes_g = size_t(nqb_h) * sizeof(block_q8_0);
    const size_t row_bytes_d = size_t(nqb_i) * sizeof(block_q8_0);
    const size_t expert_stride_g = size_t(inter) * row_bytes_g;
    const size_t expert_stride_u = expert_stride_g;
    const size_t expert_stride_d = size_t(hidden) * row_bytes_d;

    std::vector<char> gate_w(size_t(experts) * expert_stride_g);
    std::vector<char> up_w(size_t(experts) * expert_stride_u);
    std::vector<char> down_w(size_t(experts) * expert_stride_d);
    std::vector<float> x(size_t(ntokens) * size_t(hidden));
    std::vector<int32_t> ids(size_t(ntokens) * size_t(slots));
    std::vector<float> weights(size_t(ntokens) * size_t(slots));
    std::vector<float> dst_gpu(size_t(ntokens) * size_t(hidden));
    std::vector<float> dst_cpu(size_t(ntokens) * size_t(hidden));

    srand(1);
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < inter; ++r) {
            std::vector<float> wr(hidden);
            for (int i = 0; i < hidden; ++i) {
                wr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            pack_q8_0_row(wr.data(), hidden,
                reinterpret_cast<block_q8_0 *>(gate_w.data() + size_t(e) * expert_stride_g + size_t(r) * row_bytes_g));
            for (int i = 0; i < hidden; ++i) {
                wr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            pack_q8_0_row(wr.data(), hidden,
                reinterpret_cast<block_q8_0 *>(up_w.data() + size_t(e) * expert_stride_u + size_t(r) * row_bytes_g));
        }
        for (int o = 0; o < hidden; ++o) {
            std::vector<float> wr(inter);
            for (int i = 0; i < inter; ++i) {
                wr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            pack_q8_0_row(wr.data(), inter,
                reinterpret_cast<block_q8_0 *>(down_w.data() + size_t(e) * expert_stride_d + size_t(o) * row_bytes_d));
        }
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * amax;
    }
    for (int t = 0; t < ntokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            ids[size_t(t) * slots + s] = (t * slots + s) % experts;
            weights[size_t(t) * slots + s] = 0.25f + 0.5f * ((float)rand() / (float)RAND_MAX);
        }
    }

    const size_t x_nb1 = size_t(hidden) * sizeof(float);
    const size_t ids_nb1 = size_t(slots) * sizeof(int32_t);
    const size_t w_nb1 = size_t(slots) * sizeof(float);
    const size_t dst_nb1 = size_t(hidden) * sizeof(float);

    host_oracle_ffn(gate_w.data(), up_w.data(), down_w.data(), x.data(), ids.data(), weights.data(),
            dst_cpu.data(), hidden, inter, experts, slots, ntokens,
            expert_stride_g, row_bytes_g, expert_stride_u, row_bytes_g,
            expert_stride_d, row_bytes_d, x_nb1, ids_nb1, w_nb1, dst_nb1);

    char * d_g = nullptr;
    char * d_u = nullptr;
    char * d_d = nullptr;
    float * d_x = nullptr;
    int * d_ids = nullptr;
    float * d_w = nullptr;
    float * d_dst = nullptr;
    cudaMalloc(&d_g, gate_w.size());
    cudaMalloc(&d_u, up_w.size());
    cudaMalloc(&d_d, down_w.size());
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMalloc(&d_w, weights.size() * sizeof(float));
    cudaMalloc(&d_dst, dst_gpu.size() * sizeof(float));
    cudaMemcpy(d_g, gate_w.data(), gate_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, up_w.data(), up_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, down_w.data(), down_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);

    const int threads = 128;
    const size_t shmem_mega = size_t(hidden + inter) * sizeof(float);

    auto launch_mega = [&]() {
        moe_ffn_mega_q8_0_f32<<<ntokens, threads, shmem_mega>>>(
            d_g, d_u, d_d, d_x, d_ids, d_w, d_dst,
            hidden, inter, experts, slots, ntokens,
            expert_stride_g, row_bytes_g, expert_stride_u, row_bytes_g,
            expert_stride_d, row_bytes_d, x_nb1, ids_nb1, w_nb1, dst_nb1);
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i) {
        launch_mega();
    }
    cudaDeviceSynchronize();
    const int runs = ntokens >= 64 ? 50 : (ntokens == 1 ? 200 : 100);
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        launch_mega();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    const double mega_us = (double(ms) * 1000.0) / double(runs);

    launch_mega();
    cudaDeviceSynchronize();
    cudaMemcpy(dst_gpu.data(), d_dst, dst_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    int rc = 0;
    if (has_nan(dst_gpu.data(), dst_gpu.size())) {
        rc = 1;
    }
    const double err = nmse(dst_gpu.data(), dst_cpu.data(), dst_gpu.size());
    printf("n=%d NMSE=%.9g mega_us=%.2f\n", ntokens, err, mega_us);
    if (err > 5e-4) {
        rc = 1;
    }

    cudaFree(d_g);
    cudaFree(d_u);
    cudaFree(d_d);
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_w);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return rc;
}

int main() {
    printf("PATH=mega\n");
    int rc = 0;
    rc |= run_case(256, 256, 4, 2, 1, 1e5f);
    rc |= run_case(256, 256, 4, 2, 4, 1e5f);
    printf("LAB_EXIT:%d\n", rc);
    return rc;
}
#endif
