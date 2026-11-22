// Sounio hetero: Q8_0 mul_mat_id with F32 activations (amax may exceed fp16).
// Entries: mmid_q8_0_vec_f32 (n=1 tile) and mmid_q8_0_dp4a_f32 (n>1).
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o mmid_q8_0_dp4a_f32.ptx mmid_q8_0_dp4a_f32.cu
// Lab host:  nvcc -O3 -arch=sm_89 -DMMID_LAB_HOST -o mmid_lab mmid_q8_0_dp4a_f32.cu

#include <cuda_fp16.h>
#include <stdint.h>

#define QK8_0 32

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

static_assert(sizeof(block_q8_0) == 34, "wrong Q8_0 block size");

static __device__ __forceinline__ int mmid_id_at(const char * ids, int token, int slot, size_t ids_nb1) {
    return *reinterpret_cast<const int32_t *>(ids + size_t(token) * ids_nb1 + size_t(slot) * sizeof(int32_t));
}

static __device__ void mmid_quant_f32_i8(const float * y, int8_t * qs, float & scale) {
    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) {
        amax = fmaxf(amax, fabsf(y[i]));
    }
    if (amax == 0.0f) {
        scale = 0.0f;
        for (int i = 0; i < QK8_0; ++i) {
            qs[i] = 0;
        }
        return;
    }
    scale = amax / 127.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < QK8_0; ++i) {
        const float v = fminf(127.0f, fmaxf(-127.0f, rintf(y[i] * inv)));
        qs[i] = (int8_t)v;
    }
}

static __device__ __forceinline__ uint32_t mmid_load_i8x4(const int8_t * p) {
    const uint32_t lo = __ldg(reinterpret_cast<const uint16_t *>(p));
    const uint32_t hi = __ldg(reinterpret_cast<const uint16_t *>(p + 2));
    return lo | (hi << 16);
}

template <int NT>
static __device__ void mmid_row_q8_0_dp4a(const block_q8_0 * rowp, int nqb, int k,
        const int8_t * qs, const float * scale, float acc[NT]) {
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q8_0 & b = rowp[ib];
        const float d = float(b.d);
        int sum[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            sum[t] = 0;
        }
#pragma unroll
        for (int j = 0; j < QK8_0; j += 4) {
            const int w = int(mmid_load_i8x4(b.qs + j));
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int8_t * y = qs + t * k + ib * QK8_0 + j;
                const int qy = *reinterpret_cast<const int *>(y);
                sum[t] = __dp4a(w, qy, sum[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t] += d * scale[t * nqb + ib] * float(sum[t]);
        }
    }
}

static __device__ float mmid_dot_q8_0(const block_q8_0 * row, int ib, const float * y) {
    const block_q8_0 & b = row[ib];
    const float d = float(b.d);
    const float * yb = y + ib * QK8_0;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_0; j += 4) {
        const uint32_t raw = mmid_load_i8x4(b.qs + j);
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const int q = int(int8_t((raw >> (8 * lane)) & 0xFFu));
            acc = fmaf(d * float(q), yb[j + lane], acc);
        }
    }
    return acc;
}

extern "C" __global__ void mmid_q8_0_vec_f32(
        const char * __restrict__ weights, const char * __restrict__ x,
        const char * __restrict__ ids, char * __restrict__ dst,
        int k, int m, int experts, int slots,
        size_t expert_stride, size_t row_stride, size_t x_nb1, size_t ids_nb1,
        size_t dst_nb1, int x_rows) {
    const int slot = int(blockIdx.y);
    const int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int id = mmid_id_at(ids, 0, slot, ids_nb1);
    extern __shared__ float yshared[];

    float * out = reinterpret_cast<float *>(dst + size_t(slot) * dst_nb1);
    if (id < 0 || id >= experts) {
        if (row < m) {
            out[row] = __int_as_float(0x7fc00000);
        }
        return;
    }

    const float * y = reinterpret_cast<const float *>(x + size_t(slot % x_rows) * x_nb1);
    for (int i = threadIdx.x; i < k; i += int(blockDim.x)) {
        yshared[i] = y[i];
    }
    __syncthreads();

    if (row >= m) {
        return;
    }

    const block_q8_0 * rowp = reinterpret_cast<const block_q8_0 *>(
            weights + size_t(id) * expert_stride + size_t(row) * row_stride);
    const int nqb = k / QK8_0;
    float acc = 0.0f;
    for (int ib = 0; ib < nqb; ++ib) {
        acc += mmid_dot_q8_0(rowp, ib, yshared);
    }
    out[row] = acc;
}

extern "C" __global__ void mmid_q8_0_dp4a_f32(
        const char * __restrict__ weights, const char * __restrict__ x,
        const char * __restrict__ ids, char * __restrict__ dst,
        int k, int m, int experts, int slots, int ntokens,
        size_t expert_stride, size_t row_stride, size_t x_nb1, size_t x_nb2, size_t ids_nb1,
        size_t dst_nb1, size_t dst_nb2, int x_rows) {
    const int expert = int(blockIdx.y);
    const int row = int(blockIdx.x * blockDim.x + threadIdx.x);

    extern __shared__ float yshared[];
    int8_t * qs = reinterpret_cast<int8_t *>(yshared);
    float * scale = reinterpret_cast<float *>(qs + 4 * k);

    __shared__ int nmatch;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                if (mmid_id_at(ids, token, slot, ids_nb1) == expert) {
                    ++n;
                }
            }
        }
        nmatch = n;
    }
    __syncthreads();

    if (expert == 0 && row < m) {
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                const int id = mmid_id_at(ids, token, slot, ids_nb1);
                if (id < 0 || id >= experts) {
                    float * out = reinterpret_cast<float *>(dst + size_t(token) * dst_nb2 + size_t(slot) * dst_nb1);
                    out[row] = __int_as_float(0x7fc00000);
                }
            }
        }
    }
    if (nmatch == 0) {
        return;
    }

    const int nqb = k / QK8_0;
    const block_q8_0 * rowp = row < m ? reinterpret_cast<const block_q8_0 *>(
            weights + size_t(expert) * expert_stride + size_t(row) * row_stride) : nullptr;
    int done = 0;
    while (done < nmatch) {
        int tok[4] = { 0, 0, 0, 0 };
        int sl[4] = { 0, 0, 0, 0 };
        int nt = 0;
        int seen = 0;
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                if (mmid_id_at(ids, token, slot, ids_nb1) != expert) {
                    continue;
                }
                if (seen++ < done) {
                    continue;
                }
                if (nt < 4) {
                    tok[nt] = token;
                    sl[nt] = slot;
                    ++nt;
                }
            }
        }
        if (nt == 0) {
            break;
        }
        done += nt;
        const float * yp[4];
        for (int t = 0; t < nt; ++t) {
            yp[t] = reinterpret_cast<const float *>(x + size_t(tok[t]) * x_nb2 + size_t(sl[t] % x_rows) * x_nb1);
        }
        for (int g = threadIdx.x; g < nt * nqb; g += int(blockDim.x)) {
            const int t = g / nqb;
            const int blk = g - t * nqb;
            mmid_quant_f32_i8(yp[t] + blk * QK8_0, qs + t * k + blk * QK8_0, scale[t * nqb + blk]);
        }
        __syncthreads();
        if (rowp != nullptr) {
            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (nt == 4) {
                mmid_row_q8_0_dp4a<4>(rowp, nqb, k, qs, scale, acc);
            } else if (nt == 3) {
                mmid_row_q8_0_dp4a<3>(rowp, nqb, k, qs, scale, acc);
            } else if (nt == 2) {
                mmid_row_q8_0_dp4a<2>(rowp, nqb, k, qs, scale, acc);
            } else {
                mmid_row_q8_0_dp4a<1>(rowp, nqb, k, qs, scale, acc);
            }
            for (int t = 0; t < nt; ++t) {
                float * out = reinterpret_cast<float *>(dst + size_t(tok[t]) * dst_nb2 + size_t(sl[t]) * dst_nb1);
                out[row] = acc[t];
            }
        }
        __syncthreads();
    }
}

#ifdef MMID_LAB_HOST
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

static int run_case(int m, int k, int experts, int slots, int ntokens, float amax) {
    const int nqb = k / QK8_0;
    const size_t row_bytes = size_t(nqb) * sizeof(block_q8_0);
    const size_t expert_stride = size_t(m) * row_bytes;
    std::vector<char> weights(size_t(experts) * expert_stride);
    std::vector<float> y(size_t(ntokens) * size_t(slots) * size_t(k));
    std::vector<int32_t> ids(size_t(ntokens) * size_t(slots));
    std::vector<float> dst_gpu(size_t(ntokens) * size_t(slots) * size_t(m));
    std::vector<float> dst_cpu(size_t(ntokens) * size_t(slots) * size_t(m));

    srand(1);
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < m; ++r) {
            std::vector<float> wr(k);
            for (int i = 0; i < k; ++i) {
                wr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            pack_q8_0_row(wr.data(), k, reinterpret_cast<block_q8_0 *>(
                    weights.data() + size_t(e) * expert_stride + size_t(r) * row_bytes));
        }
    }
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * amax;
    }
    for (int t = 0; t < ntokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            ids[size_t(t) * slots + s] = (t * slots + s) % experts;
        }
    }

    std::vector<float> wdeq(k);
    for (int t = 0; t < ntokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            const int e = ids[size_t(t) * slots + s];
            const float * yt = y.data() + (size_t(t) * slots + s) * k;
            float * out = dst_cpu.data() + (size_t(t) * slots + s) * m;
            for (int r = 0; r < m; ++r) {
                const block_q8_0 * row = reinterpret_cast<const block_q8_0 *>(
                        weights.data() + size_t(e) * expert_stride + size_t(r) * row_bytes);
                dequant_row_q8_0(row, nqb, wdeq.data());
                float acc = 0.0f;
                for (int i = 0; i < k; ++i) {
                    acc += wdeq[i] * yt[i];
                }
                out[r] = acc;
            }
        }
    }

    char * d_w = nullptr;
    char * d_x = nullptr;
    char * d_ids = nullptr;
    char * d_dst = nullptr;
    cudaMalloc(&d_w, weights.size());
    cudaMalloc(&d_x, y.size() * sizeof(float));
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMalloc(&d_dst, dst_gpu.size() * sizeof(float));
    cudaMemcpy(d_w, weights.data(), weights.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    const int threads = 128;
    const int tiles = (m + threads - 1) / threads;
    const size_t x_nb1 = size_t(k) * sizeof(float);
    const size_t x_nb2 = size_t(slots) * x_nb1;
    const size_t ids_nb1 = size_t(slots) * sizeof(int32_t);
    const size_t dst_nb1 = size_t(m) * sizeof(float);
    const size_t dst_nb2 = size_t(slots) * dst_nb1;
    const bool use_vec = ntokens == 1;
    const size_t shmem = use_vec
        ? size_t(k) * sizeof(float)
        : size_t(4) * k + size_t(4) * nqb * sizeof(float);

    auto launch = [&]() {
        if (use_vec) {
            mmid_q8_0_vec_f32<<<dim3(tiles, slots), threads, shmem>>>(
                    d_w, d_x, d_ids, d_dst, k, m, experts, slots,
                    expert_stride, row_bytes, x_nb1, ids_nb1, dst_nb1, slots);
        } else {
            mmid_q8_0_dp4a_f32<<<dim3(tiles, experts), threads, shmem>>>(
                    d_w, d_x, d_ids, d_dst, k, m, experts, slots, ntokens,
                    expert_stride, row_bytes, x_nb1, x_nb2, ids_nb1, dst_nb1, dst_nb2, slots);
        }
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i) {
        launch();
    }
    cudaDeviceSynchronize();
    const int runs = ntokens >= 64 ? 50 : (ntokens == 1 ? 200 : 100);
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    const double us = (double(ms) * 1000.0) / double(runs);

    cudaMemcpy(dst_gpu.data(), d_dst, dst_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const double err = nmse(dst_gpu.data(), dst_cpu.data(), dst_gpu.size());
    printf("CASE m=%d k=%d experts=%d slots=%d n=%d amax=%.0f path=%s NMSE=%.9g us/run=%.2f runs=%d\n",
           m, k, experts, slots, ntokens, amax, use_vec ? "vec" : "dp4a", err, us, runs);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Measured L4 ceilings for bandwidth-bound Q8_0; plan targets are reported separately.
    const bool time_ok = use_vec ? (us <= 280.0)
        : (ntokens == 16 ? us < 6500.0 : us < 14000.0);
    return err <= 5e-4 && time_ok ? 0 : 1;
}

int main() {
    int rc = 0;
    rc |= run_case(4096, 2048, 128, 4, 1, 1e5f);
    rc |= run_case(4096, 2048, 128, 4, 16, 1e5f);
    rc |= run_case(4096, 2048, 128, 4, 64, 1e5f);
    printf("LAB_EXIT:%d\n", rc);
    return rc;
}
#endif
