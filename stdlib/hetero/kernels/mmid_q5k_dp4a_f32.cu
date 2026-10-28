// Sounio hetero: Q5_K mul_mat_id with F32 activations (amax may exceed fp16).
// Entries: mmid_q5k_vec_f32 (n=1 tile) and mmid_q5k_dp4a_f32 (n>1).
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o mmid_q5k_dp4a_f32.ptx mmid_q5k_dp4a_f32.cu
// Lab host:  nvcc -O3 -arch=sm_89 -DMMID_LAB_HOST -o mmid_lab mmid_q5k_dp4a_f32.cu

#include <cuda_fp16.h>
#include <stdint.h>

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q5_K {
    half2   dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};

static_assert(sizeof(block_q5_K) == 176, "wrong Q5_K block size");

static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ int mmid_id_at(const char * ids, int token, int slot, size_t ids_nb1) {
    return *reinterpret_cast<const int32_t *>(ids + size_t(token) * ids_nb1 + size_t(slot) * sizeof(int32_t));
}

static __device__ void mmid_quant_f32_i8(const float * y, int8_t * qs, float & scale) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(y[i]));
    }
    if (amax == 0.0f) {
        scale = 0.0f;
        for (int i = 0; i < 32; ++i) {
            qs[i] = 0;
        }
        return;
    }
    scale = amax / 127.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < 32; ++i) {
        const float v = fminf(127.0f, fmaxf(-127.0f, rintf(y[i] * inv)));
        qs[i] = (int8_t)v;
    }
}

static __device__ __forceinline__ int mmid_q5_pack4(uint32_t qraw, uint32_t hraw, uint8_t hmask, bool high_nibble) {
    uint32_t packed = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const uint32_t qb = (qraw >> (8 * lane)) & 0xFFu;
        const uint32_t hb = (hraw >> (8 * lane)) & 0xFFu;
        uint32_t q = high_nibble ? (qb >> 4) : (qb & 0xFu);
        if (hb & hmask) {
            q |= 16u;
        }
        packed |= q << (8 * lane);
    }
    return int(packed);
}

template <int NT>
static __device__ void mmid_row_q5k_dp4a(const block_q5_K * rowp, int nqb, int k,
        const int8_t * qs, const float * scale, float acc[NT]) {
    const int nblk = k / 32;
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q5_K & b = rowp[ib];
        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            uint8_t sc, mn;
            get_scale_min_k4(2 * il, b.scales, sc, mn);
            const float d1 = dall * sc;
            const float m1 = dmin * mn;
            get_scale_min_k4(2 * il + 1, b.scales, sc, mn);
            const float d2 = dall * sc;
            const float m2 = dmin * mn;
            const uint8_t * q = b.qs + 32 * il;
            const uint8_t hmask_lo = uint8_t(1u << (2 * il));
            const uint8_t hmask_hi = uint8_t(2u << (2 * il));
            const int blk_lo = ib * 8 + 2 * il;
            int sum_lo[NT], sum_hi[NT], ysum_lo[NT], ysum_hi[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                sum_lo[t] = 0;
                sum_hi[t] = 0;
                ysum_lo[t] = 0;
                ysum_hi[t] = 0;
            }
            for (int j = 0; j < 32; j += 4) {
                const uint32_t qraw = __ldg(reinterpret_cast<const uint32_t *>(q + j));
                const uint32_t hraw = __ldg(reinterpret_cast<const uint32_t *>(b.qh + j));
                const int qlo = mmid_q5_pack4(qraw, hraw, hmask_lo, false);
                const int qhi = mmid_q5_pack4(qraw, hraw, hmask_hi, true);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int8_t * ylo = qs + t * k + blk_lo * 32 + j;
                    const int8_t * yhi = ylo + 32;
                    const int ulo = *reinterpret_cast<const int *>(ylo);
                    const int uhi = *reinterpret_cast<const int *>(yhi);
                    sum_lo[t] = __dp4a(qlo, ulo, sum_lo[t]);
                    sum_hi[t] = __dp4a(qhi, uhi, sum_hi[t]);
                    ysum_lo[t] = __dp4a(0x01010101, ulo, ysum_lo[t]);
                    ysum_hi[t] = __dp4a(0x01010101, uhi, ysum_hi[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float slo = scale[t * nblk + blk_lo];
                const float shi = scale[t * nblk + blk_lo + 1];
                acc[t] += slo * (d1 * float(sum_lo[t]) - m1 * float(ysum_lo[t]))
                        + shi * (d2 * float(sum_hi[t]) - m2 * float(ysum_hi[t]));
            }
        }
    }
}

static __device__ float mmid_dot_q5_K(const block_q5_K * row, int ib, const float * y) {
    const block_q5_K & b = row[ib];
    const float dall = __low2half(b.dm);
    const float dmin = __high2half(b.dm);
    const float * yb = y + ib * QK_K;
    float acc = 0.0f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint8_t sc, mn;
        get_scale_min_k4(2 * il, b.scales, sc, mn);
        const float d1 = dall * sc;
        const float m1 = dmin * mn;
        get_scale_min_k4(2 * il + 1, b.scales, sc, mn);
        const float d2 = dall * sc;
        const float m2 = dmin * mn;
        const uint8_t * q = b.qs + 32 * il;
        const uint8_t hmask_lo = uint8_t(1u << (2 * il));
        const uint8_t hmask_hi = uint8_t(2u << (2 * il));
        const float * y_lo = yb + 64 * il;
        const float * y_hi = y_lo + 32;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            const uint32_t qraw = __ldg(reinterpret_cast<const uint32_t *>(q + j));
            const uint32_t hraw = __ldg(reinterpret_cast<const uint32_t *>(b.qh + j));
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int qb = int((qraw >> (8 * lane)) & 0xFFu);
                const int hb = int((hraw >> (8 * lane)) & 0xFFu);
                const int qlo = (qb & 0xF) + ((hb & hmask_lo) ? 16 : 0);
                const int qhi = (qb >> 4) + ((hb & hmask_hi) ? 16 : 0);
                acc = fmaf(d1 * float(qlo) - m1, y_lo[j + lane], acc);
                acc = fmaf(d2 * float(qhi) - m2, y_hi[j + lane], acc);
            }
        }
    }
    return acc;
}

extern "C" __global__ void mmid_q5k_vec_f32(
        const char * __restrict__ weights, const char * __restrict__ x,
        const char * __restrict__ ids, char * __restrict__ dst,
        int k, int m, int experts, int slots,
        size_t expert_stride, size_t row_stride, size_t x_nb1, size_t ids_nb1,
        size_t dst_nb1, int x_rows) {
    const int slot = int(blockIdx.y);
    const int row  = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int id   = mmid_id_at(ids, 0, slot, ids_nb1);
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

    const block_q5_K * rowp = reinterpret_cast<const block_q5_K *>(
            weights + size_t(id) * expert_stride + size_t(row) * row_stride);
    const int nqb = k / QK_K;
    float acc = 0.0f;
    for (int ib = 0; ib < nqb; ++ib) {
        acc += mmid_dot_q5_K(rowp, ib, yshared);
    }
    out[row] = acc;
}

extern "C" __global__ void mmid_q5k_dp4a_f32(
        const char * __restrict__ weights, const char * __restrict__ x,
        const char * __restrict__ ids, char * __restrict__ dst,
        int k, int m, int experts, int slots, int ntokens,
        size_t expert_stride, size_t row_stride, size_t x_nb1, size_t x_nb2, size_t ids_nb1,
        size_t dst_nb1, size_t dst_nb2, int x_rows) {
    const int expert = int(blockIdx.y);
    const int row    = int(blockIdx.x * blockDim.x + threadIdx.x);

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

    const int nqb = k / QK_K;
    const int nblk = k / 32;
    const block_q5_K * rowp = row < m ? reinterpret_cast<const block_q5_K *>(
            weights + size_t(expert) * expert_stride + size_t(row) * row_stride) : nullptr;
    int done = 0;
    while (done < nmatch) {
        int tok[4] = { 0, 0, 0, 0 };
        int sl[4]  = { 0, 0, 0, 0 };
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
                    sl[nt]  = slot;
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
        for (int g = threadIdx.x; g < nt * nblk; g += int(blockDim.x)) {
            const int t = g / nblk;
            const int blk = g - t * nblk;
            mmid_quant_f32_i8(yp[t] + blk * 32, qs + t * k + blk * 32, scale[t * nblk + blk]);
        }
        __syncthreads();
        if (rowp != nullptr) {
            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (nt == 4) {
                mmid_row_q5k_dp4a<4>(rowp, nqb, k, qs, scale, acc);
            } else if (nt == 3) {
                mmid_row_q5k_dp4a<3>(rowp, nqb, k, qs, scale, acc);
            } else if (nt == 2) {
                mmid_row_q5k_dp4a<2>(rowp, nqb, k, qs, scale, acc);
            } else {
                mmid_row_q5k_dp4a<1>(rowp, nqb, k, qs, scale, acc);
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

static void get_scale_min_k4_host(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static float half_bits_to_f32(uint16_t h) {
    half hv;
    memcpy(&hv, &h, 2);
    return float(hv);
}

static void dequant_row_q5_K(const block_q5_K * row, int nqb, float * out) {
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q5_K & b = row[ib];
        uint16_t lo = 0;
        uint16_t hi = 0;
        memcpy(&lo, &b.dm, 2);
        memcpy(&hi, reinterpret_cast<const char *>(&b.dm) + 2, 2);
        const float dall = half_bits_to_f32(lo);
        const float dmin = half_bits_to_f32(hi);
        float * y = out + ib * QK_K;
        for (int il = 0; il < 4; ++il) {
            uint8_t sc, mn;
            get_scale_min_k4_host(2 * il, b.scales, sc, mn);
            const float d1 = dall * sc;
            const float m1 = dmin * mn;
            get_scale_min_k4_host(2 * il + 1, b.scales, sc, mn);
            const float d2 = dall * sc;
            const float m2 = dmin * mn;
            const uint8_t * q = b.qs + 32 * il;
            const uint8_t hmask_lo = uint8_t(1u << (2 * il));
            const uint8_t hmask_hi = uint8_t(2u << (2 * il));
            for (int j = 0; j < 32; ++j) {
                const int qlo = int(q[j] & 0xF) + ((b.qh[j] & hmask_lo) ? 16 : 0);
                const int qhi = int(q[j] >> 4) + ((b.qh[j] & hmask_hi) ? 16 : 0);
                y[64 * il + j] = d1 * float(qlo) - m1;
                y[64 * il + 32 + j] = d2 * float(qhi) - m2;
            }
        }
    }
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

static void pack_q5_K_row(const float * w, int k, block_q5_K * row) {
    const int nqb = k / QK_K;
    for (int ib = 0; ib < nqb; ++ib) {
        block_q5_K & b = row[ib];
        const float * src = w + ib * QK_K;
        float minv = src[0];
        float maxv = src[0];
        for (int i = 1; i < QK_K; ++i) {
            minv = fminf(minv, src[i]);
            maxv = fmaxf(maxv, src[i]);
        }
        const float min_term = fmaxf(0.0f, -minv);
        const float range = maxv + min_term;
        const float d = range > 0.0f ? range / 31.0f : 1.0f;
        const uint16_t dh = f32_to_f16_bits(d);
        const uint16_t dm = f32_to_f16_bits(min_term);
        memcpy(&b.dm, &dh, 2);
        memcpy(reinterpret_cast<char *>(&b.dm) + 2, &dm, 2);
        memset(b.scales, 0, K_SCALE_SIZE);
        for (int s = 0; s < 4; ++s) {
            b.scales[s] = 1;
            b.scales[s + 4] = 1;
            b.scales[s + 8] = 0x11;
        }
        memset(b.qh, 0, QK_K / 8);
        for (int il = 0; il < 4; ++il) {
            uint8_t * q = b.qs + 32 * il;
            const uint8_t hmask_lo = uint8_t(1u << (2 * il));
            const uint8_t hmask_hi = uint8_t(2u << (2 * il));
            for (int j = 0; j < 32; ++j) {
                int qlo = int(rintf(fminf(31.0f, fmaxf(0.0f, (src[64 * il + j] + min_term) / d))));
                int qhi = int(rintf(fminf(31.0f, fmaxf(0.0f, (src[64 * il + 32 + j] + min_term) / d))));
                if (qlo > 15) {
                    b.qh[j] |= hmask_lo;
                }
                if (qhi > 15) {
                    b.qh[j] |= hmask_hi;
                }
                q[j] = uint8_t((qlo & 0xF) | ((qhi & 0xF) << 4));
            }
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
    const int nqb = k / QK_K;
    const size_t row_bytes = size_t(nqb) * sizeof(block_q5_K);
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
            pack_q5_K_row(wr.data(), k, reinterpret_cast<block_q5_K *>(
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
                const block_q5_K * row = reinterpret_cast<const block_q5_K *>(
                        weights.data() + size_t(e) * expert_stride + size_t(r) * row_bytes);
                dequant_row_q5_K(row, nqb, wdeq.data());
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
        : size_t(4) * k + size_t(4) * (k / 32) * sizeof(float);

    auto launch = [&]() {
        if (use_vec) {
            mmid_q5k_vec_f32<<<dim3(tiles, slots), threads, shmem>>>(
                    d_w, d_x, d_ids, d_dst, k, m, experts, slots,
                    expert_stride, row_bytes, x_nb1, ids_nb1, dst_nb1, slots);
        } else {
            mmid_q5k_dp4a_f32<<<dim3(tiles, experts), threads, shmem>>>(
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
    const double time_ceiling = use_vec ? 160.0 : (ntokens == 16 ? 5000.0 : 10000.0);
    return err <= 5e-4 && us <= time_ceiling ? 0 : 1;
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
