// Sounio hetero: fused MoE gate+up SiLU on Q4_K expert packs (F32 x).
// Build PTX: nvcc -O3 -ptx -arch=sm_89 -o moe_gate_up_silu_q4k_f32.ptx moe_gate_up_silu_q4k_f32.cu
// Lab host:  nvcc -O3 -arch=sm_89 -DMOE_GUS_LAB_HOST -o moe_gus_lab moe_gate_up_silu_q4k_f32.cu

#include <cuda_fp16.h>
#include <stdint.h>

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_K {
    half2   dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static __device__ __forceinline__ int gus_id_at(const char * ids, int token, int slot, size_t ids_nb1) {
    return *reinterpret_cast<const int32_t *>(ids + size_t(token) * ids_nb1 + size_t(slot) * sizeof(int32_t));
}

static __device__ float gus_dot_q4_K(const block_q4_K * row, int nqb, const float * y) {
    float acc = 0.0f;
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q4_K & b = row[ib];
        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);
        const float * yb = y + ib * QK_K;
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
            const float * y_lo = yb + 64 * il;
            const float * y_hi = y_lo + 32;
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const uint32_t raw = __ldg(reinterpret_cast<const uint32_t *>(q + j));
#pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    const int qq = int((raw >> (8 * lane)) & 0xFFu);
                    acc = fmaf(d1 * float(qq & 0xF) - m1, y_lo[j + lane], acc);
                    acc = fmaf(d2 * float(qq >> 4) - m2, y_hi[j + lane], acc);
                }
            }
        }
    }
    return acc;
}

static __device__ __forceinline__ float gus_silu(float g) {
    return g / (1.0f + expf(0.0f - g));
}

extern "C" __global__ void moe_gate_up_silu_q4k_f32(
        const char * __restrict__ gate_w, const char * __restrict__ up_w,
        const float * __restrict__ x, const int * __restrict__ ids,
        float * __restrict__ act, int hidden, int inter, int experts, int slots, int ntokens,
        size_t expert_stride_g, size_t row_stride_g,
        size_t expert_stride_u, size_t row_stride_u,
        size_t x_nb1, size_t ids_nb1, size_t act_nb1, size_t act_nb2) {
    const int ts = int(blockIdx.y);
    const int token = ts / slots;
    const int slot = ts - token * slots;
    const int row = int(blockIdx.x * blockDim.x + threadIdx.x);

    extern __shared__ float yshared[];
    const float * yin = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(x) + size_t(token) * x_nb1);
    for (int i = threadIdx.x; i < hidden; i += int(blockDim.x)) {
        yshared[i] = yin[i];
    }
    __syncthreads();

    if (row >= inter || token >= ntokens) {
        return;
    }

    const int id = gus_id_at(reinterpret_cast<const char *>(ids), token, slot, ids_nb1);
    float * out = reinterpret_cast<float *>(
            reinterpret_cast<char *>(act) + size_t(token) * act_nb2 + size_t(slot) * act_nb1);
    if (id < 0 || id >= experts) {
        out[row] = __int_as_float(0x7fc00000);
        return;
    }

    const int nqb = hidden / QK_K;
    const block_q4_K * gate_row = reinterpret_cast<const block_q4_K *>(
            gate_w + size_t(id) * expert_stride_g + size_t(row) * row_stride_g);
    const block_q4_K * up_row = reinterpret_cast<const block_q4_K *>(
            up_w + size_t(id) * expert_stride_u + size_t(row) * row_stride_u);
    const float g = gus_dot_q4_K(gate_row, nqb, yshared);
    const float u = gus_dot_q4_K(up_row, nqb, yshared);
    out[row] = gus_silu(g) * u;
}

#ifdef MOE_GUS_LAB_HOST
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
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static float half_bits_to_f32(uint16_t h) {
    half hv;
    memcpy(&hv, &h, 2);
    return float(hv);
}

static void dequant_row_q4_K(const block_q4_K * row, int nqb, float * out) {
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q4_K & b = row[ib];
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
            for (int j = 0; j < 32; ++j) {
                y[64 * il + j] = d1 * float(q[j] & 0xF) - m1;
                y[64 * il + 32 + j] = d2 * float(q[j] >> 4) - m2;
            }
        }
    }
}

static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00);
    }
    return (uint16_t)(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

static void pack_q4_K_row(const float * w, int k, block_q4_K * row) {
    const int nqb = k / QK_K;
    for (int ib = 0; ib < nqb; ++ib) {
        block_q4_K & b = row[ib];
        const float * src = w + ib * QK_K;
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            amax = fmaxf(amax, fabsf(src[i]));
        }
        const float d = amax > 0.0f ? amax / 15.0f : 1.0f;
        const float dmin = 0.0f;
        uint16_t dh = f32_to_f16_bits(d);
        uint16_t dm = f32_to_f16_bits(dmin);
        memcpy(&b.dm, &dh, 2);
        memcpy(reinterpret_cast<char *>(&b.dm) + 2, &dm, 2);
        memset(b.scales, 0, K_SCALE_SIZE);
        for (int s = 0; s < 4; ++s) {
            b.scales[s] = 1;
            b.scales[s + 4] = 0;
            b.scales[s + 8] = 1;
        }
        for (int il = 0; il < 4; ++il) {
            uint8_t * q = b.qs + 32 * il;
            for (int j = 0; j < 32; ++j) {
                int q0 = int(rintf(fminf(15.0f, fmaxf(0.0f, src[64 * il + j] / d))));
                int q1 = int(rintf(fminf(15.0f, fmaxf(0.0f, src[64 * il + 32 + j] / d))));
                q[j] = (uint8_t)((q1 << 4) | (q0 & 0xF));
            }
        }
    }
}

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
    }
    return den > 0.0 ? num / den : num;
}

static float host_silu(float g) {
    return g / (1.0f + expf(0.0f - g));
}

static float dot_dequant_row(const block_q4_K * row, int nqb, const float * x, float * wdeq) {
    dequant_row_q4_K(row, nqb, wdeq);
    float acc = 0.0f;
    const int k = nqb * QK_K;
    for (int i = 0; i < k; ++i) {
        acc += wdeq[i] * x[i];
    }
    return acc;
}

static void host_silu_mul(const float * gate, const float * up, float * out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = host_silu(gate[i]) * up[i];
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

template <int NT>
static __device__ void mmid_row_q4k_dp4a(const block_q4_K * rowp, int nqb, int k,
        const int8_t * qs, const float * scale, float acc[NT]) {
    const int nblk = k / 32;
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q4_K & b = rowp[ib];
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
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(q + j);
                const int lo = int(raw & 0x0F0F0F0F);
                const int hi = int((raw >> 4) & 0x0F0F0F0F);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int8_t * ylo = qs + t * k + blk_lo * 32 + j;
                    const int8_t * yhi = ylo + 32;
                    const int ulo = *reinterpret_cast<const int *>(ylo);
                    const int uhi = *reinterpret_cast<const int *>(yhi);
                    sum_lo[t] = __dp4a(lo, ulo, sum_lo[t]);
                    sum_hi[t] = __dp4a(hi, uhi, sum_hi[t]);
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

static __device__ float mmid_dot_q4_K(const block_q4_K * row, int ib, const float * y) {
    const block_q4_K & b = row[ib];
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
        const float * y_lo = yb + 64 * il;
        const float * y_hi = y_lo + 32;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            const uint32_t raw = __ldg(reinterpret_cast<const uint32_t *>(q + j));
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int qq = int((raw >> (8 * lane)) & 0xFFu);
                acc = fmaf(d1 * float(qq & 0xF) - m1, y_lo[j + lane], acc);
                acc = fmaf(d2 * float(qq >> 4) - m2, y_hi[j + lane], acc);
            }
        }
    }
    return acc;
}

extern "C" __global__ void mmid_q4k_vec_f32(
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
    const block_q4_K * rowp = reinterpret_cast<const block_q4_K *>(
            weights + size_t(id) * expert_stride + size_t(row) * row_stride);
    const int nqb = k / QK_K;
    float acc = 0.0f;
    for (int ib = 0; ib < nqb; ++ib) {
        acc += mmid_dot_q4_K(rowp, ib, yshared);
    }
    out[row] = acc;
}

extern "C" __global__ void mmid_q4k_dp4a_f32(
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
    if (nmatch == 0) {
        return;
    }
    const int nqb = k / QK_K;
    const block_q4_K * rowp = (row < m)
        ? reinterpret_cast<const block_q4_K *>(weights + size_t(expert) * expert_stride + size_t(row) * row_stride)
        : nullptr;
    const int nt = nmatch > 4 ? 4 : nmatch;
    __shared__ int tok[4];
    __shared__ int sl[4];
    if (threadIdx.x == 0) {
        int idx = 0;
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                if (mmid_id_at(ids, token, slot, ids_nb1) == expert) {
                    tok[idx] = token;
                    sl[idx] = slot;
                    ++idx;
                    if (idx >= nt) {
                        break;
                    }
                }
            }
            if (idx >= nt) {
                break;
            }
        }
    }
    __syncthreads();
    for (int round = 0; round < (nmatch + nt - 1) / nt; ++round) {
        const int base = round * nt;
        const int nthis = (base + nt <= nmatch) ? nt : (nmatch - base);
        const float * yp[4];
        for (int t = 0; t < nthis; ++t) {
            yp[t] = reinterpret_cast<const float *>(x + size_t(tok[t]) * x_nb2 + size_t(sl[t] % x_rows) * x_nb1);
        }
        for (int g = threadIdx.x; g < nthis * nqb; g += int(blockDim.x)) {
            const int t = g / nqb;
            const int blk = g - t * nqb;
            mmid_quant_f32_i8(yp[t] + blk * 32, qs + t * k + blk * 32, scale[t * nqb + blk]);
        }
        __syncthreads();
        if (rowp != nullptr) {
            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (nthis == 4) {
                mmid_row_q4k_dp4a<4>(rowp, nqb, k, qs, scale, acc);
            } else if (nthis == 3) {
                mmid_row_q4k_dp4a<3>(rowp, nqb, k, qs, scale, acc);
            } else if (nthis == 2) {
                mmid_row_q4k_dp4a<2>(rowp, nqb, k, qs, scale, acc);
            } else {
                mmid_row_q4k_dp4a<1>(rowp, nqb, k, qs, scale, acc);
            }
            for (int t = 0; t < nthis; ++t) {
                float * out = reinterpret_cast<float *>(dst + size_t(tok[t]) * dst_nb2 + size_t(sl[t]) * dst_nb1);
                out[row] = acc[t];
            }
        }
        __syncthreads();
    }
}

static int run_case(int hidden, int inter, int experts, int slots, int ntokens, float amax) {
    const int nqb = hidden / QK_K;
    const size_t row_bytes = size_t(nqb) * sizeof(block_q4_K);
    const size_t expert_stride_g = size_t(inter) * row_bytes;
    const size_t expert_stride_u = expert_stride_g;
    std::vector<char> gate_w(size_t(experts) * expert_stride_g);
    std::vector<char> up_w(size_t(experts) * expert_stride_u);
    std::vector<float> x(size_t(ntokens) * size_t(hidden));
    std::vector<int32_t> ids(size_t(ntokens) * size_t(slots));
    std::vector<float> act_gpu(size_t(ntokens) * size_t(slots) * size_t(inter));
    std::vector<float> act_cpu(size_t(ntokens) * size_t(slots) * size_t(inter));

    srand(1);
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < inter; ++r) {
            std::vector<float> wr(hidden);
            for (int i = 0; i < hidden; ++i) {
                wr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            pack_q4_K_row(wr.data(), hidden,
                reinterpret_cast<block_q4_K *>(gate_w.data() + size_t(e) * expert_stride_g + size_t(r) * row_bytes));
            for (int i = 0; i < hidden; ++i) {
                wr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            pack_q4_K_row(wr.data(), hidden,
                reinterpret_cast<block_q4_K *>(up_w.data() + size_t(e) * expert_stride_u + size_t(r) * row_bytes));
        }
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * amax;
    }
    for (int t = 0; t < ntokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            ids[size_t(t) * slots + s] = (t * slots + s) % experts;
        }
    }

    std::vector<float> wdeq(hidden);
    for (int t = 0; t < ntokens; ++t) {
        const float * xt = x.data() + size_t(t) * hidden;
        for (int s = 0; s < slots; ++s) {
            const int e = ids[size_t(t) * slots + s];
            float * out = act_cpu.data() + (size_t(t) * slots + s) * inter;
            for (int r = 0; r < inter; ++r) {
                const block_q4_K * grow = reinterpret_cast<const block_q4_K *>(
                    gate_w.data() + size_t(e) * expert_stride_g + size_t(r) * row_bytes);
                const block_q4_K * urow = reinterpret_cast<const block_q4_K *>(
                    up_w.data() + size_t(e) * expert_stride_u + size_t(r) * row_bytes);
                const float g = dot_dequant_row(grow, nqb, xt, wdeq.data());
                const float u = dot_dequant_row(urow, nqb, xt, wdeq.data());
                out[r] = host_silu(g) * u;
            }
        }
    }

    char * d_g = nullptr;
    char * d_u = nullptr;
    float * d_x = nullptr;
    int * d_ids = nullptr;
    float * d_act = nullptr;
    cudaMalloc(&d_g, gate_w.size());
    cudaMalloc(&d_u, up_w.size());
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMalloc(&d_act, act_gpu.size() * sizeof(float));
    cudaMemcpy(d_g, gate_w.data(), gate_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, up_w.data(), up_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    const size_t x_nb1 = size_t(hidden) * sizeof(float);
    const size_t ids_nb1 = size_t(slots) * sizeof(int32_t);
    const size_t act_nb1 = size_t(inter) * sizeof(float);
    const size_t act_nb2 = size_t(slots) * act_nb1;
    const int threads = 128;
    const int row_tiles = (inter + threads - 1) / threads;
    const int ts_count = ntokens * slots;
    const size_t shmem_gus = size_t(hidden) * sizeof(float);

    auto launch_gus = [&]() {
        moe_gate_up_silu_q4k_f32<<<dim3(row_tiles, ts_count), threads, shmem_gus>>>(
            d_g, d_u, d_x, d_ids, d_act, hidden, inter, experts, slots, ntokens,
            expert_stride_g, row_bytes, expert_stride_u, row_bytes,
            x_nb1, ids_nb1, act_nb1, act_nb2);
    };

    const bool use_vec = (ntokens == 1);
    const size_t x_nb2 = x_nb1;
    const size_t dst_nb1 = act_nb1;
    const size_t dst_nb2 = act_nb2;
    std::vector<float> gate_buf(size_t(ntokens) * size_t(slots) * size_t(inter));
    std::vector<float> up_buf(gate_buf.size());
    std::vector<float> act_base(gate_buf.size());
    char * d_gate = nullptr;
    char * d_up = nullptr;
    cudaMalloc(&d_gate, gate_buf.size() * sizeof(float));
    cudaMalloc(&d_up, up_buf.size() * sizeof(float));

    auto launch_baseline = [&]() {
        if (use_vec) {
            const size_t shmem = size_t(hidden) * sizeof(float);
            mmid_q4k_vec_f32<<<dim3(row_tiles, slots), threads, shmem>>>(
                d_g, reinterpret_cast<char *>(d_x), reinterpret_cast<char *>(d_ids),
                reinterpret_cast<char *>(d_gate), hidden, inter, experts, slots,
                expert_stride_g, row_bytes, x_nb1, ids_nb1, dst_nb1, 1);
            mmid_q4k_vec_f32<<<dim3(row_tiles, slots), threads, shmem>>>(
                d_u, reinterpret_cast<char *>(d_x), reinterpret_cast<char *>(d_ids),
                reinterpret_cast<char *>(d_up), hidden, inter, experts, slots,
                expert_stride_u, row_bytes, x_nb1, ids_nb1, dst_nb1, 1);
        } else {
            const size_t shmem = size_t(4) * hidden + size_t(4) * (hidden / 32) * sizeof(float);
            mmid_q4k_dp4a_f32<<<dim3(row_tiles, experts), threads, shmem>>>(
                d_g, reinterpret_cast<char *>(d_x), reinterpret_cast<char *>(d_ids),
                reinterpret_cast<char *>(d_gate), hidden, inter, experts, slots, ntokens,
                expert_stride_g, row_bytes, x_nb1, x_nb2, ids_nb1, dst_nb1, dst_nb2, 1);
            mmid_q4k_dp4a_f32<<<dim3(row_tiles, experts), threads, shmem>>>(
                d_u, reinterpret_cast<char *>(d_x), reinterpret_cast<char *>(d_ids),
                reinterpret_cast<char *>(d_up), hidden, inter, experts, slots, ntokens,
                expert_stride_u, row_bytes, x_nb1, x_nb2, ids_nb1, dst_nb1, dst_nb2, 1);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(gate_buf.data(), d_gate, gate_buf.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(up_buf.data(), d_up, up_buf.size() * sizeof(float), cudaMemcpyDeviceToHost);
        host_silu_mul(gate_buf.data(), up_buf.data(), act_base.data(), act_base.size());
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i) {
        launch_gus();
    }
    cudaDeviceSynchronize();
    const int runs = ntokens >= 64 ? 50 : (ntokens == 1 ? 200 : 100);
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        launch_gus();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    const double us_gus = (double(ms) * 1000.0) / double(runs);

    for (int i = 0; i < 3; ++i) {
        launch_baseline();
    }
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        launch_baseline();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    const double us_base = (double(ms) * 1000.0) / double(runs);

    launch_gus();
    cudaDeviceSynchronize();
    cudaMemcpy(act_gpu.data(), d_act, act_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const double err = nmse(act_gpu.data(), act_cpu.data(), act_gpu.size());
    printf("n=%d NMSE=%.9g us/run=%.2f baseline_us=%.2f\n", ntokens, err, us_gus, us_base);

    cudaFree(d_g);
    cudaFree(d_u);
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_act);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (err <= 5e-4) ? 0 : 1;
}

int main() {
    int rc = 0;
    rc |= run_case(2048, 4096, 128, 4, 1, 1e5f);
    rc |= run_case(2048, 4096, 128, 4, 4, 1e5f);
    printf("LAB_EXIT:%d\n", rc);
    return rc;
}
#endif
