/* Freestanding (zero-header) CUDA Driver API ABI witness shared library.
 * Compiles cleanly even on minimal Linux VMs without libc-dev headers. */

typedef __UINT16_TYPE__ u16;
typedef __UINT32_TYPE__ u32;
typedef __UINT64_TYPE__ u64;
typedef __UINTPTR_TYPE__ uptr;
typedef __SIZE_TYPE__ usize;

typedef u64 CUdeviceptr;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;

static int g_initialized = 0;

/* Bump allocator over 4 MiB BSS (no calloc/free/libc headers needed). */
static unsigned char g_mock_vram[4194304];
static usize g_mock_vram_used = 0;

static int mock_strcmp(const char* a, const char* b) {
    while (*a && (*a == *b)) {
        a++;
        b++;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

static int mock_contains(const char* hay, const char* needle) {
    if (!*needle) return 1;
    for (const char* p = hay; *p; ++p) {
        const char* h = p;
        const char* n = needle;
        while (*h && *n && (*h == *n)) {
            h++;
            n++;
        }
        if (!*n) return 1;
    }
    return 0;
}

static void mock_memcpy(void* dst, const void* src, usize n) {
    unsigned char* d = (unsigned char*)dst;
    const unsigned char* s = (const unsigned char*)src;
    for (usize i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

/* Freestanding IEEE-754 float math implementations (zero external library calls). */

static float mock_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    u32 i = *(u32*)&x;
    i = 0x1fbb4000 + (i >> 1);
    float y = *(float*)&i;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}

static float mock_rsqrtf(float x) {
    float s = mock_sqrtf(x);
    return (s > 0.0f) ? (1.0f / s) : 0.0f;
}

static float mock_expf(float x) {
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) return 1e38f;
    float div = x * 1.4426950408889634f;
    int k = (int)(div >= 0.0f ? (div + 0.5f) : (div - 0.5f));
    float r = x - (float)k * 0.6931471805599453f;
    float poly = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666666666666666f + r * (0.041666666666666664f + r * (0.008333333333333333f + r * 0.001388888888888889f)))));
    int exp_biased = k + 127;
    if (exp_biased <= 0) return 0.0f;
    if (exp_biased >= 255) return 1e38f;
    u32 exp_k = ((u32)exp_biased) << 23;
    float scale = *(float*)&exp_k;
    return poly * scale;
}

static float mock_logf(float x) {
    if (x <= 0.0f) return -1e38f;
    u32 bx = *(u32*)&x;
    int k = (int)((bx >> 23) & 0xFF) - 127;
    u32 bm = (bx & 0x007FFFFF) | 0x3F800000;
    float m = *(float*)&bm;
    if (m > 1.4142135623730951f) {
        m = m * 0.5f;
        k = k + 1;
    }
    float z = (m - 1.0f) / (m + 1.0f);
    float z2 = z * z;
    float lz = 2.0f * z * (1.0f + z2 * (0.3333333333333333f + z2 * (0.2f + z2 * (0.14285714285714285f + z2 * 0.1111111111111111f))));
    return (float)k * 0.6931471805599453f + lz;
}

static float mock_powf(float base, float exp) {
    if (base <= 0.0f) return 0.0f;
    return mock_expf(exp * mock_logf(base));
}

static void mock_sin_cos(float angle, float* s, float* c) {
    /* Reduce angle to [-pi/4, pi/4] using quadrant reduction:
     * angle = k * (pi/2) + r */
    float half_pi = 1.5707963267948966f;
    float inv_half_pi = 0.6366197723675814f;
    float div = angle * inv_half_pi;
    int k = (int)(div >= 0.0f ? (div + 0.5f) : (div - 0.5f));
    float r = angle - (float)k * half_pi;
    float r2 = r * r;

    /* Taylor series for sin(r) on [-pi/4, pi/4] */
    float s_r = r * (1.0f - r2 * (0.16666666666666666f - r2 * (0.008333333333333333f - r2 * 0.0001984126984126984f)));
    /* Taylor series for cos(r) on [-pi/4, pi/4] */
    float c_r = 1.0f - r2 * (0.5f - r2 * (0.041666666666666664f - r2 * (0.001388888888888889f - r2 * 0.0000248015873015873f)));

    int q = k & 3;
    if (q < 0) q += 4;

    if (q == 0) {
        *s = s_r;
        *c = c_r;
    } else if (q == 1) {
        *s = c_r;
        *c = -s_r;
    } else if (q == 2) {
        *s = -s_r;
        *c = -c_r;
    } else {
        *s = -c_r;
        *c = s_r;
    }
}

static float mock_bf16_to_f32(u16 u) {
    u32 bits = ((u32)u) << 16;
    return *(float*)&bits;
}

static float mock_f16_to_f32(u16 h) {
    u32 sign = ((u32)(h & 0x8000)) << 16;
    u32 exp = (h >> 10) & 0x1F;
    u32 mant = h & 0x03FF;
    if (exp == 31) {
        u32 f_bits = sign | 0x7F800000 | (mant << 13);
        return *(float*)&f_bits;
    }
    if (exp == 0) {
        if (mant == 0) return *(float*)&sign;
        exp = 1;
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x03FF;
    }
    u32 exp_f = exp + 112;
    u32 f_bits = sign | (exp_f << 23) | (mant << 13);
    return *(float*)&f_bits;
}

typedef struct {
    float c0, c1, c2, c3, c4, c5, c6, c7;
} MockOctF32;

static inline MockOctF32 mock_oct_mul(MockOctF32 a, MockOctF32 b) {
    MockOctF32 r;
    r.c0 = a.c0*b.c0 - a.c1*b.c1 - a.c2*b.c2 - a.c3*b.c3 - a.c4*b.c4 - a.c5*b.c5 - a.c6*b.c6 - a.c7*b.c7;
    r.c1 = a.c0*b.c1 + a.c1*b.c0 + a.c2*b.c3 - a.c3*b.c2 + a.c4*b.c5 - a.c5*b.c4 - a.c6*b.c7 + a.c7*b.c6;
    r.c2 = a.c0*b.c2 - a.c1*b.c3 + a.c2*b.c0 + a.c3*b.c1 + a.c4*b.c6 + a.c5*b.c7 - a.c6*b.c4 - a.c7*b.c5;
    r.c3 = a.c0*b.c3 + a.c1*b.c2 - a.c2*b.c1 + a.c3*b.c0 + a.c4*b.c7 - a.c5*b.c6 + a.c6*b.c5 - a.c7*b.c4;
    r.c4 = a.c0*b.c4 - a.c1*b.c5 - a.c2*b.c6 - a.c3*b.c7 + a.c4*b.c0 + a.c5*b.c1 + a.c6*b.c2 + a.c7*b.c3;
    r.c5 = a.c0*b.c5 + a.c1*b.c4 - a.c2*b.c7 + a.c3*b.c6 - a.c4*b.c1 + a.c5*b.c0 - a.c6*b.c3 + a.c7*b.c2;
    r.c6 = a.c0*b.c6 + a.c1*b.c7 + a.c2*b.c4 - a.c3*b.c5 - a.c4*b.c2 + a.c5*b.c3 + a.c6*b.c0 - a.c7*b.c1;
    r.c7 = a.c0*b.c7 - a.c1*b.c6 + a.c2*b.c5 + a.c3*b.c4 - a.c4*b.c3 - a.c5*b.c2 + a.c6*b.c1 + a.c7*b.c0;
    return r;
}

int cuInit(unsigned int flags) {
    (void)flags;
    g_initialized = 1;
    return 0;
}

int cuDeviceGet(int* device, int ordinal) {
    if (!g_initialized || !device || ordinal != 0) return 101;
    *device = 0;
    return 0;
}

int cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, int dev) {
    (void)flags;
    if (!g_initialized || !pctx || dev != 0) return 201;
    g_mock_vram_used = 0;
    *pctx = (CUcontext)(uptr)0xC0DE0001ULL;
    return 0;
}

int cuCtxDestroy_v2(CUcontext ctx) {
    g_mock_vram_used = 0;
    return ctx ? 0 : 201;
}

int cuModuleLoadData(CUmodule* module, const void* image) {
    if (!module || !image) return 200;
    const char* ptx = (const char*)image;
    if (!mock_contains(ptx, ".version 8.0") && !mock_contains(ptx, ".visible .entry")) {
        return 200;
    }
    *module = (CUmodule)(uptr)0x0D01E001ULL;
    return 0;
}

int cuModuleUnload(CUmodule hmod) {
    return hmod ? 0 : 400;
}

int cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (!hfunc || !hmod || !name) return 500;
    if (mock_strcmp(name, "bf16_to_f32") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00001ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_rmsnorm_bf16") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00002ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_rope_f32") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00003ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_bf16_gemv_f32") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00004ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_swiglu_f32") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00005ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_paged_gqa_f32") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00006ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_gemv_q4_eisa") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00007ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_gemv_q8_eisa") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00008ULL;
        return 0;
    }
    if (mock_strcmp(name, "qwen3_ossm_step_f32") == 0) {
        *hfunc = (CUfunction)(uptr)0xF0C00009ULL;
        return 0;
    }
    return 500;
}

int cuMemAlloc_v2(CUdeviceptr* dptr, usize bytesize) {
    if (!dptr || bytesize == 0) return 2;
    usize aligned = (bytesize + 15U) & ~(usize)15U;
    if (g_mock_vram_used + aligned > sizeof(g_mock_vram)) return 2;
    void* mem = &g_mock_vram[g_mock_vram_used];
    g_mock_vram_used += aligned;
    for (usize i = 0; i < aligned; ++i) ((unsigned char*)mem)[i] = 0;
    *dptr = (CUdeviceptr)(uptr)mem;
    return 0;
}

int cuMemFree_v2(CUdeviceptr dptr) {
    return dptr ? 0 : 400;
}

int cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, usize byteCount) {
    if (!dstDevice || !srcHost || byteCount == 0) return 1;
    mock_memcpy((void*)(uptr)dstDevice, srcHost, byteCount);
    return 0;
}

int cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, usize byteCount) {
    if (!dstHost || !srcDevice || byteCount == 0) return 1;
    mock_memcpy(dstHost, (const void*)(uptr)srcDevice, byteCount);
    return 0;
}

int cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra
) {
    (void)sharedMemBytes;
    (void)hStream;
    (void)extra;
    (void)gridDimY;
    (void)gridDimZ;
    (void)blockDimY;
    (void)blockDimZ;
    if (!f || !kernelParams || gridDimX == 0 || blockDimX == 0) {
        return 1;
    }

    uptr fn_id = (uptr)f;

    /* Phase 0: bf16_to_f32 */
    if (fn_id == (uptr)0xF0C00001ULL) {
        const u16* src = (const u16*)(uptr)(*(const u64*)kernelParams[0]);
        u32* dst = (u32*)(uptr)(*(const u64*)kernelParams[1]);
        u64 n = *(const u64*)kernelParams[2];
        u64 total = (u64)gridDimX * (u64)blockDimX;
        if (!src || !dst || total < n) return 1;
        for (u64 i = 0; i < n; ++i) {
            dst[i] = ((u32)src[i]) << 16;
        }
        return 0;
    }

    /* Phase 2: qwen3_rmsnorm_bf16 (6-param) */
    if (fn_id == (uptr)0xF0C00002ULL) {
        const float* x = (const float*)(uptr)(*(const u64*)kernelParams[0]);
        const u16* w = (const u16*)(uptr)(*(const u64*)kernelParams[1]);
        float* out = (float*)(uptr)(*(const u64*)kernelParams[2]);
        u64 n_groups = *(const u64*)kernelParams[3];
        u64 dim = *(const u64*)kernelParams[4];
        u64 eps_bits = *(const u64*)kernelParams[5];
        if (!x || !w || !out || n_groups == 0 || dim == 0) return 1;
        u32 eb = (u32)eps_bits;
        float eps = *(float*)&eb;
        for (u64 g = 0; g < n_groups; ++g) {
            float sum_sq = 0.0f;
            for (u64 i = 0; i < dim; ++i) {
                float v = x[g * dim + i];
                sum_sq += v * v;
            }
            float mean_sq = sum_sq / (float)dim;
            float inv_rms = mock_rsqrtf(mean_sq + eps);
            for (u64 i = 0; i < dim; ++i) {
                float weight = mock_bf16_to_f32(w[i]);
                out[g * dim + i] = x[g * dim + i] * inv_rms * weight;
            }
        }
        return 0;
    }

    /* Phase 2: qwen3_rope_f32 (6-param) */
    if (fn_id == (uptr)0xF0C00003ULL) {
        float* v = (float*)(uptr)(*(const u64*)kernelParams[0]);
        u64 n_heads = *(const u64*)kernelParams[1];
        u64 head_dim = *(const u64*)kernelParams[2];
        u64 pos = *(const u64*)kernelParams[3];
        u64 theta_bits = *(const u64*)kernelParams[4];
        if (!v || n_heads == 0 || head_dim == 0) return 1;
        u32 tb = (u32)theta_bits;
        float theta = *(float*)&tb;
        u64 half = head_dim / 2;
        for (u64 h = 0; h < n_heads; ++h) {
            float* head_ptr = v + h * head_dim;
            for (u64 j = 0; j < half; ++j) {
                float freq = mock_powf(theta, -((float)(2 * j)) / (float)head_dim);
                float angle = (float)pos * freq;
                float s = 0.0f, c = 1.0f;
                mock_sin_cos(angle, &s, &c);
                float x0 = head_ptr[j];
                float x1 = head_ptr[j + half];
                head_ptr[j] = x0 * c - x1 * s;
                head_ptr[j + half] = x1 * c + x0 * s;
            }
        }
        return 0;
    }

    /* Phase 2: qwen3_bf16_gemv_f32 (6-param) */
    if (fn_id == (uptr)0xF0C00004ULL) {
        const u16* w = (const u16*)(uptr)(*(const u64*)kernelParams[0]);
        const float* x = (const float*)(uptr)(*(const u64*)kernelParams[1]);
        float* y = (float*)(uptr)(*(const u64*)kernelParams[2]);
        u64 out_dim = *(const u64*)kernelParams[3];
        u64 in_dim = *(const u64*)kernelParams[4];
        if (!w || !x || !y || out_dim == 0 || in_dim == 0) return 1;
        for (u64 r = 0; r < out_dim; ++r) {
            const u16* row_w = w + r * in_dim;
            float sum = 0.0f;
            for (u64 c = 0; c < in_dim; ++c) {
                sum += mock_bf16_to_f32(row_w[c]) * x[c];
            }
            y[r] = sum;
        }
        return 0;
    }

    /* Phase 2: qwen3_swiglu_f32 (6-param) */
    if (fn_id == (uptr)0xF0C00005ULL) {
        const float* g = (const float*)(uptr)(*(const u64*)kernelParams[0]);
        const float* u = (const float*)(uptr)(*(const u64*)kernelParams[1]);
        float* o = (float*)(uptr)(*(const u64*)kernelParams[2]);
        u64 inter_dim = *(const u64*)kernelParams[3];
        if (!g || !u || !o || inter_dim == 0) return 1;
        for (u64 i = 0; i < inter_dim; ++i) {
            float g_val = g[i];
            float u_val = u[i];
            float silu_g = g_val / (1.0f + mock_expf(-g_val));
            o[i] = silu_g * u_val;
        }
        return 0;
    }

    /* Phase 2: qwen3_paged_gqa_f32 (8-param) */
    if (fn_id == (uptr)0xF0C00006ULL) {
        const float* q = (const float*)(uptr)(*(const u64*)kernelParams[0]);
        const float* k = (const float*)(uptr)(*(const u64*)kernelParams[1]);
        const float* v = (const float*)(uptr)(*(const u64*)kernelParams[2]);
        const u32* table = (const u32*)(uptr)(*(const u64*)kernelParams[3]);
        float* out = (float*)(uptr)(*(const u64*)kernelParams[4]);
        u64 n_q = *(const u64*)kernelParams[5];
        u64 seq_len = *(const u64*)kernelParams[6];
        u64 cfg = *(const u64*)kernelParams[7];
        if (!q || !k || !v || !table || !out || n_q == 0) return 1;
        u64 n_kv = cfg & 0xFFFF;
        u64 head_dim = (cfg >> 16) & 0xFFFF;
        u64 block_size = (cfg >> 32) & 0xFFFF;
        if (n_kv == 0) n_kv = 8;
        if (head_dim == 0) head_dim = 128;
        if (block_size == 0) block_size = 16;
        float scale = mock_rsqrtf((float)head_dim);
        u64 group_size = n_q / n_kv;
        for (u64 q_h = 0; q_h < n_q; ++q_h) {
            u64 kv_h = q_h / group_size;
            const float* q_vec = q + q_h * head_dim;
            float* out_vec = out + q_h * head_dim;
            float acc[256];
            for (u64 d = 0; d < head_dim && d < 256; ++d) acc[d] = 0.0f;
            float max_s = -1e38f;
            float l = 0.0f;
            for (u64 pos = 0; pos < seq_len; ++pos) {
                u64 logical_blk = pos / block_size;
                u32 phys_blk = table[logical_blk];
                u64 slot = pos % block_size;
                u64 kv_elem_idx = (((u64)phys_blk * block_size + slot) * n_kv + kv_h) * head_dim;
                const float* k_vec = k + kv_elem_idx;
                const float* v_vec = v + kv_elem_idx;
                float dot = 0.0f;
                for (u64 d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                float s = dot * scale;
                if (s > max_s) {
                    float alpha = mock_expf(max_s - s);
                    l = l * alpha + 1.0f;
                    for (u64 d = 0; d < head_dim && d < 256; ++d) {
                        acc[d] = acc[d] * alpha + v_vec[d];
                    }
                    max_s = s;
                } else {
                    float beta = mock_expf(s - max_s);
                    l = l + beta;
                    for (u64 d = 0; d < head_dim && d < 256; ++d) {
                        acc[d] = acc[d] + beta * v_vec[d];
                    }
                }
            }
            if (l > 0.0f) {
                float inv_l = 1.0f / l;
                for (u64 d = 0; d < head_dim && d < 256; ++d) {
                    out_vec[d] = acc[d] * inv_l;
                }
            } else {
                for (u64 d = 0; d < head_dim && d < 256; ++d) {
                    out_vec[d] = 0.0f;
                }
            }
        }
        return 0;
    }

    /* Phase 3: qwen3_gemv_q4_eisa (8-param) */
    if (fn_id == (uptr)0xF0C00007ULL) {
        const unsigned char* w = (const unsigned char*)(uptr)(*(const u64*)kernelParams[0]);
        const float* x = (const float*)(uptr)(*(const u64*)kernelParams[1]);
        float* val = (float*)(uptr)(*(const u64*)kernelParams[2]);
        float* err = (float*)(uptr)(*(const u64*)kernelParams[3]);
        u64 out_dim = *(const u64*)kernelParams[4];
        u64 in_dim = *(const u64*)kernelParams[5];
        if (!w || !x || !val || !err || out_dim == 0 || in_dim == 0) return 1;
        u64 n_blocks = (in_dim + 31) / 32;
        for (u64 r = 0; r < out_dim; ++r) {
            const unsigned char* row_w = w + r * n_blocks * 20;
            float val_sum = 0.0f;
            float err_sum = 0.0f;
            for (u64 b = 0; b < n_blocks; ++b) {
                const unsigned char* blk = row_w + b * 20;
                u16 s_bits = (u16)blk[0] | ((u16)blk[1] << 8);
                u16 rho_bits = (u16)blk[2] | ((u16)blk[3] << 8);
                float s = mock_f16_to_f32(s_bits);
                float rho = mock_f16_to_f32(rho_bits);
                float sum_sq_x = 0.0f;
                float sum_sq_w = 0.0f;
                u64 base_k = b * 32;
                for (u64 j = 0; j < 16; ++j) {
                    unsigned char byte_val = blk[4 + j];
                    int q0 = (int)(byte_val & 0x0F) - 8;
                    int q1 = (int)((byte_val >> 4) & 0x0F) - 8;
                    float w0 = s * (float)q0;
                    float w1 = s * (float)q1;
                    u64 k0 = base_k + 2 * j;
                    u64 k1 = k0 + 1;
                    float x0 = (k0 < in_dim) ? x[k0] : 0.0f;
                    float x1 = (k1 < in_dim) ? x[k1] : 0.0f;
                    val_sum += w0 * x0 + w1 * x1;
                    sum_sq_x += x0 * x0 + x1 * x1;
                    sum_sq_w += w0 * w0 + w1 * w1;
                }
                float norm_x = mock_sqrtf(sum_sq_x);
                float norm_w = mock_sqrtf(sum_sq_w);
                err_sum += rho * norm_x + 0.00001f * (norm_w + rho) * norm_x;
            }
            val[r] = val_sum;
            err[r] = err_sum;
        }
        return 0;
    }

    /* Phase 3: qwen3_gemv_q8_eisa (8-param) */
    if (fn_id == (uptr)0xF0C00008ULL) {
        const unsigned char* w = (const unsigned char*)(uptr)(*(const u64*)kernelParams[0]);
        const float* x = (const float*)(uptr)(*(const u64*)kernelParams[1]);
        float* val = (float*)(uptr)(*(const u64*)kernelParams[2]);
        float* err = (float*)(uptr)(*(const u64*)kernelParams[3]);
        u64 out_dim = *(const u64*)kernelParams[4];
        u64 in_dim = *(const u64*)kernelParams[5];
        if (!w || !x || !val || !err || out_dim == 0 || in_dim == 0) return 1;
        u64 n_blocks = (in_dim + 31) / 32;
        for (u64 r = 0; r < out_dim; ++r) {
            const unsigned char* row_w = w + r * n_blocks * 36;
            float val_sum = 0.0f;
            float err_sum = 0.0f;
            for (u64 b = 0; b < n_blocks; ++b) {
                const unsigned char* blk = row_w + b * 36;
                u16 s_bits = (u16)blk[0] | ((u16)blk[1] << 8);
                u16 rho_bits = (u16)blk[2] | ((u16)blk[3] << 8);
                float s = mock_f16_to_f32(s_bits);
                float rho = mock_f16_to_f32(rho_bits);
                float sum_sq_x = 0.0f;
                float sum_sq_w = 0.0f;
                u64 base_k = b * 32;
                for (u64 k = 0; k < 32; ++k) {
                    signed char q = (signed char)blk[4 + k];
                    float w_val = s * (float)q;
                    u64 idx = base_k + k;
                    float x_val = (idx < in_dim) ? x[idx] : 0.0f;
                    val_sum += w_val * x_val;
                    sum_sq_x += x_val * x_val;
                    sum_sq_w += w_val * w_val;
                }
                float norm_x = mock_sqrtf(sum_sq_x);
                float norm_w = mock_sqrtf(sum_sq_w);
                err_sum += rho * norm_x + 0.00001f * (norm_w + rho) * norm_x;
            }
            val[r] = val_sum;
            err[r] = err_sum;
        }
        return 0;
    }

    /* Phase 4: qwen3_ossm_step_f32 (8-param) */
    if (fn_id == (uptr)0xF0C00009ULL) {
        float* st = (float*)(uptr)(*(const u64*)kernelParams[0]);
        const float* u = (const float*)(uptr)(*(const u64*)kernelParams[1]);
        const float* alpha = (const float*)(uptr)(*(const u64*)kernelParams[2]);
        const float* b = (const float*)(uptr)(*(const u64*)kernelParams[3]);
        const float* dt = (const float*)(uptr)(*(const u64*)kernelParams[4]);
        const float* c = (const float*)(uptr)(*(const u64*)kernelParams[5]);
        float* y = (float*)(uptr)(*(const u64*)kernelParams[6]);
        u64 n_oct = *(const u64*)kernelParams[7];
        if (!st || !u || !alpha || !b || !dt || !c || !y || n_oct == 0) return 1;
        for (u64 ch = 0; ch < n_oct; ++ch) {
            float* h_ptr = st + ch * 8;
            const float* u_ptr = u + ch * 8;
            const float* alpha_ptr = alpha + ch * 8;
            const float* b_ptr = b + ch * 8;
            const float* dt_ptr = dt + ch * 8;
            const float* c_ptr = c + ch * 8;
            float* y_ptr = y + ch * 8;

            float decay = 1.0f / (1.0f + mock_expf(-alpha_ptr[0]));
            MockOctF32 B = {b_ptr[0], b_ptr[1], b_ptr[2], b_ptr[3], b_ptr[4], b_ptr[5], b_ptr[6], b_ptr[7]};
            MockOctF32 U = {u_ptr[0], u_ptr[1], u_ptr[2], u_ptr[3], u_ptr[4], u_ptr[5], u_ptr[6], u_ptr[7]};
            MockOctF32 DT = {dt_ptr[0], dt_ptr[1], dt_ptr[2], dt_ptr[3], dt_ptr[4], dt_ptr[5], dt_ptr[6], dt_ptr[7]};
            MockOctF32 C = {c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3], c_ptr[4], c_ptr[5], c_ptr[6], c_ptr[7]};

            MockOctF32 BU = mock_oct_mul(B, U);
            MockOctF32 INJ = mock_oct_mul(BU, DT);

            MockOctF32 H_new;
            H_new.c0 = decay * h_ptr[0] + INJ.c0;
            H_new.c1 = decay * h_ptr[1] + INJ.c1;
            H_new.c2 = decay * h_ptr[2] + INJ.c2;
            H_new.c3 = decay * h_ptr[3] + INJ.c3;
            H_new.c4 = decay * h_ptr[4] + INJ.c4;
            H_new.c5 = decay * h_ptr[5] + INJ.c5;
            H_new.c6 = decay * h_ptr[6] + INJ.c6;
            H_new.c7 = decay * h_ptr[7] + INJ.c7;

            h_ptr[0] = H_new.c0;
            h_ptr[1] = H_new.c1;
            h_ptr[2] = H_new.c2;
            h_ptr[3] = H_new.c3;
            h_ptr[4] = H_new.c4;
            h_ptr[5] = H_new.c5;
            h_ptr[6] = H_new.c6;
            h_ptr[7] = H_new.c7;

            MockOctF32 Y = mock_oct_mul(C, H_new);
            y_ptr[0] = Y.c0;
            y_ptr[1] = Y.c1;
            y_ptr[2] = Y.c2;
            y_ptr[3] = Y.c3;
            y_ptr[4] = Y.c4;
            y_ptr[5] = Y.c5;
            y_ptr[6] = Y.c6;
            y_ptr[7] = Y.c7;
        }
        return 0;
    }

    return 1;
}

int cuCtxSynchronize(void) {
    return g_initialized ? 0 : 3;
}
