// scripts/gpu/kaxi_ptx_runner.c
//
// Minimal CUDA Driver API runner for K-AXI → PTX kernels.
//
// Loads a PTX file (or CUBIN — cuModuleLoadData autodetects), allocates one
// or two device buffers (basic mode = mem only, epistemic mode = mem + var),
// launches the kernel with the requested grid/block, copies back, and prints
// a deterministic single-line summary the submission script can grep on.
//
// Build:
//   cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o /tmp/kaxi_ptx_runner
//
// Usage:
//   kaxi_ptx_runner <ptx_or_cubin> [--kernel NAME] [--mode basic|epistemic]
//                                  [--threads T] [--mem-words W]
//                                  [--init-mem v0,v1,...] [--init-var v0,v1,...]
//                                  [--init-file PATH]      [--init-var-file PATH]
//
// --init-file / --init-var-file read RAW binary device-image bytes
// (mem_words * sizeof(elem)) directly from disk — required when the
// inline CSV form would exceed ARG_MAX (e.g. 1.3M-thread connectomics
// sweep, ~125 MB f32 input). Takes precedence over --init-mem if both set.
//
// Defaults: kernel=kaxi_kernel, mode=basic, threads=1, mem-words=16,
//           init-mem=zeros, init-var=zeros.
//
// Output (stdout, machine-grep friendly):
//   sounio_kaxi_runtime status=pass reason=launch_pass stage=cuMemcpyDtoH cuda_result=0
//   MEM: <w0> <w1> ... <wW-1>
//   VAR: <v0> <v1> ... <vW-1>             (epistemic only)
//   device=<name> cc=<major>.<minor>

#include <dlfcn.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef uint64_t CUdeviceptr;
typedef int CUresult;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDriverGetVersion_t)(int *);
typedef CUresult (*cuDeviceGetCount_t)(int *);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*cuDeviceComputeCapability_t)(int *, int *, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuModuleLoadData_t)(CUmodule *, const void *);
typedef CUresult (*cuModuleLoadDataEx_t)(CUmodule *, const void *, unsigned int, int *, void **);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, void *, void **, void **);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemsetD8_t)(CUdeviceptr, unsigned char, size_t);

// Phase W: streams + pinned-host symbols (loaded only when --cohort-size set).
typedef void *CUstream;
typedef CUresult (*cuStreamCreate_t)(CUstream *, unsigned int);
typedef CUresult (*cuStreamDestroy_t)(CUstream);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);
typedef CUresult (*cuMemcpyHtoDAsync_t)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_t)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemHostAlloc_t)(void **, size_t, unsigned int);
typedef CUresult (*cuMemFreeHost_t)(void *);
typedef CUresult (*cuLaunchKernelStream_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int,
                                           unsigned int, CUstream, void **, void **);

// FNV-1a 64-bit digest over arbitrary bytes.
static uint64_t fnv1a64(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ULL; }
    return h;
}

static unsigned char *read_all(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return NULL; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (n != (size_t)sz) { free(buf); return NULL; }
    buf[sz] = 0;
    *out_len = (size_t)sz;
    return buf;
}

static void emit_status(const char *status, const char *reason,
                        const char *stage, int cuda_result) {
    printf("sounio_kaxi_runtime status=%s reason=%s stage=%s cuda_result=%d\n",
           status, reason, stage, cuda_result);
}

static int parse_csv_i64(const char *s, int64_t *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end = NULL;
        long long v = strtoll(p, &end, 10);
        if (end == p) break;
        out[n++] = (int64_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

static int parse_csv_i32(const char *s, int32_t *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

static int parse_csv_f32(const char *s, float *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end = NULL;
        float v = strtof(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

static int parse_csv_f64(const char *s, double *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <ptx_or_cubin> [--kernel NAME] [--mode basic|epistemic]\n", argv[0]);
        fprintf(stderr, "                          [--threads T] [--mem-words W]\n");
        fprintf(stderr, "                          [--init-mem v0,v1,...] [--init-var v0,v1,...]\n");
        fprintf(stderr, "                          [--init-file PATH] [--init-var-file PATH]\n");
        return 2;
    }

    const char *path = argv[1];
    const char *kname = "kaxi_kernel";
    int epistemic = 0;
    int value_type = 0;       // 0 = i64, 1 = f32, 2 = f64, 3 = i32
    unsigned int threads = 1;
    unsigned int blocks = 1;
    int mem_words = 16;
    const char *init_mem_csv = NULL;
    const char *init_var_csv = NULL;
    const char *init_mem_file = NULL;
    const char *init_var_file = NULL;
    int verify_init_seq = 0;  // if set, init mem to [1..mem_words] before launch
    int print_count = -1;     // override print count (default = mem_words)
    // Phase W: streamed multi-launch over a 1M-patient cohort.
    long cohort_size = 0;     // if > 0 override mem_words and enter streamed path
    int n_streams = 0;        // if > 0 use this many CUDA streams (else default-stream)
    int n_chunks = 0;         // if > 0 number of chunks; else cohort/threads (1 launch/chunk per Phase V shape)
    // Phase X: after D2H, count f32 values in mem[0..cohort_size-1] that fall
    // in [classify_low, classify_high]. Reports in_window + out_of_window in PHX line.
    float classify_low = 0.0f, classify_high = 0.0f;
    int classify_window = 0;
    // Phase Y: 4-buffer GUM mode (C1, C2, V11, V22). Single-launch only.
    int gum_mode = 0;
    const char *init_v11_file = NULL, *init_v22_file = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) { kname = argv[++i]; }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++; epistemic = (strcmp(argv[i], "epistemic") == 0);
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) { threads = (unsigned)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) { blocks = (unsigned)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--mem-words") == 0 && i + 1 < argc) { mem_words = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--init-mem") == 0 && i + 1 < argc) { init_mem_csv = argv[++i]; }
        else if (strcmp(argv[i], "--init-var") == 0 && i + 1 < argc) { init_var_csv = argv[++i]; }
        else if (strcmp(argv[i], "--init-file") == 0 && i + 1 < argc) { init_mem_file = argv[++i]; }
        else if (strcmp(argv[i], "--init-var-file") == 0 && i + 1 < argc) { init_var_file = argv[++i]; }
        else if (strcmp(argv[i], "--init-seq") == 0) { verify_init_seq = 1; }
        else if (strcmp(argv[i], "--print-count") == 0 && i + 1 < argc) { print_count = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--cohort-size") == 0 && i + 1 < argc) { cohort_size = strtol(argv[++i], NULL, 10); }
        else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc) { n_streams = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) { n_chunks = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--classify-window") == 0 && i + 1 < argc) {
            i++;
            if (sscanf(argv[i], "%f,%f", &classify_low, &classify_high) != 2) {
                fprintf(stderr, "--classify-window requires LOW,HIGH (e.g. 0.4,2.5)\n"); return 2;
            }
            classify_window = 1;
        }
        else if (strcmp(argv[i], "--epistemic") == 0) { epistemic = 1; }
        else if (strcmp(argv[i], "--gum") == 0) { gum_mode = 1; epistemic = 1; }
        else if (strcmp(argv[i], "--init-v11-file") == 0 && i + 1 < argc) { init_v11_file = argv[++i]; }
        else if (strcmp(argv[i], "--init-v22-file") == 0 && i + 1 < argc) { init_v22_file = argv[++i]; }
        else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "f32") == 0) value_type = 1;
            else if (strcmp(argv[i], "f64") == 0) value_type = 2;
            else if (strcmp(argv[i], "i64") == 0) value_type = 0;
            else if (strcmp(argv[i], "i32") == 0) value_type = 3;
            else { fprintf(stderr, "unknown --type: %s\n", argv[i]); return 2; }
        }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    // Phase W: --cohort-size overrides mem_words for streamed multi-launch.
    if (cohort_size > 0) {
        if (cohort_size < 1 || cohort_size > 2000000000L) {
            fprintf(stderr, "cohort-size out of range (1..2000000000)\n"); return 2;
        }
        mem_words = (int)cohort_size;
        if (print_count < 0) print_count = 16;   // sanity-print first 16 only
    }
    if (print_count < 0) print_count = mem_words;
    if (print_count > mem_words) print_count = mem_words;
    // --mem-words small-scale cap only applies when not using --cohort-size.
    if (cohort_size == 0 && (mem_words < 1 || mem_words > (1 << 28))) {
        fprintf(stderr, "mem-words out of range (1..2^28)\n"); return 2;
    }
    if (threads < 1 || threads > 1024) {
        fprintf(stderr, "threads out of range\n"); return 2;
    }
    if (n_streams < 0 || n_streams > 32) {
        fprintf(stderr, "streams out of range (0..32)\n"); return 2;
    }
    if (n_chunks < 0) { fprintf(stderr, "chunks must be >= 0\n"); return 2; }

    size_t img_len = 0;
    unsigned char *img = read_all(path, &img_len);
    if (!img) { emit_status("not_run", "image_read_failed", "read_all", 0); return 1; }

    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) { emit_status("not_run", "cuda_driver_missing", "dlopen", 0); free(img); return 1; }

#define LOAD_SYM(name) name##_t name = (name##_t)dlsym(lib, #name); \
    if (!name) { emit_status("not_run", "cuda_symbol_missing", #name, 0); free(img); return 1; }

    LOAD_SYM(cuInit);
    LOAD_SYM(cuDriverGetVersion);
    LOAD_SYM(cuDeviceGetCount);
    LOAD_SYM(cuDeviceGet);
    LOAD_SYM(cuDeviceGetName);
    LOAD_SYM(cuDeviceComputeCapability);
    LOAD_SYM(cuCtxCreate);
    LOAD_SYM(cuCtxDestroy);
    LOAD_SYM(cuCtxSynchronize);
    LOAD_SYM(cuModuleLoadData);
    LOAD_SYM(cuModuleLoadDataEx);
    LOAD_SYM(cuModuleGetFunction);
    LOAD_SYM(cuModuleUnload);
    LOAD_SYM(cuLaunchKernel);
    LOAD_SYM(cuMemAlloc);
    LOAD_SYM(cuMemFree);
    LOAD_SYM(cuMemcpyHtoD);
    LOAD_SYM(cuMemcpyDtoH);
    LOAD_SYM(cuMemsetD8);
#undef LOAD_SYM

    int rc;
    rc = cuInit(0);
    if (rc != 0) { emit_status("not_run", "cuInit_failed", "cuInit", rc); free(img); return 1; }

    int dev_count = 0;
    rc = cuDeviceGetCount(&dev_count);
    if (rc != 0 || dev_count <= 0) { emit_status("not_run", "no_cuda_device", "cuDeviceGetCount", rc); free(img); return 1; }

    CUdevice dev;
    rc = cuDeviceGet(&dev, 0);
    if (rc != 0) { emit_status("not_run", "cuDeviceGet_failed", "cuDeviceGet", rc); free(img); return 1; }

    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name) - 1, dev);
    int cc_major = 0, cc_minor = 0;
    cuDeviceComputeCapability(&cc_major, &cc_minor, dev);

    CUcontext ctx;
    rc = cuCtxCreate(&ctx, 0, dev);
    if (rc != 0) { emit_status("fail", "cuCtxCreate_failed", "cuCtxCreate", rc); free(img); return 1; }

    CUmodule mod;
    // Try cuModuleLoadDataEx with JIT log to get detailed PTX errors.
    enum { CU_JIT_ERROR_LOG_BUFFER = 5, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6, CU_JIT_TARGET = 8 };
    char jit_log[8192];
    void *opt_vals[3];
    int opt_keys[3];
    opt_keys[0] = CU_JIT_ERROR_LOG_BUFFER;        opt_vals[0] = jit_log;
    opt_keys[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; opt_vals[1] = (void *)(size_t)sizeof(jit_log);
    opt_keys[2] = CU_JIT_TARGET;                  opt_vals[2] = (void *)(size_t)((cc_major << 4) | cc_minor);
    jit_log[0] = '\0';
    rc = cuModuleLoadDataEx(&mod, img, 3, opt_keys, opt_vals);
    if (rc != 0) {
        fprintf(stderr, "cuModuleLoadDataEx error log: %s\n", jit_log);
        emit_status("fail", "cuModuleLoadData_rejected", "cuModuleLoadDataEx", rc);
        cuCtxDestroy(ctx); free(img); return 1;
    }

    CUfunction fn;
    rc = cuModuleGetFunction(&fn, mod, kname);
    if (rc != 0) { emit_status("fail", "cuModuleGetFunction_rejected", "cuModuleGetFunction", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

    size_t elem = value_type == 1 ? sizeof(float) : (value_type == 2 ? sizeof(double) : (value_type == 3 ? sizeof(int32_t) : sizeof(int64_t)));
    // Phase W: when streamed, pad allocation so chunk launches with grid =
    // ceil(chunk_words / threads) never overshoot device buffers. Digest is
    // taken only over the first cohort_size words; the trailing pad slots are
    // ignored. With non-streamed runs alloc_words == mem_words.
    long alloc_words = mem_words;
    if (cohort_size > 0 && n_streams >= 1 && threads > 0) {
        alloc_words = ((long)mem_words + threads - 1) / threads * threads;
    }
    size_t bytes = (size_t)alloc_words * elem;
    size_t cohort_bytes = (size_t)mem_words * elem;  // for digest scoping
    // Phase W.1: when streamed, we route inputs through pinned host memory
    // and do async H2D per chunk inside the stream loop instead of one big
    // sync H2D up front. This is the only way streams genuinely overlap
    // with H2D/D2H.
    int phase_w1_async = (cohort_size > 0 && n_streams >= 1);
    // Phase Y: GUM mode requires single-launch (no streaming).
    if (gum_mode && phase_w1_async) {
        fprintf(stderr, "error: --gum is incompatible with --streams (use single-launch)\n");
        cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 2;
    }
    CUdeviceptr d_mem = 0, d_var = 0;
    CUdeviceptr d_v11 = 0, d_v22 = 0;  // Phase Y GUM extra buffers
    rc = cuMemAlloc(&d_mem, bytes);
    if (rc != 0) { emit_status("fail", "cuMemAlloc_failed", "cuMemAlloc(mem)", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
    cuMemsetD8(d_mem, 0, bytes);

    if (!phase_w1_async && init_mem_file) {
        FILE *f = fopen(init_mem_file, "rb");
        if (!f) { emit_status("fail", "init_file_open_failed", "fopen(init_mem)", 0); cuMemFree(d_mem); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        void *host = calloc(1, bytes);  // zeroed; pad slots (alloc_words - mem_words) stay 0
        if (!host) { fclose(f); emit_status("fail", "init_file_malloc_failed", "malloc(init_mem)", 0); cuMemFree(d_mem); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        size_t got = fread(host, 1, cohort_bytes, f);
        fclose(f);
        if (got != cohort_bytes) { free(host); emit_status("fail", "init_file_short_read", "fread(init_mem)", (int)got); cuMemFree(d_mem); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        cuMemcpyHtoD(d_mem, host, bytes);
        free(host);
    } else if (!phase_w1_async && init_mem_csv) {
        if (value_type == 1) {
            float *host = (float *)calloc(mem_words, sizeof(float));
            parse_csv_f32(init_mem_csv, host, mem_words);
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        } else if (value_type == 2) {
            double *host = (double *)calloc(mem_words, sizeof(double));
            parse_csv_f64(init_mem_csv, host, mem_words);
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        } else if (value_type == 3) {
            int32_t *host = (int32_t *)calloc(mem_words, sizeof(int32_t));
            parse_csv_i32(init_mem_csv, host, mem_words);
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        } else {
            int64_t *host = (int64_t *)calloc(mem_words, sizeof(int64_t));
            parse_csv_i64(init_mem_csv, host, mem_words);
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        }
    } else if (!phase_w1_async && verify_init_seq) {
        if (value_type == 1) {
            float *host = (float *)calloc(mem_words, sizeof(float));
            for (int k = 0; k < mem_words; k++) host[k] = (float)(k + 1);
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        } else if (value_type == 2) {
            double *host = (double *)calloc(mem_words, sizeof(double));
            for (int k = 0; k < mem_words; k++) host[k] = (double)(k + 1);
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        } else if (value_type == 3) {
            int32_t *host = (int32_t *)calloc(mem_words, sizeof(int32_t));
            for (int k = 0; k < mem_words; k++) host[k] = k + 1;
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        } else {
            int64_t *host = (int64_t *)calloc(mem_words, sizeof(int64_t));
            for (int k = 0; k < mem_words; k++) host[k] = k + 1;
            cuMemcpyHtoD(d_mem, host, bytes);
            free(host);
        }
    }

    if (epistemic) {
        rc = cuMemAlloc(&d_var, bytes);
        if (rc != 0) { emit_status("fail", "cuMemAlloc_failed", "cuMemAlloc(var)", rc); cuMemFree(d_mem); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        cuMemsetD8(d_var, 0, bytes);
        if (!phase_w1_async && init_var_file) {
            FILE *f = fopen(init_var_file, "rb");
            if (!f) { emit_status("fail", "init_file_open_failed", "fopen(init_var)", 0); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            void *host = calloc(1, bytes);  // pad slots stay 0
            if (!host) { fclose(f); emit_status("fail", "init_file_malloc_failed", "malloc(init_var)", 0); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            size_t got = fread(host, 1, cohort_bytes, f);
            fclose(f);
            if (got != cohort_bytes) { free(host); emit_status("fail", "init_file_short_read", "fread(init_var)", (int)got); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            cuMemcpyHtoD(d_var, host, bytes);
            free(host);
        } else if (!phase_w1_async && init_var_csv) {
            if (value_type == 1) {
                float *host = (float *)calloc(mem_words, sizeof(float));
                parse_csv_f32(init_var_csv, host, mem_words);
                cuMemcpyHtoD(d_var, host, bytes);
                free(host);
            } else if (value_type == 2) {
                double *host = (double *)calloc(mem_words, sizeof(double));
                parse_csv_f64(init_var_csv, host, mem_words);
                cuMemcpyHtoD(d_var, host, bytes);
                free(host);
            } else if (value_type == 3) {
                int32_t *host = (int32_t *)calloc(mem_words, sizeof(int32_t));
                parse_csv_i32(init_var_csv, host, mem_words);
                cuMemcpyHtoD(d_var, host, bytes);
                free(host);
            } else {
                int64_t *host = (int64_t *)calloc(mem_words, sizeof(int64_t));
                parse_csv_i64(init_var_csv, host, mem_words);
                cuMemcpyHtoD(d_var, host, bytes);
                free(host);
            }
        }
    }

    // Phase Y: allocate and load V11/V22 GUM buffers.
    if (gum_mode) {
        rc = cuMemAlloc(&d_v11, bytes);
        if (rc != 0) { emit_status("fail","cuMemAlloc_failed","cuMemAlloc(v11)",rc); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        cuMemsetD8(d_v11, 0, bytes);
        rc = cuMemAlloc(&d_v22, bytes);
        if (rc != 0) { emit_status("fail","cuMemAlloc_failed","cuMemAlloc(v22)",rc); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        cuMemsetD8(d_v22, 0, bytes);

        // Load V11 init file
        if (init_v11_file) {
            FILE *f = fopen(init_v11_file, "rb");
            if (!f) { emit_status("fail","init_file_open_failed","fopen(init_v11)",0); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            void *host = calloc(1, bytes);
            if (!host) { fclose(f); emit_status("fail","init_file_malloc_failed","malloc(v11)",0); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            size_t got = fread(host, 1, cohort_bytes, f); fclose(f);
            if (got != cohort_bytes) { free(host); emit_status("fail","init_file_short_read","fread(v11)",(int)got); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            cuMemcpyHtoD(d_v11, host, bytes); free(host);
        }
        // Load V22 init file
        if (init_v22_file) {
            FILE *f = fopen(init_v22_file, "rb");
            if (!f) { emit_status("fail","init_file_open_failed","fopen(init_v22)",0); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            void *host = calloc(1, bytes);
            if (!host) { fclose(f); emit_status("fail","init_file_malloc_failed","malloc(v22)",0); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            size_t got = fread(host, 1, cohort_bytes, f); fclose(f);
            if (got != cohort_bytes) { free(host); emit_status("fail","init_file_short_read","fread(v22)",(int)got); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            cuMemcpyHtoD(d_v22, host, bytes); free(host);
        }
    }

    void *args[4];
    args[0] = &d_mem;
    args[1] = epistemic ? (void *)&d_var : NULL;
    args[2] = gum_mode ? (void *)&d_v11 : NULL;
    args[3] = gum_mode ? (void *)&d_v22 : NULL;

    // h_mem / h_var allocation strategy:
    //   Phase V (single-launch): plain calloc, freed with free().
    //   Phase W.1 (streamed): pinned via cuMemHostAlloc, freed with cuMemFreeHost.
    //   The pinned path also reads init files DIRECTLY into the pinned region,
    //   so no separate sync H2D up front — the stream loop does H2D async.
    void *h_mem = NULL;
    void *h_var = NULL;
    long stream_wall_us = 0;
    long stream_chunks_run = 0;
    cuMemHostAlloc_t cuMemHostAlloc = NULL;
    cuMemFreeHost_t  cuMemFreeHost  = NULL;
    cuMemcpyHtoDAsync_t cuMemcpyHtoDAsync = NULL;

    if (cohort_size > 0 && n_streams >= 1) {
        // ---- Phase W.1: streamed multi-launch w/ pinned-host async H2D ----
        cuStreamCreate_t cuStreamCreate = (cuStreamCreate_t)dlsym(lib, "cuStreamCreate");
        cuStreamDestroy_t cuStreamDestroy = (cuStreamDestroy_t)dlsym(lib, "cuStreamDestroy");
        cuStreamSynchronize_t cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(lib, "cuStreamSynchronize");
        cuMemcpyDtoHAsync_t cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_t)dlsym(lib, "cuMemcpyDtoHAsync");
        cuMemHostAlloc = (cuMemHostAlloc_t)dlsym(lib, "cuMemHostAlloc");
        cuMemFreeHost  = (cuMemFreeHost_t) dlsym(lib, "cuMemFreeHost");
        cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_t)dlsym(lib, "cuMemcpyHtoDAsync");
        if (!cuStreamCreate || !cuStreamDestroy || !cuStreamSynchronize || !cuMemcpyDtoHAsync
            || !cuMemHostAlloc || !cuMemFreeHost || !cuMemcpyHtoDAsync) {
            emit_status("fail", "cuda_stream_symbols_missing", "dlsym(streams+pinned)", 0);
            cuMemFree(d_mem); if (d_var) cuMemFree(d_var);
            cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1;
        }
        // Allocate pinned host I/O buffers.
        rc = cuMemHostAlloc(&h_mem, bytes, 0);
        if (rc != 0) {
            emit_status("fail", "cuMemHostAlloc_failed", "cuMemHostAlloc(mem)", rc);
            cuMemFree(d_mem); if (d_var) cuMemFree(d_var);
            cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1;
        }
        memset(h_mem, 0, bytes);
        if (epistemic) {
            rc = cuMemHostAlloc(&h_var, bytes, 0);
            if (rc != 0) {
                emit_status("fail", "cuMemHostAlloc_failed", "cuMemHostAlloc(var)", rc);
                cuMemFreeHost(h_mem); cuMemFree(d_mem); if (d_var) cuMemFree(d_var);
                cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1;
            }
            memset(h_var, 0, bytes);
        }
        // Read init files DIRECTLY into pinned buffers (no extra copy).
        if (init_mem_file) {
            FILE *f = fopen(init_mem_file, "rb");
            if (!f) { emit_status("fail", "init_file_open_failed", "fopen(init_mem)", 0); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            size_t got = fread(h_mem, 1, cohort_bytes, f);
            fclose(f);
            if (got != cohort_bytes) { emit_status("fail", "init_file_short_read", "fread(init_mem)", (int)got); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        }
        if (epistemic && init_var_file) {
            FILE *f = fopen(init_var_file, "rb");
            if (!f) { emit_status("fail", "init_file_open_failed", "fopen(init_var)", 0); cuMemFreeHost(h_mem); cuMemFreeHost(h_var); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            size_t got = fread(h_var, 1, cohort_bytes, f);
            fclose(f);
            if (got != cohort_bytes) { emit_status("fail", "init_file_short_read", "fread(init_var)", (int)got); cuMemFreeHost(h_mem); cuMemFreeHost(h_var); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        }
        int chunks_req = (n_chunks > 0) ? n_chunks : (n_streams * 4);
        if (chunks_req > mem_words) chunks_req = mem_words;
        // Round chunk_words up to a multiple of threads so adjacent chunks
        // never overlap (otherwise a chunk's grid = ceil(chunk_words/threads)
        // overshoots into the next chunk's slice and races other streams).
        long chunk_words = ((long)mem_words + chunks_req - 1) / chunks_req;
        chunk_words = (chunk_words + threads - 1) / threads * threads;
        if (chunk_words < (long)threads) chunk_words = threads;
        long chunks = (alloc_words + chunk_words - 1) / chunk_words;

        CUstream *streams = (CUstream *)calloc((size_t)n_streams, sizeof(CUstream));
        for (int s = 0; s < n_streams; s++) {
            rc = cuStreamCreate(&streams[s], 0);
            if (rc != 0) { emit_status("fail", "cuStreamCreate_failed", "cuStreamCreate", rc); free(streams); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long c = 0; c < chunks; c++) {
            int s = (int)(c % n_streams);
            long off_words = c * chunk_words;
            if (off_words >= alloc_words) break;
            // Kernel grid = chunk_words/threads (chunk_words is threads-aligned),
            // so the launch writes exactly chunk_words slots starting at off_words.
            // The D2H mirrors that range; the pad region between mem_words and
            // alloc_words absorbs the last chunk's overshoot.
            long copy_words = chunk_words;
            if (off_words + copy_words > alloc_words) copy_words = alloc_words - off_words;
            size_t off_bytes = (size_t)off_words * elem;
            size_t this_bytes = (size_t)copy_words * elem;
            unsigned blocks_this = (unsigned)(chunk_words / threads);

            CUdeviceptr d_mem_c = d_mem + off_bytes;
            CUdeviceptr d_var_c = epistemic ? (d_var + off_bytes) : 0;
            void *args_c[2]; args_c[0] = &d_mem_c; args_c[1] = epistemic ? (void *)&d_var_c : NULL;

            // Phase W.1: per-chunk async H2D from pinned host. Stream s
            // serializes (H2D s_c → launch s_c → D2H s_c) but stream s+1
            // can run its H2D concurrently on a separate copy engine.
            rc = cuMemcpyHtoDAsync(d_mem_c, (const char *)h_mem + off_bytes, this_bytes, streams[s]);
            if (rc != 0) { emit_status("fail", "cuMemcpyHtoDAsync_failed", "cuMemcpyHtoDAsync(mem)", rc); for (int x = 0; x < n_streams; x++) cuStreamDestroy(streams[x]); free(streams); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            if (epistemic) {
                rc = cuMemcpyHtoDAsync(d_var_c, (const char *)h_var + off_bytes, this_bytes, streams[s]);
                if (rc != 0) { emit_status("fail", "cuMemcpyHtoDAsync_failed", "cuMemcpyHtoDAsync(var)", rc); for (int x = 0; x < n_streams; x++) cuStreamDestroy(streams[x]); free(streams); cuMemFreeHost(h_mem); cuMemFreeHost(h_var); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            }

            rc = cuLaunchKernel(fn, blocks_this, 1, 1, threads, 1, 1, 0, (void *)streams[s], args_c, NULL);
            if (rc != 0) { emit_status("fail", "cuLaunchKernel_rejected", "cuLaunchKernel(streamed)", rc); for (int x = 0; x < n_streams; x++) cuStreamDestroy(streams[x]); free(streams); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

            rc = cuMemcpyDtoHAsync((char *)h_mem + off_bytes, d_mem_c, this_bytes, streams[s]);
            if (rc != 0) { emit_status("fail", "cuMemcpyDtoHAsync_failed", "cuMemcpyDtoHAsync(mem)", rc); for (int x = 0; x < n_streams; x++) cuStreamDestroy(streams[x]); free(streams); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            if (epistemic) {
                rc = cuMemcpyDtoHAsync((char *)h_var + off_bytes, d_var_c, this_bytes, streams[s]);
                if (rc != 0) { emit_status("fail", "cuMemcpyDtoHAsync_failed", "cuMemcpyDtoHAsync(var)", rc); for (int x = 0; x < n_streams; x++) cuStreamDestroy(streams[x]); free(streams); cuMemFreeHost(h_mem); cuMemFreeHost(h_var); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            }
            stream_chunks_run++;
        }
        for (int s = 0; s < n_streams; s++) {
            rc = cuStreamSynchronize(streams[s]);
            if (rc != 0) { emit_status("fail", "cuStreamSynchronize_failed", "cuStreamSynchronize", rc); for (int x = 0; x < n_streams; x++) cuStreamDestroy(streams[x]); free(streams); cuMemFreeHost(h_mem); if (h_var) cuMemFreeHost(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        stream_wall_us = (long)((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000L);

        for (int s = 0; s < n_streams; s++) cuStreamDestroy(streams[s]);
        free(streams);
    } else {
        // ---- Phase V (and earlier): single default-stream launch ----
        h_mem = calloc((size_t)alloc_words, elem);
        h_var = epistemic ? calloc((size_t)alloc_words, elem) : NULL;
        if (!h_mem || (epistemic && !h_var)) {
            emit_status("fail", "host_alloc_failed", "calloc(h_mem|h_var)", 0);
            if (h_mem) free(h_mem);
            if (h_var) free(h_var);
            cuMemFree(d_mem); if (d_var) cuMemFree(d_var);
            if (d_v11) cuMemFree(d_v11);
            if (d_v22) cuMemFree(d_v22);
            cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1;
        }
        // Phase Y GUM: host read-back buffers for V11 and V22 final state.
        void *h_v11 = gum_mode ? calloc((size_t)alloc_words, elem) : NULL;
        void *h_v22 = gum_mode ? calloc((size_t)alloc_words, elem) : NULL;
        if (gum_mode && (!h_v11 || !h_v22)) {
            emit_status("fail","host_alloc_failed","calloc(h_v11|h_v22)",0);
            free(h_mem); if (h_var) free(h_var); if (h_v11) free(h_v11); if (h_v22) free(h_v22);
            cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22);
            cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1;
        }

        rc = cuLaunchKernel(fn,
            /*grid*/ blocks, 1, 1,
            /*block*/ threads, 1, 1,
            /*shmem*/ 0,
            /*stream*/ NULL,
            args, NULL);  // kernel reads args[0..1] or args[0..3] per its param count
        if (rc != 0) { emit_status("fail", "cuLaunchKernel_rejected", "cuLaunchKernel", rc); free(h_mem); if (h_var) free(h_var); if (h_v11) free(h_v11); if (h_v22) free(h_v22); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); if (d_v11) cuMemFree(d_v11); if (d_v22) cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

        rc = cuCtxSynchronize();
        if (rc != 0) { emit_status("fail", "cuCtxSynchronize_failed", "cuCtxSynchronize", rc); free(h_mem); if (h_var) free(h_var); if (h_v11) free(h_v11); if (h_v22) free(h_v22); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); if (d_v11) cuMemFree(d_v11); if (d_v22) cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

        rc = cuMemcpyDtoH(h_mem, d_mem, bytes);
        if (rc != 0) { emit_status("fail", "cuMemcpyDtoH_failed", "cuMemcpyDtoH(mem)", rc); free(h_mem); if (h_var) free(h_var); if (h_v11) free(h_v11); if (h_v22) free(h_v22); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); if (d_v11) cuMemFree(d_v11); if (d_v22) cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        if (epistemic) {
            rc = cuMemcpyDtoH(h_var, d_var, bytes);
            if (rc != 0) { emit_status("fail", "cuMemcpyDtoH_failed", "cuMemcpyDtoH(var)", rc); free(h_mem); free(h_var); if (h_v11) free(h_v11); if (h_v22) free(h_v22); cuMemFree(d_mem); cuMemFree(d_var); if (d_v11) cuMemFree(d_v11); if (d_v22) cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        }
        if (gum_mode) {
            rc = cuMemcpyDtoH(h_v11, d_v11, bytes);
            if (rc != 0) { emit_status("fail","cuMemcpyDtoH_failed","cuMemcpyDtoH(v11)",rc); free(h_mem); free(h_var); free(h_v11); free(h_v22); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
            rc = cuMemcpyDtoH(h_v22, d_v22, bytes);
            if (rc != 0) { emit_status("fail","cuMemcpyDtoH_failed","cuMemcpyDtoH(v22)",rc); free(h_mem); free(h_var); free(h_v11); free(h_v22); cuMemFree(d_mem); cuMemFree(d_var); cuMemFree(d_v11); cuMemFree(d_v22); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        }

        // Phase Y: emit GUM stats before freeing.
        if (gum_mode && cohort_size > 0 && value_type == 1) {
            const float *c1f = (const float *)h_mem;
            const float *v11f = (const float *)h_v11;
            uint64_t c1_dig  = fnv1a64(h_mem,  cohort_bytes);
            uint64_t v11_dig = fnv1a64(h_v11, cohort_bytes);
            double sum_v11 = 0.0;
            for (long i = 0; i < (long)cohort_size; i++) sum_v11 += (double)v11f[i];
            double mean_v11 = sum_v11 / (double)cohort_size;
            double mean_u_b = sqrt(mean_v11 > 0.0 ? mean_v11 : 0.0);
            printf("PHY cohort=%ld c1_digest=%016llx v11_digest=%016llx "
                   "mean_u_b=%.6g u95_k2=%.6g\n",
                   (long)cohort_size,
                   (unsigned long long)c1_dig, (unsigned long long)v11_dig,
                   mean_u_b, 2.0 * mean_u_b);
            (void)c1f;  // digest covers it
        }

        free(h_v11); free(h_v22);
    }

    emit_status("pass", "launch_pass", "cuMemcpyDtoH", 0);

    printf("MEM:");
    if (value_type == 1) {
        float *m = (float *)h_mem;
        for (int i = 0; i < print_count; i++) printf(" %.6g", m[i]);
    } else if (value_type == 2) {
        double *m = (double *)h_mem;
        for (int i = 0; i < print_count; i++) printf(" %.12g", m[i]);
    } else if (value_type == 3) {
        int32_t *m = (int32_t *)h_mem;
        for (int i = 0; i < print_count; i++) printf(" %d", m[i]);
    } else {
        int64_t *m = (int64_t *)h_mem;
        for (int i = 0; i < print_count; i++) printf(" %lld", (long long)m[i]);
    }
    printf("\n");
    if (epistemic) {
        printf("VAR:");
        if (value_type == 1) {
            float *v = (float *)h_var;
            for (int i = 0; i < print_count; i++) printf(" %.6g", v[i]);
        } else if (value_type == 2) {
            double *v = (double *)h_var;
            for (int i = 0; i < print_count; i++) printf(" %.12g", v[i]);
        } else if (value_type == 3) {
            int32_t *v = (int32_t *)h_var;
            for (int i = 0; i < print_count; i++) printf(" %d", v[i]);
        } else {
            int64_t *v = (int64_t *)h_var;
            for (int i = 0; i < print_count; i++) printf(" %lld", (long long)v[i]);
        }
        printf("\n");
    }
    printf("device=%s cc=%d.%d\n", name, cc_major, cc_minor);

    if (cohort_size > 0) {
        // Digest only the cohort range; ignore thread-alignment padding.
        uint64_t mem_digest = fnv1a64(h_mem, cohort_bytes);
        uint64_t var_digest = epistemic ? fnv1a64(h_var, cohort_bytes) : 0;
        // Phase X.1: count NaN sentinels (f32 quiet-NaN = 0x7FC00000) in the
        // GPU's output buffer for the f32 dialect. This is the runtime count
        // of patients the GPU's compute-then-gate kernel marked as outside
        // budget. The complement (cohort - nan_count) is the in-budget count.
        // For other dialects the counts are reported as -1 (n/a).
        long nan_count = -1, in_budget = -1;
        if (value_type == 1) {
            const uint32_t SENTINEL = 0x7FC00000u;
            const uint32_t *m = (const uint32_t *)h_mem;
            long nc = 0;
            for (long i = 0; i < (long)cohort_size; i++) {
                if (m[i] == SENTINEL) nc++;
            }
            nan_count = nc;
            in_budget = (long)cohort_size - nc;
        }
        // Phase X: optional therapeutic-window classification on f32 output.
        long in_window = -1, out_of_window = -1;
        if (classify_window && value_type == 1) {
            const float *m = (const float *)h_mem;
            long iw = 0;
            for (long i = 0; i < (long)cohort_size; i++) {
                if (m[i] >= classify_low && m[i] <= classify_high) iw++;
            }
            in_window = iw;
            out_of_window = (long)cohort_size - iw;
        }
        printf("PHW cohort=%ld streams=%d chunks_run=%ld wall_us=%ld "
               "mem_digest=%016llx var_digest=%016llx "
               "nan_count=%ld in_budget=%ld "
               "in_window=%ld out_of_window=%ld\n",
               (long)cohort_size, n_streams, stream_chunks_run, stream_wall_us,
               (unsigned long long)mem_digest, (unsigned long long)var_digest,
               nan_count, in_budget, in_window, out_of_window);
    }

    if (phase_w1_async) {
        if (h_mem && cuMemFreeHost) cuMemFreeHost(h_mem);
        if (h_var && cuMemFreeHost) cuMemFreeHost(h_var);
    } else {
        if (h_mem) free(h_mem);
        if (h_var) free(h_var);
    }
    cuMemFree(d_mem); if (d_var) cuMemFree(d_var);
    if (d_v11) cuMemFree(d_v11);
    if (d_v22) cuMemFree(d_v22);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(img);
    return 0;
}
