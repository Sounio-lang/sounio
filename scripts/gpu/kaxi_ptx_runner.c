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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <ptx_or_cubin> [--kernel NAME] [--mode basic|epistemic]\n", argv[0]);
        fprintf(stderr, "                          [--threads T] [--mem-words W]\n");
        fprintf(stderr, "                          [--init-mem v0,v1,...] [--init-var v0,v1,...]\n");
        return 2;
    }

    const char *path = argv[1];
    const char *kname = "kaxi_kernel";
    int epistemic = 0;
    unsigned int threads = 1;
    int mem_words = 16;
    const char *init_mem_csv = NULL;
    const char *init_var_csv = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) { kname = argv[++i]; }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++; epistemic = (strcmp(argv[i], "epistemic") == 0);
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) { threads = (unsigned)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--mem-words") == 0 && i + 1 < argc) { mem_words = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--init-mem") == 0 && i + 1 < argc) { init_mem_csv = argv[++i]; }
        else if (strcmp(argv[i], "--init-var") == 0 && i + 1 < argc) { init_var_csv = argv[++i]; }
        else if (strcmp(argv[i], "--epistemic") == 0) { epistemic = 1; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (mem_words < 1 || mem_words > 1024) {
        fprintf(stderr, "mem-words out of range\n"); return 2;
    }
    if (threads < 1 || threads > 1024) {
        fprintf(stderr, "threads out of range\n"); return 2;
    }

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
    rc = cuModuleLoadData(&mod, img);
    if (rc != 0) { emit_status("fail", "cuModuleLoadData_rejected", "cuModuleLoadData", rc); cuCtxDestroy(ctx); free(img); return 1; }

    CUfunction fn;
    rc = cuModuleGetFunction(&fn, mod, kname);
    if (rc != 0) { emit_status("fail", "cuModuleGetFunction_rejected", "cuModuleGetFunction", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

    size_t bytes = (size_t)mem_words * sizeof(int64_t);
    CUdeviceptr d_mem = 0, d_var = 0;
    rc = cuMemAlloc(&d_mem, bytes);
    if (rc != 0) { emit_status("fail", "cuMemAlloc_failed", "cuMemAlloc(mem)", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
    cuMemsetD8(d_mem, 0, bytes);

    if (init_mem_csv) {
        int64_t *host = (int64_t *)calloc(mem_words, sizeof(int64_t));
        parse_csv_i64(init_mem_csv, host, mem_words);
        cuMemcpyHtoD(d_mem, host, bytes);
        free(host);
    }

    if (epistemic) {
        rc = cuMemAlloc(&d_var, bytes);
        if (rc != 0) { emit_status("fail", "cuMemAlloc_failed", "cuMemAlloc(var)", rc); cuMemFree(d_mem); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
        cuMemsetD8(d_var, 0, bytes);
        if (init_var_csv) {
            int64_t *host = (int64_t *)calloc(mem_words, sizeof(int64_t));
            parse_csv_i64(init_var_csv, host, mem_words);
            cuMemcpyHtoD(d_var, host, bytes);
            free(host);
        }
    }

    void *args[2];
    args[0] = &d_mem;
    args[1] = epistemic ? (void *)&d_var : NULL;

    rc = cuLaunchKernel(fn,
        /*grid*/ 1, 1, 1,
        /*block*/ threads, 1, 1,
        /*shmem*/ 0,
        /*stream*/ NULL,
        args, NULL);
    if (rc != 0) { emit_status("fail", "cuLaunchKernel_rejected", "cuLaunchKernel", rc); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

    rc = cuCtxSynchronize();
    if (rc != 0) { emit_status("fail", "cuCtxSynchronize_failed", "cuCtxSynchronize", rc); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }

    int64_t *h_mem = (int64_t *)calloc(mem_words, sizeof(int64_t));
    int64_t *h_var = epistemic ? (int64_t *)calloc(mem_words, sizeof(int64_t)) : NULL;
    rc = cuMemcpyDtoH(h_mem, d_mem, bytes);
    if (rc != 0) { emit_status("fail", "cuMemcpyDtoH_failed", "cuMemcpyDtoH(mem)", rc); free(h_mem); if (h_var) free(h_var); cuMemFree(d_mem); if (d_var) cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
    if (epistemic) {
        rc = cuMemcpyDtoH(h_var, d_var, bytes);
        if (rc != 0) { emit_status("fail", "cuMemcpyDtoH_failed", "cuMemcpyDtoH(var)", rc); free(h_mem); free(h_var); cuMemFree(d_mem); cuMemFree(d_var); cuModuleUnload(mod); cuCtxDestroy(ctx); free(img); return 1; }
    }

    emit_status("pass", "launch_pass", "cuMemcpyDtoH", 0);

    printf("MEM:");
    for (int i = 0; i < mem_words; i++) printf(" %lld", (long long)h_mem[i]);
    printf("\n");
    if (epistemic) {
        printf("VAR:");
        for (int i = 0; i < mem_words; i++) printf(" %lld", (long long)h_var[i]);
        printf("\n");
    }
    printf("device=%s cc=%d.%d\n", name, cc_major, cc_minor);

    free(h_mem); if (h_var) free(h_var);
    cuMemFree(d_mem); if (d_var) cuMemFree(d_var);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(img);
    return 0;
}
