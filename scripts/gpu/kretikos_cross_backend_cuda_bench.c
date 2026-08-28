// CUDA Driver API microbenchmark for Kretikos cross-backend PTX-side artifacts.
//
// This measures the same CUBINs validated by
// kretikos_cross_backend_cuda_runtime_gate.sh. It intentionally reports a
// narrow kernel-level timing, not an end-to-end application benchmark.

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*cuDeviceComputeCapability_t)(int *, int *, CUdevice);
typedef CUresult (*cuDriverGetVersion_t)(int *);
typedef CUresult (*cuCtxCreate_t)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuModuleLoad_t)(CUmodule *, const char *);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemsetD32_t)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, void *, void **, void **);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuEventCreate_t)(CUevent *, unsigned int);
typedef CUresult (*cuEventDestroy_t)(CUevent);
typedef CUresult (*cuEventRecord_t)(CUevent, void *);
typedef CUresult (*cuEventSynchronize_t)(CUevent);
typedef CUresult (*cuEventElapsedTime_t)(float *, CUevent, CUevent);

struct CudaApi {
    void *lib;
    cuInit_t cuInit;
    cuDeviceGet_t cuDeviceGet;
    cuDeviceGetName_t cuDeviceGetName;
    cuDeviceComputeCapability_t cuDeviceComputeCapability;
    cuDriverGetVersion_t cuDriverGetVersion;
    cuCtxCreate_t cuCtxCreate;
    cuCtxDestroy_t cuCtxDestroy;
    cuModuleLoad_t cuModuleLoad;
    cuModuleUnload_t cuModuleUnload;
    cuModuleGetFunction_t cuModuleGetFunction;
    cuMemAlloc_t cuMemAlloc;
    cuMemFree_t cuMemFree;
    cuMemcpyHtoD_t cuMemcpyHtoD;
    cuMemcpyDtoH_t cuMemcpyDtoH;
    cuMemsetD32_t cuMemsetD32;
    cuLaunchKernel_t cuLaunchKernel;
    cuCtxSynchronize_t cuCtxSynchronize;
    cuEventCreate_t cuEventCreate;
    cuEventDestroy_t cuEventDestroy;
    cuEventRecord_t cuEventRecord;
    cuEventSynchronize_t cuEventSynchronize;
    cuEventElapsedTime_t cuEventElapsedTime;
};

#define LOAD_CUDA(name) do { \
    api->name = (name##_t)dlsym(api->lib, #name); \
    if (!api->name) { fprintf(stderr, "missing CUDA symbol: %s\n", #name); return 1; } \
} while (0)

static int load_cuda(struct CudaApi *api) {
    memset(api, 0, sizeof(*api));
    api->lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!api->lib) api->lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!api->lib) {
        fprintf(stderr, "failed to load libcuda: %s\n", dlerror());
        return 1;
    }
    LOAD_CUDA(cuInit);
    LOAD_CUDA(cuDeviceGet);
    LOAD_CUDA(cuDeviceGetName);
    LOAD_CUDA(cuDeviceComputeCapability);
    LOAD_CUDA(cuDriverGetVersion);
    LOAD_CUDA(cuCtxCreate);
    LOAD_CUDA(cuCtxDestroy);
    LOAD_CUDA(cuModuleLoad);
    LOAD_CUDA(cuModuleUnload);
    LOAD_CUDA(cuModuleGetFunction);
    LOAD_CUDA(cuMemAlloc);
    LOAD_CUDA(cuMemFree);
    LOAD_CUDA(cuMemcpyHtoD);
    LOAD_CUDA(cuMemcpyDtoH);
    LOAD_CUDA(cuMemsetD32);
    LOAD_CUDA(cuLaunchKernel);
    LOAD_CUDA(cuCtxSynchronize);
    LOAD_CUDA(cuEventCreate);
    LOAD_CUDA(cuEventDestroy);
    LOAD_CUDA(cuEventRecord);
    LOAD_CUDA(cuEventSynchronize);
    LOAD_CUDA(cuEventElapsedTime);
    return 0;
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static void sanitize_device_name(char *s) {
    for (size_t i = 0; s[i] != '\0'; i++) {
        if (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') s[i] = '_';
    }
}

static int launch_timed(struct CudaApi *api, CUfunction fn, void **params, uint32_t n, float *elapsed_ms) {
    const uint32_t block_threads = 256;
    const uint32_t grid_blocks = n / block_threads;
    CUevent start = NULL, stop = NULL;
    CUresult rc = 0;
    if ((rc = api->cuEventCreate(&start, 0)) != 0) return rc;
    if ((rc = api->cuEventCreate(&stop, 0)) != 0) return rc;
    if ((rc = api->cuEventRecord(start, NULL)) != 0) return rc;
    if ((rc = api->cuLaunchKernel(fn, grid_blocks, 1, 1, block_threads, 1, 1, 0, NULL, params, NULL)) != 0) return rc;
    if ((rc = api->cuEventRecord(stop, NULL)) != 0) return rc;
    if ((rc = api->cuEventSynchronize(stop)) != 0) return rc;
    if ((rc = api->cuEventElapsedTime(elapsed_ms, start, stop)) != 0) return rc;
    api->cuEventDestroy(stop);
    api->cuEventDestroy(start);
    return 0;
}

static int bench_vec_add(struct CudaApi *api, CUfunction fn, uint32_t n, int warmup, int iters,
                         const char *device_name, int driver_version, int cc_major, int cc_minor) {
    size_t bytes = (size_t)n * sizeof(float);
    float *a = (float *)calloc(n, sizeof(float));
    float *b = (float *)calloc(n, sizeof(float));
    float *out = (float *)calloc(n, sizeof(float));
    float *times = (float *)calloc((size_t)iters, sizeof(float));
    CUdeviceptr d_a = 0, d_b = 0, d_out = 0;
    CUresult rc = 0;
    float max_abs_err = 0.0f;
    if (!a || !b || !out || !times) return 2;
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (float)((int)(i % 17) - 8) * 0.5f;
        b[i] = (float)((int)(i % 11) - 5) * 1.25f;
    }
    if ((rc = api->cuMemAlloc(&d_a, bytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_b, bytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_out, bytes)) != 0) return rc;
    if ((rc = api->cuMemcpyHtoD(d_a, a, bytes)) != 0) return rc;
    if ((rc = api->cuMemcpyHtoD(d_b, b, bytes)) != 0) return rc;
    if ((rc = api->cuMemsetD32(d_out, 0, n)) != 0) return rc;
    void *params[] = { &d_a, &d_b, &d_out };
    for (int i = 0; i < warmup; i++) {
        float ignored = 0.0f;
        if ((rc = launch_timed(api, fn, params, n, &ignored)) != 0) return rc;
    }
    for (int i = 0; i < iters; i++) {
        if ((rc = launch_timed(api, fn, params, n, &times[i])) != 0) return rc;
    }
    if ((rc = api->cuMemcpyDtoH(out, d_out, bytes)) != 0) return rc;
    for (uint32_t i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        float err = fabsf(out[i] - expected);
        if (err > max_abs_err) max_abs_err = err;
    }
    qsort(times, (size_t)iters, sizeof(float), cmp_float);
    float median = times[iters / 2];
    float q1 = times[iters / 4];
    float q3 = times[(iters * 3) / 4];
    double gscalarops = ((double)n) / ((double)median * 1000000.0);
    printf("kretikos_cuda_bench status=pass kernel=vec_add n=%u warmup=%d iterations=%d median_ms=%.9g iqr_ms=%.9g scalar_ops_per_elem=1 gscalarops_per_s=%.9g max_abs_err=%.9g device_name=%s driver_version=%d cc=%d.%d\n",
           n, warmup, iters, median, q3 - q1, gscalarops, max_abs_err, device_name, driver_version, cc_major, cc_minor);
    api->cuMemFree(d_out); api->cuMemFree(d_b); api->cuMemFree(d_a);
    free(times); free(out); free(b); free(a);
    return max_abs_err > 0.000001f ? 1 : 0;
}

static int bench_epistemic(struct CudaApi *api, CUfunction fn, uint32_t n, int warmup, int iters,
                           const char *device_name, int driver_version, int cc_major, int cc_minor) {
    size_t fbytes = (size_t)n * sizeof(float);
    size_t ubytes = (size_t)n * sizeof(uint32_t);
    float *a_value = (float *)calloc(n, sizeof(float));
    float *b_value = (float *)calloc(n, sizeof(float));
    float *a_eps = (float *)calloc(n, sizeof(float));
    float *b_eps = (float *)calloc(n, sizeof(float));
    float *out_value = (float *)calloc(n, sizeof(float));
    float *out_eps = (float *)calloc(n, sizeof(float));
    uint32_t *a_valid = (uint32_t *)calloc(n, sizeof(uint32_t));
    uint32_t *b_valid = (uint32_t *)calloc(n, sizeof(uint32_t));
    uint32_t *out_valid = (uint32_t *)calloc(n, sizeof(uint32_t));
    uint32_t *out_prov = (uint32_t *)calloc(n, sizeof(uint32_t));
    float *times = (float *)calloc((size_t)iters, sizeof(float));
    CUdeviceptr d_av = 0, d_bv = 0, d_ae = 0, d_be = 0, d_aok = 0, d_bok = 0, d_ov = 0, d_oe = 0, d_ook = 0, d_oprov = 0;
    CUresult rc = 0;
    float value_max_abs_err = 0.0f, eps_max_abs_err = 0.0f;
    uint32_t valid_mismatch = 0, prov_mismatch = 0;
    if (!a_value || !b_value || !a_eps || !b_eps || !out_value || !out_eps || !a_valid || !b_valid || !out_valid || !out_prov || !times) return 2;
    for (uint32_t i = 0; i < n; i++) {
        a_value[i] = (float)((int)(i % 13) - 6) * 0.25f;
        b_value[i] = (float)((int)(i % 7) - 3) * 0.5f;
        a_eps[i] = 0.01f * (float)((i % 5) + 1);
        b_eps[i] = 0.02f * (float)((i % 3) + 1);
        a_valid[i] = (i % 4) == 0 ? 0U : 1U;
        b_valid[i] = (i % 6) == 0 ? 0U : 1U;
    }
    if ((rc = api->cuMemAlloc(&d_av, fbytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_bv, fbytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_ae, fbytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_be, fbytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_aok, ubytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_bok, ubytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_ov, fbytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_oe, fbytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_ook, ubytes)) != 0) return rc;
    if ((rc = api->cuMemAlloc(&d_oprov, ubytes)) != 0) return rc;
    api->cuMemcpyHtoD(d_av, a_value, fbytes); api->cuMemcpyHtoD(d_bv, b_value, fbytes);
    api->cuMemcpyHtoD(d_ae, a_eps, fbytes); api->cuMemcpyHtoD(d_be, b_eps, fbytes);
    api->cuMemcpyHtoD(d_aok, a_valid, ubytes); api->cuMemcpyHtoD(d_bok, b_valid, ubytes);
    api->cuMemsetD32(d_ov, 0, n); api->cuMemsetD32(d_oe, 0, n); api->cuMemsetD32(d_ook, 0, n); api->cuMemsetD32(d_oprov, 0, n);
    void *params[] = { &d_av, &d_bv, &d_ae, &d_be, &d_aok, &d_bok, &d_ov, &d_oe, &d_ook, &d_oprov };
    for (int i = 0; i < warmup; i++) { float ignored = 0.0f; if ((rc = launch_timed(api, fn, params, n, &ignored)) != 0) return rc; }
    for (int i = 0; i < iters; i++) { if ((rc = launch_timed(api, fn, params, n, &times[i])) != 0) return rc; }
    api->cuMemcpyDtoH(out_value, d_ov, fbytes); api->cuMemcpyDtoH(out_eps, d_oe, fbytes);
    api->cuMemcpyDtoH(out_valid, d_ook, ubytes); api->cuMemcpyDtoH(out_prov, d_oprov, ubytes);
    for (uint32_t i = 0; i < n; i++) {
        float expected_value = a_value[i] + b_value[i];
        float expected_eps = a_eps[i] * a_eps[i] + b_eps[i] * b_eps[i];
        uint32_t expected_valid = a_valid[i] & b_valid[i];
        uint32_t expected_prov = (i ^ 0x12345678U) ^ expected_valid;
        float verr = fabsf(out_value[i] - expected_value);
        float eerr = fabsf(out_eps[i] - expected_eps);
        if (verr > value_max_abs_err) value_max_abs_err = verr;
        if (eerr > eps_max_abs_err) eps_max_abs_err = eerr;
        if (out_valid[i] != expected_valid) valid_mismatch++;
        if (out_prov[i] != expected_prov) prov_mismatch++;
    }
    qsort(times, (size_t)iters, sizeof(float), cmp_float);
    float median = times[iters / 2];
    float q1 = times[iters / 4];
    float q3 = times[(iters * 3) / 4];
    double gscalarops = ((double)n * 7.0) / ((double)median * 1000000.0);
    printf("kretikos_cuda_bench status=pass kernel=epistemic_dual_output_f32 n=%u warmup=%d iterations=%d median_ms=%.9g iqr_ms=%.9g scalar_ops_per_elem=7 gscalarops_per_s=%.9g value_max_abs_err=%.9g eps_max_abs_err=%.9g valid_mismatch=%u prov_mismatch=%u device_name=%s driver_version=%d cc=%d.%d\n",
           n, warmup, iters, median, q3 - q1, gscalarops, value_max_abs_err, eps_max_abs_err, valid_mismatch, prov_mismatch, device_name, driver_version, cc_major, cc_minor);
    return (value_max_abs_err > 0.000001f || eps_max_abs_err > 0.000001f || valid_mismatch || prov_mismatch) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <cubin> <kernel> <mode>\n", argv[0]);
        return 2;
    }
    struct CudaApi api;
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    char device_name[128] = "unknown";
    int driver_version = 0, cc_major = 0, cc_minor = 0;
    const uint32_t n = 1048576;
    const int warmup = 10;
    const int iters = 100;
    if (load_cuda(&api) != 0) return 1;
    if (api.cuInit(0) != 0) return 1;
    if (api.cuDeviceGet(&dev, 0) != 0) return 1;
    if (api.cuDeviceGetName(device_name, (int)sizeof(device_name), dev) != 0) return 1;
    sanitize_device_name(device_name);
    if (api.cuDriverGetVersion(&driver_version) != 0) return 1;
    if (api.cuDeviceComputeCapability(&cc_major, &cc_minor, dev) != 0) return 1;
    if (api.cuCtxCreate(&ctx, 0, dev) != 0) return 1;
    if (api.cuModuleLoad(&mod, argv[1]) != 0) return 1;
    if (api.cuModuleGetFunction(&fn, mod, argv[2]) != 0) return 1;
    int rc = 1;
    if (strcmp(argv[3], "vec_add_f32") == 0) {
        rc = bench_vec_add(&api, fn, n, warmup, iters, device_name, driver_version, cc_major, cc_minor);
    } else if (strcmp(argv[3], "epistemic_dual_output_f32") == 0) {
        rc = bench_epistemic(&api, fn, n, warmup, iters, device_name, driver_version, cc_major, cc_minor);
    } else {
        fprintf(stderr, "unsupported mode: %s\n", argv[3]);
        rc = 2;
    }
    api.cuModuleUnload(mod);
    api.cuCtxDestroy(ctx);
    return rc;
}
