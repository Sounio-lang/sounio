// CUDA Driver API runner for Kretikos cross-backend semantic artifacts.
//
// This runner intentionally matches the exact PTX emitter signatures used by
// scripts/ci/kretikos_cross_backend_semantic_gate.sh. It does not use CUDA
// runtime headers; symbols are loaded from libcuda.so through dlsym.

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
    return 0;
}

static char g_device_name[128] = "unknown";
static int g_driver_version = 0;
static int g_cc_major = 0;
static int g_cc_minor = 0;

static void sanitize_device_name(char *s) {
    for (size_t i = 0; s[i] != '\0'; i++) {
        if (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') s[i] = '_';
    }
}

static void emit_fail(const char *kernel, const char *reason, const char *stage, int code) {
    printf("cross_backend_cuda_runtime status=fail kernel=%s reason=%s stage=%s cuda_result=%d\n",
           kernel, reason, stage, code);
}

static int run_vec_add(struct CudaApi *api, CUfunction fn, uint32_t n) {
    const char *kernel = "vec_add";
    size_t bytes = (size_t)n * sizeof(float);
    float *a = (float *)calloc(n, sizeof(float));
    float *b = (float *)calloc(n, sizeof(float));
    float *out = (float *)calloc(n, sizeof(float));
    CUdeviceptr d_a = 0, d_b = 0, d_out = 0;
    CUresult rc = 0;
    float max_abs_err = 0.0f;

    if (!a || !b || !out) { emit_fail(kernel, "host_alloc_failed", "calloc", 0); return 1; }
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (float)((int)(i % 17) - 8) * 0.5f;
        b[i] = (float)((int)(i % 11) - 5) * 1.25f;
    }

#define VEC_FAIL(reason, stage, code) do { emit_fail(kernel, reason, stage, code); rc = code ? code : 1; goto vec_cleanup; } while (0)
    if ((rc = api->cuMemAlloc(&d_a, bytes)) != 0) VEC_FAIL("cuMemAlloc_a_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_b, bytes)) != 0) VEC_FAIL("cuMemAlloc_b_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_out, bytes)) != 0) VEC_FAIL("cuMemAlloc_out_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemcpyHtoD(d_a, a, bytes)) != 0) VEC_FAIL("cuMemcpyHtoD_a_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemcpyHtoD(d_b, b, bytes)) != 0) VEC_FAIL("cuMemcpyHtoD_b_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemsetD32(d_out, 0, n)) != 0) VEC_FAIL("cuMemsetD32_out_failed", "cuMemsetD32", rc);

    void *params[] = { &d_a, &d_b, &d_out };
    uint32_t block_threads = 256;
    uint32_t grid_blocks = n / block_threads;
    if ((rc = api->cuLaunchKernel(fn, grid_blocks, 1, 1, block_threads, 1, 1, 0, NULL, params, NULL)) != 0)
        VEC_FAIL("cuLaunchKernel_failed", "cuLaunchKernel", rc);
    if ((rc = api->cuCtxSynchronize()) != 0) VEC_FAIL("cuCtxSynchronize_failed", "cuCtxSynchronize", rc);
    if ((rc = api->cuMemcpyDtoH(out, d_out, bytes)) != 0) VEC_FAIL("cuMemcpyDtoH_failed", "cuMemcpyDtoH", rc);

    for (uint32_t i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        float err = fabsf(out[i] - expected);
        if (err > max_abs_err) max_abs_err = err;
    }
    if (max_abs_err > 0.000001f) VEC_FAIL("mismatch", "verify", 0);

    printf("cross_backend_cuda_runtime status=pass kernel=%s reason=runtime_vec_add_pass n=%u max_abs_err=%.9g observed0=%.9g expected0=%.9g device_name=%s driver_version=%d cc=%d.%d\n",
           kernel, n, max_abs_err, out[0], a[0] + b[0], g_device_name, g_driver_version, g_cc_major, g_cc_minor);

vec_cleanup:
    if (d_out) api->cuMemFree(d_out);
    if (d_b) api->cuMemFree(d_b);
    if (d_a) api->cuMemFree(d_a);
    free(out);
    free(b);
    free(a);
    return rc ? 1 : 0;
#undef VEC_FAIL
}

static int run_epistemic_dual(struct CudaApi *api, CUfunction fn, uint32_t n) {
    const char *kernel = "epistemic_dual_output_f32";
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
    CUdeviceptr d_av = 0, d_bv = 0, d_ae = 0, d_be = 0, d_aok = 0, d_bok = 0, d_ov = 0, d_oe = 0, d_ook = 0, d_oprov = 0;
    CUresult rc = 0;
    float value_max_abs_err = 0.0f;
    float eps_max_abs_err = 0.0f;
    uint32_t valid_mismatch = 0;
    uint32_t prov_mismatch = 0;

    if (!a_value || !b_value || !a_eps || !b_eps || !out_value || !out_eps || !a_valid || !b_valid || !out_valid || !out_prov) {
        emit_fail(kernel, "host_alloc_failed", "calloc", 0);
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        a_value[i] = (float)((int)(i % 13) - 6) * 0.25f;
        b_value[i] = (float)((int)(i % 7) - 3) * 0.5f;
        a_eps[i] = 0.01f * (float)((i % 5) + 1);
        b_eps[i] = 0.02f * (float)((i % 3) + 1);
        a_valid[i] = (i % 4) == 0 ? 0U : 1U;
        b_valid[i] = (i % 6) == 0 ? 0U : 1U;
    }

#define EPI_FAIL(reason, stage, code) do { emit_fail(kernel, reason, stage, code); rc = code ? code : 1; goto epi_cleanup; } while (0)
    if ((rc = api->cuMemAlloc(&d_av, fbytes)) != 0) EPI_FAIL("cuMemAlloc_a_value_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_bv, fbytes)) != 0) EPI_FAIL("cuMemAlloc_b_value_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_ae, fbytes)) != 0) EPI_FAIL("cuMemAlloc_a_eps_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_be, fbytes)) != 0) EPI_FAIL("cuMemAlloc_b_eps_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_aok, ubytes)) != 0) EPI_FAIL("cuMemAlloc_a_valid_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_bok, ubytes)) != 0) EPI_FAIL("cuMemAlloc_b_valid_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_ov, fbytes)) != 0) EPI_FAIL("cuMemAlloc_out_value_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_oe, fbytes)) != 0) EPI_FAIL("cuMemAlloc_out_eps_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_ook, ubytes)) != 0) EPI_FAIL("cuMemAlloc_out_valid_failed", "cuMemAlloc", rc);
    if ((rc = api->cuMemAlloc(&d_oprov, ubytes)) != 0) EPI_FAIL("cuMemAlloc_out_prov_failed", "cuMemAlloc", rc);

    if ((rc = api->cuMemcpyHtoD(d_av, a_value, fbytes)) != 0) EPI_FAIL("cuMemcpyHtoD_a_value_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemcpyHtoD(d_bv, b_value, fbytes)) != 0) EPI_FAIL("cuMemcpyHtoD_b_value_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemcpyHtoD(d_ae, a_eps, fbytes)) != 0) EPI_FAIL("cuMemcpyHtoD_a_eps_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemcpyHtoD(d_be, b_eps, fbytes)) != 0) EPI_FAIL("cuMemcpyHtoD_b_eps_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemcpyHtoD(d_aok, a_valid, ubytes)) != 0) EPI_FAIL("cuMemcpyHtoD_a_valid_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemcpyHtoD(d_bok, b_valid, ubytes)) != 0) EPI_FAIL("cuMemcpyHtoD_b_valid_failed", "cuMemcpyHtoD", rc);
    if ((rc = api->cuMemsetD32(d_ov, 0, n)) != 0) EPI_FAIL("cuMemsetD32_out_value_failed", "cuMemsetD32", rc);
    if ((rc = api->cuMemsetD32(d_oe, 0, n)) != 0) EPI_FAIL("cuMemsetD32_out_eps_failed", "cuMemsetD32", rc);
    if ((rc = api->cuMemsetD32(d_ook, 0, n)) != 0) EPI_FAIL("cuMemsetD32_out_valid_failed", "cuMemsetD32", rc);
    if ((rc = api->cuMemsetD32(d_oprov, 0, n)) != 0) EPI_FAIL("cuMemsetD32_out_prov_failed", "cuMemsetD32", rc);

    void *params[] = { &d_av, &d_bv, &d_ae, &d_be, &d_aok, &d_bok, &d_ov, &d_oe, &d_ook, &d_oprov };
    uint32_t block_threads = 256;
    uint32_t grid_blocks = n / block_threads;
    if ((rc = api->cuLaunchKernel(fn, grid_blocks, 1, 1, block_threads, 1, 1, 0, NULL, params, NULL)) != 0)
        EPI_FAIL("cuLaunchKernel_failed", "cuLaunchKernel", rc);
    if ((rc = api->cuCtxSynchronize()) != 0) EPI_FAIL("cuCtxSynchronize_failed", "cuCtxSynchronize", rc);
    if ((rc = api->cuMemcpyDtoH(out_value, d_ov, fbytes)) != 0) EPI_FAIL("cuMemcpyDtoH_out_value_failed", "cuMemcpyDtoH", rc);
    if ((rc = api->cuMemcpyDtoH(out_eps, d_oe, fbytes)) != 0) EPI_FAIL("cuMemcpyDtoH_out_eps_failed", "cuMemcpyDtoH", rc);
    if ((rc = api->cuMemcpyDtoH(out_valid, d_ook, ubytes)) != 0) EPI_FAIL("cuMemcpyDtoH_out_valid_failed", "cuMemcpyDtoH", rc);
    if ((rc = api->cuMemcpyDtoH(out_prov, d_oprov, ubytes)) != 0) EPI_FAIL("cuMemcpyDtoH_out_prov_failed", "cuMemcpyDtoH", rc);

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
    if (value_max_abs_err > 0.000001f || eps_max_abs_err > 0.000001f || valid_mismatch != 0 || prov_mismatch != 0)
        EPI_FAIL("mismatch", "verify", 0);

    printf("cross_backend_cuda_runtime status=pass kernel=%s reason=runtime_epistemic_dual_output_pass n=%u value_max_abs_err=%.9g eps_max_abs_err=%.9g valid_mismatch=%u prov_mismatch=%u observed_value0=%.9g observed_eps_last=%.9g observed_prov0=0x%08x device_name=%s driver_version=%d cc=%d.%d\n",
           kernel, n, value_max_abs_err, eps_max_abs_err, valid_mismatch, prov_mismatch, out_value[0], out_eps[n - 1], out_prov[0], g_device_name, g_driver_version, g_cc_major, g_cc_minor);

epi_cleanup:
    if (d_oprov) api->cuMemFree(d_oprov);
    if (d_ook) api->cuMemFree(d_ook);
    if (d_oe) api->cuMemFree(d_oe);
    if (d_ov) api->cuMemFree(d_ov);
    if (d_bok) api->cuMemFree(d_bok);
    if (d_aok) api->cuMemFree(d_aok);
    if (d_be) api->cuMemFree(d_be);
    if (d_ae) api->cuMemFree(d_ae);
    if (d_bv) api->cuMemFree(d_bv);
    if (d_av) api->cuMemFree(d_av);
    free(out_valid);
    free(out_prov);
    free(b_valid);
    free(a_valid);
    free(out_eps);
    free(out_value);
    free(b_eps);
    free(a_eps);
    free(b_value);
    free(a_value);
    return rc ? 1 : 0;
#undef EPI_FAIL
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <cubin> <kernel> <mode>\n", argv[0]);
        return 2;
    }

    const char *cubin = argv[1];
    const char *kernel = argv[2];
    const char *mode = argv[3];
    uint32_t n = 4096;
    struct CudaApi api;
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    CUresult rc = 0;
    int failed = 0;

    if (load_cuda(&api) != 0) return 1;
    if ((rc = api.cuInit(0)) != 0) { emit_fail(kernel, "cuInit_failed", "cuInit", rc); return 1; }
    if ((rc = api.cuDeviceGet(&dev, 0)) != 0) { emit_fail(kernel, "cuDeviceGet_failed", "cuDeviceGet", rc); return 1; }
    if ((rc = api.cuDeviceGetName(g_device_name, (int)sizeof(g_device_name), dev)) != 0) { emit_fail(kernel, "cuDeviceGetName_failed", "cuDeviceGetName", rc); return 1; }
    sanitize_device_name(g_device_name);
    if ((rc = api.cuDriverGetVersion(&g_driver_version)) != 0) { emit_fail(kernel, "cuDriverGetVersion_failed", "cuDriverGetVersion", rc); return 1; }
    if ((rc = api.cuDeviceComputeCapability(&g_cc_major, &g_cc_minor, dev)) != 0) { emit_fail(kernel, "cuDeviceComputeCapability_failed", "cuDeviceComputeCapability", rc); return 1; }
    if ((rc = api.cuCtxCreate(&ctx, 0, dev)) != 0) { emit_fail(kernel, "cuCtxCreate_failed", "cuCtxCreate", rc); return 1; }
    if ((rc = api.cuModuleLoad(&mod, cubin)) != 0) { emit_fail(kernel, "cuModuleLoad_failed", "cuModuleLoad", rc); api.cuCtxDestroy(ctx); return 1; }
    if ((rc = api.cuModuleGetFunction(&fn, mod, kernel)) != 0) {
        emit_fail(kernel, "cuModuleGetFunction_failed", "cuModuleGetFunction", rc);
        api.cuModuleUnload(mod);
        api.cuCtxDestroy(ctx);
        return 1;
    }

    if (strcmp(mode, "vec_add_f32") == 0) {
        failed = run_vec_add(&api, fn, n);
    } else if (strcmp(mode, "epistemic_dual_output_f32") == 0) {
        failed = run_epistemic_dual(&api, fn, n);
    } else {
        emit_fail(kernel, "unsupported_mode", "argparse", 0);
        failed = 1;
    }

    api.cuModuleUnload(mod);
    api.cuCtxDestroy(ctx);
    return failed;
}
