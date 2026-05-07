// CUDA Driver API loader harness for Sounio-owned CUBIN runtime rungs.
//
// Build on hosts with libdl and a CUDA driver available at runtime, for example:
//   cc -O2 scripts/gpu/nvidia_bare_driver_loader.c -ldl -o /tmp/sounio_bare_loader
//
// Modes:
//   admission  cuModuleLoadData + cuModuleGetFunction only
//   launch     admission + cuLaunchKernel with dummy params
//   store_u32_const  launch + device writeback verification for 42
//   vec_add_f32  launch + device vector add verification against CPU oracle
//   vec_add_f64  launch + device vector add verification against CPU oracle
//   vec_sub_f64  launch + device vector subtract verification against CPU oracle
//   vec_mul_f64  launch + device vector multiply verification against CPU oracle
//   vec_div_f64  launch + device vector divide verification against CPU oracle
//   fma_f64  launch + device affine fused multiply-add verification
//   vec_sub_f32  launch + device vector subtract verification against CPU oracle
//   vec_mul_f32  launch + device vector multiply verification against CPU oracle
//   vec_div_f32  launch + device vector divide verification against CPU oracle
//   fma_f32  launch + device affine fused multiply-add verification
//   epistemic_elementwise_f32  vec add kernel with nonzero epsilon lane
//   epistemic_dual_lane_f32  two-launch value lane + uncertainty lane oracle
//   epistemic_dual_output_f32  one launch, value + uncertainty output buffers
//   epistemic_dual_output_f32_sm89_l4  same oracle against the SM89/L4 artifact
//   epistemic_dual_output_f32_l4cc_alias  same oracle against same-length L4 alias artifact
//   epistemic_dual_output_f32_sm89_l4_layout  same oracle against repaired long-name layout
//   epistemic_dual_output_f32_sm89_l4_flags  same oracle against repaired layout with SM89 flags

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
typedef CUresult (*cuModuleLoadData_t)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, void *, void **, void **);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemsetD32_t)(CUdeviceptr, unsigned int, size_t);

#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define CU_LAUNCH_PARAM_END ((void *)0x00)

static int g_driver_version = -1;
static int g_device_count = -1;
static int g_device_ordinal = -1;
static int g_cc_major = -1;
static int g_cc_minor = -1;
static char g_device_name[128] = "unknown";

static int read_env_int(const char *name, int fallback, int min_value, int max_value) {
    const char *raw = getenv(name);
    if (!raw || !raw[0]) return fallback;
    char *end = NULL;
    long value = strtol(raw, &end, 10);
    if (end == raw || *end != '\0') return fallback;
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return (int)value;
}

static void sanitize_device_name(void) {
    for (size_t i = 0; i < sizeof(g_device_name) && g_device_name[i]; i++) {
        char c = g_device_name[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) {
            g_device_name[i] = '_';
        }
    }
}

static void print_cuda_probe_suffix(void) {
    printf(" driver_version=%d device_count=%d device_ordinal=%d cc_major=%d cc_minor=%d device_name=%s",
           g_driver_version, g_device_count, g_device_ordinal, g_cc_major, g_cc_minor, g_device_name);
}

static void emit_status(const char *status, const char *reason, const char *stage, int cuda_result) {
    printf("sounio_nvidia_bare_runtime status=%s reason=%s stage=%s cuda_result=%d",
           status, reason, stage, cuda_result);
    print_cuda_probe_suffix();
    printf("\n");
}

static int is_dual_output_mode(const char *mode) {
    return strcmp(mode, "epistemic_dual_output_f32") == 0 ||
           strcmp(mode, "epistemic_dual_output_f32_sm89_l4") == 0 ||
           strcmp(mode, "epistemic_dual_output_f32_l4cc_alias") == 0 ||
           strcmp(mode, "epistemic_dual_output_f32_sm89_l4_layout") == 0 ||
           strcmp(mode, "epistemic_dual_output_f32_sm89_l4_flags") == 0;
}

static unsigned char *read_all(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    unsigned char *buf = (unsigned char *)malloc((size_t)n);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

#define LOAD_SYM(name) do { \
    name = (name##_t)dlsym(cuda, #name); \
    if (!name) { emit_status("not_run", "cuda_symbol_missing", #name, 0); return 77; } \
} while (0)

#define LOAD_SYM_ALIAS(var, type, primary, fallback) do { \
    var = (type)dlsym(cuda, primary); \
    if (!var) var = (type)dlsym(cuda, fallback); \
    if (!var) { emit_status("not_run", "cuda_symbol_missing", primary, 0); return 77; } \
} while (0)

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <cubin> <kernel> [admission|launch|store_u32_const|vec_add_f32|vec_add_f64|vec_sub_f64|vec_mul_f64|vec_div_f64|fma_f64|vec_sub_f32|vec_mul_f32|vec_div_f32|fma_f32|epistemic_elementwise_f32|epistemic_dual_lane_f32|epistemic_dual_output_f32|epistemic_dual_output_f32_l4cc_alias|epistemic_dual_output_f32_sm89_l4|epistemic_dual_output_f32_sm89_l4_layout|epistemic_dual_output_f32_sm89_l4_flags]\n", argv[0]);
        return 2;
    }
    const char *mode = argc >= 4 ? argv[3] : "admission";
    if (strcmp(mode, "admission") != 0 &&
        strcmp(mode, "launch") != 0 &&
        strcmp(mode, "store_u32_const") != 0 &&
        strcmp(mode, "vec_add_f32") != 0 &&
        strcmp(mode, "vec_add_f64") != 0 &&
        strcmp(mode, "vec_sub_f64") != 0 &&
        strcmp(mode, "vec_mul_f64") != 0 &&
        strcmp(mode, "vec_div_f64") != 0 &&
        strcmp(mode, "fma_f64") != 0 &&
        strcmp(mode, "vec_sub_f32") != 0 &&
        strcmp(mode, "vec_mul_f32") != 0 &&
        strcmp(mode, "vec_div_f32") != 0 &&
        strcmp(mode, "fma_f32") != 0 &&
        strcmp(mode, "epistemic_elementwise_f32") != 0 &&
        strcmp(mode, "epistemic_dual_lane_f32") != 0 &&
        !is_dual_output_mode(mode)) {
        fprintf(stderr, "usage: mode must be admission, launch, store_u32_const, vec_add_f32, vec_add_f64, vec_sub_f64, vec_mul_f64, vec_div_f64, fma_f64, vec_sub_f32, vec_mul_f32, vec_div_f32, fma_f32, epistemic_elementwise_f32, epistemic_dual_lane_f32, or a supported epistemic_dual_output_f32 diagnostic rung\n");
        return 2;
    }

    size_t cubin_len = 0;
    unsigned char *cubin = read_all(argv[1], &cubin_len);
    if (!cubin || cubin_len == 0) {
        emit_status("not_run", "cubin_read_failed", "read_all", 0);
        return 77;
    }

    void *cuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (!cuda) cuda = dlopen("libcuda.so", RTLD_NOW);
    if (!cuda) {
        emit_status("not_run", "cuda_driver_missing", "dlopen", 0);
        free(cubin);
        return 77;
    }

    cuInit_t cuInit;
    cuDriverGetVersion_t cuDriverGetVersion;
    cuDeviceGetCount_t cuDeviceGetCount;
    cuDeviceGet_t cuDeviceGet;
    cuDeviceGetName_t cuDeviceGetName;
    cuDeviceComputeCapability_t cuDeviceComputeCapability;
    cuCtxCreate_t cuCtxCreate;
    cuModuleLoadData_t cuModuleLoadData;
    cuModuleGetFunction_t cuModuleGetFunction;
    cuLaunchKernel_t cuLaunchKernel;
    cuCtxSynchronize_t cuCtxSynchronize;
    cuModuleUnload_t cuModuleUnload;
    cuCtxDestroy_t cuCtxDestroy;
    cuMemAlloc_t cuMemAlloc = NULL;
    cuMemFree_t cuMemFree = NULL;
    cuMemcpyHtoD_t cuMemcpyHtoD = NULL;
    cuMemcpyDtoH_t cuMemcpyDtoH = NULL;
    cuMemsetD32_t cuMemsetD32 = NULL;

    LOAD_SYM(cuInit);
    LOAD_SYM(cuDriverGetVersion);
    LOAD_SYM(cuDeviceGetCount);
    LOAD_SYM(cuDeviceGet);
    LOAD_SYM(cuDeviceGetName);
    LOAD_SYM(cuDeviceComputeCapability);
    LOAD_SYM_ALIAS(cuCtxCreate, cuCtxCreate_t, "cuCtxCreate_v2", "cuCtxCreate");
    LOAD_SYM(cuModuleLoadData);
    LOAD_SYM(cuModuleGetFunction);
    LOAD_SYM(cuLaunchKernel);
    LOAD_SYM(cuCtxSynchronize);
    LOAD_SYM(cuModuleUnload);
    LOAD_SYM_ALIAS(cuCtxDestroy, cuCtxDestroy_t, "cuCtxDestroy_v2", "cuCtxDestroy");
    if (strcmp(mode, "store_u32_const") == 0 ||
        strcmp(mode, "vec_add_f32") == 0 ||
        strcmp(mode, "vec_add_f64") == 0 ||
        strcmp(mode, "vec_sub_f64") == 0 ||
        strcmp(mode, "vec_mul_f64") == 0 ||
        strcmp(mode, "vec_div_f64") == 0 ||
        strcmp(mode, "fma_f64") == 0 ||
        strcmp(mode, "vec_sub_f32") == 0 ||
        strcmp(mode, "vec_mul_f32") == 0 ||
        strcmp(mode, "vec_div_f32") == 0 ||
        strcmp(mode, "fma_f32") == 0 ||
        strcmp(mode, "epistemic_elementwise_f32") == 0 ||
        strcmp(mode, "epistemic_dual_lane_f32") == 0 ||
        is_dual_output_mode(mode)) {
        LOAD_SYM_ALIAS(cuMemAlloc, cuMemAlloc_t, "cuMemAlloc_v2", "cuMemAlloc");
        LOAD_SYM_ALIAS(cuMemFree, cuMemFree_t, "cuMemFree_v2", "cuMemFree");
        LOAD_SYM_ALIAS(cuMemcpyHtoD, cuMemcpyHtoD_t, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
        LOAD_SYM_ALIAS(cuMemcpyDtoH, cuMemcpyDtoH_t, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
        LOAD_SYM_ALIAS(cuMemsetD32, cuMemsetD32_t, "cuMemsetD32_v2", "cuMemsetD32");
    }

    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    CUresult rc = cuInit(0);
    if (rc != 0) { emit_status("not_run", "cuInit_failed", "cuInit", rc); free(cubin); return 77; }
    (void)cuDriverGetVersion(&g_driver_version);
    rc = cuDeviceGetCount(&g_device_count);
    if (rc != 0) { emit_status("not_run", "cuDeviceGetCount_failed", "cuDeviceGetCount", rc); free(cubin); return 77; }
    if (g_device_count <= 0) { emit_status("not_run", "no_cuda_device", "cuDeviceGetCount", 0); free(cubin); return 77; }
    int requested_device = 0;
    const char *raw_device = getenv("SOUNIO_NVIDIA_BARE_DEVICE_ORDINAL");
    if (raw_device && raw_device[0]) {
        char *end = NULL;
        long parsed = strtol(raw_device, &end, 10);
        if (end == raw_device || *end != '\0') {
            emit_status("not_run", "invalid_cuda_device_ordinal", "SOUNIO_NVIDIA_BARE_DEVICE_ORDINAL", 0);
            free(cubin);
            return 77;
        }
        if (parsed < 0 || parsed >= g_device_count) {
            g_device_ordinal = (int)parsed;
            emit_status("not_run", "cuda_device_ordinal_out_of_range", "cuDeviceGetCount", 0);
            free(cubin);
            return 77;
        }
        requested_device = (int)parsed;
    }
    g_device_ordinal = requested_device;
    rc = cuDeviceGet(&dev, requested_device);
    if (rc != 0) { emit_status("not_run", "cuDeviceGet_failed", "cuDeviceGet", rc); free(cubin); return 77; }
    if (cuDeviceGetName(g_device_name, (int)sizeof(g_device_name), dev) != 0) {
        strcpy(g_device_name, "unknown");
    }
    g_device_name[sizeof(g_device_name) - 1] = '\0';
    sanitize_device_name();
    (void)cuDeviceComputeCapability(&g_cc_major, &g_cc_minor, dev);
    rc = cuCtxCreate(&ctx, 0, dev);
    if (rc != 0) { emit_status("not_run", "cuCtxCreate_failed", "cuCtxCreate", rc); free(cubin); return 77; }
    rc = cuModuleLoadData(&mod, cubin);
    if (rc != 0) { emit_status("fail", "cuModuleLoadData_rejected", "cuModuleLoadData", rc); cuCtxDestroy(ctx); free(cubin); return 1; }
    rc = cuModuleGetFunction(&fn, mod, argv[2]);
    if (rc != 0) { emit_status("fail", "cuModuleGetFunction_missing", "cuModuleGetFunction", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }

    if (strcmp(mode, "admission") == 0) {
        emit_status("pass", "runtime_admission_pass", "cuModuleGetFunction", 0);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        return 0;
    }

    if (strcmp(mode, "store_u32_const") == 0) {
        CUdeviceptr d_out = 0;
        uint32_t seed = 0;
        uint64_t out_arg = 0;
        uint32_t n = 1;
        uint32_t observed = 0;

        rc = cuMemAlloc(&d_out, sizeof(uint32_t));
        if (rc != 0) { emit_status("fail", "cuMemAlloc_failed", "cuMemAlloc", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }
        rc = cuMemsetD32(d_out, 0, 1);
        if (rc != 0) { emit_status("fail", "cuMemsetD32_failed", "cuMemsetD32", rc); cuMemFree(d_out); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }

        out_arg = (uint64_t)d_out;
        void *store_params[] = { &seed, &out_arg, &n };
        rc = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, store_params, NULL);
        if (rc != 0) { emit_status("fail", "cuLaunchKernel_failed", "cuLaunchKernel", rc); cuMemFree(d_out); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }
        rc = cuCtxSynchronize();
        if (rc != 0) { emit_status("fail", "cuCtxSynchronize_failed", "cuCtxSynchronize", rc); cuMemFree(d_out); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }
        rc = cuMemcpyDtoH(&observed, d_out, sizeof(observed));
        if (rc != 0) { emit_status("fail", "cuMemcpyDtoH_failed", "cuMemcpyDtoH", rc); cuMemFree(d_out); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }
        rc = cuMemFree(d_out);
        if (rc != 0) { emit_status("fail", "cuMemFree_failed", "cuMemFree", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }
        if (observed != 42u) {
            printf("sounio_nvidia_bare_runtime status=fail reason=store_u32_const_mismatch stage=verify_store_u32_const cuda_result=0");
            print_cuda_probe_suffix();
            printf(" observed_u32=%u expected_u32=42\n", observed);
            cuModuleUnload(mod);
            cuCtxDestroy(ctx);
            free(cubin);
            return 1;
        }
        printf("sounio_nvidia_bare_runtime status=pass reason=runtime_store_u32_const_pass stage=cuMemcpyDtoH cuda_result=0");
        print_cuda_probe_suffix();
        printf(" observed_u32=%u expected_u32=42\n", observed);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        return 0;
    }

    if (is_dual_output_mode(mode)) {
        const char *dual_output_label = mode;
        uint32_t n_arg = (uint32_t)read_env_int("SOUNIO_NVIDIA_BARE_VEC_N", 64, 1, 256);
        size_t bytes = (size_t)n_arg * sizeof(float);
        float *h_x = (float *)calloc(n_arg, sizeof(float));
        float *h_y = (float *)calloc(n_arg, sizeof(float));
        float *h_ux = (float *)calloc(n_arg, sizeof(float));
        float *h_uy = (float *)calloc(n_arg, sizeof(float));
        float *h_eps = (float *)calloc(n_arg, sizeof(float));
        float *h_value_out = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_out = (float *)calloc(n_arg, sizeof(float));
        float *h_value_expected = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_expected = (float *)calloc(n_arg, sizeof(float));
        CUdeviceptr d_x = 0;
        CUdeviceptr d_y = 0;
        CUdeviceptr d_ux = 0;
        CUdeviceptr d_uy = 0;
        CUdeviceptr d_eps = 0;
        CUdeviceptr d_value_out = 0;
        CUdeviceptr d_uncert_out = 0;
        uint64_t x_arg = 0;
        uint64_t y_arg = 0;
        uint64_t ux_arg = 0;
        uint64_t uy_arg = 0;
        uint64_t eps_arg = 0;
        uint64_t value_out_arg = 0;
        uint64_t uncert_out_arg = 0;
        float value_max_abs_err = 0.0f;
        float uncertainty_max_abs_err = 0.0f;
        int dual_output_failed = 0;

        if (!h_x || !h_y || !h_ux || !h_uy || !h_eps || !h_value_out ||
            !h_uncert_out || !h_value_expected || !h_uncert_expected) {
            emit_status("fail", "host_alloc_failed", "calloc", 0);
            dual_output_failed = 1;
            goto dual_output_cleanup;
        }

        for (uint32_t i = 0; i < n_arg; i++) {
            h_x[i] = (float)((int)(i % 17) - 8) * 0.5f;
            h_y[i] = (float)((int)(i % 11) - 5) * 0.25f;
            h_ux[i] = (float)((int)(i % 5) + 1) * 0.0625f;
            h_uy[i] = (float)((int)(i % 7) + 1) * 0.03125f;
            h_eps[i] = (float)((int)(i % 3) + 1) * 0.0625f;
            h_value_expected[i] = h_x[i] + h_y[i];
            h_uncert_expected[i] = h_ux[i] + h_uy[i] * (1.0f + h_eps[i]);
        }

#define DUAL_OUT_FAIL(reason, stage, code) do { \
            emit_status("fail", reason, stage, code); \
            dual_output_failed = 1; \
            goto dual_output_cleanup; \
        } while (0)

        rc = cuMemAlloc(&d_x, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_x_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_y, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_y_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_ux, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_ux_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_uy, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_uy_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_eps, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_eps_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_value_out, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_value_out_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_uncert_out, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemAlloc_uncertainty_out_failed", "cuMemAlloc", rc);

        rc = cuMemcpyHtoD(d_x, h_x, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyHtoD_x_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_y, h_y, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyHtoD_y_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_ux, h_ux, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyHtoD_ux_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_uy, h_uy, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyHtoD_uy_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_eps, h_eps, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyHtoD_eps_failed", "cuMemcpyHtoD", rc);
        rc = cuMemsetD32(d_value_out, 0, n_arg);
        if (rc != 0) DUAL_OUT_FAIL("cuMemsetD32_value_out_failed", "cuMemsetD32", rc);
        rc = cuMemsetD32(d_uncert_out, 0, n_arg);
        if (rc != 0) DUAL_OUT_FAIL("cuMemsetD32_uncertainty_out_failed", "cuMemsetD32", rc);

        x_arg = (uint64_t)d_x;
        y_arg = (uint64_t)d_y;
        ux_arg = (uint64_t)d_ux;
        uy_arg = (uint64_t)d_uy;
        eps_arg = (uint64_t)d_eps;
        value_out_arg = (uint64_t)d_value_out;
        uncert_out_arg = (uint64_t)d_uncert_out;
        unsigned char dual_output_param_buffer[64] = {0};
        size_t dual_output_param_size = sizeof(dual_output_param_buffer);
        memcpy(dual_output_param_buffer + 0, &x_arg, sizeof(x_arg));
        memcpy(dual_output_param_buffer + 8, &y_arg, sizeof(y_arg));
        memcpy(dual_output_param_buffer + 16, &ux_arg, sizeof(ux_arg));
        memcpy(dual_output_param_buffer + 24, &uy_arg, sizeof(uy_arg));
        memcpy(dual_output_param_buffer + 32, &eps_arg, sizeof(eps_arg));
        memcpy(dual_output_param_buffer + 40, &value_out_arg, sizeof(value_out_arg));
        memcpy(dual_output_param_buffer + 48, &uncert_out_arg, sizeof(uncert_out_arg));
        memcpy(dual_output_param_buffer + 56, &n_arg, sizeof(n_arg));
        void *dual_output_extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, dual_output_param_buffer,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &dual_output_param_size,
            CU_LAUNCH_PARAM_END,
        };
        rc = cuLaunchKernel(fn, 1, 1, 1, n_arg, 1, 1, 0, NULL, NULL, dual_output_extra);
        if (rc != 0) DUAL_OUT_FAIL("cuLaunchKernel_failed", "cuLaunchKernel", rc);
        rc = cuCtxSynchronize();
        if (rc != 0) DUAL_OUT_FAIL("cuCtxSynchronize_failed", "cuCtxSynchronize", rc);
        rc = cuMemcpyDtoH(h_value_out, d_value_out, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyDtoH_value_failed", "cuMemcpyDtoH", rc);
        rc = cuMemcpyDtoH(h_uncert_out, d_uncert_out, bytes);
        if (rc != 0) DUAL_OUT_FAIL("cuMemcpyDtoH_uncertainty_failed", "cuMemcpyDtoH", rc);

        for (uint32_t i = 0; i < n_arg; i++) {
            float value_err = h_value_out[i] - h_value_expected[i];
            float uncertainty_err = h_uncert_out[i] - h_uncert_expected[i];
            if (value_err < 0.0f) value_err = -value_err;
            if (uncertainty_err < 0.0f) uncertainty_err = -uncertainty_err;
            if (value_err > value_max_abs_err) value_max_abs_err = value_err;
            if (uncertainty_err > uncertainty_max_abs_err) uncertainty_max_abs_err = uncertainty_err;
        }

        if (value_max_abs_err > 0.000001f || uncertainty_max_abs_err > 0.000001f) {
            printf("sounio_nvidia_bare_runtime status=fail reason=%s_mismatch stage=verify_%s cuda_result=0",
                   dual_output_label, dual_output_label);
            print_cuda_probe_suffix();
            printf(" %s_n=%u value_max_abs_err=%.9g uncertainty_max_abs_err=%.9g uncertainty_eps_nonzero=1 observed_value0=%.9g expected_value0=%.9g observed_uncertainty_last=%.9g expected_uncertainty_last=%.9g\n",
                   dual_output_label, n_arg, value_max_abs_err, uncertainty_max_abs_err,
                   h_value_out[0], h_value_expected[0], h_uncert_out[n_arg - 1], h_uncert_expected[n_arg - 1]);
            dual_output_failed = 1;
            goto dual_output_cleanup;
        }

        printf("sounio_nvidia_bare_runtime status=pass reason=runtime_%s_pass stage=cuMemcpyDtoH cuda_result=0",
               dual_output_label);
        print_cuda_probe_suffix();
        printf(" %s_n=%u value_max_abs_err=%.9g uncertainty_max_abs_err=%.9g uncertainty_eps_nonzero=1 observed_value0=%.9g expected_value0=%.9g observed_uncertainty_last=%.9g expected_uncertainty_last=%.9g\n",
               dual_output_label, n_arg, value_max_abs_err, uncertainty_max_abs_err,
               h_value_out[0], h_value_expected[0], h_uncert_out[n_arg - 1], h_uncert_expected[n_arg - 1]);

dual_output_cleanup:
        if (d_uncert_out) cuMemFree(d_uncert_out);
        if (d_value_out) cuMemFree(d_value_out);
        if (d_eps) cuMemFree(d_eps);
        if (d_uy) cuMemFree(d_uy);
        if (d_ux) cuMemFree(d_ux);
        if (d_y) cuMemFree(d_y);
        if (d_x) cuMemFree(d_x);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        free(h_uncert_expected);
        free(h_value_expected);
        free(h_uncert_out);
        free(h_value_out);
        free(h_eps);
        free(h_uy);
        free(h_ux);
        free(h_y);
        free(h_x);
#undef DUAL_OUT_FAIL
        return dual_output_failed ? 1 : 0;
    }

    if (strcmp(mode, "epistemic_dual_lane_f32") == 0) {
        uint32_t n_arg = (uint32_t)read_env_int("SOUNIO_NVIDIA_BARE_VEC_N", 64, 1, 256);
        size_t bytes = (size_t)n_arg * sizeof(float);
        float *h_value_x = (float *)calloc(n_arg, sizeof(float));
        float *h_value_y = (float *)calloc(n_arg, sizeof(float));
        float *h_value_eps = (float *)calloc(n_arg, sizeof(float));
        float *h_value_out = (float *)calloc(n_arg, sizeof(float));
        float *h_value_expected = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_x = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_y = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_eps = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_out = (float *)calloc(n_arg, sizeof(float));
        float *h_uncert_expected = (float *)calloc(n_arg, sizeof(float));
        CUdeviceptr d_x = 0;
        CUdeviceptr d_y = 0;
        CUdeviceptr d_eps = 0;
        CUdeviceptr d_value_out = 0;
        CUdeviceptr d_uncert_out = 0;
        uint64_t x_arg = 0;
        uint64_t y_arg = 0;
        uint64_t eps_arg = 0;
        uint64_t out_arg = 0;
        float value_max_abs_err = 0.0f;
        float uncertainty_max_abs_err = 0.0f;
        int dual_failed = 0;

        if (!h_value_x || !h_value_y || !h_value_eps || !h_value_out || !h_value_expected ||
            !h_uncert_x || !h_uncert_y || !h_uncert_eps || !h_uncert_out || !h_uncert_expected) {
            emit_status("fail", "host_alloc_failed", "calloc", 0);
            dual_failed = 1;
            goto dual_cleanup;
        }

        for (uint32_t i = 0; i < n_arg; i++) {
            h_value_x[i] = (float)((int)(i % 17) - 8) * 0.5f;
            h_value_y[i] = (float)((int)(i % 11) - 5) * 0.25f;
            h_value_eps[i] = 0.0f;
            h_value_expected[i] = h_value_x[i] + h_value_y[i];

            h_uncert_x[i] = (float)((int)(i % 5) + 1) * 0.0625f;
            h_uncert_y[i] = (float)((int)(i % 7) + 1) * 0.03125f;
            h_uncert_eps[i] = (float)((int)(i % 3) + 1) * 0.0625f;
            h_uncert_expected[i] = h_uncert_x[i] + h_uncert_y[i] * (1.0f + h_uncert_eps[i]);
        }

#define DUAL_FAIL(reason, stage, code) do { \
            emit_status("fail", reason, stage, code); \
            dual_failed = 1; \
            goto dual_cleanup; \
        } while (0)

        rc = cuMemAlloc(&d_x, bytes);
        if (rc != 0) DUAL_FAIL("cuMemAlloc_x_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_y, bytes);
        if (rc != 0) DUAL_FAIL("cuMemAlloc_y_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_eps, bytes);
        if (rc != 0) DUAL_FAIL("cuMemAlloc_eps_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_value_out, bytes);
        if (rc != 0) DUAL_FAIL("cuMemAlloc_value_out_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_uncert_out, bytes);
        if (rc != 0) DUAL_FAIL("cuMemAlloc_uncertainty_out_failed", "cuMemAlloc", rc);

        rc = cuMemcpyHtoD(d_x, h_value_x, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyHtoD_value_x_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_y, h_value_y, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyHtoD_value_y_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_eps, h_value_eps, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyHtoD_value_eps_failed", "cuMemcpyHtoD", rc);
        rc = cuMemsetD32(d_value_out, 0, n_arg);
        if (rc != 0) DUAL_FAIL("cuMemsetD32_value_out_failed", "cuMemsetD32", rc);

        x_arg = (uint64_t)d_x;
        y_arg = (uint64_t)d_y;
        eps_arg = (uint64_t)d_eps;
        out_arg = (uint64_t)d_value_out;
        void *value_params[] = { &x_arg, &y_arg, &eps_arg, &out_arg, &n_arg };
        rc = cuLaunchKernel(fn, 1, 1, 1, n_arg, 1, 1, 0, NULL, value_params, NULL);
        if (rc != 0) DUAL_FAIL("cuLaunchKernel_value_failed", "cuLaunchKernel", rc);
        rc = cuCtxSynchronize();
        if (rc != 0) DUAL_FAIL("cuCtxSynchronize_value_failed", "cuCtxSynchronize", rc);
        rc = cuMemcpyDtoH(h_value_out, d_value_out, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyDtoH_value_failed", "cuMemcpyDtoH", rc);

        rc = cuMemcpyHtoD(d_x, h_uncert_x, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyHtoD_uncertainty_x_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_y, h_uncert_y, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyHtoD_uncertainty_y_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_eps, h_uncert_eps, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyHtoD_uncertainty_eps_failed", "cuMemcpyHtoD", rc);
        rc = cuMemsetD32(d_uncert_out, 0, n_arg);
        if (rc != 0) DUAL_FAIL("cuMemsetD32_uncertainty_out_failed", "cuMemsetD32", rc);

        out_arg = (uint64_t)d_uncert_out;
        void *uncert_params[] = { &x_arg, &y_arg, &eps_arg, &out_arg, &n_arg };
        rc = cuLaunchKernel(fn, 1, 1, 1, n_arg, 1, 1, 0, NULL, uncert_params, NULL);
        if (rc != 0) DUAL_FAIL("cuLaunchKernel_uncertainty_failed", "cuLaunchKernel", rc);
        rc = cuCtxSynchronize();
        if (rc != 0) DUAL_FAIL("cuCtxSynchronize_uncertainty_failed", "cuCtxSynchronize", rc);
        rc = cuMemcpyDtoH(h_uncert_out, d_uncert_out, bytes);
        if (rc != 0) DUAL_FAIL("cuMemcpyDtoH_uncertainty_failed", "cuMemcpyDtoH", rc);

        for (uint32_t i = 0; i < n_arg; i++) {
            float value_err = h_value_out[i] - h_value_expected[i];
            float uncertainty_err = h_uncert_out[i] - h_uncert_expected[i];
            if (value_err < 0.0f) value_err = -value_err;
            if (uncertainty_err < 0.0f) uncertainty_err = -uncertainty_err;
            if (value_err > value_max_abs_err) value_max_abs_err = value_err;
            if (uncertainty_err > uncertainty_max_abs_err) uncertainty_max_abs_err = uncertainty_err;
        }

        if (value_max_abs_err > 0.000001f || uncertainty_max_abs_err > 0.000001f) {
            printf("sounio_nvidia_bare_runtime status=fail reason=epistemic_dual_lane_f32_mismatch stage=verify_epistemic_dual_lane_f32 cuda_result=0");
            print_cuda_probe_suffix();
            printf(" epistemic_dual_lane_f32_n=%u value_max_abs_err=%.9g uncertainty_max_abs_err=%.9g uncertainty_eps_nonzero=1 observed_value0=%.9g expected_value0=%.9g observed_uncertainty_last=%.9g expected_uncertainty_last=%.9g\n",
                   n_arg, value_max_abs_err, uncertainty_max_abs_err,
                   h_value_out[0], h_value_expected[0], h_uncert_out[n_arg - 1], h_uncert_expected[n_arg - 1]);
            dual_failed = 1;
            goto dual_cleanup;
        }

        printf("sounio_nvidia_bare_runtime status=pass reason=runtime_epistemic_dual_lane_f32_pass stage=cuMemcpyDtoH cuda_result=0");
        print_cuda_probe_suffix();
        printf(" epistemic_dual_lane_f32_n=%u value_max_abs_err=%.9g uncertainty_max_abs_err=%.9g uncertainty_eps_nonzero=1 observed_value0=%.9g expected_value0=%.9g observed_uncertainty_last=%.9g expected_uncertainty_last=%.9g\n",
               n_arg, value_max_abs_err, uncertainty_max_abs_err,
               h_value_out[0], h_value_expected[0], h_uncert_out[n_arg - 1], h_uncert_expected[n_arg - 1]);

dual_cleanup:
        if (d_uncert_out) cuMemFree(d_uncert_out);
        if (d_value_out) cuMemFree(d_value_out);
        if (d_eps) cuMemFree(d_eps);
        if (d_y) cuMemFree(d_y);
        if (d_x) cuMemFree(d_x);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        free(h_uncert_expected);
        free(h_uncert_out);
        free(h_uncert_eps);
        free(h_uncert_y);
        free(h_uncert_x);
        free(h_value_expected);
        free(h_value_out);
        free(h_value_eps);
        free(h_value_y);
        free(h_value_x);
#undef DUAL_FAIL
        return dual_failed ? 1 : 0;
    }

    if (strcmp(mode, "vec_add_f64") == 0 ||
        strcmp(mode, "vec_sub_f64") == 0 ||
        strcmp(mode, "vec_mul_f64") == 0 ||
        strcmp(mode, "vec_div_f64") == 0 ||
        strcmp(mode, "fma_f64") == 0) {
        const char *vec_label = mode;
        int subtract = strcmp(mode, "vec_sub_f64") == 0;
        int multiply = strcmp(mode, "vec_mul_f64") == 0;
        int divide = strcmp(mode, "vec_div_f64") == 0;
        int fma = strcmp(mode, "fma_f64") == 0;
        uint32_t n_arg = (uint32_t)read_env_int("SOUNIO_NVIDIA_BARE_VEC_N", 64, 1, 256);
        size_t bytes = (size_t)n_arg * sizeof(double);
        double *h_x = (double *)calloc(n_arg, sizeof(double));
        double *h_y = (double *)calloc(n_arg, sizeof(double));
        double *h_z = (double *)calloc(n_arg, sizeof(double));
        double *h_out = (double *)calloc(n_arg, sizeof(double));
        double *h_expected = (double *)calloc(n_arg, sizeof(double));
        CUdeviceptr d_x = 0;
        CUdeviceptr d_y = 0;
        CUdeviceptr d_z = 0;
        CUdeviceptr d_out = 0;
        uint64_t x_arg = 0;
        uint64_t y_arg = 0;
        uint64_t z_arg = 0;
        uint64_t out_arg = 0;
        double max_abs_err = 0.0;
        int vec_failed = 0;

        if (!h_x || !h_y || !h_z || !h_out || !h_expected) {
            emit_status("fail", "host_alloc_failed", "calloc", 0);
            vec_failed = 1;
            goto vec_f64_cleanup;
        }

        for (uint32_t i = 0; i < n_arg; i++) {
            h_x[i] = (double)((int)(i % 17) - 8) * 0.5;
            h_y[i] = (double)((int)(i % 11) - 5) * 1.25;
            h_z[i] = (double)((int)(i % 13) - 6) * 0.125;
            if (multiply || divide || fma) {
                h_x[i] = (double)((int)(i % 19) + 1) * 0.25;
                h_y[i] = (double)((int)(i % 7) + 1) * 0.5;
            }
            if (subtract) {
                h_expected[i] = h_x[i] - h_y[i];
            } else if (multiply) {
                h_expected[i] = h_x[i] * h_y[i];
            } else if (divide) {
                h_expected[i] = h_x[i] / h_y[i];
            } else if (fma) {
                h_expected[i] = (h_x[i] * h_y[i]) + h_z[i];
            } else {
                h_expected[i] = h_x[i] + h_y[i];
            }
        }

#define VEC_F64_FAIL(reason, stage, code) do { \
            emit_status("fail", reason, stage, code); \
            vec_failed = 1; \
            goto vec_f64_cleanup; \
        } while (0)

        rc = cuMemAlloc(&d_x, bytes);
        if (rc != 0) VEC_F64_FAIL("cuMemAlloc_x_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_y, bytes);
        if (rc != 0) VEC_F64_FAIL("cuMemAlloc_y_failed", "cuMemAlloc", rc);
        if (fma) {
            rc = cuMemAlloc(&d_z, bytes);
            if (rc != 0) VEC_F64_FAIL("cuMemAlloc_z_failed", "cuMemAlloc", rc);
        }
        rc = cuMemAlloc(&d_out, bytes);
        if (rc != 0) VEC_F64_FAIL("cuMemAlloc_out_failed", "cuMemAlloc", rc);
        rc = cuMemcpyHtoD(d_x, h_x, bytes);
        if (rc != 0) VEC_F64_FAIL("cuMemcpyHtoD_x_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_y, h_y, bytes);
        if (rc != 0) VEC_F64_FAIL("cuMemcpyHtoD_y_failed", "cuMemcpyHtoD", rc);
        if (fma) {
            rc = cuMemcpyHtoD(d_z, h_z, bytes);
            if (rc != 0) VEC_F64_FAIL("cuMemcpyHtoD_z_failed", "cuMemcpyHtoD", rc);
        }
        rc = cuMemsetD32(d_out, 0, n_arg * 2);
        if (rc != 0) VEC_F64_FAIL("cuMemsetD32_out_failed", "cuMemsetD32", rc);

        x_arg = (uint64_t)d_x;
        y_arg = (uint64_t)d_y;
        z_arg = (uint64_t)d_z;
        out_arg = (uint64_t)d_out;
        void *vec3_params[] = { &x_arg, &y_arg, &out_arg };
        void *fma_params[] = { &x_arg, &y_arg, &z_arg, &out_arg };
        rc = cuLaunchKernel(fn, 1, 1, 1, n_arg, 1, 1, 0, NULL, fma ? fma_params : vec3_params, NULL);
        if (rc != 0) VEC_F64_FAIL("cuLaunchKernel_failed", "cuLaunchKernel", rc);
        rc = cuCtxSynchronize();
        if (rc != 0) VEC_F64_FAIL("cuCtxSynchronize_failed", "cuCtxSynchronize", rc);
        rc = cuMemcpyDtoH(h_out, d_out, bytes);
        if (rc != 0) VEC_F64_FAIL("cuMemcpyDtoH_failed", "cuMemcpyDtoH", rc);

        for (uint32_t i = 0; i < n_arg; i++) {
            double err = h_out[i] - h_expected[i];
            if (err < 0.0) err = -err;
            if (err > max_abs_err) max_abs_err = err;
        }
        if (max_abs_err > 0.000000000001) {
            printf("sounio_nvidia_bare_runtime status=fail reason=%s_mismatch stage=verify_%s cuda_result=0",
                   vec_label,
                   vec_label);
            print_cuda_probe_suffix();
            printf(" %s_n=%u %s_max_abs_err=%.17g observed0=%.17g expected0=%.17g observed_last=%.17g expected_last=%.17g\n",
                   vec_label, n_arg,
                   vec_label, max_abs_err,
                   h_out[0], h_expected[0], h_out[n_arg - 1], h_expected[n_arg - 1]);
            vec_failed = 1;
            goto vec_f64_cleanup;
        }
        printf("sounio_nvidia_bare_runtime status=pass reason=%s stage=cuMemcpyDtoH cuda_result=0",
               divide ? "runtime_vec_div_f64_pass" : (multiply ? "runtime_vec_mul_f64_pass" : (fma ? "runtime_fma_f64_pass" : (subtract ? "runtime_vec_sub_f64_pass" : "runtime_vec_add_f64_pass"))));
        print_cuda_probe_suffix();
        printf(" %s_n=%u %s_max_abs_err=%.17g observed0=%.17g expected0=%.17g observed_last=%.17g expected_last=%.17g\n",
               vec_label, n_arg,
               vec_label, max_abs_err,
               h_out[0], h_expected[0], h_out[n_arg - 1], h_expected[n_arg - 1]);

vec_f64_cleanup:
        if (d_out) cuMemFree(d_out);
        if (d_z) cuMemFree(d_z);
        if (d_y) cuMemFree(d_y);
        if (d_x) cuMemFree(d_x);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        free(h_expected);
        free(h_out);
        free(h_z);
        free(h_y);
        free(h_x);
#undef VEC_F64_FAIL
        return vec_failed ? 1 : 0;
    }

    if (strcmp(mode, "vec_add_f32") == 0 ||
        strcmp(mode, "vec_sub_f32") == 0 ||
        strcmp(mode, "vec_mul_f32") == 0 ||
        strcmp(mode, "vec_div_f32") == 0 ||
        strcmp(mode, "fma_f32") == 0 ||
        strcmp(mode, "epistemic_elementwise_f32") == 0) {
        const char *vec_label = mode;
        int subtract = strcmp(mode, "vec_sub_f32") == 0;
        int multiply = strcmp(mode, "vec_mul_f32") == 0;
        int divide = strcmp(mode, "vec_div_f32") == 0;
        int fma = strcmp(mode, "fma_f32") == 0;
        int epistemic = strcmp(mode, "epistemic_elementwise_f32") == 0;
        uint32_t n_arg = (uint32_t)read_env_int("SOUNIO_NVIDIA_BARE_VEC_N", (epistemic || fma || multiply || divide) ? 64 : 16, 1, 256);
        size_t bytes = (size_t)n_arg * sizeof(float);
        float *h_x = (float *)calloc(n_arg, sizeof(float));
        float *h_y = (float *)calloc(n_arg, sizeof(float));
        float *h_eps = (float *)calloc(n_arg, sizeof(float));
        float *h_out = (float *)calloc(n_arg, sizeof(float));
        float *h_expected = (float *)calloc(n_arg, sizeof(float));
        CUdeviceptr d_x = 0;
        CUdeviceptr d_y = 0;
        CUdeviceptr d_eps = 0;
        CUdeviceptr d_out = 0;
        uint64_t x_arg = 0;
        uint64_t y_arg = 0;
        uint64_t eps_arg = 0;
        uint64_t out_arg = 0;
        float max_abs_err = 0.0f;
        int vec_failed = 0;

        if (!h_x || !h_y || !h_eps || !h_out || !h_expected) {
            emit_status("fail", "host_alloc_failed", "calloc", 0);
            vec_failed = 1;
            goto vec_cleanup;
        }

        for (uint32_t i = 0; i < n_arg; i++) {
            h_x[i] = (multiply || divide) ? 0.0f : (float)((int)(i % 17) - 8) * 0.5f;
            h_y[i] = (float)((int)(i % 11) - 5) * 1.25f;
            if (divide) {
                float denom = (float)(1U << (i % 4));
                h_eps[i] = (1.0f / denom) - 1.0f;
            } else {
                h_eps[i] = subtract ? -2.0f : ((multiply || epistemic || fma) ? ((float)((int)(i % 7) - 3) * 0.03125f) - (multiply ? 1.0f : 0.0f) : 0.0f);
            }
            h_expected[i] = h_x[i] + h_y[i] * (1.0f + h_eps[i]);
        }

#define VEC_FAIL(reason, stage, code) do { \
            emit_status("fail", reason, stage, code); \
            vec_failed = 1; \
            goto vec_cleanup; \
        } while (0)

        rc = cuMemAlloc(&d_x, bytes);
        if (rc != 0) VEC_FAIL("cuMemAlloc_x_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_y, bytes);
        if (rc != 0) VEC_FAIL("cuMemAlloc_y_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_eps, bytes);
        if (rc != 0) VEC_FAIL("cuMemAlloc_eps_failed", "cuMemAlloc", rc);
        rc = cuMemAlloc(&d_out, bytes);
        if (rc != 0) VEC_FAIL("cuMemAlloc_out_failed", "cuMemAlloc", rc);
        rc = cuMemcpyHtoD(d_x, h_x, bytes);
        if (rc != 0) VEC_FAIL("cuMemcpyHtoD_x_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_y, h_y, bytes);
        if (rc != 0) VEC_FAIL("cuMemcpyHtoD_y_failed", "cuMemcpyHtoD", rc);
        rc = cuMemcpyHtoD(d_eps, h_eps, bytes);
        if (rc != 0) VEC_FAIL("cuMemcpyHtoD_eps_failed", "cuMemcpyHtoD", rc);
        rc = cuMemsetD32(d_out, 0, n_arg);
        if (rc != 0) VEC_FAIL("cuMemsetD32_out_failed", "cuMemsetD32", rc);

        x_arg = (uint64_t)d_x;
        y_arg = (uint64_t)d_y;
        eps_arg = (uint64_t)d_eps;
        out_arg = (uint64_t)d_out;
        void *vec_params[] = { &x_arg, &y_arg, &eps_arg, &out_arg, &n_arg };
        rc = cuLaunchKernel(fn, 1, 1, 1, n_arg, 1, 1, 0, NULL, vec_params, NULL);
        if (rc != 0) VEC_FAIL("cuLaunchKernel_failed", "cuLaunchKernel", rc);
        rc = cuCtxSynchronize();
        if (rc != 0) VEC_FAIL("cuCtxSynchronize_failed", "cuCtxSynchronize", rc);
        rc = cuMemcpyDtoH(h_out, d_out, bytes);
        if (rc != 0) VEC_FAIL("cuMemcpyDtoH_failed", "cuMemcpyDtoH", rc);

        for (uint32_t i = 0; i < n_arg; i++) {
            float err = h_out[i] - h_expected[i];
            if (err < 0.0f) err = -err;
            if (err > max_abs_err) max_abs_err = err;
        }
        if (max_abs_err > 0.000001f) {
            printf("sounio_nvidia_bare_runtime status=fail reason=%s_mismatch stage=verify_%s cuda_result=0",
                   vec_label,
                   vec_label);
            print_cuda_probe_suffix();
            printf(" %s_n=%u %s_max_abs_err=%.9g %s_eps_nonzero=%d observed0=%.9g expected0=%.9g observed_last=%.9g expected_last=%.9g\n",
                   vec_label, n_arg,
                   vec_label, max_abs_err,
                   vec_label, (epistemic || subtract || multiply || divide || fma),
                   h_out[0], h_expected[0], h_out[n_arg - 1], h_expected[n_arg - 1]);
            vec_failed = 1;
            goto vec_cleanup;
        }
        printf("sounio_nvidia_bare_runtime status=pass reason=%s stage=cuMemcpyDtoH cuda_result=0",
               divide ? "runtime_vec_div_f32_pass" : (multiply ? "runtime_vec_mul_f32_pass" : (fma ? "runtime_fma_f32_pass" : (subtract ? "runtime_vec_sub_f32_pass" : (epistemic ? "runtime_epistemic_elementwise_f32_pass" : "runtime_vec_add_f32_pass")))));
        print_cuda_probe_suffix();
        printf(" %s_n=%u %s_max_abs_err=%.9g %s_eps_nonzero=%d observed0=%.9g expected0=%.9g observed_last=%.9g expected_last=%.9g\n",
               vec_label, n_arg,
               vec_label, max_abs_err,
               vec_label, (epistemic || subtract || multiply || divide || fma),
               h_out[0], h_expected[0], h_out[n_arg - 1], h_expected[n_arg - 1]);

vec_cleanup:
        if (d_out) cuMemFree(d_out);
        if (d_eps) cuMemFree(d_eps);
        if (d_y) cuMemFree(d_y);
        if (d_x) cuMemFree(d_x);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        free(h_expected);
        free(h_out);
        free(h_eps);
        free(h_y);
        free(h_x);
#undef VEC_FAIL
        return vec_failed ? 1 : 0;
    }

    uint64_t dummy0 = 0;
    uint64_t dummy1 = 0;
    uint64_t dummy2 = 0;
    uint64_t dummy3 = 0;
    uint64_t dummy4 = 0;
    uint64_t dummy5 = 0;
    uint64_t dummy6 = 0;
    uint64_t dummy7 = 0;
    void *kernel_params[] = {
        &dummy0, &dummy1, &dummy2, &dummy3,
        &dummy4, &dummy5, &dummy6, &dummy7,
    };
    rc = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, kernel_params, NULL);
    if (rc != 0) { emit_status("fail", "cuLaunchKernel_failed", "cuLaunchKernel", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }
    rc = cuCtxSynchronize();
    if (rc != 0) { emit_status("fail", "cuCtxSynchronize_failed", "cuCtxSynchronize", rc); cuModuleUnload(mod); cuCtxDestroy(ctx); free(cubin); return 1; }

    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(cubin);
    emit_status("pass", "runtime_launch_pass", "cuLaunchKernel", 0);
    return 0;
}
