// HIP runtime loader harness for Sounio-owned HSACO runtime rungs.
//
// Build: cc -O2 scripts/amd/hip_bare_driver_loader.c -ldl -o /tmp/sounio_hip_loader
// Run:   /tmp/sounio_hip_loader <hsaco> <kernel> [admission|vec_add_f32]
//
// This loader dlopen()s libamdhip64.so so it can be built on hosts without
// the ROCm dev headers, as long as the target runtime node has the library.

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal HIP types
typedef int hipDevice_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipStream_t;
typedef uint64_t hipDeviceptr_t;
typedef int hipError_t;
#define hipSuccess 0

// Function pointers loaded dynamically
typedef hipError_t (*hipInit_t)(unsigned int);
typedef hipError_t (*hipDeviceGet_t)(hipDevice_t *, int);
typedef hipError_t (*hipDeviceGetName_t)(char *, int, hipDevice_t);
typedef hipError_t (*hipSetDevice_t)(int);
typedef hipError_t (*hipModuleLoadData_t)(hipModule_t *, const void *);
typedef hipError_t (*hipModuleGetFunction_t)(hipFunction_t *, hipModule_t, const char *);
typedef hipError_t (*hipMalloc_t)(hipDeviceptr_t *, size_t);
typedef hipError_t (*hipFree_t)(hipDeviceptr_t);
typedef hipError_t (*hipMemcpyHtoD_t)(hipDeviceptr_t, const void *, size_t);
typedef hipError_t (*hipMemcpyDtoH_t)(void *, hipDeviceptr_t, size_t);
typedef hipError_t (*hipModuleLaunchKernel_t)(
    hipFunction_t,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, hipStream_t, void **, void **);
typedef hipError_t (*hipDeviceSynchronize_t)(void);

static int g_device_count = -1;
static char g_device_name[128] = "unknown";

static void emit_status(const char *status, const char *reason, const char *stage, hipError_t hip_result) {
    printf("sounio_hip_bare_runtime status=%s reason=%s stage=%s hip_result=%d device_count=%d device_name=%s\n",
           status, reason, stage, hip_result, g_device_count, g_device_name);
}

static unsigned char *read_all(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    unsigned char *buf = (unsigned char *)malloc((size_t)n);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <hsaco> <kernel> [admission|vec_add_f32]\n", argv[0]);
        return 2;
    }
    const char *mode = argc >= 4 ? argv[3] : "admission";

    size_t hsaco_len = 0;
    unsigned char *hsaco = read_all(argv[1], &hsaco_len);
    if (!hsaco || hsaco_len == 0) {
        emit_status("not_run", "hsaco_read_failed", "read_all", 0);
        return 77;
    }

    void *hip = dlopen("libamdhip64.so", RTLD_NOW);
    if (!hip) hip = dlopen("libamdhip64.so.6", RTLD_NOW);
    if (!hip) hip = dlopen("libamdhip64.so.5", RTLD_NOW);
    if (!hip) {
        emit_status("not_run", "hip_driver_missing", "dlopen", 0);
        free(hsaco);
        return 77;
    }

    hipInit_t hipInit = (hipInit_t)dlsym(hip, "hipInit");
    hipDeviceGet_t hipDeviceGet = (hipDeviceGet_t)dlsym(hip, "hipDeviceGet");
    hipDeviceGetName_t hipDeviceGetName = (hipDeviceGetName_t)dlsym(hip, "hipDeviceGetName");
    hipSetDevice_t hipSetDevice = (hipSetDevice_t)dlsym(hip, "hipSetDevice");
    hipModuleLoadData_t hipModuleLoadData = (hipModuleLoadData_t)dlsym(hip, "hipModuleLoadData");
    hipModuleGetFunction_t hipModuleGetFunction = (hipModuleGetFunction_t)dlsym(hip, "hipModuleGetFunction");
    hipMalloc_t hipMalloc = (hipMalloc_t)dlsym(hip, "hipMalloc");
    hipFree_t hipFree = (hipFree_t)dlsym(hip, "hipFree");
    hipMemcpyHtoD_t hipMemcpyHtoD = (hipMemcpyHtoD_t)dlsym(hip, "hipMemcpyHtoD");
    hipMemcpyDtoH_t hipMemcpyDtoH = (hipMemcpyDtoH_t)dlsym(hip, "hipMemcpyDtoH");
    hipModuleLaunchKernel_t hipModuleLaunchKernel = (hipModuleLaunchKernel_t)dlsym(hip, "hipModuleLaunchKernel");
    hipDeviceSynchronize_t hipDeviceSynchronize = (hipDeviceSynchronize_t)dlsym(hip, "hipDeviceSynchronize");

    if (!hipInit || !hipDeviceGet || !hipSetDevice || !hipModuleLoadData || !hipModuleGetFunction ||
        !hipMalloc || !hipFree || !hipMemcpyHtoD || !hipMemcpyDtoH || !hipModuleLaunchKernel || !hipDeviceSynchronize) {
        emit_status("not_run", "hip_symbol_missing", "dlsym", 0);
        free(hsaco);
        return 77;
    }

    hipError_t r = hipInit(0);
    if (r != hipSuccess) { emit_status("fail", "hipInit", "init", r); free(hsaco); return 1; }

    hipDevice_t dev;
    r = hipDeviceGet(&dev, 0);
    if (r != hipSuccess) { emit_status("fail", "hipDeviceGet", "init", r); free(hsaco); return 1; }

    r = hipSetDevice(0);
    if (r != hipSuccess) { emit_status("fail", "hipSetDevice", "init", r); free(hsaco); return 1; }

    g_device_count = 1;
    hipDeviceGetName(g_device_name, sizeof(g_device_name), dev);

    hipModule_t mod;
    r = hipModuleLoadData(&mod, hsaco);
    if (r != hipSuccess) { emit_status("fail", "hipModuleLoadData", "load", r); free(hsaco); return 1; }

    hipFunction_t func;
    r = hipModuleGetFunction(&func, mod, argv[2]);
    if (r != hipSuccess) { emit_status("fail", "hipModuleGetFunction", "load", r); free(hsaco); return 1; }

    if (strcmp(mode, "admission") == 0) {
        emit_status("PASS", "admission_only", "admission", 0);
        free(hsaco);
        return 0;
    }

    if (strcmp(mode, "vec_add_f32") == 0) {
        const int N = 1024;
        float *h_a = (float *)malloc(N * sizeof(float));
        float *h_b = (float *)malloc(N * sizeof(float));
        float *h_out = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i) {
            h_a[i] = (float)i;
            h_b[i] = (float)(N - i);
        }

        hipDeviceptr_t d_a, d_b, d_out;
        r = hipMalloc(&d_a, N * sizeof(float));
        if (r != hipSuccess) { emit_status("fail", "hipMalloc", "launch", r); free(hsaco); return 1; }
        r = hipMalloc(&d_b, N * sizeof(float));
        if (r != hipSuccess) { emit_status("fail", "hipMalloc", "launch", r); free(hsaco); return 1; }
        r = hipMalloc(&d_out, N * sizeof(float));
        if (r != hipSuccess) { emit_status("fail", "hipMalloc", "launch", r); free(hsaco); return 1; }

        hipMemcpyHtoD(d_a, h_a, N * sizeof(float));
        hipMemcpyHtoD(d_b, h_b, N * sizeof(float));

        int n = N;
        void *args[] = { &d_a, &d_b, &d_out, &n };
        r = hipModuleLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        if (r != hipSuccess) { emit_status("fail", "hipModuleLaunchKernel", "launch", r); free(hsaco); return 1; }

        r = hipDeviceSynchronize();
        if (r != hipSuccess) { emit_status("fail", "hipDeviceSynchronize", "sync", r); free(hsaco); return 1; }

        hipMemcpyDtoH(h_out, d_out, N * sizeof(float));

        int errors = 0;
        for (int i = 0; i < N; ++i) {
            float expected = h_a[i] + h_b[i];
            if (h_out[i] != expected) errors++;
        }

        hipFree(d_a); hipFree(d_b); hipFree(d_out);
        free(h_a); free(h_b); free(h_out);

        if (errors == 0) {
            emit_status("PASS", "vec_add_f32_oracle_match", "verify", 0);
            free(hsaco);
            return 0;
        } else {
            emit_status("FAIL", "oracle_mismatch", "verify", 0);
            free(hsaco);
            return 1;
        }
    }

    emit_status("FAIL", "unknown_mode", "usage", 0);
    free(hsaco);
    return 2;
}
