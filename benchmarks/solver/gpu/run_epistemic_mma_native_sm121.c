// Driver-API launcher: load a NATIVE sm_121 cubin via cuModuleLoad (no JIT).
// Identical A=0 verification to run_epistemic_mma.c:
//   A=0 => D = A*B + C = C = [1,2,3,4], eps_C = sqrt(7) ~ 2.64575, prov = 0x5
//
// Usage:
//   gcc -O2 run_native_sm121.c -o run_native_sm121 -lcuda -lm
//   ./run_native_sm121 /path/to/epistemic_mma.cubin
//
// cuModuleLoad(CUmodule*, const char* path) reads a fatbin/cubin directly from
// disk; no JIT step, no PTX intermediary — pure SASS for sm_121 executes as-is.
// The .cubin must have been compiled with -arch=sm_121 (or a compatible fatbin).
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static void chk(CUresult r, const char* w){
    if(r != CUDA_SUCCESS){
        const char* m; cuGetErrorString(r, &m);
        fprintf(stderr, "ERR %s: %s\n", w, m); exit(1);
    }
}

int main(int argc, char** argv){
    const char* cubinpath = argc > 1 ? argv[1] : "/tmp/epistemic_mma.cubin";

    chk(cuInit(0), "init");
    CUdevice dev; chk(cuDeviceGet(&dev, 0), "devget");
    char name[128]; cuDeviceGetName(name, sizeof name, dev);
    int ccM = 0, ccm = 0;
    cuDeviceGetAttribute(&ccM, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&ccm, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    printf("device: %s  sm_%d%d\n", name, ccM, ccm);

    CUcontext ctx;
    chk(cuDevicePrimaryCtxRetain(&ctx, dev), "ctxRetain");
    chk(cuCtxSetCurrent(ctx), "ctxset");

    // Load native cubin directly from disk — no JIT, no PTX.
    // cuModuleLoad(CUmodule*, const char*) signature is stable through CUDA 13.
    CUmodule mod;
    chk(cuModuleLoad(&mod, cubinpath), "cuModuleLoad(cubin)");
    CUfunction fn;
    chk(cuModuleGetFunction(&fn, mod, "epistemic_mma_kernel"), "getfn");
    printf("Loaded native cubin from: %s\n", cubinpath);

    // Device buffers matching kernel signature:
    //   A_val[16B f16], A_eps[f32], B_val[8B f16], B_eps[f32],
    //   C_val[16B 4xf32 in/out], C_eps[f32 out], C_vld[u64], C_prv[u64]
    CUdeviceptr dAv, dAe, dBv, dBe, dCv, dCe, dVl, dPr;
    chk(cuMemAlloc(&dAv, 16), "aAv"); chk(cuMemAlloc(&dAe, 4), "aAe");
    chk(cuMemAlloc(&dBv,  8), "aBv"); chk(cuMemAlloc(&dBe, 4), "aBe");
    chk(cuMemAlloc(&dCv, 16), "aCv"); chk(cuMemAlloc(&dCe, 4), "aCe");
    chk(cuMemAlloc(&dVl,  8), "aVl"); chk(cuMemAlloc(&dPr, 8), "aPr");

    // A=0 (f16 zeros, 8 half-precision values packed into 16 bytes)
    // B=0 (f16 zeros, 4 half-precision values packed into 8 bytes)
    unsigned char zA[16] = {0};
    unsigned char zB[8]  = {0};
    float  Cv[4]  = {1.f, 2.f, 3.f, 4.f};
    float  epsA   = 0.5f;
    float  epsB   = 0.25f;
    float  Ce     = -1.f;           // output slot; kernel overwrites
    unsigned long long vld = 1ULL;
    unsigned long long prv = 0x5ULL;

    chk(cuMemcpyHtoD(dAv, zA, 16),  "hAv");
    chk(cuMemcpyHtoD(dAe, &epsA, 4),"hAe");
    chk(cuMemcpyHtoD(dBv, zB, 8),   "hBv");
    chk(cuMemcpyHtoD(dBe, &epsB, 4),"hBe");
    chk(cuMemcpyHtoD(dCv, Cv, 16),  "hCv");
    chk(cuMemcpyHtoD(dCe, &Ce, 4),  "hCe");
    chk(cuMemcpyHtoD(dVl, &vld, 8), "hVl");
    chk(cuMemcpyHtoD(dPr, &prv, 8), "hPr");

    void* args[8] = {&dAv, &dAe, &dBv, &dBe, &dCv, &dCe, &dVl, &dPr};
    chk(cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, 0), "launch");
    chk(cuCtxSynchronize(), "sync");

    float Dout[4]; float Ceout; unsigned long long prout;
    chk(cuMemcpyDtoH(Dout,  dCv, 16), "dCv");
    chk(cuMemcpyDtoH(&Ceout, dCe, 4), "dCe");
    chk(cuMemcpyDtoH(&prout, dPr, 8), "dPr");

    printf("D (expect 1,2,3,4): %.4f %.4f %.4f %.4f\n",
           Dout[0], Dout[1], Dout[2], Dout[3]);
    printf("eps_C (expect sqrt(7)=%.5f): %.5f\n", sqrtf(7.f), Ceout);
    printf("prov  (expect 0x5): 0x%llx\n", prout);

    int okD = (Dout[0]==1.f && Dout[1]==2.f && Dout[2]==3.f && Dout[3]==4.f);
    int okE = fabsf(Ceout - sqrtf(7.f)) < 1e-3f;
    int okP = (prout == 0x5ULL);
    printf("RESULT: dataPath=%s epistemicShadow=%s provenance=%s\n",
           okD?"PASS":"FAIL", okE?"PASS":"FAIL", okP?"PASS":"FAIL");

    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return (okD && okE && okP) ? 0 : 2;
}
