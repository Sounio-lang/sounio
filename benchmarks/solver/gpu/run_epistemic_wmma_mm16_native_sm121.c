// Driver-API launcher for the COMPILER-GENERATED epistemic WMMA kernel
// (self-hosted/gpu/kernel_ir.sio: gpu_build_epistemic_wmma_matmul_16x16_ir,
// emitted via self-hosted/gpu/kretikos_emit_epistemic_wmma.sio "epi_wmma_mm16").
//
// This is DISTINCT from run_epistemic_mma_native_sm121.c, which loads the
// hand-written self-hosted/gpu/epistemic_mma_reference.ptx kernel
// ("epistemic_mma_kernel") — a different formula, not compiler output. Do not
// conflate the two receipts.
//
// A=0, B=0 verification case (broadcast: D = A*B + C = C), matching the
// GUM RSS-quadrature epsilon law fixed in kernel_ir.sio:
//   term1 = |D[0]| * eps_B = 1 * 0.25 = 0.25
//   term2 = |D[2]| * eps_A = 3 * 0.5  = 1.5
//   U^2   = term1^2 + term2^2 = 0.0625 + 2.25 = 2.3125
//   eps_C = sqrt(16 * U^2) = sqrt(37) ~= 6.08276
//
// Provenance is NOT asserted to an exact value here: unlike the hand-written
// reference kernel (which passes a fixed input through), this kernel computes
// C_prv = A_val_ptr OR B_val_ptr from real device pointers (see the
// KNOWN LIMITATION comment in kernel_ir.sio) — allocation-order dependent,
// so it is printed for inspection only, not compared to a constant.
//
// Usage:
//   souc build self-hosted/gpu/kretikos_emit_epistemic_wmma.sio -o /tmp/kew.elf
//   /tmp/kew.elf epi_wmma_mm16 > epi_wmma_mm16.ptx
//   $CUDA/bin/ptxas -arch=sm_121 epi_wmma_mm16.ptx -o epi_wmma_mm16.cubin
//   gcc -O2 run_epistemic_wmma_mm16_native_sm121.c -o run_epi_wmma -lcuda -lm
//   ./run_epi_wmma epi_wmma_mm16.cubin
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
    const char* cubinpath = argc > 1 ? argv[1] : "/tmp/epi_wmma_mm16.cubin";

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

    CUmodule mod;
    chk(cuModuleLoad(&mod, cubinpath), "cuModuleLoad(cubin)");
    CUfunction fn;
    chk(cuModuleGetFunction(&fn, mod, "epi_wmma_mm16"), "getfn");
    printf("Loaded native cubin from: %s\n", cubinpath);

    // Params (8, all pointers): a_val, a_eps, b_val, b_eps, c_val, c_eps, c_vld, c_prv
    CUdeviceptr dAv, dAe, dBv, dBe, dCv, dCe, dVl, dPr;
    chk(cuMemAlloc(&dAv, 16), "aAv"); chk(cuMemAlloc(&dAe, 4), "aAe");
    chk(cuMemAlloc(&dBv,  8), "aBv"); chk(cuMemAlloc(&dBe, 4), "aBe");
    chk(cuMemAlloc(&dCv, 16), "aCv"); chk(cuMemAlloc(&dCe, 4), "aCe");
    chk(cuMemAlloc(&dVl,  8), "aVl"); chk(cuMemAlloc(&dPr, 8), "aPr");

    unsigned char zA[16] = {0};   // A = 0 (packed f16)
    unsigned char zB[8]  = {0};   // B = 0 (packed f16)
    float  Cv[4]  = {1.f, 2.f, 3.f, 4.f};
    float  epsA   = 0.5f;
    float  epsB   = 0.25f;
    float  Ce     = -1.f;
    unsigned long long vld = 0ULL;
    unsigned long long prv = 0ULL;

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

    float Dout[4]; float Ceout; unsigned long long vldout, prout;
    chk(cuMemcpyDtoH(Dout,  dCv, 16), "dCv");
    chk(cuMemcpyDtoH(&Ceout, dCe, 4), "dCe");
    chk(cuMemcpyDtoH(&vldout, dVl, 8), "dVl");
    chk(cuMemcpyDtoH(&prout, dPr, 8), "dPr");

    const float eps_expected = sqrtf(37.f);

    printf("D (expect 1,2,3,4): %.4f %.4f %.4f %.4f\n",
           Dout[0], Dout[1], Dout[2], Dout[3]);
    printf("eps_C (expect sqrt(37)=%.5f): %.5f\n", eps_expected, Ceout);
    printf("valid (informational): %llu\n", vldout);
    printf("prov  (informational, pointer-dependent): 0x%llx\n", prout);

    int okD = (Dout[0]==1.f && Dout[1]==2.f && Dout[2]==3.f && Dout[3]==4.f);
    int okE = fabsf(Ceout - eps_expected) < 1e-3f;
    printf("RESULT: dataPath=%s epistemicShadow(RSS-quadrature)=%s\n",
           okD?"PASS":"FAIL", okE?"PASS":"FAIL");

    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return (okD && okE) ? 0 : 2;
}
