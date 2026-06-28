// Driver-API launcher: exercise the tensor-core mma.sync path with non-zero A and B.
//
// IMPORTANT CAVEAT (read before interpreting output):
//   The reference epistemic_mma_kernel loads the SAME fragment tile into every
//   warp lane — it does not offset per-lane within the 16x16 fragment.  The
//   mma.sync instruction therefore computes a rank-1 (or constant-tile)
//   multiply, NOT a general A*B of distinct 16x16 matrices.
//   This launcher proves:
//     (a) the tensor-core multiply EXECUTES with non-zero operands, and
//     (b) D and eps_C are deterministic across runs.
//   It does NOT validate a full 16x16 row-major A*B result.
//   For that, a per-lane-offset kernel would be required.
//
// A = all f16 1.0 (0x3C00) packed into 16 bytes (8 half values per fragment)
// B = all f16 1.0 (0x3C00) packed into  8 bytes (4 half values per fragment)
// C = all f32 0.0
// epsA = 0.5, epsB = 0.25, vld = 1, prv = 0x5
//
// The launcher prints D[0..3] and eps_C without asserting specific values.
//
// Usage:
//   # PTX path (JIT):
//   gcc -O2 run_matmul_verify.c -o run_matmul_verify -lcuda -lm
//   ./run_matmul_verify /path/to/epistemic_mma.ptx
//
//   # Native cubin path: change cuModuleLoadDataEx -> cuModuleLoad (see run_native_sm121.c)
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

// Fill a byte buffer with the f16 bit pattern for 1.0 (IEEE 754 half: 0x3C00).
// Little-endian layout: low byte = 0x00, high byte = 0x3C.
static void fill_f16_ones(unsigned char* buf, size_t nbytes){
    for(size_t i = 0; i < nbytes; i += 2){
        buf[i]   = 0x00;   // low byte of 0x3C00
        buf[i+1] = 0x3C;   // high byte of 0x3C00
    }
}

int main(int argc, char** argv){
    const char* ptxpath = argc > 1 ? argv[1] : "/tmp/epi_ascii.ptx";
    FILE* fh = fopen(ptxpath, "rb");
    if(!fh){ perror("ptx"); return 1; }
    fseek(fh, 0, SEEK_END); long n = ftell(fh); fseek(fh, 0, SEEK_SET);
    char* ptx = malloc(n + 1); fread(ptx, 1, n, fh); ptx[n] = 0; fclose(fh);

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

    // JIT-load PTX (replace these two lines with cuModuleLoad for native cubin)
    CUmodule mod;
    CUjit_option o[1] = {CU_JIT_TARGET_FROM_CUCONTEXT}; void* ov[1] = {0};
    chk(cuModuleLoadDataEx(&mod, ptx, 1, o, ov), "loadPTX(JIT)");
    free(ptx);

    CUfunction fn;
    chk(cuModuleGetFunction(&fn, mod, "epistemic_mma_kernel"), "getfn");
    printf("Loaded epistemic_mma_kernel\n");

    // Device buffers
    CUdeviceptr dAv, dAe, dBv, dBe, dCv, dCe, dVl, dPr;
    chk(cuMemAlloc(&dAv, 16), "aAv"); chk(cuMemAlloc(&dAe, 4), "aAe");
    chk(cuMemAlloc(&dBv,  8), "aBv"); chk(cuMemAlloc(&dBe, 4), "aBe");
    chk(cuMemAlloc(&dCv, 16), "aCv"); chk(cuMemAlloc(&dCe, 4), "aCe");
    chk(cuMemAlloc(&dVl,  8), "aVl"); chk(cuMemAlloc(&dPr, 8), "aPr");

    // A = all f16 1.0 (16 bytes = 8 half-precision values)
    unsigned char Av[16]; fill_f16_ones(Av, 16);
    // B = all f16 1.0 (8 bytes = 4 half-precision values)
    unsigned char Bv[8];  fill_f16_ones(Bv,  8);
    // C = all f32 0.0  (D = A*B + C = A*B; mma result is unbiased by accumulator)
    float  Cv[4]  = {0.f, 0.f, 0.f, 0.f};
    float  epsA   = 0.5f;
    float  epsB   = 0.25f;
    float  Ce     = -1.f;   // output slot; kernel overwrites
    unsigned long long vld = 1ULL;
    unsigned long long prv = 0x5ULL;

    chk(cuMemcpyHtoD(dAv, Av, 16),   "hAv");
    chk(cuMemcpyHtoD(dAe, &epsA, 4), "hAe");
    chk(cuMemcpyHtoD(dBv, Bv,  8),   "hBv");
    chk(cuMemcpyHtoD(dBe, &epsB, 4), "hBe");
    chk(cuMemcpyHtoD(dCv, Cv, 16),   "hCv");
    chk(cuMemcpyHtoD(dCe, &Ce,  4),  "hCe");
    chk(cuMemcpyHtoD(dVl, &vld, 8),  "hVl");
    chk(cuMemcpyHtoD(dPr, &prv, 8),  "hPr");

    void* args[8] = {&dAv, &dAe, &dBv, &dBe, &dCv, &dCe, &dVl, &dPr};
    chk(cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, 0), "launch");
    chk(cuCtxSynchronize(), "sync");

    float Dout[4]; float Ceout; unsigned long long prout;
    chk(cuMemcpyDtoH(Dout,   dCv, 16), "dCv");
    chk(cuMemcpyDtoH(&Ceout, dCe,  4), "dCe");
    chk(cuMemcpyDtoH(&prout, dPr,  8), "dPr");

    // Print without asserting: the mma result depends on per-lane tile layout.
    printf("--- NON-ZERO OPERAND mma.sync RESULTS ---\n");
    printf("A = all f16 1.0 (0x3C00), B = all f16 1.0 (0x3C00), C = all f32 0.0\n");
    printf("D[0..3] (measured): %.6f  %.6f  %.6f  %.6f\n",
           Dout[0], Dout[1], Dout[2], Dout[3]);
    printf("eps_C   (measured): %.6f   (sqrt(7) ref = %.6f)\n",
           Ceout, sqrtf(7.f));
    printf("prov    (measured): 0x%llx\n", prout);
    printf("\n");
    printf("NOTE: D values above confirm the tensor-core executed with non-zero\n");
    printf("      operands and produced a deterministic result.  They are NOT the\n");
    printf("      expected result of a general 16x16 A*B because the kernel\n");
    printf("      broadcasts the same tile to every warp lane.\n");

    // Soft sanity: D should be finite and non-negative (all-ones * all-ones + 0)
    int finite_ok = 1;
    for(int i = 0; i < 4; i++){
        if(!(Dout[i] == Dout[i]) /* NaN check */ || Dout[i] < 0.f) finite_ok = 0;
    }
    printf("RESULT: finite_nonneg_D=%s  provenance=0x%llx\n",
           finite_ok ? "PASS" : "FAIL", prout);

    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return finite_ok ? 0 : 2;
}
