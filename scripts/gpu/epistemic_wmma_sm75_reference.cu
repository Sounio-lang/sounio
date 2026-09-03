// epistemic_wmma_sm75_reference.cu
// ───────────────────────────────────────────────────────────────────────────
// GUM-correct uncertainty propagation through Turing (sm_75) tensor cores.
//
// Validates the variance-space double-matmul law from
//   docs/design/EPISTEMIC_TENSOR_CORE_GUM_TURING.md
// on real RTX 8000 / Quadro Turing hardware, BEFORE the Sounio emitter is wired
// to produce equivalent PTX (the emitter currently emits Ampere-only m16n8k16 /
// TF32 forms — see docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md).
//
// This is a HARDWARE VALIDATION HARNESS, not the science path. The end goal is
// Sounio-emitted PTX; this C++/wmma reference exists only to confirm the math and
// Turing-legality on silicon (consistent with the existing C runners in scripts/gpu/).
//
// LAW (GUM JCGM 100:2008 §5.1.2, uncorrelated inputs), for D = A·B:
//   u²(D[i,j]) = Σ_k [ B[k,j]²·u²(A[i,k]) + A[i,k]²·u²(B[k,j]) ]
//   In matrix form (VA=u²(A), VB=u²(B), ⊙ = elementwise square):
//       U2 = VA·(B⊙B) + (A⊙A)·VB          ← two extra matmuls, same contraction
//       u(D) = sqrt(U2)
//
// Tensor core: wmma m16n16k16, f16 inputs, f32 accumulate — supported on sm_70+,
// hence valid on Turing sm_75. (m16n8k16 mma.sync and TF32 wmma are Ampere+ and
// would be REJECTED by ptxas -arch=sm_75.)
//
// Build (on the RTX 8000 host):
//   nvcc -arch=sm_75 -o epi_wmma scripts/gpu/epistemic_wmma_sm75_reference.cu
// Run:
//   ./epi_wmma
// Expect: "PASS" with small relative error vs the CPU GUM oracle (f16 tolerance).
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

static const int M = 16, N = 16, K = 16;   // single warp tile

// Kernel: one warp computes value D = A·B and uncertainty U2 = VA·Bsq + Asq·VB.
// Inputs are pre-squared on the host for the variance terms (Asq, Bsq) so this
// kernel is pure tensor-core contraction — exactly what the emitter will lower.
__global__ void epistemic_wmma(const half* A,  const half* B,
                               const half* Asq, const half* Bsq,
                               const half* VA, const half* VB,
                               float* D, float* U2) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> d_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> u_frag;

    // ── value path: D = A·B ─────────────────────────────────────────────
    wmma::fill_fragment(d_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, N);
    wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

    // ── variance term 1: VA · Bsq ───────────────────────────────────────
    wmma::fill_fragment(u_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, VA, K);
    wmma::load_matrix_sync(b_frag, Bsq, N);
    wmma::mma_sync(u_frag, a_frag, b_frag, u_frag);   // u += VA·Bsq

    // ── variance term 2: + Asq · VB  (accumulate into same u_frag) ───────
    wmma::load_matrix_sync(a_frag, Asq, K);
    wmma::load_matrix_sync(b_frag, VB, N);
    wmma::mma_sync(u_frag, a_frag, b_frag, u_frag);   // u += Asq·VB  ⇒ U2

    wmma::store_matrix_sync(D,  d_frag, N, wmma::mem_row_major);
    wmma::store_matrix_sync(U2, u_frag, N, wmma::mem_row_major);
    // u(D) = sqrt(U2) is applied at read-out (host) to keep the kernel a pure
    // contraction; in the fused emitter, sqrt.approx.f32 is the store-time op.
}

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } }while(0)

int main() {
    int dev = 0; cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) { printf("no CUDA device\n"); return 1; }
    printf("GPU: %s  sm_%d%d\n", p.name, p.major, p.minor);
    if (p.major*10 + p.minor < 75) printf("WARN: expected sm_75+ (Turing); got sm_%d%d\n", p.major, p.minor);

    const int n = M*N;
    float hA[n], hB[n], hVA[n], hVB[n];
    half hAh[n], hBh[n], hAsq[n], hBsq[n], hVAh[n], hVBh[n];

    // Deterministic, bounded inputs (no Math.random): structured ramps.
    for (int i=0;i<M;i++) for (int j=0;j<K;j++){
        float a = 0.5f + 0.01f*((i*K+j)%7);     // ~[0.5,1.16]
        float va = 0.001f*(1 + ((i*K+j)%5));    // u²(A) small, varied
        hA[i*K+j]=a; hVA[i*K+j]=va;
        hAh[i*K+j]=__float2half(a); hAsq[i*K+j]=__float2half(a*a); hVAh[i*K+j]=__float2half(va);
    }
    for (int i=0;i<K;i++) for (int j=0;j<N;j++){
        float b = 0.4f + 0.02f*((i*N+j)%6);
        float vb = 0.002f*(1 + ((i*N+j)%4));
        hB[i*N+j]=b; hVB[i*N+j]=vb;
        hBh[i*N+j]=__float2half(b); hBsq[i*N+j]=__float2half(b*b); hVBh[i*N+j]=__float2half(vb);
    }

    half *dA,*dB,*dAsq,*dBsq,*dVA,*dVB; float *dD,*dU2;
    CK(cudaMalloc(&dA,n*2)); CK(cudaMalloc(&dB,n*2)); CK(cudaMalloc(&dAsq,n*2));
    CK(cudaMalloc(&dBsq,n*2)); CK(cudaMalloc(&dVA,n*2)); CK(cudaMalloc(&dVB,n*2));
    CK(cudaMalloc(&dD,n*4)); CK(cudaMalloc(&dU2,n*4));
    CK(cudaMemcpy(dA,hAh,n*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hBh,n*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAsq,hAsq,n*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBsq,hBsq,n*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVA,hVAh,n*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVB,hVBh,n*2,cudaMemcpyHostToDevice));

    epistemic_wmma<<<1,32>>>(dA,dB,dAsq,dBsq,dVA,dVB,dD,dU2);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    float gD[n], gU2[n];
    CK(cudaMemcpy(gD,dD,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gU2,dU2,n*4,cudaMemcpyDeviceToHost));

    // ── CPU GUM oracle (f32) ─────────────────────────────────────────────
    float maxRelD=0, maxRelU=0;
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        float d=0, u2=0;
        for (int k=0;k<K;k++){
            float a=hA[i*K+k], b=hB[k*N+j], va=hVA[i*K+k], vb=hVB[k*N+j];
            d  += a*b;
            u2 += b*b*va + a*a*vb;            // exact first-order GUM
        }
        float rd = fabsf(gD[i*N+j]-d)/(fabsf(d)+1e-6f);
        float ru = fabsf(gU2[i*N+j]-u2)/(fabsf(u2)+1e-6f);
        if (rd>maxRelD) maxRelD=rd;
        if (ru>maxRelU) maxRelU=ru;
    }
    printf("max rel err  value D = %.4f   uncertainty U2 = %.4f\n", maxRelD, maxRelU);
    // f16 inputs ⇒ ~1e-2 rel error is expected and acceptable for the reference.
    bool pass = (maxRelD < 0.03f) && (maxRelU < 0.05f);
    printf("%s — GUM variance-space propagation on Turing tensor cores\n", pass?"PASS":"FAIL");
    return pass?0:2;
}
