// Tensor-core (WMMA) octonion multiply in canonical Convention X, validated on GB10.
// Octonion product a·b is linear in b: (a·b) = L(a)·b, where the 8x8 left-mul matrix bakes in
// the Cayley-Dickson signs:  L(a)[k][j] = sigma(k^j, j) * a[k^j].  A plain signed matmul then
// yields the correct X product (e1·e2=+e3) with NO separate sign correction.
// We pad L(a) to 16x16 and multiply by B (16x16, columns = up to 16 different b vectors) using
// one wmma m16n16k16 f16 tile, then compare D against the exact X reference on host.
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// recursive Cayley-Dickson twist sign (host)
int cd_sigma(int a,int b,int bits){
    if(a==0||b==0) return 1;
    if(bits<=1) return -1;
    int h=1<<(bits-1), ah=a>=h, bh=b>=h, al=a&(h-1), bl=b&(h-1);
    if(!ah&&!bh) return cd_sigma(al,bl,bits-1);
    if(!ah&&bh)  return cd_sigma(bl,al,bits-1);
    if(ah&&!bh)  return bl==0? cd_sigma(al,0,bits-1) : -cd_sigma(al,bl,bits-1);
    if(bl==0) return -cd_sigma(0,al,bits-1);
    return cd_sigma(bl,al,bits-1);
}
void octmul_ref(const float*a,const float*b,float*r){ // exact X reference
    for(int k=0;k<8;k++) r[k]=0;
    for(int i=0;i<8;i++)for(int j=0;j<8;j++) r[i^j]+=cd_sigma(i,j,3)*a[i]*b[j];
}

__global__ void oct_mul_wmma(const half* L, const half* B, float* D){
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> aF;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bF;
    wmma::fragment<wmma::accumulator,16,16,16,float> cF;
    wmma::fill_fragment(cF,0.0f);
    wmma::load_matrix_sync(aF,L,16);
    wmma::load_matrix_sync(bF,B,16);
    wmma::mma_sync(cF,aF,bF,cF);
    wmma::store_matrix_sync(D,cF,16,wmma::mem_row_major);
}

int main(){
    // fixed left operand a (nontrivial), 16 different right operands b_col
    float a[8]={1,2,0,-1,3,0,1,-2};
    float bcols[16][8];
    for(int c=0;c<16;c++) for(int j=0;j<8;j++) bcols[c][j]=((c+1)*0.5f + j*0.25f)*((j%2)?-1:1);
    // build L(a) 16x16 row-major (only 8x8 nonzero)
    half hL[256]; for(int i=0;i<256;i++) hL[i]=__float2half(0.0f);
    for(int k=0;k<8;k++)for(int j=0;j<8;j++){ int i=k^j; hL[k*16+j]=__float2half((float)cd_sigma(i,j,3)*a[i]); }
    // B 16x16 col-major: column c holds b_c in first 8 rows
    half hB[256]; for(int i=0;i<256;i++) hB[i]=__float2half(0.0f);
    for(int c=0;c<16;c++)for(int j=0;j<8;j++) hB[c*16+j]=__float2half(bcols[c][j]); // col-major: [col*16+row]
    half *dL,*dB; float *dD; cudaMalloc(&dL,256*2); cudaMalloc(&dB,256*2); cudaMalloc(&dD,256*4);
    cudaMemcpy(dL,hL,256*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,256*2,cudaMemcpyHostToDevice);
    oct_mul_wmma<<<1,32>>>(dL,dB,dD);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("CUDA ERROR: %s\n",cudaGetErrorString(e)); return 2; }
    float hD[256]; cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    // D is row-major 16x16: D[k*16+c] = (L(a)·B)[k][c] = (a·b_c)[k]
    int fails=0; float maxerr=0;
    for(int c=0;c<16;c++){
        float ref[8]; octmul_ref(a,bcols[c],ref);
        for(int k=0;k<8;k++){ float got=hD[k*16+c]; float err=fabsf(got-ref[k]); if(err>maxerr)maxerr=err;
            if(err>0.15f){ fails++; if(fails<=4) printf("  mismatch c=%d k=%d got=%.3f ref=%.3f\n",c,k,got,ref[k]); } }
    }
    // discriminator: e1·e2 must land in component 3 (X), not 4 (Y)
    float e1[8]={0,1,0,0,0,0,0,0}, e2[8]={0,0,1,0,0,0,0,0};
    half hL2[256]; for(int i=0;i<256;i++)hL2[i]=__float2half(0.0f);
    for(int k=0;k<8;k++)for(int j=0;j<8;j++){int i=k^j; hL2[k*16+j]=__float2half((float)cd_sigma(i,j,3)*e1[i]);}
    half hB2[256]; for(int i=0;i<256;i++)hB2[i]=__float2half(0.0f);
    for(int j=0;j<8;j++) hB2[0*16+j]=__float2half(e2[j]);
    cudaMemcpy(dL,hL2,256*2,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB2,256*2,cudaMemcpyHostToDevice);
    oct_mul_wmma<<<1,32>>>(dL,dB,dD); cudaDeviceSynchronize();
    cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    float c3=hD[3*16+0], c4=hD[4*16+0];
    printf("e1*e2 on tensor core: comp3=%.2f comp4=%.2f  (X: comp3=+1,comp4=0)\n",c3,c4);
    printf("batch: %d/128 comps mismatch, maxerr=%.3f (f16 tile precision)\n",fails,maxerr);
    if(fails==0 && c3>0.7f && c4<0.3f){ printf("PASS: WMMA octonion multiply is Convention X on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
