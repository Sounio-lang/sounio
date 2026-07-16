// Self-contained tensor-core octonion multiply (Convention X): the kernel takes RAW octonions
// a (8 f32) and B (16 columns × 8 f32), builds the left-multiply matrix L(a)[k][j]=σ(k⊕j,j)·a[k⊕j]
// IN-KERNEL into shared memory (f16), stages B in shared, runs one wmma m16n16k16 tile, and stores
// D = L(a)·B = the 16 octonion products a·b_c. Proof + nvcc-ptx ground truth for the Sounio emitter.
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__device__ int cd_sigma_dev(int a,int b){         // bits=3 (octonion), iterative-safe recursion
    if(a==0||b==0) return 1;
    // depth <= 3
    int sign=1, bits=3;
    while(bits>0){
        if(a==0||b==0) return sign;
        if(bits==1) return -sign;                  // a,b both nonzero at bits==1 -> e·e=-1
        int h=1<<(bits-1), ah=a>=h, bh=b>=h, al=a&(h-1), bl=b&(h-1);
        if(!ah&&!bh){ a=al; b=bl; }
        else if(!ah&&bh){ a=bl; b=al; }
        else if(ah&&!bh){ if(bl==0){ a=al; b=0; } else { sign=-sign; a=al; b=bl; } }
        else { if(bl==0){ sign=-sign; a=0; b=al; } else { a=bl; b=al; } }
        bits--;
    }
    return sign;
}

__global__ void oct_mul_full(const float* a, const float* B, float* D){
    __shared__ half sL[256];   // L(a) 16x16 row-major
    __shared__ half sB[256];   // B    16x16 col-major
    int t = threadIdx.x;       // one warp: 0..31
    for(int idx=t; idx<256; idx+=32){
        int k=idx>>4, j=idx&15;
        if(k<8 && j<8){ int i=k^j; sL[idx]=__float2half((float)cd_sigma_dev(i,j)*a[i]); }
        else sL[idx]=__float2half(0.0f);
    }
    for(int idx=t; idx<256; idx+=32){
        int c=idx>>4, r=idx&15;                    // col-major: sB[c*16+r] = B[c][r]
        sB[idx]= (r<8) ? __float2half(B[c*8+r]) : __float2half(0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> aF;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bF;
    wmma::fragment<wmma::accumulator,16,16,16,float> cF;
    wmma::fill_fragment(cF,0.0f);
    wmma::load_matrix_sync(aF,sL,16);
    wmma::load_matrix_sync(bF,sB,16);
    wmma::mma_sync(cF,aF,bF,cF);
    wmma::store_matrix_sync(D,cF,16,wmma::mem_row_major);
}

int cd_sigma(int a,int b,int bits){
    if(a==0||b==0) return 1; if(bits<=1) return -1;
    int h=1<<(bits-1), ah=a>=h, bh=b>=h, al=a&(h-1), bl=b&(h-1);
    if(!ah&&!bh) return cd_sigma(al,bl,bits-1);
    if(!ah&&bh)  return cd_sigma(bl,al,bits-1);
    if(ah&&!bh)  return bl==0? cd_sigma(al,0,bits-1) : -cd_sigma(al,bl,bits-1);
    if(bl==0) return -cd_sigma(0,al,bits-1);
    return cd_sigma(bl,al,bits-1);
}
void octmul_ref(const float*a,const float*b,float*r){ for(int k=0;k<8;k++)r[k]=0;
    for(int i=0;i<8;i++)for(int j=0;j<8;j++) r[i^j]+=cd_sigma(i,j,3)*a[i]*b[j]; }

int main(){
    float a[8]={1,2,0,-1,3,0,1,-2};
    float Bc[16*8]; for(int c=0;c<16;c++)for(int j=0;j<8;j++) Bc[c*8+j]=((c+1)*0.5f+j*0.25f)*((j%2)?-1:1);
    float *da,*dB,*dD; cudaMalloc(&da,8*4); cudaMalloc(&dB,16*8*4); cudaMalloc(&dD,256*4);
    cudaMemcpy(da,a,8*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,Bc,16*8*4,cudaMemcpyHostToDevice);
    oct_mul_full<<<1,32>>>(da,dB,dD);
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("ERR %s\n",cudaGetErrorString(e));return 2;}
    float hD[256]; cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    int fails=0; float maxerr=0;
    for(int c=0;c<16;c++){ float ref[8]; octmul_ref(a,&Bc[c*8],ref);
        for(int k=0;k<8;k++){ float err=fabsf(hD[k*16+c]-ref[k]); if(err>maxerr)maxerr=err; if(err>0.15f)fails++; } }
    // e1·e2 discriminator (a=e1, B col0 = e2)
    float a2[8]={0,1,0,0,0,0,0,0}, B2[16*8]={0}; B2[0*8+2]=1.0f;
    cudaMemcpy(da,a2,8*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,B2,16*8*4,cudaMemcpyHostToDevice);
    oct_mul_full<<<1,32>>>(da,dB,dD); cudaDeviceSynchronize(); cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    printf("in-kernel L(a): e1*e2 comp3=%.2f comp4=%.2f | batch %d/128 mismatch maxerr=%.3f\n",hD[3*16+0],hD[4*16+0],fails,maxerr);
    if(fails==0 && hD[3*16+0]>0.7f && hD[4*16+0]<0.3f){ printf("PASS: in-kernel WMMA octonion multiply is Convention X on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
