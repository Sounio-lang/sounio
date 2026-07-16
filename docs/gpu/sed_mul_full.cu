// Self-contained tensor-core SEDENION multiply (Convention X): kernel takes raw a (16 f32) and
// B (16 cols × 16 f32), builds L(a)[k][j]=σ(k⊕j,j,4)·a[k⊕j] in shared (f16) — L(a) is 16×16, EXACTLY
// one wmma m16n16k16 tile, NO padding — runs the tile, stores D = L(a)·B = the 16 sedenion products.
// Validates the X product AND the canonical zero divisor (e3+e10)·(e6−e15)=0 on the tensor core.
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__device__ int cd_sigma_dev(int a,int b,int bits){
    int sign=1;
    while(bits>0){
        if(a==0||b==0) return sign;
        if(bits==1) return -sign;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;}
        else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else {sign=-sign;a=al;b=bl;} }
        else { if(bl==0){sign=-sign;a=0;b=al;} else {a=bl;b=al;} }
        bits--;
    }
    return sign;
}
__global__ void sed_mul_full(const float* a, const float* B, float* D){
    __shared__ half sL[256];   // L(a) 16x16 row-major  (full tile, no padding)
    __shared__ half sB[256];   // B    16x16 col-major
    int t=threadIdx.x;
    for(int idx=t; idx<256; idx+=32){
        int k=idx>>4, j=idx&15, i=k^j;
        sL[idx]=__float2half((float)cd_sigma_dev(i,j,4)*a[i]);
    }
    for(int idx=t; idx<256; idx+=32){
        int c=idx>>4, r=idx&15;                // col-major sB[c*16+r]=B[c][r]
        sB[idx]=__float2half(B[c*16+r]);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> aF;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bF;
    wmma::fragment<wmma::accumulator,16,16,16,float> cF;
    wmma::fill_fragment(cF,0.0f);
    wmma::load_matrix_sync(aF,sL,16); wmma::load_matrix_sync(bF,sB,16);
    wmma::mma_sync(cF,aF,bF,cF);
    wmma::store_matrix_sync(D,cF,16,wmma::mem_row_major);
}
int cd_sigma(int a,int b,int bits){ if(a==0||b==0)return 1; if(bits<=1)return -1;
  int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
  if(!ah&&!bh)return cd_sigma(al,bl,bits-1); if(!ah&&bh)return cd_sigma(bl,al,bits-1);
  if(ah&&!bh)return bl==0?cd_sigma(al,0,bits-1):-cd_sigma(al,bl,bits-1);
  if(bl==0)return -cd_sigma(0,al,bits-1); return cd_sigma(bl,al,bits-1); }
void ref(const float*a,const float*b,float*r){for(int k=0;k<16;k++)r[k]=0;
  for(int i=0;i<16;i++)for(int j=0;j<16;j++)r[i^j]+=cd_sigma(i,j,4)*a[i]*b[j];}

int main(){
    float *da,*dB,*dD; cudaMalloc(&da,16*4); cudaMalloc(&dB,16*16*4); cudaMalloc(&dD,256*4);
    // batch test: random a, 16 columns of b
    float a[16]; for(int i=0;i<16;i++)a[i]=(i%3)-1 + 0.5f*(i%2?-1:1);
    float B[256]; for(int c=0;c<16;c++)for(int r=0;r<16;r++)B[c*16+r]=((c+1)*0.25f+r*0.125f)*((r%2)?-1:1);
    cudaMemcpy(da,a,16*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,B,256*4,cudaMemcpyHostToDevice);
    sed_mul_full<<<1,32>>>(da,dB,dD);
    if(cudaDeviceSynchronize()){printf("ERR\n");return 2;}
    float hD[256]; cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    int fails=0; float mx=0;
    for(int c=0;c<16;c++){float rr[16];ref(a,&B[c*16],rr);for(int k=0;k<16;k++){float er=fabsf(hD[k*16+c]-rr[k]);if(er>mx)mx=er;if(er>0.3f)fails++;}}
    // e2·e5 discriminator (a=e2, col0=e5) -> comp7
    float a2[16]={0}; a2[2]=1; float B2[256]={0}; B2[5]=1; // B2 col0 row5
    cudaMemcpy(da,a2,16*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,B2,256*4,cudaMemcpyHostToDevice);
    sed_mul_full<<<1,32>>>(da,dB,dD); cudaDeviceSynchronize(); cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    float e25=hD[7*16+0];
    // ZERO DIVISOR: a=e3+e10, col0=e6-e15  -> product must be all zero
    float aZ[16]={0}; aZ[3]=1; aZ[10]=1; float BZ[256]={0}; BZ[6]=1; BZ[15]=-1;
    cudaMemcpy(da,aZ,16*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,BZ,256*4,cudaMemcpyHostToDevice);
    sed_mul_full<<<1,32>>>(da,dB,dD); cudaDeviceSynchronize(); cudaMemcpy(hD,dD,256*4,cudaMemcpyDeviceToHost);
    int znz=0; float zmax=0; for(int k=0;k<16;k++){float v=fabsf(hD[k*16+0]); if(v>zmax)zmax=v; if(v>0.05f)znz++;}
    printf("SEDENION tensor-core GB10: e2*e5->comp7=%.2f | batch %d/256 mismatch maxerr=%.3f\n",e25,fails,mx);
    printf("ZERO DIVISOR (e3+e10)(e6-e15) on tensor core: %d/16 nonzero comps, max|comp|=%.4f\n",znz,zmax);
    if(fails==0 && e25>0.7f && znz==0){ printf("PASS: tensor-core sedenion multiply is Convention X + ZD annihilates on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
