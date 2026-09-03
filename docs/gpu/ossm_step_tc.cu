// Tensor-core O-SSM octonion linear recurrence step, for a BATCH.
// The O-SSM forward step (self-hosted/gpu/kernels/ossm_forward.sio) does per (batch,head):
//   ah = A ⊗ h_prev  (octonion multiply);  bx = B * x;  sum = ah + bx;  h = sigmoid(sum); ...
// The octonion multiply A⊗h_prev, across a BATCH of states H, is L(A)·H — a tensor-core matmul.
// This kernel computes the LINEAR part  S[b] = A⊗H[b] + B·x[b]  for a 16-wide batch on tensor cores,
// and we validate it matches the scalar O-SSM reference (per-element octonion multiply + scale-add).
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
__device__ __host__ int cds(int a,int b,int bits){ int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; } return s; }
// kernel: build L(A) (8x8 in 16x16) in shared; H_batch staged in shared; AH=L(A)·H via wmma;
// then S = AH + B·x  (bx[k][b] = B[k]*x[b]); store S (16x16 row-major, first 8 rows valid).
__global__ void ossm_step(const float* A, const float* B, const float* x, const float* H, float* S){
    __shared__ half sL[256]; __shared__ half sH[256];
    int t=threadIdx.x;
    for(int idx=t; idx<256; idx+=32){ int k=idx>>4,j=idx&15; if(k<8&&j<8){int i=k^j; sL[idx]=__float2half((float)cds(i,j,3)*A[i]);} else sL[idx]=__float2half(0.0f); }
    for(int idx=t; idx<256; idx+=32){ int c=idx>>4,r=idx&15; sH[idx]= (r<8) ? __float2half(H[c*8+r]) : __float2half(0.0f); } // col-major H[c][r]
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> aF;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> hF;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.0f);
    wmma::load_matrix_sync(aF,sL,16); wmma::load_matrix_sync(hF,sH,16); wmma::mma_sync(acc,aF,hF,acc);
    wmma::store_matrix_sync(S,acc,16,wmma::mem_row_major);   // S[k*16+b] = (A⊗H[b])[k]
    __syncwarp();
    // add bx: S[k*16+b] += B[k]*x[b]   (thread 0, straightforward)
    if(t==0){ for(int b=0;b<16;b++) for(int k=0;k<8;k++) S[k*16+b]+=B[k]*x[b]; }
}
void om(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cds(i,j,3)*a[i]*b[j];}
int main(){
    float A[8]={0.5f,-0.2f,0.3f,0,0.1f,0,-0.4f,0.2f}, B[8]={0.1f,0.2f,-0.1f,0.3f,0,0.1f,0,-0.2f};
    float x[16], H[16*8];
    for(int b=0;b<16;b++){ x[b]=0.5f+0.1f*b; for(int k=0;k<8;k++) H[b*8+k]=((b+1)*0.2f+k*0.1f)*((k%2)?-1:1); }
    float *dA,*dB,*dx,*dH,*dS; cudaMalloc(&dA,8*4);cudaMalloc(&dB,8*4);cudaMalloc(&dx,16*4);cudaMalloc(&dH,16*8*4);cudaMalloc(&dS,256*4);
    cudaMemcpy(dA,A,8*4,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,8*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dx,x,16*4,cudaMemcpyHostToDevice);cudaMemcpy(dH,H,16*8*4,cudaMemcpyHostToDevice);
    ossm_step<<<1,32>>>(dA,dB,dx,dH,dS); if(cudaDeviceSynchronize()){printf("ERR\n");return 2;}
    float hS[256]; cudaMemcpy(hS,dS,256*4,cudaMemcpyDeviceToHost);
    int fails=0; float mx=0;
    for(int b=0;b<16;b++){ float ah[8]; om(A,&H[b*8],ah);
        for(int k=0;k<8;k++){ float ref=ah[k]+B[k]*x[b]; float got=hS[k*16+b]; float e=fabsf(got-ref); if(e>mx)mx=e; if(e>0.05f)fails++; } }
    printf("O-SSM tensor-core step on GB10: S=A⊗H+B·x  batch %d/128 mismatch maxerr=%.4f\n",fails,mx);
    if(!fails){ printf("PASS: tensor-core O-SSM octonion linear step matches scalar reference on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
