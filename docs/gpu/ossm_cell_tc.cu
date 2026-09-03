// Full O-SSM octonion forward CELL on tensor cores, batched, with T-step recurrence.
// Per timestep t, for each batch column b (16 independent sequences), dim=8 octonion state:
//   S    = A⊗h_{t-1} + B·x_t                     (linear step: L(A)·H tensor-core tile + B·x post-add)
//   h_t  = sigmoidcubic(S)  = (1/64)S³ + 0.25 S + 0.5   (matches ossm_emit_sigmoid_f64 in ossm_forward.sio)
//   y_t  = Re(C⊗h_t) = (L(C)·h_t)[0]             (output projection: second tensor-core tile)
// L(A) and L(C) are built once (fixed across time); H evolves in shared across T steps. Two wmma
// m16n16k16 tiles per step. Validates the full forward cell + recurrence vs the scalar reference on GB10.
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#define T 6
#define NB 16
__device__ __host__ int cds(int a,int b,int bits){ int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; } return s; }
__device__ __host__ float sigc(float x){ return 0.015625f*x*x*x + 0.25f*x + 0.5f; }

__global__ void ossm_cell(const float* A,const float* B,const float* C,const float* x,const float* H0,float* Y){
    __shared__ half sLA[256], sLC[256], sH[256]; __shared__ float sS[256];
    int t=threadIdx.x;
    // build L(A), L(C): L[k][j]=σ(k⊕j,j)·M[k⊕j]  (8x8 in 16x16, row-major)
    for(int idx=t; idx<256; idx+=32){ int k=idx>>4,j=idx&15;
        if(k<8&&j<8){int i=k^j; sLA[idx]=__float2half((float)cds(i,j,3)*A[i]); sLC[idx]=__float2half((float)cds(i,j,3)*C[i]);}
        else { sLA[idx]=__float2half(0.f); sLC[idx]=__float2half(0.f); } }
    // stage H0 col-major: sH[b*16+r]=H0[b*8+r], r<8
    for(int idx=t; idx<256; idx+=32){ int b=idx>>4,r=idx&15; sH[idx]= (r<8)?__float2half(H0[b*8+r]):__float2half(0.f); }
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> mA;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> mH;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    for(int step=0; step<T; step++){
        // S = L(A)·H
        wmma::fill_fragment(acc,0.f);
        wmma::load_matrix_sync(mA,sLA,16); wmma::load_matrix_sync(mH,sH,16); wmma::mma_sync(acc,mA,mH,acc);
        wmma::store_matrix_sync(sS,acc,16,wmma::mem_row_major);   // sS[k*16+b]
        __syncwarp();
        // post-add B·x, cubic sigmoid, write h_t back into sH col-major (thread 0)
        if(t==0){ const float* xt=&x[step*NB];
            for(int b=0;b<NB;b++) for(int k=0;k<8;k++){ float s=sS[k*16+b]+B[k]*xt[b]; sH[b*16+k]=__float2half(sigc(s)); } }
        __syncwarp();
        // CH = L(C)·h_t ;  y_t[b] = Re = CH[0][b]
        wmma::fill_fragment(acc,0.f);
        wmma::load_matrix_sync(mA,sLC,16); wmma::load_matrix_sync(mH,sH,16); wmma::mma_sync(acc,mA,mH,acc);
        wmma::store_matrix_sync(sS,acc,16,wmma::mem_row_major);
        __syncwarp();
        if(t==0) for(int b=0;b<NB;b++) Y[step*NB+b]=sS[0*16+b];
        __syncwarp();
    }
}
// scalar octonion multiply (Convention X)
void om(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cds(i,j,3)*a[i]*b[j];}
int main(){
    float A[8]={0.5f,-0.2f,0.3f,0,0.1f,0,-0.4f,0.2f}, B[8]={0.1f,0.2f,-0.1f,0.3f,0,0.1f,0,-0.2f};
    float C[8]={0.3f,0.1f,-0.2f,0.15f,-0.05f,0.2f,0.1f,-0.1f};
    float x[T*NB], H0[NB*8], Yref[T*NB];
    for(int b=0;b<NB;b++){ for(int k=0;k<8;k++) H0[b*8+k]=((b+1)*0.15f+k*0.07f)*((k%2)?-1:1); }
    for(int s=0;s<T;s++)for(int b=0;b<NB;b++) x[s*NB+b]=0.3f+0.05f*b-0.02f*s;
    // scalar reference recurrence
    float Href[NB*8]; for(int i=0;i<NB*8;i++)Href[i]=H0[i];
    for(int s=0;s<T;s++){ const float* xt=&x[s*NB];
        for(int b=0;b<NB;b++){ float ah[8]; om(A,&Href[b*8],ah); float ht[8];
            for(int k=0;k<8;k++) ht[k]=sigc(ah[k]+B[k]*xt[b]);
            for(int k=0;k<8;k++) Href[b*8+k]=ht[k];
            float ch[8]; om(C,ht,ch); Yref[s*NB+b]=ch[0]; } }
    float *dA,*dB,*dC,*dx,*dH,*dY; cudaMalloc(&dA,8*4);cudaMalloc(&dB,8*4);cudaMalloc(&dC,8*4);
    cudaMalloc(&dx,T*NB*4);cudaMalloc(&dH,NB*8*4);cudaMalloc(&dY,T*NB*4);
    cudaMemcpy(dA,A,8*4,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,8*4,cudaMemcpyHostToDevice);cudaMemcpy(dC,C,8*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dx,x,T*NB*4,cudaMemcpyHostToDevice);cudaMemcpy(dH,H0,NB*8*4,cudaMemcpyHostToDevice);
    ossm_cell<<<1,32>>>(dA,dB,dC,dx,dH,dY); if(cudaDeviceSynchronize()){printf("ERR\n");return 2;}
    float Y[T*NB]; cudaMemcpy(Y,dY,T*NB*4,cudaMemcpyDeviceToHost);
    int fails=0; float mx=0;
    for(int i=0;i<T*NB;i++){ float e=fabsf(Y[i]-Yref[i]); if(e>mx)mx=e; if(e>0.02f)fails++; }
    printf("O-SSM full cell tensor-core GB10: T=%d batch=%d  y mismatch %d/%d maxerr=%.4f\n",T,NB,fails,T*NB,mx);
    if(!fails){ printf("PASS: tensor-core full O-SSM octonion cell (sigmoid+C output+recurrence) matches scalar reference on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
