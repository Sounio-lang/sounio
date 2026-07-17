// Octonion ASSOCIATOR [a,b,c]=(ab)c−a(bc) on tensor cores. Since the associator is linear in b,
//   [a,b,c] = M·b,  M = R(c)·L(a) − L(a)·R(c)   (matrix commutator of left/right multiply matrices),
// so M is computed by TWO wmma m16n16k16 matmuls of the 8×8 (padded 16×16) matrices
//   L(a)[k][j]=σ(k⊕j,j)·a[k⊕j],  R(c)[k][i]=σ(i,k⊕i)·c[k⊕i],
// plus an f32 subtract. Validates: [a,b,c]=M·b vs direct; the 168 non-associative basis triples;
// the norm dichotomy ‖[e_i,e_j,e_k]‖∈{0,2}. On the GB10 tensor cores.
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
__device__ __host__ int cds(int a,int b,int bits){
    int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; }
    return s; }
// kernel: build L(a),R(c) row-major (16x16, 8x8 nonzero) in shared; M = R·L - L·R via two wmma; store M.
__global__ void assoc_kernel(const float* a, const float* c, float* Mout){
    __shared__ half sL[256]; __shared__ half sR[256];
    int t=threadIdx.x;
    for(int idx=t; idx<256; idx+=32){
        int k=idx>>4, j=idx&15;
        if(k<8&&j<8){ int i=k^j; sL[idx]=__float2half((float)cds(i,j,3)*a[i]); } else sL[idx]=__float2half(0.0f);
    }
    for(int idx=t; idx<256; idx+=32){
        int k=idx>>4, i=idx&15;
        if(k<8&&i<8){ int j=k^i; sR[idx]=__float2half((float)cds(i,j,3)*c[j]); } else sR[idx]=__float2half(0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> Ra,La;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> Lb,Rb;
    wmma::fragment<wmma::accumulator,16,16,16,float> T1,T2;
    wmma::fill_fragment(T1,0.0f); wmma::fill_fragment(T2,0.0f);
    wmma::load_matrix_sync(Ra,sR,16); wmma::load_matrix_sync(Lb,sL,16); wmma::mma_sync(T1,Ra,Lb,T1); // T1 = R·L
    wmma::load_matrix_sync(La,sL,16); wmma::load_matrix_sync(Rb,sR,16); wmma::mma_sync(T2,La,Rb,T2); // T2 = L·R
    for(int e=0;e<T1.num_elements;e++) T1.x[e]-=T2.x[e];                                             // M = T1 - T2
    wmma::store_matrix_sync(Mout,T1,16,wmma::mem_row_major);
}
void octmul(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;
    for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cds(i,j,3)*a[i]*b[j];}
int main(){
    float *da,*dc,*dM; cudaMalloc(&da,8*4);cudaMalloc(&dc,8*4);cudaMalloc(&dM,256*4); float hM[256];
    // random check: [a,b,c]=M·b vs direct
    float mxA=0; unsigned seed=12345;
    auto rnd=[&](){ seed=seed*1103515245+12345; return ((seed>>16)&0x7fff)/16384.0f-1.0f; };
    for(int trial=0; trial<50; trial++){
        float a[8],b[8],c[8]; for(int i=0;i<8;i++){a[i]=rnd();b[i]=rnd();c[i]=rnd();}
        cudaMemcpy(da,a,8*4,cudaMemcpyHostToDevice); cudaMemcpy(dc,c,8*4,cudaMemcpyHostToDevice);
        assoc_kernel<<<1,32>>>(da,dc,dM); if(cudaDeviceSynchronize()){printf("ERR\n");return 2;}
        cudaMemcpy(hM,dM,256*4,cudaMemcpyDeviceToHost);
        float ab[8],bc[8],abc[8],abc2[8],ref[8],Mb[8];
        octmul(a,b,ab); octmul(ab,c,abc); octmul(b,c,bc); octmul(a,bc,abc2);
        for(int k=0;k<8;k++){ ref[k]=abc[k]-abc2[k]; Mb[k]=0; for(int j=0;j<8;j++)Mb[k]+=hM[k*16+j]*b[j]; if(fabsf(Mb[k]-ref[k])>mxA)mxA=fabsf(Mb[k]-ref[k]); }
    }
    // 168 count + norm dichotomy over basis triples i,j,k in 1..7
    int cnt=0; int norm0=0,norm2=0,other=0;
    for(int i=1;i<8;i++)for(int k=1;k<8;k++){
        float a[8]={0},c[8]={0}; a[i]=1; c[k]=1;
        cudaMemcpy(da,a,8*4,cudaMemcpyHostToDevice); cudaMemcpy(dc,c,8*4,cudaMemcpyHostToDevice);
        assoc_kernel<<<1,32>>>(da,dc,dM); cudaDeviceSynchronize(); cudaMemcpy(hM,dM,256*4,cudaMemcpyDeviceToHost);
        for(int j=1;j<8;j++){ float n2=0; for(int kk=0;kk<8;kk++){float v=hM[kk*16+j]; n2+=v*v;}
            if(n2>0.5f) cnt++; float nr=sqrtf(n2); if(nr<0.3f)norm0++; else if(fabsf(nr-2.0f)<0.3f)norm2++; else other++; }
    }
    printf("ASSOCIATOR on GB10 tensor cores: [a,b,c]=M·b vs direct max err=%.4f\n",mxA);
    printf("  non-associative basis triples (i,j,k in 1..7): %d (expect 168)\n",cnt);
    printf("  associator norms: #(‖·‖≈0)=%d  #(‖·‖≈2)=%d  #(other)=%d  (dichotomy {0,2})\n",norm0,norm2,other);
    if(mxA<0.15f && cnt==168 && other==0){ printf("PASS: tensor-core octonion associator — 168 + norm dichotomy {0,2} on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
