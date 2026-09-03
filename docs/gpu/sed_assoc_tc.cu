// SEDENION associator [a,b,c]=(ab)c−a(bc) on tensor cores. Linear in b: [a,b,c]=M·b,
// M = R(c)·L(a) − L(a)·R(c). For sedenions (bits=4) L(a),R(c) are 16×16 = EXACTLY one wmma tile
// (no padding). Two wmma m16n16k16 matmuls + f32 subtract → M. Validates: [a,b,c]=M·b vs direct;
// the 1848 non-associative sedenion basis triples; the norm dichotomy ‖·‖∈{0,2}. On the GB10.
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
__global__ void sed_assoc(const float* a, const float* c, float* Mout){
    __shared__ half sL[256]; __shared__ half sR[256];
    int t=threadIdx.x;
    for(int idx=t; idx<256; idx+=32){ int k=idx>>4, j=idx&15, i=k^j; sL[idx]=__float2half((float)cds(i,j,4)*a[i]); }
    for(int idx=t; idx<256; idx+=32){ int k=idx>>4, i=idx&15, j=k^i; sR[idx]=__float2half((float)cds(i,j,4)*c[j]); }
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> Ra,La;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> Lb,Rb;
    wmma::fragment<wmma::accumulator,16,16,16,float> T1,T2;
    wmma::fill_fragment(T1,0.0f); wmma::fill_fragment(T2,0.0f);
    wmma::load_matrix_sync(Ra,sR,16); wmma::load_matrix_sync(Lb,sL,16); wmma::mma_sync(T1,Ra,Lb,T1);
    wmma::load_matrix_sync(La,sL,16); wmma::load_matrix_sync(Rb,sR,16); wmma::mma_sync(T2,La,Rb,T2);
    for(int e=0;e<T1.num_elements;e++) T1.x[e]-=T2.x[e];
    wmma::store_matrix_sync(Mout,T1,16,wmma::mem_row_major);
}
void om(const float*a,const float*b,float*r){for(int k=0;k<16;k++)r[k]=0;
    for(int i=0;i<16;i++)for(int j=0;j<16;j++)r[i^j]+=cds(i,j,4)*a[i]*b[j];}
int main(){
    float *da,*dc,*dM; cudaMalloc(&da,16*4);cudaMalloc(&dc,16*4);cudaMalloc(&dM,256*4); float hM[256];
    unsigned s=54321; auto rn=[&](){s=s*1103515245+12345; return ((s>>16)&0x7fff)/16384.0f-1.0f;};
    float mxA=0;
    for(int t=0;t<40;t++){ float a[16],b[16],c[16]; for(int i=0;i<16;i++){a[i]=rn();b[i]=rn();c[i]=rn();}
        cudaMemcpy(da,a,16*4,cudaMemcpyHostToDevice);cudaMemcpy(dc,c,16*4,cudaMemcpyHostToDevice);
        sed_assoc<<<1,32>>>(da,dc,dM); if(cudaDeviceSynchronize()){printf("ERR\n");return 2;}
        cudaMemcpy(hM,dM,256*4,cudaMemcpyDeviceToHost);
        float ab[16],bc[16],p1[16],p2[16]; om(a,b,ab);om(ab,c,p1);om(b,c,bc);om(a,bc,p2);
        for(int k=0;k<16;k++){float Mb=0;for(int j=0;j<16;j++)Mb+=hM[k*16+j]*b[j]; float e=fabsf(Mb-(p1[k]-p2[k])); if(e>mxA)mxA=e;} }
    // 1848 count + norm dichotomy over basis triples i,j,k in 1..15
    int cnt=0,n0=0,n2=0,oth=0;
    for(int i=1;i<16;i++)for(int k=1;k<16;k++){ float a[16]={0},c[16]={0}; a[i]=1; c[k]=1;
        cudaMemcpy(da,a,16*4,cudaMemcpyHostToDevice);cudaMemcpy(dc,c,16*4,cudaMemcpyHostToDevice);
        sed_assoc<<<1,32>>>(da,dc,dM); cudaDeviceSynchronize(); cudaMemcpy(hM,dM,256*4,cudaMemcpyDeviceToHost);
        for(int j=1;j<16;j++){ float q=0; for(int r=0;r<16;r++){float v=hM[r*16+j]; q+=v*v;}
            if(q>0.5f)cnt++; float nr=sqrtf(q); if(nr<0.3f)n0++; else if(fabsf(nr-2)<0.3f)n2++; else oth++; } }
    printf("SEDENION associator on GB10: [a,b,c]=M·b vs direct max=%.4f\n",mxA);
    printf("  non-associative basis triples (i,j,k in 1..15): %d (expect 1848)\n",cnt);
    printf("  norms: #(≈0)=%d #(≈2)=%d #(other)=%d  (dichotomy {0,2})\n",n0,n2,oth);
    if(mxA<0.25f && cnt==1848 && oth==0){ printf("PASS: tensor-core sedenion associator — 1848 + norm dichotomy {0,2} on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
