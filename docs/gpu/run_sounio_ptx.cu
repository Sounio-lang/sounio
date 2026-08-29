// Host harness that loads the SOUNIO-EMITTED wmma PTX and launches it on the GB10 via the
// CUDA Driver API. Builds L(a) (Convention X) + B on host, runs sounio_oct_mul_wmma, checks
// the output equals the exact X octonion product. Validates the Sounio-emitted PTX end-to-end.
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

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
void octmul_ref(const float*a,const float*b,float*r){
    for(int k=0;k<8;k++) r[k]=0;
    for(int i=0;i<8;i++)for(int j=0;j<8;j++) r[i^j]+=cd_sigma(i,j,3)*a[i]*b[j];
}
#define CK(x) do{ CUresult e=(x); if(e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(e,&s); printf("CUDA ERR %s at %d: %s\n",#x,__LINE__,s); return 2; } }while(0)

int main(int argc,char**argv){
    const char* ptxpath = argc>1?argv[1]:"/tmp/sounio_oct_wmma.ptx";
    FILE*f=fopen(ptxpath,"rb"); if(!f){printf("no ptx\n");return 2;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char*ptx=(char*)malloc(n+1); fread(ptx,1,n,f); ptx[n]=0; fclose(f);

    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
    CUmodule mod; CK(cuModuleLoadData(&mod,ptx));
    CUfunction k; CK(cuModuleGetFunction(&k,mod,"sounio_oct_mul_wmma"));

    float a[8]={1,2,0,-1,3,0,1,-2};
    float bcols[16][8];
    for(int c=0;c<16;c++)for(int j=0;j<8;j++) bcols[c][j]=((c+1)*0.5f + j*0.25f)*((j%2)?-1:1);
    // L(a) 16x16 row-major f16
    half hL[256]; for(int i=0;i<256;i++)hL[i]=__float2half(0.0f);
    for(int kk=0;kk<8;kk++)for(int j=0;j<8;j++){int i=kk^j; hL[kk*16+j]=__float2half((float)cd_sigma(i,j,3)*a[i]);}
    // B 16x16 col-major f16: column c = b_c
    half hB[256]; for(int i=0;i<256;i++)hB[i]=__float2half(0.0f);
    for(int c=0;c<16;c++)for(int j=0;j<8;j++) hB[c*16+j]=__float2half(bcols[c][j]);
    float hD[256];

    CUdeviceptr dL,dB,dD; CK(cuMemAlloc(&dL,256*2)); CK(cuMemAlloc(&dB,256*2)); CK(cuMemAlloc(&dD,256*4));
    CK(cuMemcpyHtoD(dL,hL,256*2)); CK(cuMemcpyHtoD(dB,hB,256*2));
    void*args[]={&dL,&dB,&dD};
    CK(cuLaunchKernel(k, 1,1,1, 32,1,1, 0, 0, args, 0));   // one warp
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(hD,dD,256*4));

    int fails=0; float maxerr=0;
    for(int c=0;c<16;c++){ float ref[8]; octmul_ref(a,bcols[c],ref);
        for(int kk=0;kk<8;kk++){ float got=hD[kk*16+c],err=fabsf(got-ref[kk]); if(err>maxerr)maxerr=err;
            if(err>0.15f){fails++; if(fails<=4)printf("  mismatch c=%d k=%d got=%.3f ref=%.3f\n",c,kk,got,ref[kk]);} } }
    // e1·e2 discriminator
    half hL2[256]; for(int i=0;i<256;i++)hL2[i]=__float2half(0.0f);
    { float e1[8]={0,1,0,0,0,0,0,0}; for(int kk=0;kk<8;kk++)for(int j=0;j<8;j++){int i=kk^j; hL2[kk*16+j]=__float2half((float)cd_sigma(i,j,3)*e1[i]);} }
    half hB2[256]; for(int i=0;i<256;i++)hB2[i]=__float2half(0.0f); hB2[0*16+2]=__float2half(1.0f); // b=e2 in col0
    CK(cuMemcpyHtoD(dL,hL2,256*2)); CK(cuMemcpyHtoD(dB,hB2,256*2));
    CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,args,0)); CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(hD,dD,256*4));
    float c3=hD[3*16+0], c4=hD[4*16+0];
    printf("SOUNIO-emitted PTX on GB10: e1*e2 comp3=%.2f comp4=%.2f | batch %d/128 mismatch maxerr=%.3f\n",c3,c4,fails,maxerr);
    if(fails==0 && c3>0.7f && c4<0.3f){ printf("PASS: Sounio-emitted WMMA octonion multiply is Convention X on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
