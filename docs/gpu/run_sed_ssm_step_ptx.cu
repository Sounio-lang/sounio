// Driver-API harness: load Sounio-emitted PTX, launch sounio_sed_ssm_step, validate (1) random batch
// vs scalar sedenion reference and (2) zero-divisor gating, on the GB10.
#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){ int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; } return s; }
static void sm(const float*a,const float*b,float*r){for(int k=0;k<16;k++)r[k]=0;for(int i=0;i<16;i++)for(int j=0;j<16;j++)r[i^j]+=cds(i,j,4)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);printf("ERR %s @%d: %s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int argc,char**argv){
    const char* ptx = argc>1?argv[1]:"/tmp/sounio_sed_ssm_step.ptx";
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
    CUmodule mod; CK(cuModuleLoad(&mod,ptx)); CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"sounio_sed_ssm_step"));
    CUdeviceptr dA,dB,dx,dH,dS;
    CK(cuMemAlloc(&dA,16*4));CK(cuMemAlloc(&dB,16*4));CK(cuMemAlloc(&dx,16*4));CK(cuMemAlloc(&dH,256*4));CK(cuMemAlloc(&dS,256*4));
    float hS[256];
    // (1) random validation
    unsigned s=1234567; auto rn=[&](){s=s*1103515245+12345; return ((s>>16)&0x7fff)/16384.0f-1.0f;};
    float A[16],B[16],x[16],H[256]; for(int i=0;i<16;i++){A[i]=rn();B[i]=rn();x[i]=rn();} for(int i=0;i<256;i++)H[i]=rn();
    CK(cuMemcpyHtoD(dA,A,16*4));CK(cuMemcpyHtoD(dB,B,16*4));CK(cuMemcpyHtoD(dx,x,16*4));CK(cuMemcpyHtoD(dH,H,256*4));
    void* a1[]={&dA,&dB,&dx,&dH,&dS};
    CK(cuLaunchKernel(fn,1,1,1,32,1,1,0,0,a1,0)); CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(hS,dS,256*4));
    int fails=0; float mx=0;
    for(int b=0;b<16;b++){ float ah[16]; sm(A,&H[b*16],ah);
        for(int k=0;k<16;k++){ float ref=ah[k]+B[k]*x[b]; float e=fabsf(hS[k*16+b]-ref); if(e>mx)mx=e; if(e>0.05f)fails++; } }
    // (2) zero-divisor gating: A=(e3+e10), B=0, x=0; col0 h=(e6−e15) annihilates, col1 h=e1 passes
    float Az[16]={0}; Az[3]=1; Az[10]=1; float Bz[16]={0}, xz[16]={0}, Hz[256]={0};
    Hz[0*16+6]=1; Hz[0*16+15]=-1; Hz[1*16+1]=1;
    CK(cuMemcpyHtoD(dA,Az,16*4));CK(cuMemcpyHtoD(dB,Bz,16*4));CK(cuMemcpyHtoD(dx,xz,16*4));CK(cuMemcpyHtoD(dH,Hz,256*4));
    CK(cuLaunchKernel(fn,1,1,1,32,1,1,0,0,a1,0)); CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(hS,dS,256*4));
    float gate=0,ctrl=0; for(int k=0;k<16;k++){ gate+=hS[k*16+0]*hS[k*16+0]; ctrl+=hS[k*16+1]*hS[k*16+1]; }
    gate=sqrtf(gate); ctrl=sqrtf(ctrl);
    printf("Sounio SEDENION S-SSM step PTX GB10: A⊗H+B·x  mismatch %d/256 maxerr=%.4f\n",fails,mx);
    printf("  zero-divisor gate: ||(e3+e10)⊗(e6−e15)||=%.4f (~0)  ||(e3+e10)⊗e1||=%.4f (>0)\n",gate,ctrl);
    if(!fails && gate<0.05f && ctrl>0.5f){ printf("PASS: Sounio-emitted tensor-core sedenion S-SSM step + zero-divisor gating on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
