// Driver-API harness: souc-emitted sedenion multiply kernel step(pa,ph,pout) = pa ⊗ ph.
// Validates (1) random bit-parity vs scalar cd_sigma(bits=4) and (2) the canonical
// zero divisor (e3+e10)(e6−e15) = 0 — all on GB10.
#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){ int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; } return s; }
static void sm(const double*a,const double*b,double*r){for(int k=0;k<16;k++)r[k]=0;
    for(int i=0;i<16;i++)for(int j=0;j<16;j++)r[i^j]+=(double)cds(i,j,4)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);printf("ERR %s @%d: %s\n",#x,__LINE__,s);return 2;}}while(0)
static CUfunction FN; static CUdeviceptr da,dh,dp;
static int runk(const double*a,const double*h,double*out){
    if(cuMemcpyHtoD(da,a,16*8))return 1; if(cuMemcpyHtoD(dh,h,16*8))return 1;
    double z[16]={0}; cuMemcpyHtoD(dp,z,16*8);
    void*args[]={&da,&dh,&dp};
    if(cuLaunchKernel(FN,1,1,1,1,1,1,0,0,args,0))return 1; if(cuCtxSynchronize())return 1;
    return cuMemcpyDtoH(out,dp,16*8)!=CUDA_SUCCESS;
}
int main(int argc,char**argv){
    const char* ptx = argc>1?argv[1]:"/tmp/sed_mul.ptx";
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
    CUmodule mod; CK(cuModuleLoad(&mod,ptx)); CK(cuModuleGetFunction(&FN,mod,"step"));
    CK(cuMemAlloc(&da,16*8)); CK(cuMemAlloc(&dh,16*8)); CK(cuMemAlloc(&dp,16*8));
    // (1) random parity
    unsigned s=12345; auto rn=[&](){s=s*1103515245+12345; return ((s>>16)&0x7fff)/16384.0-1.0;};
    double a[16],h[16],ref[16],got[16]; for(int i=0;i<16;i++){a[i]=rn();h[i]=rn();}
    sm(a,h,ref); if(runk(a,h,got)){printf("launch/copy fail\n");return 2;}
    int fails=0; double mx=0; for(int k=0;k<16;k++){double e=fabs(got[k]-ref[k]);if(e>mx)mx=e;if(e>1e-9)fails++;}
    // (2) zero divisor: (e3+e10)(e6−e15)=0
    double za[16]={0},zh[16]={0},zo[16]; za[3]=1;za[10]=1; zh[6]=1;zh[15]=-1;
    runk(za,zh,zo); double zn=0; for(int k=0;k<16;k++)zn+=zo[k]*zo[k]; zn=sqrt(zn);
    printf("Sounio source-level sedenion `a*h` PTX GB10: pout=pa⊗ph  mismatch %d/16 maxerr=%.3e\n",fails,mx);
    printf("  zero divisor ||(e3+e10)(e6-e15)|| = %.3e (expect ~0)\n",zn);
    if(!fails && zn<1e-9){ printf("PASS: compiler-lowered source-level sedenion multiply + zero-divisor on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
