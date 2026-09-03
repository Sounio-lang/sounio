// Driver-API harness: load souc-emitted PTX for the source-level octonion multiply kernel
//   kernel fn step(pa:&Hyper<Octonion,f64>, ph:&Hyper<Octonion,f64>, pout:&!Hyper<Octonion,f64>)
// which computes pout = pa ⊗ ph. Validate bit/near parity vs the scalar cd_sigma reference on GB10.
#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){ int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; } return s; }
static void om(const double*a,const double*b,double*r){for(int k=0;k<8;k++)r[k]=0;
    for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=(double)cds(i,j,3)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);printf("ERR %s @%d: %s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int argc,char**argv){
    const char* ptx = argc>1?argv[1]:"/tmp/oct_mul.ptx";
    const char* fn_name = argc>2?argv[2]:"step";
    double a[8]={0.5,-0.2,0.3,0.0,0.1,0.0,-0.4,0.2}, h[8]={0.1,0.2,-0.1,0.3,0.05,0.1,-0.2,0.15};
    double ref[8]; om(a,h,ref);
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
    CUmodule mod; CK(cuModuleLoad(&mod,ptx)); CUfunction fn; CK(cuModuleGetFunction(&fn,mod,fn_name));
    CUdeviceptr da,dh,dp; CK(cuMemAlloc(&da,8*8)); CK(cuMemAlloc(&dh,8*8)); CK(cuMemAlloc(&dp,8*8));
    CK(cuMemcpyHtoD(da,a,8*8)); CK(cuMemcpyHtoD(dh,h,8*8));
    double zero[8]={0}; CK(cuMemcpyHtoD(dp,zero,8*8));
    void* args[]={&da,&dh,&dp};
    CK(cuLaunchKernel(fn,1,1,1, 1,1,1, 0,0,args,0)); CK(cuCtxSynchronize());
    double got[8]; CK(cuMemcpyDtoH(got,dp,8*8));
    int fails=0; double mx=0;
    for(int k=0;k<8;k++){ double e=fabs(got[k]-ref[k]); if(e>mx)mx=e; if(e>1e-9)fails++; }
    printf("Sounio source-level octonion `a*h` PTX GB10: pout=pa⊗ph  mismatch %d/8 maxerr=%.3e\n",fails,mx);
    printf("  ref = ["); for(int k=0;k<8;k++)printf("%.4f%s",ref[k],k<7?", ":"");
    printf("]\n  got = ["); for(int k=0;k<8;k++)printf("%.4f%s",got[k],k<7?", ":""); printf("]\n");
    if(!fails){ printf("PASS: compiler-lowered source-level octonion multiply matches scalar reference on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
