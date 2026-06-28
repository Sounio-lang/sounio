// Clean launcher for the Sounio K-AXI epistemic scorer kernel on Blackwell.
// Kernel: .visible .entry kaxi_kernel(.param .u64 param_mem); N = ntid.x (single block).
// Packed f32 buffer: [0..N)=act_mean, [N..2N)=act_var, [2N..3N)=score(out), [3N]=beta.
// Per thread i: score[i] = mem[2N+i] = mem[i] + mem[3N]*sqrt(mem[N+i]).
// Loads a NATIVE .cubin (cuModuleLoad, no JIT) OR a .ptx (cuModuleLoadDataEx).
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
static void chk(CUresult r,const char*w){if(r!=CUDA_SUCCESS){const char*m;cuGetErrorString(r,&m);fprintf(stderr,"ERR %s: %s\n",w,m);exit(1);}}
int main(int argc,char**argv){
    const char* path = argc>1?argv[1]:"/tmp/scorer_real.cubin";
    int isPtx = strstr(path,".ptx")!=NULL;
    chk(cuInit(0),"init");
    CUdevice dev; chk(cuDeviceGet(&dev,0),"dev");
    char nm[128]; cuDeviceGetName(nm,sizeof nm,dev);
    int M,m; cuDeviceGetAttribute(&M,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
    cuDeviceGetAttribute(&m,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
    printf("device: %s sm_%d%d\n",nm,M,m);
    CUcontext ctx; chk(cuDevicePrimaryCtxRetain(&ctx,dev),"ctx"); chk(cuCtxSetCurrent(ctx),"setctx");
    CUmodule mod;
    if(isPtx){
        FILE*f=fopen(path,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char*buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){return 3;} buf[n]=0; fclose(f);
        CUjit_option o[1]={CU_JIT_TARGET_FROM_CUCONTEXT}; void*ov[1]={0};
        chk(cuModuleLoadDataEx(&mod,buf,1,o,ov),"loadPTX");
    } else {
        chk(cuModuleLoad(&mod,path),"loadCUBIN(native)");
    }
    CUfunction fn; chk(cuModuleGetFunction(&fn,mod,"kaxi_kernel"),"getfn");
    printf("loaded kaxi_kernel from %s\n", isPtx?"PTX(JIT)":"native cubin");

    const int N=4;
    float host[3*N+1];
    float mean[4]={0.1f,0.2f,0.3f,0.4f}, var[4]={0.04f,0.09f,0.16f,0.25f}, beta=0.6f;
    for(int i=0;i<N;i++){host[i]=mean[i]; host[N+i]=var[i]; host[2*N+i]=0.f;}
    host[3*N]=beta;
    CUdeviceptr d; chk(cuMemAlloc(&d,sizeof host),"alloc");
    chk(cuMemcpyHtoD(d,host,sizeof host),"htod");
    void* args[1]={&d};
    chk(cuLaunchKernel(fn,1,1,1, N,1,1, 0,0, args,0),"launch");
    chk(cuCtxSynchronize(),"sync");
    chk(cuMemcpyDtoH(host,d,sizeof host),"dtoh");

    printf("--- epistemic scorer: score[i] = mean[i] + beta*sqrt(var[i]),  beta=%.2f ---\n",beta);
    printf("  i  mean   var    measured   expected   delta\n");
    int ok=1;
    for(int i=0;i<N;i++){
        float exp=mean[i]+beta*sqrtf(var[i]);
        float got=host[2*N+i];
        float d2=fabsf(got-exp);
        printf("  %d  %.3f  %.4f  %.6f   %.6f   %.2e %s\n",i,mean[i],var[i],got,exp,d2, d2<1e-3f?"":"<-- FAIL");
        if(d2>=1e-3f) ok=0;
    }
    printf("RESULT: %s\n", ok?"PASS":"FAIL");
    return ok?0:2;
}
