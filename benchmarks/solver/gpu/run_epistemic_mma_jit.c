// Driver-API launcher: JIT-load Sounio epistemic-MMA PTX, run on the GPU, verify.
// Verifiable case: A=0 => D = A*B + C = C ; shadow eps_C = sqrt(4*(|D0|*epsB + |D2|*epsA)).
// With C=[1,2,3,4], epsA=0.5, epsB=0.25 -> D=[1,2,3,4], eps_C = sqrt(4*(1*0.25+3*0.5)) = sqrt(7) ~ 2.6458.
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static void chk(CUresult r, const char* w){ if(r!=CUDA_SUCCESS){ const char* m; cuGetErrorString(r,&m); fprintf(stderr,"ERR %s: %s\n",w,m); exit(1);} }

int main(int argc, char** argv){
    const char* ptxpath = argc>1? argv[1] : "/tmp/epi_ascii.ptx";
    FILE* f=fopen(ptxpath,"rb"); if(!f){perror("ptx");return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(n+1); fread(ptx,1,n,f); ptx[n]=0; fclose(f);

    chk(cuInit(0),"init");
    CUdevice dev; chk(cuDeviceGet(&dev,0),"devget");
    char name[128]; cuDeviceGetName(name,sizeof name,dev);
    int ccM=0,ccm=0; cuDeviceGetAttribute(&ccM,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
    cuDeviceGetAttribute(&ccm,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
    printf("device: %s  sm_%d%d\n",name,ccM,ccm);
    CUcontext ctx; chk(cuDevicePrimaryCtxRetain(&ctx,dev),"ctxRetain"); chk(cuCtxSetCurrent(ctx),"ctxset");

    // JIT-load PTX (proves forward-compat PTX->this-arch SASS at load time)
    CUmodule mod; CUjit_option o[1]={CU_JIT_TARGET_FROM_CUCONTEXT}; void* ov[1]={0};
    chk(cuModuleLoadDataEx(&mod, ptx, 1, o, ov),"loadPTX(JIT)");
    CUfunction fn; chk(cuModuleGetFunction(&fn,mod,"epistemic_mma_kernel"),"getfn");
    printf("JIT-loaded epistemic_mma_kernel from PTX\n");

    // device buffers
    CUdeviceptr dAv,dAe,dBv,dBe,dCv,dCe,dVl,dPr;
    chk(cuMemAlloc(&dAv,16),"aAv"); chk(cuMemAlloc(&dAe,4),"aAe");
    chk(cuMemAlloc(&dBv,8),"aBv");  chk(cuMemAlloc(&dBe,4),"aBe");
    chk(cuMemAlloc(&dCv,16),"aCv"); chk(cuMemAlloc(&dCe,4),"aCe");
    chk(cuMemAlloc(&dVl,8),"aVl");  chk(cuMemAlloc(&dPr,8),"aPr");

    unsigned char zA[16]={0}, zB[8]={0};               // A=0 (f16 zeros), B=0
    float Cv[4]={1.f,2.f,3.f,4.f}, epsA=0.5f, epsB=0.25f, Ce=-1.f;
    unsigned long long vld=1ULL, prv=0x5ULL;
    chk(cuMemcpyHtoD(dAv,zA,16),"hAv"); chk(cuMemcpyHtoD(dAe,&epsA,4),"hAe");
    chk(cuMemcpyHtoD(dBv,zB,8),"hBv");  chk(cuMemcpyHtoD(dBe,&epsB,4),"hBe");
    chk(cuMemcpyHtoD(dCv,Cv,16),"hCv"); chk(cuMemcpyHtoD(dCe,&Ce,4),"hCe");
    chk(cuMemcpyHtoD(dVl,&vld,8),"hVl"); chk(cuMemcpyHtoD(dPr,&prv,8),"hPr");

    void* args[8]={&dAv,&dAe,&dBv,&dBe,&dCv,&dCe,&dVl,&dPr};
    chk(cuLaunchKernel(fn, 1,1,1, 32,1,1, 0,0, args, 0),"launch");
    chk(cuCtxSynchronize(),"sync");

    float Dout[4]; float Ceout; unsigned long long prout;
    chk(cuMemcpyDtoH(Dout,dCv,16),"dCv"); chk(cuMemcpyDtoH(&Ceout,dCe,4),"dCe");
    chk(cuMemcpyDtoH(&prout,dPr,8),"dPr");

    printf("D (expect 1,2,3,4): %.4f %.4f %.4f %.4f\n",Dout[0],Dout[1],Dout[2],Dout[3]);
    printf("eps_C (expect sqrt(7)=%.5f): %.5f\n", sqrtf(7.f), Ceout);
    printf("prov (expect 0x5): 0x%llx\n", prout);

    int okD = (Dout[0]==1.f&&Dout[1]==2.f&&Dout[2]==3.f&&Dout[3]==4.f);
    int okE = fabsf(Ceout - sqrtf(7.f)) < 1e-3f;
    int okP = (prout==0x5ULL);
    printf("RESULT: dataPath=%s epistemicShadow=%s provenance=%s\n",
           okD?"PASS":"FAIL", okE?"PASS":"FAIL", okP?"PASS":"FAIL");
    return (okD&&okE&&okP)?0:2;
}
