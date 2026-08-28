// Driver-API harness: load Sounio-emitted PTX, launch sounio_ossm_oct_cell, compare full O-SSM
// forward cell (linear + cubic sigmoid + C output + T-step recurrence) vs scalar reference on GB10.
#include <cstdio>
#include <cmath>
#include <cuda.h>
#define T 6
#define NB 16
static int cds(int a,int b,int bits){ int s=1;
    while(bits>0){ if(a==0||b==0)return s; if(bits==1)return -s;
        int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
        if(!ah&&!bh){a=al;b=bl;} else if(!ah&&bh){a=bl;b=al;}
        else if(ah&&!bh){ if(bl==0){a=al;b=0;} else{s=-s;a=al;b=bl;} }
        else { if(bl==0){s=-s;a=0;b=al;} else{a=bl;b=al;} } bits--; } return s; }
static float sigc(float x){ return 0.015625f*x*x*x + 0.25f*x + 0.5f; }
static void om(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cds(i,j,3)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);printf("ERR %s @%d: %s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int argc,char**argv){
    const char* ptx = argc>1?argv[1]:"/tmp/sounio_ossm_cell.ptx";
    float A[8]={0.5f,-0.2f,0.3f,0,0.1f,0,-0.4f,0.2f}, B[8]={0.1f,0.2f,-0.1f,0.3f,0,0.1f,0,-0.2f};
    float C[8]={0.3f,0.1f,-0.2f,0.15f,-0.05f,0.2f,0.1f,-0.1f};
    float x[T*NB], H0[NB*8], Yref[T*NB];
    for(int b=0;b<NB;b++){ for(int k=0;k<8;k++) H0[b*8+k]=((b+1)*0.15f+k*0.07f)*((k%2)?-1:1); }
    for(int s=0;s<T;s++)for(int b=0;b<NB;b++) x[s*NB+b]=0.3f+0.05f*b-0.02f*s;
    float Href[NB*8]; for(int i=0;i<NB*8;i++)Href[i]=H0[i];
    for(int s=0;s<T;s++){ const float* xt=&x[s*NB];
        for(int b=0;b<NB;b++){ float ah[8]; om(A,&Href[b*8],ah); float ht[8];
            for(int k=0;k<8;k++) ht[k]=sigc(ah[k]+B[k]*xt[b]);
            for(int k=0;k<8;k++) Href[b*8+k]=ht[k];
            float ch[8]; om(C,ht,ch); Yref[s*NB+b]=ch[0]; } }
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
    CUmodule mod; CK(cuModuleLoad(&mod,ptx)); CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"sounio_ossm_oct_cell"));
    CUdeviceptr dA,dB,dC,dx,dH,dY;
    CK(cuMemAlloc(&dA,8*4));CK(cuMemAlloc(&dB,8*4));CK(cuMemAlloc(&dC,8*4));
    CK(cuMemAlloc(&dx,T*NB*4));CK(cuMemAlloc(&dH,NB*8*4));CK(cuMemAlloc(&dY,T*NB*4));
    CK(cuMemcpyHtoD(dA,A,8*4));CK(cuMemcpyHtoD(dB,B,8*4));CK(cuMemcpyHtoD(dC,C,8*4));
    CK(cuMemcpyHtoD(dx,x,T*NB*4));CK(cuMemcpyHtoD(dH,H0,NB*8*4));
    int Tn=T; void* args[]={&dA,&dB,&dC,&dx,&dH,&dY,&Tn};
    CK(cuLaunchKernel(fn,1,1,1, 32,1,1, 0,0,args,0)); CK(cuCtxSynchronize());
    float Y[T*NB]; CK(cuMemcpyDtoH(Y,dY,T*NB*4));
    int fails=0; float mx=0;
    for(int i=0;i<T*NB;i++){ float e=fabsf(Y[i]-Yref[i]); if(e>mx)mx=e; if(e>0.02f)fails++; }
    printf("Sounio O-SSM full cell PTX GB10: T=%d batch=%d  y mismatch %d/%d maxerr=%.4f\n",T,NB,fails,T*NB,mx);
    if(!fails){ printf("PASS: Sounio-emitted tensor-core full O-SSM octonion cell matches scalar reference on GB10\n"); return 0; }
    printf("FAIL\n"); return 1;
}
